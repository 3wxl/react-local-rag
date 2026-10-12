import { splitQuestion, isEnoughInfo, condenseQuery } from "./agentPlanner";
import {
  searchTopK,
  releaseReranker,
  unloadEmbeddingModel,
} from "../utils/embeddingClient";
import type { SearchHit } from "../utils/retrieval";
import {
  buildEvidence,
  toVerifyPassages,
  type ContextRef,
} from "../utils/citation";
import {
  generateAnswer,
  type GenerateCallbacks,
  type GenerateHandle,
} from "../utils/generateAnswer";
import {
  verifyAnswer,
  buildRefusalResult,
  NO_RELEVANT_ANSWER,
  type VerificationResult,
} from "../utils/verifyAnswer";
import type { HistoryTurn } from "../utils/history";
import { startTimer, recordSearchResult } from "../utils/perf";
import {
  requestCloudPlanner,
  requestCloudAnswer,
  streamCloudAnswer,
  type CloudPlannerConfig,
  type CloudFallbackReason,
} from "./cloudPlanner";
import { mergeResult } from "./mergeResult";
import { DEFAULT_AGENT_CONFIG } from "./types";
import type { AgentStep, AgentStepOrigin, CloudAgentCommand } from "./types";

/** runSelfRagAgent 最终产出 */
export interface SelfRagAgentResult {
  /** 最终答案原文 */
  answer: string;
  /** 幻觉校验结果（逐句高亮用） */
  verification: VerificationResult;
  /** Agent 执行步骤链，供 UI 展示思考过程 */
  steps: AgentStep[];
  /** 实际用于生成的去重后片段（也是校验上下文） */
  chunks: string[];
  /** 编号引用卡片（与答案角标 [n] 一一对应，供 UI 渲染与点击定位） */
  contexts: ContextRef[];
  /** 实际执行的「检索-判断」轮数 */
  iterations: number;
  /** 是否达到最大轮次仍判为不足（强制终止） */
  forcedStop: boolean;
  /** 子问题拆分是否走了兜底（模型乱格式 → [原问题]） */
  splitFallback: boolean;
}
/**
 * 模仿`generateAnswer`的设计：返回一个对象，包含 promise + cancel 取消函数。
UI 层拿到这个 handle：`handle.promise`等待结果，用户点停止按钮调用`handle.cancel()`
 */
export interface SelfRagAgentHandle {
  promise: Promise<SelfRagAgentResult>;
  cancel: () => void;
}
/**
 * 继承普通 RAG 生成回调（onLoadProgress、onToken、onGenerating 等），额外增加`onStep`。
 **onStep：每走完一个 Agent 步骤就触发一次**，UI 实时渲染 Agent 思考面板（展示：正在拆分问题、检索到哪些片段、判断信息是否充足）。
 */
export interface SelfRagCallbacks extends GenerateCallbacks {
  /** 每完成一个步骤回调一次（UI 实时渲染思考链） */
  onStep?: (step: AgentStep) => void;
}

export interface SelfRagOptions {
  /** 每轮混合检索的 Top-K，默认 3 */
  topK?: number;
  /** 最大「检索-判断」轮数（防死循环硬上限），默认 3 */
  maxIterations?: number;
  /** 多轮对话：最近历史 */
  history?: HistoryTurn[];
  /** 多轮对话：滚动摘要 */
  historySummary?: string;
  /** 是否启用 reranker 两阶段精排（默认关闭，见 useAgentSettings.DEFAULT_ENABLE_RERANK） */
  rerank?: boolean;
  /**
   * 已改写好的检索用问题。由上层（Hybrid Agent）先改写后透传，
   * 避免同一问题被重复改写两次（多一次 LLM 推理）。
   */
  retrievalQuery?: string;
}

/**
 * Self-RAG Agent 主循环：
 * 拆分子问题 → 逐个子问题混合检索 → 收集去重片段 → 判断信息是否充足
 *   充足 → 汇总片段调 LLM 生成答案 → verifyAnswer 逐句校验
 *   不足 → 下一个子问题；达到最大轮次强制终止
 *
 * 纯编排函数，不触碰 React/IndexedDB，可独立调用与测试。
 */
export function runSelfRagAgent(
  indexId: string, // 知识库索引ID，用于searchTopK检索
  question: string, // 用户原始提问
  callbacks: SelfRagCallbacks = {},
  options: SelfRagOptions = {},
): SelfRagAgentHandle {
  const topK = options.topK ?? 3;
  const maxIterations = Math.max(
    1,
    options.maxIterations ?? DEFAULT_AGENT_CONFIG.maxIterations,
  );

  const steps: AgentStep[] = [];
  let stepSeq = 0;
  const emit = (step: Omit<AgentStep, "stepIndex" | "timestamp">) => {
    //**步骤发射器**。接收不带序号、时间的 step 对象自动加上自增 stepIndex、时间戳；存入 steps 数组，同时调用`onStep`推送给 UI 实时渲染。
    const full: AgentStep = {
      ...step,
      stepIndex: ++stepSeq,
      timestamp: Date.now(),
    };
    steps.push(full); //保存全部步骤，最后塞进返回结果，供 UI 查看完整思考链路。
    callbacks.onStep?.(full);
  };
  /**
 * 取消逻辑说明（注释很关键）：
splitQuestion /isEnoughInfo 调用的 llm worker 都是**一次性短任务**，就算中途 cancel，worker 还是会跑完；代码只是丢弃返回结果。
只有最后的`generateAnswer`流式生成长任务，可以调用`genHandle.cancel()`立刻中断模型。
 */
  let cancelled = false;
  let genHandle: GenerateHandle | null = null; //保存最终答案流式生成句柄，cancel 的时候用来终止 LLM 生成
  const ensureAlive = () => {
    if (cancelled) throw new Error("Agent 已取消");
  }; //每次异步操作前调用，一旦用户取消，直接抛异常中断整个流程

  const promise = (async (): Promise<SelfRagAgentResult> => {
    // 0. 多轮追问改写：把「它的参数呢？」补全成独立问题，**只用于检索**
    //    （生成仍用原问题：用户看到的答案要对应用户真正问的那句话）
    //    无历史 / 问题本身已独立时 condenseQuery 内部直接短路，不额外推理
    //    上层（Hybrid）已改写时直接用透传结果，不重复改写
    let retrievalQuestion = options.retrievalQuery;
    if (!retrievalQuestion) {
      const condensed = await condenseQuery(
        question,
        options.history ?? [],
        callbacks.onLoadProgress,
      );
      ensureAlive();
      if (!condensed.fallback && condensed.query !== question) {
        emit({
          type: "plan",
          subQuestion: question,
          thinking: `追问改写：结合最近对话把问题补全为「${condensed.query}」（仅用于检索）`,
        });
      }
      retrievalQuestion = condensed.query;
    }

    // 1. 拆分子问题（0.5B 乱格式时 splitQuestion 内部兜底为 [原问题]）
    const tPlan = startTimer("agent-plan");
    const split = await splitQuestion(retrievalQuestion, callbacks.onLoadProgress); //调用`agentPlanner.splitQuestion`，传入用户问题（已改写则用改写句），模型加载进度回调透传
    ensureAlive();
    // 子问题数同时受拆分上限（3）与最大迭代轮次约束
    const subQuestions = split.subQuestions.slice(0, maxIterations); //限制子问题数量，不会超过最大检索轮次，防止子问题太多
    tPlan.done({
      subQuestions: subQuestions.length,
      fallback: split.fallback ? 1 : 0,
    });
    emit({
      type: "plan", //emit `plan`类型步骤，UI 展示：问题拆分结果，是否触发兜底
      subQuestion: question,
      thinking: `拆分为 ${subQuestions.length} 个子问题${
        split.fallback ? "（模型输出无法解析，回退为原问题）" : ""
      }：${subQuestions.map((q, i) => `\n${i + 1}. ${q}`).join("")}`,
    });

    // 2. 逐子问题：检索 → 收集 → 充足性判断
    //    按文本内容去重，避免多次检索拿到一模一样的片段浪费上下文窗口；
    //    保留整条命中（而非只留文本），后面要拿它的 chunkIds/页码/章节做引用与校验范围
    const collected = new Map<string, SearchHit>();
    let enough = false;
    let iterations = 0;

    for (const subQ of subQuestions) {
      ensureAlive();
      iterations += 1;
      const tIter = startTimer("agent-iterate");

      // 2a. 混合检索（向量 + BM25 RRF；向量不可用时 embeddingClient 内部降级纯 BM25）
      const search = await searchTopK(indexId, subQ, topK, {
        onLoadProgress: callbacks.onLoadProgress,
        rerank: options.rerank,
      });
      ensureAlive();
      recordSearchResult(search);

      const newChunks: string[] = [];
      for (const hit of search.hits) {
        if (!collected.has(hit.content)) {
          collected.set(hit.content, hit);
          newChunks.push(hit.content);
        }
      }
      emit({
        type: "retrieve",
        subQuestion: subQ,
        retrievedChunks: search.hits.map((h) => h.content),
        thinking: `第 ${iterations} 轮检索（${search.mode}）命中 ${search.hits.length} 条，新增去重片段 ${newChunks.length} 条，累计 ${collected.size} 条`,
      }); //emit `retrieve`步骤：UI 展示当前子问题检索结果、命中片段数量、新增片段数量。

      // 2b. 信息充足性判断（本地模型，JS 兜底提取「足够/不足」）
      const chunksArr = [...collected.keys()];
      const judge = await isEnoughInfo(
        question,
        chunksArr,
        callbacks.onLoadProgress,
      ); /*⚠️重点：评估永远基于【原始用户问题】，不是子问题！判断的是：**现在手里全部资料能不能回答用户最开始的问题**，不是判断能不能回答这个子问题。*/
      ensureAlive();
      enough = judge.result === "enough"; //如果`enough=true`：直接 break 跳出循环，不再继续检索剩下子问题，节省轮次。
      emit({
        type: "evaluate",
        subQuestion: subQ,
        thinking: `信息充足性判断：${judge.result === "enough" ? "足够" : "不足"}${
          judge.fallback
            ? "（模型输出无法解析，JS 兜底判定为不足，继续检索）"
            : ""
        }`,
      });
      tIter.done({
        iteration: iterations,
        enough: enough ? 1 : 0,
        chunks: collected.size,
      });
      /*循环终止两种情况：
1. isEnoughInfo 返回 enough → 主动退出循环（信息足够）
2. 遍历完所有子问题（等于跑满 maxIterations）→ 强制退出 */
      if (enough) break;
      // 不足 → 继续下一个子问题
    }

    // 达到最大轮次仍不足：强制终止，用已收集的全部片段作答（模型会据此拒答或部分回答）
    const forcedStop = !enough;
    const hits = [...collected.values()];
    const chunks = hits.map((h) => h.content);

    // 相关度门限把全部候选都滤掉了：直接走「文档未提及」拒答分支，
    // 不再硬喂噪声片段给模型（省一次完整生成，也避免模型被无关片段带偏）
    if (chunks.length === 0) {
      emit({
        type: "synthesize",
        thinking:
          "全部子问题检索后无相关片段（相关度低于门限），直接按「文档中没有找到相关内容」拒答，不再调用模型生成",
      });
      return {
        answer: NO_RELEVANT_ANSWER,
        verification: buildRefusalResult(NO_RELEVANT_ANSWER),
        steps,
        chunks: [],
        contexts: [],
        iterations,
        forcedStop: true,
        splitFallback: split.fallback,
      };
    }

    // 检索已结束、马上加载本地 Qwen：先释放精排模型（常驻 300~400MB），
    // 避免与 Qwen 争 WASM 堆触发 OrtRun std::bad_alloc
    if (options.rerank) await releaseReranker().catch(() => {});

    // 检索全部结束、chunk 已拿到：卸载向量模型（BGE）权重，把 WASM 堆让给本地 Qwen。
    // 此刻起只依赖 LLM 流式生成，不再需要向量模型，卸载不影响继续吐 token。
    // clearIndex=false 只释放权重、保留向量索引；生成结束后的 verifyAnswer 再用时
    // getEmbedder 会按需自动重新加载模型（索引无需重建），而那时流式回答已输出完毕。
    // 卸载失败不阻断生成。
    try {
      await unloadEmbeddingModel(false);
    } catch {
      /* 卸载失败不阻断生成 */
    }

    emit({
      type: "synthesize",
      thinking: forcedStop
        ? `已达最大轮次 ${maxIterations} 仍判定信息不足，强制终止检索，使用现有 ${chunks.length} 条片段生成答案`
        : `信息充足，汇总 ${chunks.length} 条片段生成最终答案`,
    });

    // 3. 汇总命中转编号段落（每段带 [n] 文件名·章节·页码），调 LLM 流式生成最终答案
    const { passages, contexts } = buildEvidence(hits);
    emit({ type: "generate", subQuestion: question });
    genHandle = generateAnswer(
      question,
      passages,
      {
        onLoadProgress: callbacks.onLoadProgress,
        onGenerating: callbacks.onGenerating,
        onThinking: callbacks.onThinking,
        onToken: callbacks.onToken,
      },
      { history: options.history, historySummary: options.historySummary },
    );
    const answer = await genHandle.promise;
    genHandle = null;
    ensureAlive();

    // 4. 幻觉后处理校验（确定性 JS 校验，不靠模型自觉）。
    //    语义比对范围收敛到「答案实际引用的段落 + 其窗口」，不再对全篇索引取 max。
    const verification = await verifyAnswer(indexId, answer, chunks.join("\n"), {
      passages: toVerifyPassages(contexts),
    });

    return {
      answer,
      verification,
      steps,
      chunks,
      contexts,
      iterations,
      forcedStop,
      splitFallback: split.fallback,
    };
  })();

  return {
    promise,
    cancel: () => {
      cancelled = true;
      // 规划/判断阶段的 worker 是非流式短任务，会自行结束并被忽略结果；
      // 最终生成是长任务，立即终止
      genHandle?.cancel();
    },
  };
}

/* ──────────────────────────────────────────────────────────────────
 *  混合 Agent（阶段三）
 *  本地轻量检索取信号（不取片段原文上云）→ 云端规划 → 三指令分支：
 *    LOCAL_KNOWLEDGE  → runSelfRagAgent
 *    GENERAL_KNOWLEDGE→ 云端直答（不读本地文档）；云端异常降级本地
 *    MIXED            → 本地 Self-RAG 与云端通用问答并行，mergeResult 合并
 *  规划器本身任何异常都会降级 LOCAL_KNOWLEDGE，因此最坏情况等价本地 Self-RAG。
 * ────────────────────────────────────────────────────────────────── */

/** runHybridAgent 最终产出 */
export interface HybridAgentResult {
  /** 云端规划器实际下发（或降级后采用）的指令 */
  command: CloudAgentCommand;
  /** 主答案文本：LOCAL/MIXED=本地答案；GENERAL=云端答案 */
  answer: string;
  /** 云端部分文本（MIXED 的拓展 / GENERAL 的全部），UI 须带「无本地依据」警示块 */
  cloudContent?: string;
  /** 本地部分的幻觉校验结果（纯云端时为 undefined，不执行校验） */
  verification?: VerificationResult;
  /** 本地实际使用的片段（context 展示；纯云端时为空数组） */
  chunks: string[];
  /** 本地答案的编号引用卡片（纯云端时为空数组） */
  contexts: ContextRef[];
  /** 完整步骤链 */
  steps: AgentStep[];
  /** 本地 Self-RAG 实际迭代轮数（纯云端为 0） */
  iterations: number;
  /** 云端规划是否降级（true=回退到本地链路） */
  plannerFallback: boolean;
  /** 规划降级原因（plannerFallback=true 时有值） */
  plannerFallbackReason?: CloudFallbackReason;
  /** 云端直答/拓展是否失败被忽略（MIXED 云端挂掉时为 true，仅交付本地部分） */
  cloudFailed?: boolean;
}

export interface HybridAgentHandle {
  promise: Promise<HybridAgentResult>;
  cancel: () => void;
}

export interface HybridAgentOptions extends SelfRagOptions {
  /** 云端连接配置（apiKey / baseUrl / model） */
  cloud: CloudPlannerConfig;
}

export function runHybridAgent(
  indexId: string,
  question: string,
  callbacks: SelfRagCallbacks = {},
  options: HybridAgentOptions,
): HybridAgentHandle {
  const topK = options.topK ?? 3;

  // 父级步骤链：子 Self-RAG 的步骤经 onStep 拦截后重新编号并打 origin 标记
  const steps: AgentStep[] = [];
  let stepSeq = 0;
  const emit = (
    step: Omit<AgentStep, "stepIndex" | "timestamp">,
  ): AgentStep => {
    const full: AgentStep = {
      ...step,
      stepIndex: ++stepSeq,
      timestamp: Date.now(),
    };
    steps.push(full);
    callbacks.onStep?.(full);
    return full;
  };

  let cancelled = false;
  let subHandle: SelfRagAgentHandle | null = null;
  const cloudAbort = new AbortController();
  /** 本层已改写好的检索用问题（在 promise 内赋值，供 runLocal 透传给子 Agent） */
  let hybridRetrievalQuery: string | undefined;

  const ensureAlive = () => {
    if (cancelled) throw new Error("Agent 已取消");
  };

  /**
   * 委托运行本地 Self-RAG，子步骤统一打上 origin 标记并入父链重新编号。
   * 返回 SelfRag 的结果（steps 字段以父链为准，子结果内的 steps 仅备用）。
   */
  const runLocal = (origin: AgentStepOrigin): Promise<SelfRagAgentResult> => {
    const handle = runSelfRagAgent(
      indexId,
      question,
      {
        onLoadProgress: callbacks.onLoadProgress,
        onGenerating: callbacks.onGenerating,
        onThinking: callbacks.onThinking,
        onToken: callbacks.onToken,
        onStep: (sub) => {
          // 剥去子步骤自带的序号/时间，按父链重新编号并标记来源
          const { stepIndex: _ignoredIdx, timestamp: _ignoredTs, ...rest } = sub;
          void _ignoredIdx;
          void _ignoredTs;
          emit({ ...rest, origin });
        },
      },
      {
        topK,
        maxIterations: options.maxIterations,
        history: options.history,
        historySummary: options.historySummary,
        rerank: options.rerank,
        // 改写已在 Hybrid 层做过，透传给子 Agent 避免重复推理
        retrievalQuery: hybridRetrievalQuery,
      },
    );
    subHandle = handle;
    return handle.promise;
  };

  /** 把本地 Self-RAG 结果映射为 Hybrid 结果 */
  const fromLocal = (
    command: CloudAgentCommand,
    local: SelfRagAgentResult,
    extra: Partial<HybridAgentResult> = {},
  ): HybridAgentResult => ({
    command,
    answer: local.answer,
    verification: local.verification,
    chunks: local.chunks,
    contexts: local.contexts,
    steps,
    iterations: local.iterations,
    plannerFallback: false,
    ...extra,
  });

  const promise = (async (): Promise<HybridAgentResult> => {
    // 0. 追问改写：本地预检索用改写句（含代词时），生成阶段仍用原问题
    const condensed = await condenseQuery(
      question,
      options.history ?? [],
      callbacks.onLoadProgress,
    );
    ensureAlive();
    const retrievalQuestion = condensed.query;
    hybridRetrievalQuery = retrievalQuestion;

    // 1. 本地轻量检索：只用 RRF 分数构造数字信号，片段原文不进入云端请求
    //    无文档/索引未同步时 searchTopK 会失败，兜底为空信号，让云端规划走 GENERAL_KNOWLEDGE
    let signals: { hitCount: number; topScore: number; mode: string } = {
      hitCount: 0,
      topScore: 0,
      mode: "empty",
    };
    try {
      const search = await searchTopK(indexId, retrievalQuestion, topK, {
        onLoadProgress: callbacks.onLoadProgress,
        rerank: options.rerank,
      });
      ensureAlive();
      recordSearchResult(search);
      const topHit = search.hits[0];
      signals = {
        hitCount: search.hits.length,
        topScore: topHit?.rrfScore ?? 0,
        mode: search.mode,
      };
    } catch (searchErr) {
      // 无文档或索引未同步：signals 保持空，云端规划基于 hitCount=0 走 GENERAL_KNOWLEDGE
      console.warn(
        "[Hybrid] 本地预检索失败（可能未上传文档），将走云端规划",
        searchErr,
      );
    }

    // 2. 云端规划（任何失败内部已降级为 LOCAL_KNOWLEDGE，不抛异常）
    const plan = await requestCloudPlanner(
      question,
      signals,
      options.history ?? [],
      options.cloud,
    );
    ensureAlive();
    const rawCommand: CloudAgentCommand = plan.command;

    // 保底修正：云端规划器只看到数字信号（红线：chunk 原文不能上传），
    // 无法准确判断本地是否真有用户问的内容。若本地有命中却仍被路由到纯云端，
    // 容易出现"用户问文档、云端答'你忘记附上文档了'"的尴尬。
    // 命中数 > 0 时强制把 GENERAL_KNOWLEDGE 升级为 MIXED，
    // 让本地 Self-RAG 必跑一遍给有依据的答案，云端再补充通用视角，分区合并。
    const routingOverride =
      rawCommand === "GENERAL_KNOWLEDGE" && signals.hitCount > 0
        ? ("general-to-mixed" as const)
        : undefined;
    const command: CloudAgentCommand =
      routingOverride === "general-to-mixed" ? "MIXED" : rawCommand;

    emit({
      type: "cloud-route",
      origin: "cloud-plan",
      subQuestion: question,
      thinking:
        `本地预检索（${signals.mode}）命中 ${signals.hitCount} 条，最高 RRF 分 ${signals.topScore.toFixed(4)}；` +
        `云端规划指令：${rawCommand}` +
        (plan.fallback
          ? `（云端不可用，已降级本地，原因：${plan.fallbackReason}）`
          : "") +
        (routingOverride
          ? `；本地有命中，强制升级为 MIXED 以保证本地答案参与`
          : ""),
    });

    // 3. 三指令分支
    if (command === "LOCAL_KNOWLEDGE") {
      const local = await runLocal("local-selfrag");
      return fromLocal(command, local, {
        plannerFallback: plan.fallback,
        plannerFallbackReason: plan.fallbackReason,
      });
    }

    if (command === "GENERAL_KNOWLEDGE") {
      emit({
        type: "cloud-execute",
        origin: "cloud-plan",
        subQuestion: question,
        thinking: "纯云端直答：不读取本地文档，等待云端通用知识回答…",
      });
      const cloud = await requestCloudAnswer(
        question,
        options.history ?? [],
        options.cloud,
        { mode: "general", externalSignal: cloudAbort.signal },
      );
      ensureAlive();

      // 云端异常（超时/网络/key）→ 降级本地 Self-RAG
      if (cloud.fallback) {
        emit({
          type: "cloud-execute",
          origin: "cloud-plan",
          thinking: `云端直答失败（${cloud.fallbackReason}），自动降级本地 Self-RAG`,
        });
        const local = await runLocal("local-selfrag");
        // 降级后实际走的是本地知识库，指令按实际执行链路修正
        return fromLocal("LOCAL_KNOWLEDGE", local, {
          plannerFallback: true,
          plannerFallbackReason: cloud.fallbackReason,
        });
      }

      emit({
        type: "cloud-execute",
        origin: "cloud-plan",
        thinking: `云端直答完成，共 ${cloud.text.length} 字（无本地文档依据，不执行幻觉校验）`,
      });
      return {
        command,
        answer: cloud.text,
        cloudContent: cloud.text,
        chunks: [],
        contexts: [],
        steps,
        iterations: 0,
        plannerFallback: false,
      };
    }

    // command === "MIXED"：本地 Self-RAG 与云端通用问答并行
    emit({
      type: "cloud-execute",
      origin: "mixed-cloud",
      subQuestion: question,
      thinking: "MIXED：本地 Self-RAG 与云端通用知识问答并行执行中…",
    });

    const [localSettled, cloudSettled] = await Promise.allSettled([
      runLocal("local-selfrag"),
      requestCloudAnswer(question, options.history ?? [], options.cloud, {
        mode: "mixed",
        externalSignal: cloudAbort.signal,
      }),
    ]);

    // 用户取消：两路都会中止，直接抛出
    if (cancelled) throw new Error("Agent 已取消");

    // 本地链路是 MIXED 的主体，失败必须上抛；云端失败则只交付本地部分
    if (localSettled.status === "rejected") throw localSettled.reason;
    const local = localSettled.value;

    if (cloudSettled.status === "rejected") throw cloudSettled.reason;
    const cloud = cloudSettled.value;

    if (cloud.fallback || !cloud.text) {
      emit({
        type: "cloud-execute",
        origin: "mixed-cloud",
        thinking: `云端拓展失败（${cloud.fallbackReason ?? "空响应"}），仅交付本地 Self-RAG 答案`,
      });
      return fromLocal("LOCAL_KNOWLEDGE", local, { cloudFailed: true });
    }

    emit({
      type: "cloud-execute",
      origin: "mixed-cloud",
      thinking: `云端拓展完成 ${cloud.text.length} 字，与本地答案分区合并`,
    });

    const merged = mergeResult({
      command: "MIXED",
      local: { text: local.answer, verification: local.verification },
      cloud: { text: cloud.text },
    });

    return {
      command: "MIXED",
      answer: merged.local?.text ?? "",
      cloudContent: merged.cloud.text,
      verification: merged.local?.verification,
      chunks: local.chunks,
      contexts: local.contexts,
      steps,
      iterations: local.iterations,
      plannerFallback: false,
    };
  })();

  return {
    promise,
    cancel: () => {
      cancelled = true;
      // 中止云端 fetch；本地子 Agent 同步取消
      cloudAbort.abort();
      subHandle?.cancel();
    },
  };
}

/* ------------------------------------------------------------------ */
/* 纯云端模式（cloud-only）：完全绕过本地知识库                          */
/* ------------------------------------------------------------------ */

/**
 * runCloudOnlyChat 最终产出。
 * 与 SelfRagAgentResult 字段对齐（缺 chunks / verification / iterations），
 * 让 App.tsx 调用方无需根据模式分支写不同 schema。
 */
export interface CloudOnlyChatResult {
  /** 云端流式累积的完整回答 */
  answer: string;
  /** cloud-only 不执行幻觉校验，恒为 undefined */
  verification?: VerificationResult;
  /** cloud-only 不读本地文档，chunks 恒为空 */
  chunks: string[];
  /** 步骤链（cloud-only 只有一步「云端直答」） */
  steps: AgentStep[];
  /** 降级原因（config/timeout/network/auth/http）；用户取消不在此，直接抛错 */
  fallbackReason?: CloudFallbackReason;
  /** 是否降级 */
  fallback: boolean;
}

/**
 * 纯云端聊天 Agent。
 *
 * 阶段四核心：
 * - 不加载本地 Embedding 模型 / 不加载本地 Qwen LLM
 * - 不调 searchTopK / 不读 IndexedDB 向量
 * - 不执行 verifyAnswer 幻觉校验
 * - 直接流式请求云端，token 经 onToken 回调实时推到 UI
 *
 * 红线：调用 streamCloudAnswer，内部 filterContextForCloud 剥除用户问题里
 * 任何可能的 chunk 片段，确保只发用户纯问题 + 过滤后历史。
 *
 * 失败处理（与 hybrid 不同）：
 * - 用户主动取消 → throw Error("Agent 已取消")，App.tsx catch 块识别文案置 done
 * - 其余故障 → fallback 结果返回，App.tsx 在消息气泡里展示降级原因
 *   （cloud-only 不回退本地，因为用户显式选择不读本地）
 *
 * @param question  用户提问原文（filterContextForCloud 会剥除可能的 chunk 片段）
 * @param history   过滤后的对话历史（最近 4 条 × 800 字硬截断）
 * @param callbacks 流式回调（onToken/onStep/onDone 等）
 * @param options   云端配置 + 外部取消信号
 */
export interface CloudOnlyCallbacks {
  /** 流式 token 回调（每收到一段 delta 触发一次） */
  onToken?: (delta: string) => void;
  /** 步骤回调（cloud-only 仅触发一次「云端直答」步骤） */
  onStep?: (step: AgentStep) => void;
  /** 完成回调 */
  onDone?: (result: CloudOnlyChatResult) => void;
}

export interface CloudOnlyOptions {
  /** 云端配置 */
  cloud: CloudPlannerConfig;
  /** 外部取消信号（来自 UI 停止按钮） */
  externalSignal?: AbortSignal;
}

export interface CloudOnlyChatHandle {
  promise: Promise<CloudOnlyChatResult>;
  cancel: () => void;
}

export function runCloudOnlyChat(
  question: string,
  history: HistoryTurn[],
  callbacks: CloudOnlyCallbacks,
  options: CloudOnlyOptions,
  contextChunks: string[] = [],
): CloudOnlyChatHandle {
  let cancelled = false;
  const controller = new AbortController();
  const steps: AgentStep[] = [];
  let stepCounter = 0;

  const emit = (
    partial: Omit<AgentStep, "stepIndex" | "timestamp">,
  ): AgentStep => {
    stepCounter += 1;
    const step: AgentStep = {
      ...partial,
      stepIndex: stepCounter,
      timestamp: Date.now(),
    };
    steps.push(step);
    callbacks.onStep?.(step);
    return step;
  };

  const promise = (async (): Promise<CloudOnlyChatResult> => {
    const tStream = startTimer("agent-iterate");
    emit({
      type: "cloud-execute",
      origin: "cloud-plan",
      subQuestion: question,
      thinking:
        contextChunks.length > 0
          ? `云端 RAG：本地已检索到 Top-K ${contextChunks.length} 个相关分片，将这些分片发送云端，由云端大模型基于分片作答（不加载本地 Qwen、不做本地幻觉校验）…`
          : "云端直答：未上传文档，直接基于通用知识与对话历史作答…",
    });

    const result = await streamCloudAnswer(
      question,
      history,
      options.cloud,
      {
        externalSignal: controller.signal,
        onToken: (delta) => {
          if (!cancelled) callbacks.onToken?.(delta);
        },
      },
      contextChunks,
    );

    if (result.fallback) {
      tStream.done({ fallback: result.fallbackReason ?? "unknown" });
      emit({
        type: "cloud-execute",
        origin: "cloud-plan",
        subQuestion: question,
        thinking: `云端 RAG 失败（${result.fallbackReason}）；cloud-only 不回退本地，已停止`,
      });
      return {
        answer: "",
        chunks: [],
        steps,
        fallback: true,
        fallbackReason: result.fallbackReason,
      };
    }

    tStream.done({ chars: result.text.length });
    emit({
      type: "cloud-execute",
      origin: "cloud-plan",
      subQuestion: question,
      thinking:
        contextChunks.length > 0
          ? `云端 RAG 完成，共 ${result.text.length} 字（云端基于本地检索的 Top-K 分片作答，不做本地幻觉校验）`
          : `云端直答完成，共 ${result.text.length} 字（基于通用知识，无文档依据）`,
    });

    return {
      answer: result.text,
      chunks: [],
      steps,
      fallback: false,
    };
  })();

  return {
    promise,
    cancel: () => {
      cancelled = true;
      controller.abort();
    },
  };
}
