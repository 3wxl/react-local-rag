import { splitQuestion, isEnoughInfo } from "./agentPlanner";
import { searchTopK } from "../utils/embeddingClient";
import {
  generateAnswer,
  type GenerateCallbacks,
  type GenerateHandle,
} from "../utils/generateAnswer";
import { verifyAnswer, type VerificationResult } from "../utils/verifyAnswer";
import type { HistoryTurn } from "../utils/history";
import { startTimer } from "../utils/perf";
import { DEFAULT_AGENT_CONFIG } from "./types";
import type { AgentStep } from "./types";

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
    // 1. 拆分子问题（0.5B 乱格式时 splitQuestion 内部兜底为 [原问题]）
    const tPlan = startTimer("agent-plan");
    const split = await splitQuestion(question, callbacks.onLoadProgress); //调用`agentPlanner.splitQuestion`，传入用户问题，模型加载进度回调透传
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
    const collected = new Set<string>(); // **chunk 去重**。按文本内容去重，避免多次检索拿到一模一样的片段，浪费上下文窗口。
    let enough = false;
    let iterations = 0;

    for (const subQ of subQuestions) {
      ensureAlive();
      iterations += 1;
      const tIter = startTimer("agent-iterate");

      // 2a. 混合检索（向量 + BM25 RRF；向量不可用时 embeddingClient 内部降级纯 BM25）
      const search = await searchTopK(indexId, subQ, topK, {
        onLoadProgress: callbacks.onLoadProgress,
      });
      ensureAlive();

      const newChunks: string[] = [];
      for (const hit of search.hits) {
        if (!collected.has(hit.content)) {
          collected.add(hit.content);
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
      const chunksArr = [...collected];
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
    const chunks = [...collected];
    emit({
      type: "synthesize",
      thinking: forcedStop
        ? `已达最大轮次 ${maxIterations} 仍判定信息不足，强制终止检索，使用现有 ${chunks.length} 条片段生成答案`
        : `信息充足，汇总 ${chunks.length} 条片段生成最终答案`,
    });

    // 3. 汇总片段调 LLM 流式生成最终答案（沿用原普通 RAG 的生成器与思考流）
    emit({ type: "generate", subQuestion: question });
    genHandle = generateAnswer(
      question,
      chunks,
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

    // 4. 幻觉后处理校验（确定性 JS 校验，不靠模型自觉）
    const verification = await verifyAnswer(indexId, answer, chunks.join("\n"));

    return {
      answer,
      verification,
      steps,
      chunks,
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
