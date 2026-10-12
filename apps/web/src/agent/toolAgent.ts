/**
 * 工具智能体（tool-agent 模式）：云端模型经 OpenAI tool-calls 协议驱动一个
 * 「检索规划 → 本地执行 → 结果回喂 → 再规划」的循环，直到模型给出最终答复。
 *
 * 三个工具全部在**本地**执行（索引在浏览器 IndexedDB 里，云端拿不到）：
 * - search_docs：混合检索（向量+BM25 RRF），复用线上同一套 postProcessRetrieval
 * - doc_outline：从本地分块表生成文档大纲（标题 / 页码范围 / 块数）
 * - calculator：纯函数四则运算求值器（无 eval / new Function）
 *
 * 隐私红线（与原 hybrid-agent 一致，按模式区分）：
 * - **元数据模式（默认）**：发给云端的工具结果只有元数据摘要
 *   （命中条数 / 文档名 / 章节 / 页码 / 相关度），**不含片段原文**；
 *   云端只负责规划检索，最终答案由本地 Qwen 基于实际检索到的片段生成。
 * - **同意模式（用户显式开启）**：search_docs 的编号段落原文随 tool 消息上云，
 *   云端直接作答 —— 与 cloud-only 模式同一性质的知情同意例外。
 *
 * 降级契约：云端失败 → 返回 fallback 结果（不抛异常），由 UI 提示；
 * 仅用户主动取消抛 "Agent 已取消"。
 */

import { loadVectorIndex } from "../utils/db";
import {
  searchTopK,
  releaseReranker,
  unloadEmbeddingModel,
} from "../utils/embeddingClient";
import { evaluateExpression } from "../utils/calculator";
import {
  buildEvidence,
  buildCitationLabel,
  pageLabel,
  toVerifyPassages,
  type AnswerPassage,
  type ContextRef,
} from "../utils/citation";
import { generateAnswer, type GenerateHandle } from "../utils/generateAnswer";
import {
  buildRefusalResult,
  NO_RELEVANT_ANSWER,
  verifyAnswer,
  stripThinkTags,
  type VerificationResult,
} from "../utils/verifyAnswer";
import {
  filterContextForCloud,
  filterHistoryForCloud,
  requestToolStep,
  type CloudChatMessage,
  type CloudFallbackReason,
  type CloudPlannerConfig,
  type CloudToolCall,
  type CloudToolSpec,
  type FetchLike,
  type ToolStepResult,
} from "./cloudPlanner";
import {
  TOOL_AGENT_CONSENT_PROMPT,
  TOOL_AGENT_METADATA_PROMPT,
} from "./promptTemplates";
import type { AgentStep } from "./types";
import type { SearchHit } from "../utils/retrieval";
import type { HistoryTurn } from "../utils/history";
import type { VectorChunk } from "../types/doc";

/* ------------------------------------------------------------------ */
/* 工具声明（发给云端模型的 JSON Schema）                                */
/* ------------------------------------------------------------------ */

/**
 * 工具白名单。与 proxy.mjs 里的 TOOL_WHITELIST 必须一致
 * （代理侧无法 import TS，是刻意的重复；改这里要同步改 proxy.mjs）。
 */
export const TOOL_SPECS: CloudToolSpec[] = [
  {
    type: "function",
    function: {
      name: "search_docs",
      description:
        "在用户本地文档库中做混合检索（向量+关键词），返回与查询最相关的片段" +
        "（元数据模式只返回摘要信息；同意模式返回带编号的正文段落）。" +
        "多个主题请分多次调用，换用不同的关键词。",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "检索关键词或完整问题，尽量只用名词/术语/数字",
          },
          top_k: {
            type: "integer",
            minimum: 1,
            maximum: 10,
            description: "返回条数，默认 3",
          },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "doc_outline",
      description:
        "查看本地文档的结构大纲（文档名 / 章节标题 / 页码范围 / 块数），" +
        "用于确定该往哪些章节检索。不含正文内容。",
      parameters: {
        type: "object",
        properties: {
          doc_name: {
            type: "string",
            description: "只看某一篇文档时给文档名；省略则列出全部文档",
          },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "calculator",
      description:
        "算术计算器：支持 + - * / % ^（取模、乘方）与括号，例如 (5200/1.8)*100。",
      parameters: {
        type: "object",
        properties: {
          expression: {
            type: "string",
            description: "要计算的算式，只含数字、运算符与括号",
          },
        },
        required: ["expression"],
      },
    },
  },
];

/** 单轮循环内最多收集的片段数（去重后按发现顺序截断；防把本地 Qwen 上下文撑爆） */
const MAX_COLLECTED_HITS = 12;
/** 大纲输出的字符上限（防止巨型文档把一整轮上下文占满） */
const MAX_OUTLINE_CHARS = 3500;
/** 默认最大工具轮次 */
const DEFAULT_MAX_ROUNDS = 5;

/* ------------------------------------------------------------------ */
/* 依赖注入（单测替身）与执行器                                          */
/* ------------------------------------------------------------------ */

/** 工具执行结果：content 是回喂给模型的工具消息文本 */
interface ToolExecResult {
  content: string;
  ok: boolean;
  /** search_docs 命中（供后续本地合成 / 引用卡片使用） */
  hits?: SearchHit[];
}

export interface ToolAgentDeps {
  /** 单步云端请求（默认包 requestToolStep；测试注入假实现） */
  step: (
    messages: CloudChatMessage[],
    round: number,
  ) => Promise<ToolStepResult>;
  /** search_docs 执行器（默认 searchTopK） */
  searchDocs: (query: string, topK: number) => Promise<SearchHit[]>;
  /** doc_outline 执行器（默认 loadVectorIndex + buildOutlineText） */
  outline: (docName?: string) => Promise<string>;
  /** 元数据模式的本地合成（默认 generateAnswer；测试注入假实现） */
  synthesize: (
    question: string,
    passages: AnswerPassage[],
    callbacks: ToolAgentCallbacks,
    opts: { history?: HistoryTurn[]; historySummary?: string },
  ) => { promise: Promise<string>; cancel: () => void };
  /** 答案依据校验（默认 verifyAnswer + 容错；测试注入假实现） */
  verify: (
    answer: string,
    contextText: string,
    passages: { index: number; chunkIds: string[] }[],
  ) => Promise<VerificationResult | undefined>;
}

/** 解析工具参数 JSON：模型的 arguments 是字符串，非法时回喂错误让它自纠 */
function parseArgs(raw: string): { ok: true; args: Record<string, unknown> } | { ok: false; error: string } {
  try {
    const parsed = JSON.parse(raw || "{}");
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      return { ok: false, error: "参数必须是 JSON 对象" };
    }
    return { ok: true, args: parsed as Record<string, unknown> };
  } catch {
    return { ok: false, error: "参数不是合法 JSON" };
  }
}

/** 字符串参数读取 + 长度裁剪 */
function argString(args: Record<string, unknown>, key: string): string {
  const v = args[key];
  return typeof v === "string" ? v.trim() : "";
}

/** 元数据摘要：只有计数/文档名/章节/页码/相关度，绝不含片段原文 */
export function buildSearchSummary(hits: SearchHit[]): string {
  return JSON.stringify(
    {
      mode: "metadata",
      count: hits.length,
      hits: hits.map((h, i) => ({
        n: i + 1,
        doc: h.docName,
        heading: h.headingPath.filter(Boolean).slice(-1)[0] ?? "正文",
        pages: h.pageStart === h.pageEnd ? `P${h.pageStart}` : `P${h.pageStart}-${h.pageEnd}`,
        relevance: Number(h.relevance.toFixed(3)),
      })),
      note: "仅返回元数据摘要，正文不回传；检索到的片段会交给本地模型生成最终答案。",
    },
    null,
    0,
  );
}

/** 同意模式：编号段落（与 buildEvidence 的编号规则一致，模型据此标注 [n]） */
export function buildNumberedPassages(hits: SearchHit[], startIndex: number): string {
  return hits
    .map((h, i) => `[${startIndex + i}] ${buildCitationLabel(h)}\n${h.content}`)
    .join("\n\n");
}

/** 文档大纲：按文档分组，标题连续段合并为一条（标题 / 页码 / 块数） */
export function buildOutlineText(chunks: VectorChunk[], docName?: string): string {
  const filtered = docName
    ? chunks.filter((c) => c.docName === docName || c.docId === docName)
    : chunks;
  if (filtered.length === 0) {
    return JSON.stringify({
      docs: [],
      note: docName ? `未找到文档「${docName}」` : "本地暂无已索引文档",
    });
  }

  const byDoc = new Map<string, VectorChunk[]>();
  for (const c of filtered) {
    const list = byDoc.get(c.docName);
    if (list) list.push(c);
    else byDoc.set(c.docName, [c]);
  }

  interface HeadingEntry {
    heading: string;
    pages: string;
    chunks: number;
    /** 该标题连续段的页码范围（合并时取两端极值） */
    pageStart: number;
    pageEnd: number;
  }
  const docs: { doc: string; totalChunks: number; headings: HeadingEntry[] }[] = [];
  for (const [name, list] of byDoc) {
    const sorted = [...list].sort((a, b) => a.ord - b.ord);
    const headings: HeadingEntry[] = [];
    let currentKey: string | null = null;
    for (const c of sorted) {
      const key = c.headingPath.join("/");
      const label = c.headingPath.filter(Boolean).slice(-1)[0] ?? "正文";
      const last = headings[headings.length - 1];
      if (key === currentKey && last) {
        last.chunks++;
        last.pageStart = Math.min(last.pageStart, c.pageStart);
        last.pageEnd = Math.max(last.pageEnd, c.pageEnd);
        last.pages = pageLabel(last.pageStart, last.pageEnd);
      } else {
        currentKey = key;
        headings.push({
          heading: label,
          pages: pageLabel(c.pageStart, c.pageEnd),
          chunks: 1,
          pageStart: c.pageStart,
          pageEnd: c.pageEnd,
        });
      }
    }
    docs.push({ doc: name, totalChunks: sorted.length, headings });
  }

  // 超长截断：从尾部丢标题（保留每篇文档至少一条），并附提示
  let note: string | undefined;
  const size = () => JSON.stringify(note ? { docs, note } : { docs }).length;
  while (size() > MAX_OUTLINE_CHARS && docs.length > 0) {
    const last = docs[docs.length - 1];
    if (last.headings.length > 1) last.headings.pop();
    else docs.pop();
    note = "大纲过长已截断；可用 doc_name 指定单篇文档查看";
  }
  return JSON.stringify(note ? { docs, note } : { docs });
}

/** 执行一次工具调用；永不抛异常（错误以工具结果回喂，模型可自纠） */
async function executeTool(
  call: CloudToolCall,
  deps: ToolAgentDeps,
  privacy: "metadata" | "consent",
  topK: number,
): Promise<ToolExecResult> {
  const name = call.function.name;
  const parsed = parseArgs(call.function.arguments);
  if (!parsed.ok) {
    return { content: JSON.stringify({ error: parsed.error }), ok: false };
  }
  const args = parsed.args;

  if (name === "search_docs") {
    const query = argString(args, "query");
    if (!query) {
      return { content: JSON.stringify({ error: "缺少 query 参数" }), ok: false };
    }
    const rawTop = typeof args.top_k === "number" ? Math.floor(args.top_k) : topK;
    const k = Math.min(10, Math.max(1, rawTop));
    try {
      const hits = await deps.searchDocs(query, k);
      // 同意模式的编号段落要等「收集去重」之后再生成（编号必须与最终引用卡片一一对应），
      // 这里先回元数据占位，调用方收编后覆写 content
      return {
        content:
          privacy === "consent"
            ? `检索到 ${hits.length} 条`
            : buildSearchSummary(hits),
        ok: true,
        hits,
      };
    } catch (err) {
      return {
        content: JSON.stringify({
          error: `检索失败：${err instanceof Error ? err.message : String(err)}`,
        }),
        ok: false,
      };
    }
  }

  if (name === "doc_outline") {
    try {
      const docName = argString(args, "doc_name") || undefined;
      return { content: await deps.outline(docName), ok: true };
    } catch (err) {
      return {
        content: JSON.stringify({
          error: `读取大纲失败：${err instanceof Error ? err.message : String(err)}`,
        }),
        ok: false,
      };
    }
  }

  if (name === "calculator") {
    const expression = argString(args, "expression");
    if (!expression) {
      return { content: JSON.stringify({ error: "缺少 expression 参数" }), ok: false };
    }
    const result = evaluateExpression(expression);
    return result.ok
      ? {
          content: JSON.stringify({ expression, result: result.formatted }),
          ok: true,
        }
      : {
          content: JSON.stringify({ expression, error: result.error }),
          ok: false,
        };
  }

  return {
    content: JSON.stringify({ error: `未知工具：${name}` }),
    ok: false,
  };
}

/* ------------------------------------------------------------------ */
/* 运行入口                                                             */
/* ------------------------------------------------------------------ */

export interface ToolAgentCallbacks {
  onLoadProgress?: (progress: number) => void;
  onGenerating?: () => void;
  onThinking?: (delta: string) => void;
  onToken?: (delta: string) => void;
  onStep?: (step: AgentStep) => void;
}

export interface ToolAgentOptions {
  indexId: string;
  question: string;
  history?: HistoryTurn[];
  historySummary?: string;
  topK?: number;
  rerank?: boolean;
  cloud: CloudPlannerConfig;
  /** true = 用户已显式同意片段随工具结果上云（同意模式） */
  allowChunksToCloud?: boolean;
  /** 最大工具轮次，默认 5 */
  maxRounds?: number;
  /** 注入 fetch（测试用） */
  fetchImpl?: FetchLike;
  /** 依赖替身（测试用） */
  deps?: Partial<ToolAgentDeps>;
}

export interface ToolAgentResult {
  /** 本地生成的答案（元数据模式）；同意模式为空串，答案在 cloudContent */
  answer: string;
  /** 同意模式：云端最终答案 */
  cloudContent?: string;
  verification?: VerificationResult;
  /** 收集到的片段窗口文本（供消息的 context 字段） */
  chunks: string[];
  contexts: ContextRef[];
  steps: AgentStep[];
  rounds: number;
  mode: "metadata" | "consent";
  /** 云端整轮失败（未进入循环或中途失败且无产出） */
  fallback: boolean;
  fallbackReason?: CloudFallbackReason;
}

export interface ToolAgentHandle {
  promise: Promise<ToolAgentResult>;
  cancel: () => void;
}

/**
 * 运行工具智能体。返回 { promise, cancel } 句柄（与 generateAnswer 契约一致）。
 */
export function runToolAgent(
  options: ToolAgentOptions,
  callbacks: ToolAgentCallbacks = {},
): ToolAgentHandle {
  const controller = new AbortController();
  // 本地生成句柄在循环后段才创建，用可变槽让 cancel() 无论何时调用都有效
  const genSlot: { handle: GenerateHandle | null } = { handle: null };
  const promise = runLoop(options, callbacks, controller.signal, (g) => {
    genSlot.handle = g;
  });
  return {
    promise,
    cancel: () => {
      controller.abort();
      genSlot.handle?.cancel();
    },
  };
}

async function runLoop(
  options: ToolAgentOptions,
  callbacks: ToolAgentCallbacks,
  signal: AbortSignal,
  setLocalGen: (g: GenerateHandle) => void,
): Promise<ToolAgentResult> {
  const mode: "metadata" | "consent" = options.allowChunksToCloud
    ? "consent"
    : "metadata";
  const topK = options.topK ?? 3;
  const maxRounds = options.maxRounds ?? DEFAULT_MAX_ROUNDS;
  const privacy = mode;

  const steps: AgentStep[] = [];
  let stepIndex = 0;
  const emit = (step: Omit<AgentStep, "stepIndex" | "timestamp" | "origin">): void => {
    const full: AgentStep = {
      ...step,
      stepIndex: ++stepIndex,
      origin: "tool-agent",
      timestamp: Date.now(),
    };
    steps.push(full);
    callbacks.onStep?.(full);
  };

  const deps: ToolAgentDeps = {
    step:
      options.deps?.step ??
      ((messages, round) =>
        requestToolStep(messages, TOOL_SPECS, options.cloud, {
          round,
          privacy,
          fetchImpl: options.fetchImpl,
          externalSignal: signal,
        })),
    searchDocs:
      options.deps?.searchDocs ??
      ((query, k) =>
        searchTopK(options.indexId, query, k, { rerank: options.rerank }).then(
          (r) => r.hits,
        )),
    outline:
      options.deps?.outline ??
      (async (docName?: string) => buildOutlineText(await loadVectorIndex(options.indexId), docName)),
    synthesize:
      options.deps?.synthesize ??
      ((question, passages, callbacks, opts) =>
        generateAnswer(question, passages, callbacks, opts)),
    verify:
      options.deps?.verify ??
      (async (answer, contextText, passages) => {
        try {
          return await verifyAnswer(options.indexId, answer, contextText, {
            passages,
          });
        } catch {
          // 校验失败不阻断答案展示（与 App 普通 RAG 分支的容错一致）
          return undefined;
        }
      }),
  };

  // 云端消息：问题与历史都先过与 hybrid 相同的内容过滤（剥除误粘贴的片段/引用块）
  const messages: CloudChatMessage[] = [
    {
      role: "system",
      content: mode === "consent" ? TOOL_AGENT_CONSENT_PROMPT : TOOL_AGENT_METADATA_PROMPT,
    },
    ...filterHistoryForCloud(options.history ?? []).map(
      (turn): CloudChatMessage => ({ role: turn.role, content: turn.content }),
    ),
    { role: "user", content: filterContextForCloud(options.question) },
  ];

  /** 收集到的片段（去重、按发现顺序，同意模式下编号与之一一对应） */
  const collected: SearchHit[] = [];
  const seen = new Set<string>();
  /** 收编新片段，返回起始编号与实际采纳的片段（编号与最终 contexts 严格一致） */
  const collectHits = (
    hits: SearchHit[],
  ): { startIndex: number; accepted: SearchHit[] } => {
    const startIndex = collected.length + 1;
    const accepted: SearchHit[] = [];
    for (const hit of hits) {
      if (collected.length >= MAX_COLLECTED_HITS) break;
      const key = hit.content;
      if (seen.has(key)) continue;
      seen.add(key);
      collected.push(hit);
      accepted.push(hit);
    }
    return { startIndex, accepted };
  };

  let rounds = 0;
  let cloudFinal = "";

  for (let round = 1; round <= maxRounds; round++) {
    rounds = round;
    emit({ type: "cloud-route", thinking: `第 ${round} 轮：请求云端规划` });

    const resp = await deps.step(messages, round);
    if (resp.fallback) {
      // 首轮就失败 = 整轮没有产出；已有片段（后面轮失败）则继续走本地合成
      if (collected.length === 0) {
        return {
          answer: "",
          chunks: [],
          contexts: [],
          steps,
          rounds,
          mode,
          fallback: true,
          fallbackReason: resp.fallbackReason,
        };
      }
      break;
    }
    if (resp.reasoning.trim()) {
      emit({ type: "cloud-route", thinking: resp.reasoning.trim().slice(0, 500) });
    }

    if (resp.toolCalls.length === 0) {
      // 模型给出最终答复（同意模式）或声明完成（元数据模式的 DONE）
      cloudFinal = stripThinkTags(resp.content || "").trim();
      break;
    }

    // assistant 消息必须原样带回 tool_calls，否则上游会 400
    messages.push({
      role: "assistant",
      content: resp.content || "",
      tool_calls: resp.toolCalls,
    });

    for (const call of resp.toolCalls) {
      emit({
        type: "tool-call",
        tool: { name: call.function.name, args: call.function.arguments.slice(0, 160) },
        thinking: `调用工具 ${call.function.name}`,
      });
      const result = await executeTool(call, deps, privacy, topK);
      if (result.hits && result.hits.length > 0) {
        const { startIndex, accepted } = collectHits(result.hits);
        if (privacy === "consent") {
          // 编号段落必须在收编去重之后生成：模型看到的 [n] 与最终引用卡片一一对应
          if (accepted.length === 0) {
            result.content = `检索到 ${result.hits.length} 条，但均与此前结果重复，不重复编号。`;
          } else {
            const truncated =
              accepted.length < result.hits.length
                ? `（本次超上限只回传 ${accepted.length} 条）`
                : "";
            result.content = `检索到 ${result.hits.length} 条${truncated}，编号段落如下：\n\n${buildNumberedPassages(accepted, startIndex)}`;
          }
        }
      }
      emit({
        type: "tool-result",
        tool: { name: call.function.name, ok: result.ok },
        thinking: result.ok ? result.content.slice(0, 300) : result.content,
        // 同意模式的工具结果本就含原文；元数据模式不展示片段
        retrievedChunks:
          mode === "consent" && result.hits
            ? result.hits.map((h) => h.content.slice(0, 200))
            : undefined,
      });
      messages.push({
        role: "tool",
        tool_call_id: call.id,
        content: result.content,
      });
    }
  }

  // ── 合成阶段 ──
  const hits = collected;
  const chunks = hits.map((h) => h.content);

  if (mode === "consent") {
    const answer = cloudFinal;
    if (!answer) {
      const refusal = NO_RELEVANT_ANSWER;
      return {
        answer: "",
        cloudContent: refusal,
        chunks,
        contexts: [],
        steps,
        rounds,
        mode,
        fallback: false,
      };
    }
    const { contexts } = buildEvidence(hits);
    const verification =
      hits.length > 0
        ? await deps.verify(answer, chunks.join("\n\n"), toVerifyPassages(contexts))
        : undefined;
    return {
      answer: "",
      cloudContent: answer,
      verification,
      chunks,
      contexts,
      steps,
      rounds,
      mode,
      fallback: false,
    };
  }

  // 元数据模式：本地 Qwen 基于收集到的片段生成最终答案
  if (hits.length === 0) {
    const refusal = NO_RELEVANT_ANSWER;
    stepIndex++;
    steps.push({
      stepIndex,
      type: "generate",
      origin: "tool-agent",
      thinking: "未检索到相关片段，直接拒答",
      timestamp: Date.now(),
    });
    return {
      answer: refusal,
      verification: buildRefusalResult(refusal),
      chunks: [],
      contexts: [],
      steps,
      rounds,
      mode,
      fallback: false,
    };
  }

  // 释放向量模型与精排（与 self-rag 相同：生成阶段不需要，且同驻留易 bad_alloc）
  try {
    await releaseReranker();
    await unloadEmbeddingModel(false);
  } catch {
    /* 释放失败不阻断 */
  }

  const { passages, contexts } = buildEvidence(hits);
  emit({ type: "generate", thinking: "本地模型基于检索片段生成答案" });
  const genHandle = deps.synthesize(options.question, passages, callbacks, {
    history: options.history,
    historySummary: options.historySummary,
  });
  setLocalGen(genHandle);
  const rawAnswer = await genHandle.promise;
  const answer = stripThinkTags(rawAnswer || "");

  const verification = await deps.verify(
    answer,
    chunks.join("\n\n"),
    toVerifyPassages(contexts),
  );

  return {
    answer,
    verification,
    chunks,
    contexts,
    steps,
    rounds,
    mode,
    fallback: false,
  };
}
