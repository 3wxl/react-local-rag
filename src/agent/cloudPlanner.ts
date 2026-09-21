/**
 * 云端规划器（混合 Agent 阶段三）。
 *
 * 核心红线：任何情况下【文档 chunk 原文 / 检索片段全文】禁止上传云端。
 * 发给云端的内容只有三类：
 *   1. 用户原始问题 question
 *   2. 本地预检索的【统计信号】（命中条数、最高相关性分、检索模式）——全部是数字/枚举
 *   3. 过滤后的对话历史（最近若干轮 + 单条长度硬截断）
 *
 * 降级契约（与项目「降级不阻断」原则一致）：
 *   - 配置缺失（无 API Key / Base URL）→ 不发请求，直接 LOCAL_KNOWLEDGE
 *   - 超时 / 网络错误 / 401·403 key 错误 / HTTP 异常 / 指令格式无法解析
 *     → 一律降级 LOCAL_KNOWLEDGE（调用方据此回退本地 Self-RAG），不抛异常
 */

import {
  CLOUD_ROUTE_SYSTEM_PROMPT,
  CLOUD_GENERAL_ANSWER_PROMPT,
  CLOUD_MIXED_ANSWER_PROMPT,
} from "./promptTemplates";
import { DEFAULT_AGENT_CONFIG } from "./types";
import type { CloudAgentCommand } from "./types";
import type { HistoryTurn } from "../utils/history";
import { DEFAULT_CLOUD_MODEL } from "./useAgentSettings";

export { DEFAULT_CLOUD_MODEL };

/** 进入云端请求的历史最大轮数（防御性硬上限，调用方通常已过滤） */
const MAX_HISTORY_TURNS = 4;
/** 单条历史进入云端请求的字符上限（防止个别长回答外泄过多 / 撑爆上下文） */
const MAX_HISTORY_TURN_CHARS = 800;
/**
 * 路由输出上限。DeepSeek 等推理模型把 max_tokens 算进 reasoning_tokens，
 * 32 会在写出完整指令前被截断（content 只剩 "GENERAL"）。
 */
const MAX_OUTPUT_TOKENS = 512;

/** 降级原因分类（供调用方打点与日志，不直接暴露给终端用户） */
export type CloudFallbackReason =
  | "config" // 云端未配置（apiKey / baseUrl 缺失），未发请求
  | "timeout" // 请求超时（AbortController 触发）
  | "network" // 网络层失败（断网、DNS、CORS 等 fetch reject）
  | "auth" // 401/403：API Key 非法或无权限
  | "http" // 其他非 2xx / 响应体不是合法 ChatCompletion
  | "format"; // 服务正常返回，但指令文本无法解析

/** 本地预检索的相关性信号：只含数字/枚举，绝不包含片段原文 */
export interface LocalRetrievalSignals {
  /** 命中片段条数 */
  hitCount: number;
  /** 最高相关性分（RRF 融合分，越大越相关） */
  topScore: number;
  /** 检索模式：hybrid / bm25（可选） */
  mode?: string;
}

/** 云端连接配置（来自设置面板，不上传任何文档） */
export interface CloudPlannerConfig {
  apiKey: string;
  /** 本机代理根地址，如 http://localhost:8787/v1 */
  baseUrl: string;
  /** 真实上游 OpenAI 兼容根地址；经 X-Upstream-Base-Url 交给代理按请求切换 */
  upstreamBaseUrl?: string;
  /** 模型名，缺省用 DEFAULT_CLOUD_MODEL */
  model?: string;
}

/** requestCloudPlanner 的可调参数 */
export interface RequestCloudPlannerOptions {
  /** 超时毫秒，默认取 DEFAULT_AGENT_CONFIG.cloudTimeoutMs */
  timeoutMs?: number;
  /** 注入 fetch 实现（单元测试用），默认全局 fetch */
  fetchImpl?: FetchLike;
}

/** 云端规划结果：永远 resolve，失败通过 fallback 表达 */
export interface CloudPlanResult {
  /** 路由指令，任何降级场景恒为 LOCAL_KNOWLEDGE */
  command: CloudAgentCommand;
  /** 云端原始输出（调试用；降级时可能为空串） */
  raw: string;
  /** 是否走了降级（true 时调用方回退本地 Self-RAG） */
  fallback: boolean;
  /** 降级原因（fallback=false 时为 undefined） */
  fallbackReason?: CloudFallbackReason;
  /** 实际 HTTP 状态码（调试用） */
  httpStatus?: number;
}

/** fetch 的结构化最小类型，便于测试替身注入 */
export type FetchLike = (
  input: string,
  init?: RequestInit,
) => Promise<Response>;

/* ------------------------------------------------------------------ */
/* 纯函数：请求体构造（export 供单元测试验证红线：body 中无 chunk 原文） */
/* ------------------------------------------------------------------ */

/** OpenAI 兼容 ChatCompletion 的消息结构 */
export interface CloudChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

/**
 * 对历史做防御性过滤：只保留最近 MAX_HISTORY_TURNS 条、单条硬截断。
 * 即便调用方误传了未过滤历史，这里也是最后一道闸门。
 */
export function filterHistoryForCloud(history: HistoryTurn[]): HistoryTurn[] {
  return history.slice(-MAX_HISTORY_TURNS).map((turn) => {
    const content =
      turn.content.length > MAX_HISTORY_TURN_CHARS
        ? turn.content.slice(0, MAX_HISTORY_TURN_CHARS)
        : turn.content;
    return { role: turn.role, content };
  });
}

/**
 * 构造发送给云端的 user 消息文本。
 * 只包含：数字检索信号 + 过滤后的历史 + 用户问题，不含任何片段原文。
 */
export function buildPlannerUserText(
  question: string,
  signals: LocalRetrievalSignals,
  history: HistoryTurn[],
): string {
  const lines: string[] = [
    "【本地预检索信号】（仅统计数字，不含任何文档原文）",
    `命中片段数：${Math.max(0, Math.floor(signals.hitCount) || 0)}`,
    `最高相关性分：${Number.isFinite(signals.topScore) ? signals.topScore.toFixed(4) : "0.0000"}`,
    `检索模式：${signals.mode || "unknown"}`,
  ];

  const filtered = filterHistoryForCloud(history);
  if (filtered.length > 0) {
    lines.push("", "【最近对话】");
    for (const turn of filtered) {
      lines.push(`${turn.role === "user" ? "用户" : "助手"}：${turn.content}`);
    }
  }

  lines.push(
    "",
    "【当前问题】",
    question,
    "",
    "请只输出指令：LOCAL_KNOWLEDGE / GENERAL_KNOWLEDGE / MIXED",
  );
  return lines.join("\n");
}

/**
 * 构造完整 ChatCompletion 请求体。
 * export 供单元测试断言：body 中只有数字信号，不出现 chunk 内容字段。
 */
export function buildPlannerRequestBody(
  question: string,
  signals: LocalRetrievalSignals,
  history: HistoryTurn[],
  model: string = DEFAULT_CLOUD_MODEL,
): {
  model: string;
  messages: CloudChatMessage[];
  temperature: number;
  max_tokens: number;
} {
  return {
    model,
    temperature: 0, // 规划任务要确定性，温度必须 0
    max_tokens: MAX_OUTPUT_TOKENS,
    messages: [
      { role: "system", content: CLOUD_ROUTE_SYSTEM_PROMPT },
      {
        role: "user",
        content: buildPlannerUserText(question, signals, history),
      },
    ],
  };
}

/* ------------------------------------------------------------------ */
/* 纯函数：指令解析（格式异常 → 降级 LOCAL_KNOWLEDGE）                  */
/* ------------------------------------------------------------------ */

/**
 * 从云端输出中提取路由指令。
 * 容错：大小写归一、允许模型在指令外加括号/前后缀，用词边界匹配。
 * 解析不出合法指令时返回 LOCAL_KNOWLEDGE + format 降级标记。
 * export 供单元测试。
 */
const COMMANDS: CloudAgentCommand[] = [
  "GENERAL_KNOWLEDGE",
  "LOCAL_KNOWLEDGE",
  "MIXED",
];

/** 被 max_tokens 截断时，单词可能只剩指令前缀（如 GENERAL） */
function matchCommandToken(token: string): CloudAgentCommand | null {
  if (token.length < 3) return null;
  const hits = COMMANDS.filter(
    (cmd) => cmd.startsWith(token) || token.startsWith(cmd),
  );
  return hits.length === 1 ? hits[0] : null;
}

export function parseCloudCommand(raw: string): {
  command: CloudAgentCommand;
  fallback: boolean;
} {
  const text = (raw || "").trim().toUpperCase();
  if (!text) return { command: "LOCAL_KNOWLEDGE", fallback: true };

  // 完整指令优先（推理正文里常写 Answer: GENERAL_KNOWLEDGE）
  if (text.includes("LOCAL_KNOWLEDGE")) {
    return { command: "LOCAL_KNOWLEDGE", fallback: false };
  }
  if (text.includes("GENERAL_KNOWLEDGE")) {
    return { command: "GENERAL_KNOWLEDGE", fallback: false };
  }
  if (text.includes("MIXED")) {
    return { command: "MIXED", fallback: false };
  }

  // 截断容错：只认唯一匹配的词，避免一句话里同时出现多个前缀时误判
  const hits = new Set<CloudAgentCommand>();
  for (const word of text.split(/[^A-Z_]+/)) {
    const matched = matchCommandToken(word);
    if (matched) hits.add(matched);
  }
  if (hits.size === 1) {
    return { command: [...hits][0], fallback: false };
  }

  // 格式异常：安全降级为本地知识库（不会错误地把问题送去云端生成）
  return { command: "LOCAL_KNOWLEDGE", fallback: true };
}

/* ------------------------------------------------------------------ */
/* 主入口：requestCloudPlanner                                          */
/* ------------------------------------------------------------------ */

/** 拼接 Base URL 与 chat/completions 路径，容忍尾部斜杠 */
function buildEndpoint(baseUrl: string): string {
  const trimmed = baseUrl.trim().replace(/\/+$/, "");
  return `${trimmed}/chat/completions`;
}

/** 代理协议头：Key + 按请求切换的上游地址 */
function buildCloudHeaders(config: {
  apiKey: string;
  upstreamBaseUrl?: string;
}): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "X-User-Api-Key": config.apiKey,
  };
  const upstream = config.upstreamBaseUrl?.trim();
  if (upstream) {
    headers["X-Upstream-Base-Url"] = upstream.replace(/\/+$/, "");
  }
  return headers;
}

/** 统一构造降级结果 */
function degraded(
  reason: CloudFallbackReason,
  raw = "",
  httpStatus?: number,
): CloudPlanResult {
  console.warn(
    `[云端规划] 降级为本地 Self-RAG，原因：${reason}`,
    httpStatus ?? "",
  );
  return {
    command: "LOCAL_KNOWLEDGE",
    raw,
    fallback: true,
    fallbackReason: reason,
    httpStatus,
  };
}

/** ChatCompletion 响应体的最小结构（只取需要的字段） */
interface ChatCompletionResponse {
  choices?: Array<{
    message?: {
      content?: string | null;
      /** DeepSeek 推理模型把思考放在此字段，content 可能被截断 */
      reasoning_content?: string | null;
    };
  }>;
}

/**
 * 请求云端规划器，返回路由指令。
 *
 * 永不抛异常：配置缺失、超时、网络故障、key 无效、HTTP 错误、格式异常
 * 全部降级为 LOCAL_KNOWLEDGE（调用方直接回退本地 Self-RAG）。
 *
 * 安全保证：方法内部不接收、不转发任何文档 chunk / 检索片段原文，
 * 检索信息仅以 { hitCount, topScore, mode } 数字信号形式出现。
 */
export async function requestCloudPlanner(
  question: string,
  signals: LocalRetrievalSignals, //本地检索信号（只有 hitCount、topScore 这类数字，没有文档片段！隐私红线）
  history: HistoryTurn[],
  config: CloudPlannerConfig, //用户在 UI 面板填写的配置 `apiKey / baseUrl / model`
  options: RequestCloudPlannerOptions = {}, //超时、自定义 fetch、测试注入参数
): Promise<CloudPlanResult> {
  // 1. 配置缺失：不发请求，直接降级（避免泄露网络错误噪音）
  const apiKey = config.apiKey?.trim();
  const baseUrl = config.baseUrl?.trim();
  if (!apiKey || !baseUrl) {
    return degraded("config");
  }

  const timeoutMs = options.timeoutMs ?? DEFAULT_AGENT_CONFIG.cloudTimeoutMs;
  const fetchImpl: FetchLike = options.fetchImpl ?? fetch; //注入 fetch，方便单元测试 mock 网络请求，不写死全局 fetch。
  const controller = new AbortController(); //现代浏览器 API，**可以中途取消 fetch 请求**；
  const timer = setTimeout(() => controller.abort(), timeoutMs); //超时自动触发 `controller.abort()` 终止请求。

  let httpStatus: number | undefined;
  try {
    const response = await fetchImpl(buildEndpoint(baseUrl), {
      // baseUrl = 本机代理；真实厂商由 X-Upstream-Base-Url 按请求切换
      method: "POST",
      headers: buildCloudHeaders({
        apiKey,
        upstreamBaseUrl: config.upstreamBaseUrl,
      }),
      body: JSON.stringify(
        buildPlannerRequestBody(
          //`buildPlannerRequestBody`：组装发给规划大模型的消息体，**这里只传 signals 数字，不传文档 chunk**（隐私核心）
          question,
          signals,
          history,
          config.model?.trim() || DEFAULT_CLOUD_MODEL,
        ),
      ),
      signal: controller.signal, //`signal: controller.signal`：把中止信号绑定 fetch，超时就断掉网络。
    });

    httpStatus = response.status;

    // 2. 401/403：API Key 非法 / 无权限 → auth 降级
    if (response.status === 401 || response.status === 403) {
      return degraded("auth", "", httpStatus);
    }

    // 3. 其他非 2xx → http 降级
    if (!response.ok) {
      return degraded("http", "", httpStatus);
    }

    // 4. 解析响应体；非法信封 → http 降级
    let data: ChatCompletionResponse;
    try {
      data = (await response.json()) as ChatCompletionResponse;
    } catch {
      return degraded("http", "", httpStatus);
    }

    const message = data.choices?.[0]?.message;
    const content = typeof message?.content === "string" ? message.content : "";
    const reasoning =
      typeof message?.reasoning_content === "string"
        ? message.reasoning_content
        : "";
    // content 被截断时，完整指令常在 reasoning_content 里
    const fromContent = content.trim() ? parseCloudCommand(content) : null;
    const fromReasoning = reasoning.trim()
      ? parseCloudCommand(reasoning)
      : null;
    const parsed =
      fromContent && !fromContent.fallback
        ? fromContent
        : fromReasoning && !fromReasoning.fallback
          ? fromReasoning
          : fromContent ?? fromReasoning;
    const raw = content || reasoning;
    if (!parsed || !raw) {
      return degraded("http", "", httpStatus);
    }
    return {
      command: parsed.command,
      raw,
      fallback: parsed.fallback,
      fallbackReason: parsed.fallback ? "format" : undefined,
      httpStatus,
    };
  } catch (err) {
    // AbortController 触发的中止 → 超时；其余 fetch reject 视为网络故障
    if (err instanceof DOMException && err.name === "AbortError") {
      return degraded("timeout");
    }
    // 部分环境（jsdom/旧浏览器）AbortError 不以 DOMException 形式出现
    if (err instanceof Error && err.name === "AbortError") {
      return degraded("timeout");
    }
    return degraded("network");
  } finally {
    clearTimeout(timer);
  }
}

/* ------------------------------------------------------------------ */
/* 云端直答：GENERAL_KNOWLEDGE 纯云端 / MIXED 的云端拓展部分            */
/* ------------------------------------------------------------------ */

/** 云端答案默认超时：长回答比路由指令给更宽的时限 */
export const DEFAULT_CLOUD_ANSWER_TIMEOUT = 60_000;
/**
 * 云端答案最大输出 token。
 * 1024 会把稍长的回答拦腰截断；4096 覆盖大多数问答场景，
 * 同时仍是 DeepSeek/OpenAI 单次响应的常见上限，不会触发上游 400。
 */
const CLOUD_ANSWER_MAX_TOKENS = 4096;

export interface RequestCloudAnswerOptions {
  /** "general"=纯云端直答；"mixed"=仅补充文档外通用知识 */
  mode: "general" | "mixed";
  /** 超时毫秒，默认 DEFAULT_CLOUD_ANSWER_TIMEOUT */
  timeoutMs?: number;
  /** 注入 fetch 实现（单元测试用） */
  fetchImpl?: FetchLike;
  /** 外部取消信号（用户点停止按钮）；触发后 promise 以取消错误 reject */
  externalSignal?: AbortSignal;
}

export interface CloudAnswerResult {
  /** 云端回答原文（fallback 时为空串） */
  text: string;
  /** 是否降级（true 时调用方回退本地） */
  fallback: boolean;
  /** 降级原因 */
  fallbackReason?: CloudFallbackReason;
  httpStatus?: number;
}

/** 判断错误是否为 abort 类（跨 DOMException / 普通 Error 环境） */
function isAbortError(err: unknown): boolean {
  return (
    (err instanceof DOMException && err.name === "AbortError") ||
    (err instanceof Error && err.name === "AbortError")
  );
}

/**
 * 请求云端大模型直接作答。
 *
 * 红线：函数不接收任何文档 chunk / 检索片段，入参只有问题 + 过滤后的历史。
 * 任何网络/鉴权/超时故障都以 fallback 结果返回（不抛异常），调用方据此降级本地；
 * 唯独「用户主动取消」会以 Error("Agent 已取消") reject，与本地 Agent 取消语义一致。
 */
export async function requestCloudAnswer(
  question: string,
  history: HistoryTurn[],
  config: CloudPlannerConfig,
  options: RequestCloudAnswerOptions,
): Promise<CloudAnswerResult> {
  const apiKey = config.apiKey?.trim();
  const baseUrl = config.baseUrl?.trim();
  if (!apiKey || !baseUrl) {
    return { text: "", fallback: true, fallbackReason: "config" };
  }

  const timeoutMs = options.timeoutMs ?? DEFAULT_CLOUD_ANSWER_TIMEOUT;
  const fetchImpl: FetchLike = options.fetchImpl ?? fetch;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  // 外部取消 → 联动中止 fetch
  /**
   * `externalSignal` 监听：
- 外部传入的中止信号，来自 UI【停止生成】按钮
- 一旦用户点击停止，触发 `onExternalAbort`，立刻终止 fetch
- 提前判断：如果传入时信号已经是 abort 状态，直接中止
   */
  const onExternalAbort = () => controller.abort();
  if (options.externalSignal?.aborted) controller.abort();
  options.externalSignal?.addEventListener("abort", onExternalAbort);

  const systemPrompt =
    options.mode === "mixed"
      ? CLOUD_MIXED_ANSWER_PROMPT
      : CLOUD_GENERAL_ANSWER_PROMPT;

  const filtered = filterHistoryForCloud(history);
  const messages: CloudChatMessage[] = [
    { role: "system", content: systemPrompt },
    ...filtered.map(
      (turn): CloudChatMessage => ({ role: turn.role, content: turn.content }),
    ),
    { role: "user", content: question },
  ];

  let httpStatus: number | undefined;
  try {
    const response = await fetchImpl(buildEndpoint(baseUrl), {
      method: "POST",
      headers: buildCloudHeaders({
        apiKey,
        upstreamBaseUrl: config.upstreamBaseUrl,
      }),
      body: JSON.stringify({
        model: config.model?.trim() || DEFAULT_CLOUD_MODEL,
        temperature: options.mode === "mixed" ? 0.3 : 0.7,
        max_tokens: CLOUD_ANSWER_MAX_TOKENS,
        messages,
      }),
      signal: controller.signal,
    });

    httpStatus = response.status;
    if (response.status === 401 || response.status === 403) {
      return { text: "", fallback: true, fallbackReason: "auth", httpStatus };
    }
    if (!response.ok) {
      return { text: "", fallback: true, fallbackReason: "http", httpStatus };
    }

    let data: ChatCompletionResponse;
    try {
      data = (await response.json()) as ChatCompletionResponse;
    } catch {
      return { text: "", fallback: true, fallbackReason: "http", httpStatus };
    }

    const content = data.choices?.[0]?.message?.content;
    const text = typeof content === "string" ? content.trim() : "";
    if (!text) {
      return { text: "", fallback: true, fallbackReason: "http", httpStatus };
    }
    return { text, fallback: false, httpStatus };
  } catch (err) {
    if (isAbortError(err)) {
      // 区分用户取消与超时：外部信号已中止 → 取消，中断整个 Agent
      if (options.externalSignal?.aborted) {
        throw new Error("Agent 已取消");
      }
      return { text: "", fallback: true, fallbackReason: "timeout" };
    }
    return { text: "", fallback: true, fallbackReason: "network" };
  } finally {
    clearTimeout(timer);
    options.externalSignal?.removeEventListener("abort", onExternalAbort);
  }
}
