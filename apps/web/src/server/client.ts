/**
 * 服务器模式客户端（M1）：把 apps/api 的 `/v1/*` 映射成与 Lite 模式**同一批类型**。
 *
 * 为什么必须映射而不是直接用返回体：
 * 引用卡片（ContextRef）与逐句校验（VerificationResult）在前端已经被
 * MessageBubble / AgentStepPanel / PerfPanel 消费，服务器模式只要产出同样的类型，
 * 那套 UI 一行都不用改 —— 这也是"双端同口径"在 UI 层的体现（字段映射表见 docs/api-contract.md 附录）。
 *
 * 契约纪律：响应体字段是 snake_case，新增字段必须容忍（宽松解析），
 * 所以这里不校验多余字段、也不因为缺可选字段就报错。
 */

import type { AgentStep, AgentStepType } from "../agent/types";
import type { ContextRef } from "../utils/citation";
import type {
  VerificationResult,
  VerifiedSentence,
  SentenceLevel,
  VerifyVerdict,
} from "../utils/verifyAnswer";

/** 服务器错误（带错误码，UI 可据此给不同提示） */
export class ServerError extends Error {
  readonly code: string;
  readonly status: number;
  readonly detail: unknown;

  constructor(code: string, message: string, status: number, detail?: unknown) {
    super(message);
    this.name = "ServerError";
    this.code = code;
    this.status = status;
    this.detail = detail;
  }
}

/** 规范化 baseUrl：去掉尾部斜杠（默认空串 = 同源，dev 走 vite 代理） */
export function normalizeBaseUrl(baseUrl: string): string {
  return (baseUrl ?? "").trim().replace(/\/+$/, "");
}

export function apiUrl(baseUrl: string, path: string): string {
  return `${normalizeBaseUrl(baseUrl)}${path}`;
}

/* ---------------- 响应类型（契约 §0.1 / §2） ---------------- */

interface RawCitation {
  index: number;
  doc_id: string;
  doc_name: string;
  heading_path: (string | null)[];
  page_start: number;
  page_end: number;
  score: number;
  relevance: number;
  chunk_ids: string[];
  window_chunk_ids: string[];
  matched_content: string;
  content: string;
}

interface RawSentence {
  text: string;
  level: SentenceLevel;
  score: number;
  semantic: number | null;
  lexical: number;
  unknown_numbers: string[];
  citations: number[];
  invalid_citations: number[];
  /** 引用强制闭环（M2）：主张句没有有效角标。老服务端不返回时按 false 处理 */
  uncited?: boolean;
  trivial: boolean;
}

interface RawConvergence {
  uncited_claims: number;
  out_of_range_claims: number;
  citations_subset_of_passages: boolean;
}

interface RawVerification {
  source: "semantic" | "lexical";
  scoped_chunk_count: number;
  convergence?: RawConvergence;
  sentences: RawSentence[];
}

export interface RawAnswerResponse {
  answer_id: string;
  trace_id: string;
  question: string;
  answer: string;
  verdict: VerifyVerdict;
  citations: RawCitation[];
  refusal: { reason: string; message: string } | null;
  verification: RawVerification;
  cost: {
    latency_ms: number;
    latency_breakdown_ms: Record<string, number>;
    tokens: { prompt: number; completion: number; total: number };
    estimated_cost_usd: number;
  };
}

export interface ServerAnswer {
  answerId: string;
  traceId: string;
  answer: string;
  verdict: VerifyVerdict;
  contexts: ContextRef[];
  verification: VerificationResult;
  refusalMessage: string | null;
  cost: RawAnswerResponse["cost"];
}

export interface ServerHealth {
  status: string;
  storage: { backend: string; documents: number; chunks: number };
  index: { version: string; state: string };
  models: { embed: string; llm: string; llm_provider: string; llm_model: string };
}

/* ---------------- 映射 ---------------- */

/**
 * 契约 Citation → Lite 模式 ContextRef。
 * 差异只有命名：heading_path 里的 null（标题层级跳跃）原样保留，
 * 因为 UI 的 headingLabel 会过滤空值，行为与本地一致。
 */
export function mapCitation(raw: RawCitation): ContextRef {
  return {
    index: raw.index,
    docId: raw.doc_id,
    docName: raw.doc_name,
    headingPath: (raw.heading_path ?? []) as string[],
    pageStart: raw.page_start,
    pageEnd: raw.page_end,
    score: raw.score,
    relevance: raw.relevance,
    chunkIds: raw.chunk_ids ?? [],
    windowChunkIds: raw.window_chunk_ids ?? [],
    matchedContent: raw.matched_content ?? "",
    content: raw.content ?? "",
  };
}

export function mapVerification(raw: RawVerification): VerificationResult {
  const sentences: VerifiedSentence[] = (raw.sentences ?? []).map((s) => ({
    text: s.text,
    level: s.level,
    score: s.score,
    semantic: s.semantic ?? null,
    lexical: s.lexical,
    unknownNumbers: s.unknown_numbers ?? [],
    citations: s.citations ?? [],
    invalidCitations: s.invalid_citations ?? [],
    uncited: s.uncited ?? false,
    trivial: s.trivial,
  }));
  const convergence = raw.convergence;
  return {
    verdict: verdictOf(sentences),
    sentences,
    // 服务端不返回 checkedAt（时间由服务端记录在 trace 里）：这里用本地时间近似，
    // 不参与任何判定，只用于 UI 排序/展示
    checkedAt: Date.now(),
    source: raw.source === "lexical" ? "lexical" : "semantic",
    scopedChunkCount: raw.scoped_chunk_count ?? 0,
    // 服务端老版本不返回收敛块：从逐句结果本地复算（同一条规则，不会漂移）
    convergence: convergence
      ? {
          uncitedClaims: convergence.uncited_claims,
          outOfRangeClaims: convergence.out_of_range_claims,
          citationsSubsetOfPassages: convergence.citations_subset_of_passages,
        }
      : {
          uncitedClaims: sentences.filter((s) => !s.trivial && s.uncited).length,
          outOfRangeClaims: sentences.filter(
            (s) => !s.trivial && s.invalidCitations.length > 0,
          ).length,
          citationsSubsetOfPassages: sentences.every(
            (s) => s.invalidCitations.length === 0,
          ),
        },
  };
}

/**
 * 由逐句结果汇总整体裁决（与 verifyAnswer 的汇总规则一致）。
 *
 * 为什么前端还要算一遍：契约里 verdict 是顶层字段，但如果服务端用的是旧版规则
 * （例如把 trivial 句算进分母），UI 的逐句高亮与整体标记就会自相矛盾。
 * 以逐句结果为准能保证"看到的每一句"和"整体结论"永远一致。
 */
export function verdictOf(sentences: VerifiedSentence[]): VerifyVerdict {
  const claims = sentences.filter((s) => !s.trivial);
  if (claims.length === 0) return "supported";
  const unsupported = claims.filter((s) => s.level === "unsupported").length;
  const weak = claims.filter((s) => s.level === "weak").length;
  if (unsupported === claims.length) return "unsupported";
  if (unsupported > 0 || weak > 0) return "partial";
  return "supported";
}

export function mapAnswer(raw: RawAnswerResponse): ServerAnswer {
  return {
    answerId: raw.answer_id,
    traceId: raw.trace_id,
    answer: raw.answer ?? "",
    verdict: raw.verdict,
    contexts: (raw.citations ?? []).map(mapCitation),
    verification: mapVerification(raw.verification),
    refusalMessage: raw.refusal?.message ?? null,
    cost: raw.cost,
  };
}

/* ---------------- 请求 ---------------- */

async function readError(res: Response): Promise<ServerError> {
  let code = "http_error";
  let message = `请求失败（HTTP ${res.status}）`;
  let detail: unknown;
  try {
    const body = (await res.json()) as {
      error?: { code?: string; message?: string; detail?: unknown };
    };
    if (body?.error) {
      code = body.error.code ?? code;
      message = body.error.message ?? message;
      detail = body.error.detail;
    }
  } catch {
    /* 非 JSON 错误体（网关/代理返回的 HTML）：保留默认文案 */
  }
  return new ServerError(code, message, res.status, detail);
}

export async function checkHealth(
  baseUrl: string,
  signal?: AbortSignal,
): Promise<ServerHealth> {
  const res = await fetch(apiUrl(baseUrl, "/v1/health"), { signal });
  if (!res.ok) throw await readError(res);
  return (await res.json()) as ServerHealth;
}

export async function askServer(
  baseUrl: string,
  question: string,
  topK: number,
  signal?: AbortSignal,
): Promise<ServerAnswer> {
  const res = await fetch(apiUrl(baseUrl, "/v1/answer"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK }),
    signal,
  });
  if (!res.ok) throw await readError(res);
  return mapAnswer((await res.json()) as RawAnswerResponse);
}

/* ---------------- 流式（M3：SSE） ---------------- */

/** 轨迹行（与 traces 表的 nodes[] 同形；M3 起是 LangGraph 节点轨迹） */
export interface TraceNode {
  seq: number;
  name: string;
  status: string;
  duration_ms: number;
  input_summary?: string;
  output_summary?: string;
  error?: string | null;
}

export interface ServerTrace {
  id: string;
  traceId: string;
  question: string;
  nodes: TraceNode[];
  finalVerdict: VerifyVerdict;
  cost: unknown;
  evidenceSnapshot: Record<string, unknown>;
}

export interface ServerStreamHandlers {
  /** 每个图节点完成（前端据此实时画步骤链） */
  onNode?: (node: TraceNode) => void;
  /** 增量文本（provider 支持流式时才有；以 final.answer 为准） */
  onToken?: (delta: string) => void;
  /** 查询改写开关（默认 false：与 Lite 模式同口径） */
  rewrite?: boolean;
}

/**
 * 读 SSE 帧（event: 行 + 单行 data: 行，`[DONE]` 结束）。
 * 与 apps/api/app/api/sse.py 的帧格式成对；解析口径照抄 cloudPlanner 的既有实现，
 * 避免仓库里出现第二套 SSE 解析。
 */
async function* readSse(
  res: Response,
): AsyncGenerator<{ event: string; data: unknown }> {
  const reader = res.body?.getReader();
  if (!reader) {
    throw new ServerError(
      "stream_unsupported",
      "当前环境不支持流式读取（ReadableStream 不可用）",
      0,
    );
  }
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let event = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let index: number;
    while ((index = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, index).trim();
      buffer = buffer.slice(index + 1);
      if (!line || line.startsWith(":")) continue;
      if (line.startsWith("event:")) {
        event = line.slice(6).trim();
        continue;
      }
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (payload === "[DONE]") return;
      let data: unknown;
      try {
        data = JSON.parse(payload);
      } catch {
        continue; // 坏帧跳过：流式里宁可丢一帧，也不要整条流崩掉
      }
      yield { event: event || "message", data };
      event = "";
    }
  }
}

function errorFromEvent(data: unknown): ServerError {
  const body = (data ?? {}) as {
    error?: { code?: string; message?: string; detail?: unknown };
  };
  return new ServerError(
    body.error?.code ?? "stream_error",
    body.error?.message ?? "流式请求失败",
    200,
    body.error?.detail,
  );
}

/** 流式问答（M3）：事件驱动 + 最终返回与非流式同一形状的答案。 */
export async function askServerStream(
  baseUrl: string,
  question: string,
  topK: number,
  handlers: ServerStreamHandlers = {},
  signal?: AbortSignal,
): Promise<ServerAnswer> {
  const res = await fetch(apiUrl(baseUrl, "/v1/answer"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      top_k: topK,
      stream: true,
      rewrite: handlers.rewrite ?? false,
    }),
    signal,
  });
  if (!res.ok) throw await readError(res);

  let final: ServerAnswer | null = null;
  for await (const { event, data } of readSse(res)) {
    if (event === "node") handlers.onNode?.(data as TraceNode);
    else if (event === "token") {
      const delta = (data as { delta?: string }).delta;
      if (delta) handlers.onToken?.(delta);
    } else if (event === "final") final = mapAnswer(data as RawAnswerResponse);
    else if (event === "error") throw errorFromEvent(data);
  }
  if (!final) {
    throw new ServerError("stream_incomplete", "流式响应缺少 final 事件", 200);
  }
  return final;
}

/* ---------------- Agent（M3：/v1/agent/*） ---------------- */

export interface AgentPendingApproval {
  kind: string;
  call_id: string;
  tool: string;
  args_preview: string;
  reason?: string;
  /** 仅 SSE 的 interrupt 事件带（非流式响应里 run_id 在顶层） */
  run_id?: string;
}

export interface AgentToolCall {
  tool: string;
  ok: boolean;
  denied?: boolean;
  source?: string;
  latency_ms?: number;
  args_preview?: string;
}

export interface AgentRunResult {
  runId: string;
  traceId: string;
  status: "succeeded" | "interrupted";
  question?: string | null;
  answer: string;
  verdict: VerifyVerdict;
  contexts: ContextRef[];
  verification: VerificationResult | null;
  refusalMessage: string | null;
  cost: RawAnswerResponse["cost"] | null;
  toolCalls: AgentToolCall[];
  noProgress: { reason?: string } | null;
  steps: number;
  pendingApproval: AgentPendingApproval | null;
  traceNodes: TraceNode[];
}

interface RawAgentResponse {
  run_id: string;
  trace_id: string;
  status: "succeeded" | "interrupted";
  question?: string | null;
  answer?: string;
  verdict?: VerifyVerdict;
  citations?: RawCitation[];
  refusal?: { reason: string; message: string } | null;
  verification?: RawVerification;
  cost?: RawAnswerResponse["cost"];
  tool_calls?: AgentToolCall[];
  no_progress?: { reason?: string } | null;
  steps?: number;
  pending_approval?: AgentPendingApproval | null;
  trace_nodes?: TraceNode[];
}

export function mapAgentRun(raw: RawAgentResponse): AgentRunResult {
  return {
    runId: raw.run_id,
    traceId: raw.trace_id,
    status: raw.status,
    question: raw.question ?? null,
    answer: raw.answer ?? "",
    verdict: raw.verdict ?? "supported",
    contexts: (raw.citations ?? []).map(mapCitation),
    verification: raw.verification ? mapVerification(raw.verification) : null,
    refusalMessage: raw.refusal?.message ?? null,
    cost: raw.cost ?? null,
    toolCalls: raw.tool_calls ?? [],
    noProgress: raw.no_progress ?? null,
    steps: raw.steps ?? 0,
    pendingApproval: raw.pending_approval ?? null,
    traceNodes: raw.trace_nodes ?? [],
  };
}

export interface AgentStreamHandlers extends ServerStreamHandlers {
  onInterrupt?: (pending: AgentPendingApproval) => void;
}

async function consumeAgentStream(
  res: Response,
  handlers: AgentStreamHandlers,
): Promise<AgentRunResult> {
  let final: AgentRunResult | null = null;
  for await (const { event, data } of readSse(res)) {
    if (event === "node") handlers.onNode?.(data as TraceNode);
    else if (event === "token") {
      const delta = (data as { delta?: string }).delta;
      if (delta) handlers.onToken?.(delta);
    } else if (event === "interrupt") {
      handlers.onInterrupt?.(data as AgentPendingApproval);
    } else if (event === "final") {
      final = mapAgentRun({
        ...(data as object),
        status: "succeeded",
      } as RawAgentResponse);
    } else if (event === "error") throw errorFromEvent(data);
  }
  if (!final) {
    throw new ServerError("stream_incomplete", "Agent 流式响应缺少 final 事件", 200);
  }
  return final;
}

async function agentRequest(
  baseUrl: string,
  path: string,
  body: Record<string, unknown>,
  signal?: AbortSignal,
): Promise<RawAgentResponse> {
  const res = await fetch(apiUrl(baseUrl, path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw await readError(res);
  return (await res.json()) as RawAgentResponse;
}

/** 启动一次 Agent run（非流式；可能返回 interrupted + 待审批动作）。 */
export async function runAgent(
  baseUrl: string,
  question: string,
  topK: number,
  signal?: AbortSignal,
): Promise<AgentRunResult> {
  return mapAgentRun(
    await agentRequest(baseUrl, "/v1/agent/run", { question, top_k: topK }, signal),
  );
}

/** 流式启动一次 Agent run（node/token/interrupt/final/error 事件）。 */
export async function runAgentStream(
  baseUrl: string,
  question: string,
  topK: number,
  handlers: AgentStreamHandlers = {},
  signal?: AbortSignal,
): Promise<AgentRunResult> {
  const res = await fetch(apiUrl(baseUrl, "/v1/agent/run"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK, stream: true }),
    signal,
  });
  if (!res.ok) throw await readError(res);
  return consumeAgentStream(res, handlers);
}

/** 人工审批后续跑（流式）：批准/驳回都要调用，前端复用同一套事件处理。 */
export async function resumeAgentStream(
  baseUrl: string,
  runId: string,
  approved: boolean,
  handlers: AgentStreamHandlers = {},
  signal?: AbortSignal,
): Promise<AgentRunResult> {
  const res = await fetch(
    apiUrl(baseUrl, `/v1/agent/${encodeURIComponent(runId)}/resume`),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ approved, stream: true }),
      signal,
    },
  );
  if (!res.ok) throw await readError(res);
  return consumeAgentStream(res, handlers);
}

export interface AgentRunStatus {
  runId: string;
  status: "queued" | "running" | "succeeded" | "failed" | "interrupted";
  pendingApproval: AgentPendingApproval | null;
  error: string | null;
}

export async function getAgentRun(
  baseUrl: string,
  runId: string,
  signal?: AbortSignal,
): Promise<AgentRunStatus> {
  const res = await fetch(apiUrl(baseUrl, `/v1/agent/${encodeURIComponent(runId)}`), {
    signal,
  });
  if (!res.ok) throw await readError(res);
  const body = (await res.json()) as {
    run_id: string;
    status: AgentRunStatus["status"];
    pending_approval: AgentPendingApproval | null;
    error: string | null;
  };
  return {
    runId: body.run_id,
    status: body.status,
    pendingApproval: body.pending_approval ?? null,
    error: body.error ?? null,
  };
}

/* ---------------- 轨迹（M3：LangGraph 节点轨迹） ---------------- */

export async function getAnswerTrace(
  baseUrl: string,
  answerId: string,
  signal?: AbortSignal,
): Promise<ServerTrace> {
  const res = await fetch(
    apiUrl(baseUrl, `/v1/answer/${encodeURIComponent(answerId)}/trace`),
    { signal },
  );
  if (!res.ok) throw await readError(res);
  const body = (await res.json()) as {
    answer_id: string;
    trace_id: string;
    question: string;
    nodes: TraceNode[];
    final_verdict: VerifyVerdict;
    cost: unknown;
    evidence_snapshot: Record<string, unknown>;
  };
  return {
    id: body.answer_id,
    traceId: body.trace_id,
    question: body.question,
    nodes: body.nodes ?? [],
    finalVerdict: body.final_verdict,
    cost: body.cost,
    evidenceSnapshot: body.evidence_snapshot ?? {},
  };
}

export async function getAgentTrace(
  baseUrl: string,
  runId: string,
  signal?: AbortSignal,
): Promise<ServerTrace> {
  const res = await fetch(
    apiUrl(baseUrl, `/v1/agent/${encodeURIComponent(runId)}/trace`),
    { signal },
  );
  if (!res.ok) throw await readError(res);
  const body = (await res.json()) as {
    run_id: string;
    trace_id: string;
    question: string;
    nodes: TraceNode[];
    final_verdict: VerifyVerdict;
    cost: unknown;
    evidence_snapshot: Record<string, unknown>;
  };
  return {
    id: body.run_id,
    traceId: body.trace_id,
    question: body.question,
    nodes: body.nodes ?? [],
    finalVerdict: body.final_verdict,
    cost: body.cost,
    evidenceSnapshot: body.evidence_snapshot ?? {},
  };
}

/**
 * 服务端轨迹行 → 前端的 AgentStep（复用既有的步骤面板）。
 *
 * 契约附录已声明 `trace.nodes[] ≈ AgentStep[]`：这里做的是"名字 → 类型"的确定性映射，
 * 不做任何推理 —— 轨迹回放看到的必须与后端记录的完全一致。
 */
const NODE_TO_STEP_TYPE: Record<string, AgentStepType> = {
  rewrite: "plan",
  retrieve: "retrieve",
  gate: "evaluate",
  generate: "generate",
  verify: "evaluate",
  adjudicate: "evaluate",
  plan: "plan",
  act: "tool-call",
  reflect: "evaluate",
  synthesize: "synthesize",
  ingest_doc: "tool-result",
  selfcheck: "evaluate",
};

export function mapTraceNodesToAgentSteps(nodes: TraceNode[]): AgentStep[] {
  return nodes.map((node, index) => {
    const step: AgentStep = {
      stepIndex: node.seq || index + 1,
      type: NODE_TO_STEP_TYPE[node.name] ?? "evaluate",
      origin: "server-agent",
      thinking: [node.input_summary, node.output_summary]
        .filter((part) => !!part)
        .join("\n"),
      timestamp: Date.now() + index,
    };
    if (node.name === "act") {
      const input = node.input_summary ?? "";
      step.tool = {
        name: input.split(" args=")[0] || "tool",
        args: input.includes(" args=") ? input.split(" args=")[1] : undefined,
        ok: node.status === "ok",
      };
    }
    return step;
  });
}

export interface UploadResult {
  taskId: string;
  docName: string;
}

export async function uploadDocument(
  baseUrl: string,
  file: File,
  signal?: AbortSignal,
): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file, file.name);
  const res = await fetch(apiUrl(baseUrl, "/v1/documents"), {
    method: "POST",
    body: form,
    signal,
  });
  if (!res.ok) throw await readError(res);
  const body = (await res.json()) as { task_id: string; doc_name: string };
  return { taskId: body.task_id, docName: body.doc_name };
}

export interface TaskStatus {
  status: "queued" | "running" | "succeeded" | "failed" | "interrupted";
  progress: { chunks?: number; doc_name?: string; deduped?: boolean };
  error: string | null;
}

export async function getTask(
  baseUrl: string,
  taskId: string,
  signal?: AbortSignal,
): Promise<TaskStatus> {
  const res = await fetch(apiUrl(baseUrl, `/v1/tasks/${taskId}`), { signal });
  if (!res.ok) throw await readError(res);
  const body = (await res.json()) as {
    status: TaskStatus["status"];
    progress: TaskStatus["progress"];
    error: string | null;
  };
  return { status: body.status, progress: body.progress ?? {}, error: body.error };
}

/**
 * 轮询任务直到终态（摄取是异步的：202 + 轮询，见 apps/api 的路由说明）。
 * 失败/超时都抛 ServerError，交给调用方展示。
 */
export async function waitForTask(
  baseUrl: string,
  taskId: string,
  options: { timeoutMs?: number; intervalMs?: number; onProgress?: (t: TaskStatus) => void } = {},
): Promise<TaskStatus> {
  const timeoutMs = options.timeoutMs ?? 180_000;
  const intervalMs = options.intervalMs ?? 700;
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const task = await getTask(baseUrl, taskId);
    options.onProgress?.(task);
    if (task.status === "succeeded") return task;
    if (task.status === "failed" || task.status === "interrupted") {
      throw new ServerError(
        task.status,
        task.error || "服务端处理失败",
        200,
      );
    }
    if (Date.now() > deadline) {
      throw new ServerError("timeout", "服务端处理超时（可在服务端日志中查看任务进度）", 200);
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
