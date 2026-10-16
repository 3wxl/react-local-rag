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
