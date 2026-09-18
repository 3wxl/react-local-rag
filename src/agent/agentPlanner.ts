import LlmWorker from "../worker/llm.worker?worker";
import {
  modelInferenceError,
  workerCrashError,
  workerTimeoutError,
  type AppError,
} from "../utils/errors";
import { stripThinkTags } from "../utils/verifyAnswer";

const LLM_IDLE_TIMEOUT = 120_000;

// ──────────────────────────────────────────────
//  共享常驻 LLM Worker 单例
//  全局只创建一个实例，split/evaluate/summarize/generate 任务通过 id 路由，
//  推理完成不 terminate，模型只加载一次。
// ──────────────────────────────────────────────

let sharedWorker: Worker | null = null;
let workerCrashed = false;
let msgSeq = 0;

interface PendingTask {
  resolve: (text: string) => void;
  reject: (err: Error) => void;
  timeoutTimer: ReturnType<typeof setTimeout>;
  timeoutMs: number;
  onLoadProgress?: (progress: number) => void;
  /** 流式回调（仅 generate 模式使用） */
  onGenerating?: () => void;
  onToken?: (text: string) => void;
  /** 已取消：后续 token/done 忽略，promise 已 reject */
  cancelled?: boolean;
}

const pendingTasks = new Map<number, PendingTask>();

/** 重置任务的空闲超时（任何消息都证明 worker 活着） */
function armTaskTimeout(task: PendingTask, id: number): void {
  clearTimeout(task.timeoutTimer);
  task.timeoutTimer = setTimeout(() => {
    pendingTasks.delete(id);
    task.reject(workerTimeoutError("大模型", task.timeoutMs));
  }, task.timeoutMs);
}

/** 获取或创建共享 worker 实例（崩溃后自动重建） */
function getSharedWorker(): Worker {
  if (sharedWorker && !workerCrashed) return sharedWorker;

  // 崩溃后重建
  workerCrashed = false;
  sharedWorker = new LlmWorker();

  sharedWorker.onmessage = (e: MessageEvent) => {
    const msg = e.data;
    const id = msg.id;

    // load-progress / loading：重置超时 + 路由进度回调
    if (msg.type === "load-progress" || msg.type === "loading") {
      if (id != null) {
        const task = pendingTasks.get(id);
        if (task && !task.cancelled) {
          armTaskTimeout(task, id);
          task.onLoadProgress?.(msg.progress ?? 0);
        }
      } else {
        // 未带 id 的进度广播，发给所有 pending
        for (const [tid, task] of pendingTasks) {
          if (!task.cancelled) {
            armTaskTimeout(task, tid);
            task.onLoadProgress?.(msg.progress ?? 0);
          }
        }
      }
      return;
    }

    // 流式 token：重置超时 + 路由到对应 pending 的回调
    if (msg.type === "token" && id != null) {
      const task = pendingTasks.get(id);
      if (task && !task.cancelled) {
        armTaskTimeout(task, id);
        task.onToken?.(msg.text as string);
      }
      return;
    }

    // generating：重置超时 + 路由回调
    if (msg.type === "generating" && id != null) {
      const task = pendingTasks.get(id);
      if (task && !task.cancelled) {
        armTaskTimeout(task, id);
        task.onGenerating?.();
      }
      return;
    }

    // done / error 带有 id
    const task = id != null ? pendingTasks.get(id) : null;
    if (!task) return;
    clearTimeout(task.timeoutTimer);
    pendingTasks.delete(id);

    // 已取消的任务：token/done 都忽略（promise 已在 cancel 时 reject）
    if (task.cancelled) return;

    if (msg.type === "done") {
      const cleaned = stripThinkTags(String(msg.text || "")).trim();
      task.resolve(cleaned);
    } else if (msg.type === "error") {
      task.reject(modelInferenceError(new Error(msg.error || "模型推理失败")));
    }
  };

  sharedWorker.onerror = (err) => {
    // worker 崩溃，reject 所有 pending，标记 crashed 供下次重建
    workerCrashed = true;
    sharedWorker = null;
    for (const [id, task] of pendingTasks) {
      clearTimeout(task.timeoutTimer);
      task.reject(workerCrashError("大模型", err));
      pendingTasks.delete(id);
    }
  };

  return sharedWorker;
}

/**
 * 调用共享 LLM Worker 执行非流式推理（split/evaluate/summarize）。
 * 复用常驻实例，不 new / terminate，模型只加载一次。
 */
export function callWorker(
  payload: Record<string, unknown>,
  onLoadProgress?: (progress: number) => void,
  timeoutMs: number = LLM_IDLE_TIMEOUT,
): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const id = ++msgSeq;
    const task: PendingTask = {
      resolve,
      reject,
      timeoutTimer: undefined as unknown as ReturnType<typeof setTimeout>,
      timeoutMs,
      onLoadProgress,
    };
    armTaskTimeout(task, id);
    pendingTasks.set(id, task);

    const worker = getSharedWorker();
    worker.postMessage({ ...payload, id });
  });
}

/** 流式回调接口（与 generateAnswer 的 GenerateCallbacks 对齐） */
export interface StreamingCallbacks {
  onLoadProgress?: (progress: number) => void;
  onGenerating?: () => void;
  onToken?: (text: string) => void;
}

export interface StreamingHandle {
  promise: Promise<string>;
  cancel: () => void;
}

/**
 * 调用共享 LLM Worker 执行流式推理（generate 模式）。
 * token 通过 id 路由到 onToken 回调，与 split/evaluate/summarize 共享同一 worker 实例。
 * 每收到一个 token/generating/load-progress 都重置空闲超时，长回答不会被误杀。
 * cancel 不 terminate worker，只标记取消 + reject promise，worker 继续生成但结果被忽略。
 */
export function callWorkerStreaming(
  payload: Record<string, unknown>,
  callbacks: StreamingCallbacks = {},
  timeoutMs: number = LLM_IDLE_TIMEOUT,
): StreamingHandle {
  let resolve: (text: string) => void;
  let reject: (err: Error) => void;
  const promise = new Promise<string>((res, rej) => {
    resolve = res;
    reject = rej;
  });

  const id = ++msgSeq;
  const task: PendingTask = {
    resolve: resolve!,
    reject: reject!,
    timeoutTimer: undefined as unknown as ReturnType<typeof setTimeout>,
    timeoutMs,
    onLoadProgress: callbacks.onLoadProgress,
    onGenerating: callbacks.onGenerating,
    onToken: callbacks.onToken,
    cancelled: false,
  };
  armTaskTimeout(task, id);
  pendingTasks.set(id, task);

  const worker = getSharedWorker();
  worker.postMessage({ ...payload, id });

  return {
    promise,
    cancel: () => {
      const t = pendingTasks.get(id);
      if (t) {
        t.cancelled = true;
        clearTimeout(t.timeoutTimer);
        pendingTasks.delete(id);
        reject(new Error("用户取消生成"));
      }
    },
  };
}

/**
 * 手动释放共享 LLM Worker（空闲 5 分钟自动调用 / 用户手动卸载时调用）。
 * 终止 worker 并清空 pending + 引用，下次 callWorker 会重建。
 */
export function disposeSharedLlmWorker(): void {
  if (sharedWorker) {
    sharedWorker.terminate();
    sharedWorker = null;
  }
  workerCrashed = false;
  // 清理所有 pending 的超时定时器
  for (const [, task] of pendingTasks) {
    clearTimeout(task.timeoutTimer);
  }
  pendingTasks.clear();
}

// ──────────────────────────────────────────────
//  splitQuestion：子问题拆分
// ──────────────────────────────────────────────

export interface SplitResult {
  /** 拆分出的子问题列表（1~3 条）；模型不配合时回退为 [原问题] */
  subQuestions: string[];
  /** 是否走了兜底（模型输出无法解析） */
  fallback: boolean;
}

/**
 * 调用本地模型拆分用户问题为最多 3 个子问题。
 * Qwen2.5-0.5B 输出 JSON 不稳定，提供多层解析兜底：
 * 1. 尝试直接 JSON.parse
 * 2. 提取 [ ... ] 片段再 parse
 * 3. 按换行/序号正则提取
 * 4. 全部失败 → 返回原问题，fallback=true
 */
export async function splitQuestion(
  question: string,
  onLoadProgress?: (progress: number) => void,
): Promise<SplitResult> {
  const raw = await callWorker(
    { type: "split", text: question },
    onLoadProgress,
  );

  const subQuestions = parseSubQuestions(raw);
  if (subQuestions.length === 0) {
    return { subQuestions: [question], fallback: true };
  }
  return { subQuestions, fallback: false };
}

/**
 * 纯函数：从 0.5B 模型的乱格式输出中提取子问题列表。
 * 多层容错，export 供单元测试。
 */
export function parseSubQuestions(raw: string): string[] {
  const text = raw.trim();
  if (!text) return [];

  // 1. 直接 JSON.parse
  try {
    const arr = JSON.parse(text);
    if (Array.isArray(arr)) {
      const strs = arr
        .map((v) => String(v).trim())
        .filter((s) => s.length > 0);
      if (strs.length > 0 && strs.length <= 5) return strs;
    }
  } catch {
    /* 继续 */
  }

  // 2. 提取第一个 [ ... ] 片段再 parse
  const bracketMatch = text.match(/\[[\s\S]*\]/);
  if (bracketMatch) {
    try {
      const arr = JSON.parse(bracketMatch[0]);
      if (Array.isArray(arr)) {
        const strs = arr
          .map((v) => String(v).trim())
          .filter((s) => s.length > 0);
        if (strs.length > 0 && strs.length <= 5) return strs;
      }
    } catch {
      /* 继续 */
    }
  }

  // 3. 按换行分割 + 去序号前缀（1. / 2. / - / * / • 等）
  // 过滤掉原始 JSON 拼片、规则/示例行、过长文本（子问题不该超过 200 字）
  const lines = text
    .split(/\n/)
    .map((l) => l.replace(/^[\d]+[.、)]\s*/, "").trim())
    .filter(
      (l) =>
        l.length > 2 &&
        l.length <= 200 &&
        !l.match(/^(规则|示例|输出|问|答|\[)/),
    );
  if (lines.length >= 1 && lines.length <= 5) return lines;

  // 4. 全部失败
  return [];
}

// ──────────────────────────────────────────────
//  isEnoughInfo：信息充足性判断
// ──────────────────────────────────────────────

export type SufficiencyResult = "enough" | "insufficient";

export interface EnoughInfoResult {
  result: SufficiencyResult;
  /** 是否走了兜底（模型输出既不含「足够」也不含「不足」） */
  fallback: boolean;
  /** 模型原始输出（调试用） */
  raw: string;
}

/**
 * 调用本地模型判断检索到的片段是否足以回答用户问题。
 * 模型输出应只有「足够」或「不足」，但 0.5B 可能输出多余文字，
 * JS 兜底：只要输出包含「足够」即判 enough，否则 insufficient。
 */
export async function isEnoughInfo(
  question: string,
  collectedChunks: string[],
  onLoadProgress?: (progress: number) => void,
): Promise<EnoughInfoResult> {
  const raw = await callWorker(
    { type: "evaluate", text: question, contextChunks: collectedChunks },
    onLoadProgress,
  );

  const { result, fallback } = parseSufficiency(raw);
  return { result, fallback, raw };
}

/**
 * 纯函数：从模型输出中提取「足够」/「不足」。
 * 优先匹配「不足」（因为它包含「足」字，先匹配避免误判）。
 * export 供单元测试。
 */
export function parseSufficiency(raw: string): {
  result: SufficiencyResult;
  fallback: boolean;
} {
  const text = raw.trim();

  // 先判「不足」（避免被「足够」的「足」字误匹配）
  if (text.includes("不足")) return { result: "insufficient", fallback: false };
  if (text.includes("足够")) return { result: "enough", fallback: false };

  // 兜底：无法解析，默认「不足」（更安全，触发后续补充检索）
  return { result: "insufficient", fallback: true };
}
