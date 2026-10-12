import LlmWorker from "../worker/llm.worker?worker";
import {
  modelInferenceError,
  workerCrashError,
  workerTimeoutError,
} from "../utils/errors";
import { stripThinkTags } from "../utils/verifyAnswer";
import { acquireInferenceLock, type InferencePriority } from "../utils/inferenceLock";
import { reportDevice } from "../utils/perf";
import type { DeviceKind } from "../utils/deviceAccel";
import type { LlmRequest } from "../worker/llm.worker";

const LLM_IDLE_TIMEOUT = 120_000;

// ──────────────────────────────────────────────
//  共享常驻 LLM Worker 单例
//  全局只创建一个实例，split/evaluate/summarize/generate 任务通过 id 路由，
//  推理完成不 terminate，模型只加载一次。
// ──────────────────────────────────────────────

let sharedWorker: Worker | null = null; // 全局唯一LLM Worker实例
let workerCrashed = false; // 标记worker是否崩溃，崩溃后会重建
let msgSeq = 0; // 自增任务ID，用来区分多个排队任务
/** 设备偏好（WebGPU/WASM）：worker 可能尚未创建或已被卸载，先记下，创建时立刻下发 */
let pendingDevice: DeviceKind | null = null;

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
  /** 推理锁释放函数（acquire 后才有值，推理结束/超时/取消/崩溃时调用） */
  releaseLock?: () => void;
}

const pendingTasks = new Map<number, PendingTask>();

/** 重置任务的空闲超时（任何消息都证明 worker 活着） */
function armTaskTimeout(task: PendingTask, id: number): void {
  clearTimeout(task.timeoutTimer);
  task.timeoutTimer = setTimeout(() => {
    pendingTasks.delete(id);
    task.releaseLock?.();
    task.reject(workerTimeoutError("大模型", task.timeoutMs));
  }, task.timeoutMs);
}

/** 获取或创建共享 worker 实例（崩溃后自动重建） */
function getSharedWorker(): Worker {
  if (sharedWorker && !workerCrashed) return sharedWorker;

  // 崩溃后重建
  workerCrashed = false;
  sharedWorker = new LlmWorker();
  // 新实例按当前设备偏好加载（worker 内模型是惰性的，这里只下发偏好）
  if (pendingDevice) {
    sharedWorker.postMessage({ type: "set-device", device: pendingDevice });
  }

  sharedWorker.onmessage = (e: MessageEvent) => {
    const msg = e.data;
    const id = msg.id;

    // 设备回报（无 id 广播）：模型真正加载 + 预热成功后通告实际生效设备
    if (msg.type === "device") {
      reportDevice("llm", msg.device, msg.fallbackReason ?? null);
      return;
    }

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

    // 已取消的任务：token/done 都忽略（promise 已在 cancel 时 reject，锁也已释放）
    if (task.cancelled) {
      task.releaseLock?.();
      return;
    }

    if (msg.type === "done") {
      const cleaned = stripThinkTags(String(msg.text || "")).trim();
      task.releaseLock?.();
      task.resolve(cleaned);
    } else if (msg.type === "error") {
      task.releaseLock?.();
      task.reject(modelInferenceError(new Error(msg.error || "模型推理失败")));
    }
  };

  sharedWorker.onerror = (err) => {
    // worker 崩溃，reject 所有 pending，标记 crashed 供下次重建
    workerCrashed = true;
    sharedWorker = null;
    for (const [id, task] of pendingTasks) {
      clearTimeout(task.timeoutTimer);
      task.releaseLock?.();
      task.reject(workerCrashError("大模型", err));
      pendingTasks.delete(id);
    }
  };

  return sharedWorker;
}

/**
 * 调用共享 LLM Worker 执行非流式推理（split/evaluate/summarize）。
 * 复用常驻实例，不 new / terminate，模型只加载一次。
 *
 * 推理锁：acquire 后才 postMessage，防止与 BGE 向量化/检索/校验同时
 * 占用 WASM 堆触发 std::bad_alloc。锁在 done/error/timeout/crash/cancel
 * 路径统一释放（见 PendingTask.releaseLock）。
 */
export function callWorker(
  payload: LlmRequest,
  onLoadProgress?: (progress: number) => void,
  timeoutMs: number = LLM_IDLE_TIMEOUT,
  priority: InferencePriority = "high",
): Promise<string> {
  return (async (): Promise<string> => {
    const release = await acquireInferenceLock(`llm:${payload.type}`, priority);
    return new Promise<string>((resolve, reject) => {
      const id = ++msgSeq;
      const task: PendingTask = {
        resolve,
        reject,
        timeoutTimer: undefined as unknown as ReturnType<typeof setTimeout>,
        timeoutMs,
        onLoadProgress,
        releaseLock: release,
      };
      armTaskTimeout(task, id);
      pendingTasks.set(id, task);

      const worker = getSharedWorker();
      worker.postMessage({ ...payload, id });
    });
  })();
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
 *
 * 推理锁：与 callWorker 相同，acquire 后才 postMessage；cancel 路径需处理
 * "等待锁期间被取消" 的边界 —— 通过 task.cancelled 标记，acquire 完成时立即 release。
 */
export function callWorkerStreaming(
  payload: LlmRequest,
  callbacks: StreamingCallbacks = {},
  timeoutMs: number = LLM_IDLE_TIMEOUT,
  priority: InferencePriority = "high",
): StreamingHandle {
  let resolve!: (text: string) => void;
  let reject!: (err: Error) => void;
  const promise = new Promise<string>((res, rej) => {
    resolve = res;
    reject = rej;
  });

  const id = ++msgSeq;
  const task: PendingTask = {
    resolve,
    reject,
    timeoutTimer: undefined as unknown as ReturnType<typeof setTimeout>,
    timeoutMs,
    onLoadProgress: callbacks.onLoadProgress,
    onGenerating: callbacks.onGenerating,
    onToken: callbacks.onToken,
    cancelled: false,
  };

  // 异步 acquire 推理锁：拿到锁后才 postMessage，防止与 BGE 推理并发抢占 WASM 堆。
  // 若在等待锁期间被 cancel，acquire 完成时检查 cancelled 立即 release 并 return。
  acquireInferenceLock(`llm:${payload.type}`, priority).then((release) => {
    if (task.cancelled) {
      // 等待锁期间用户已取消，promise 已 reject，直接释放锁
      release();
      return;
    }
    task.releaseLock = release;
    armTaskTimeout(task, id);
    pendingTasks.set(id, task);

    const worker = getSharedWorker();
    worker.postMessage({ ...payload, id });
  });

  return {
    promise,
    cancel: () => {
      // 幂等：多次调用安全
      if (task.cancelled) return;
      task.cancelled = true;

      const t = pendingTasks.get(id);
      if (t) {
        clearTimeout(t.timeoutTimer);
        pendingTasks.delete(id);
        t.releaseLock?.();
      }
      // 无论 t 是否存在（还在 acquire 锁中），都要 reject promise 避免悬挂
      reject(new Error("用户取消生成"));
    },
  };
}

/**
 * 下发本地 LLM 的推理设备偏好（WebGPU / WASM）。
 *
 * worker 尚未创建时只记下偏好（不在设置变更时提前创建 worker），
 * 已存在则直接下发；worker 内若设备真的变了会释放当前模型，下次请求重新加载。
 */
export function setLlmDevice(device: DeviceKind): void {
  pendingDevice = device;
  if (sharedWorker && !workerCrashed) {
    sharedWorker.postMessage({ type: "set-device", device });
  }
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
  // 清理所有 pending：超时定时器 + 推理锁一并释放，避免锁悬挂导致后续推理永久阻塞
  for (const [, task] of pendingTasks) {
    clearTimeout(task.timeoutTimer);
    task.releaseLock?.();
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
//  condenseQuery：多轮追问的查询改写
// ──────────────────────────────────────────────

export interface CondenseResult {
  /** 改写后用于检索的独立问题（失败时为原问题） */
  query: string;
  /** 是否走了兜底（模型输出不可用 / 无历史） */
  fallback: boolean;
}

/** 改写任务超时：短输出任务，模型卡死时快速回退原问题，不阻塞问答 */
const CONDENSE_TIMEOUT_MS = 15_000;

/** 改写结果长度上限（超过视为模型跑偏，回退原问题） */
const MAX_CONDENSED_CHARS = 120;
/** 触发改写的条件：问题短或含指代词时才有改写价值 */
const PRONOUN_RE =
  /(它|他|她|这个|那个|这些|那些|该|此|上述|上面|前面|其|这里|那里|它俩|两者)/;

/**
 * 多轮追问的查询改写：用最近 1~2 轮对话把「它的参数呢？」补全成独立问题，**只用于检索**。
 *
 * 何时改写：
 * - 无历史 → 直接返回原问题（无上下文可补）
 * - 问题较长且不含指代词 → 大概率已独立，跳过改写省一次推理
 * 何时不改写：模型输出为空/过长/含拒绝语时回退原问题（fallback=true）
 *
 * 生成阶段仍用原问题：用户看到的回答要对应用户真正问的那句话。
 */
export async function condenseQuery(
  question: string,
  history: Array<{ role: "user" | "assistant"; content: string }>,
  onLoadProgress?: (progress: number) => void,
): Promise<CondenseResult> {
  // 只取最近 2 轮（最多 4 条），更早的语境对指代消解无益且拖慢推理
  const recent = history.slice(-4);
  if (recent.length === 0) return { query: question, fallback: true };
  if (question.length > 30 && !PRONOUN_RE.test(question)) {
    return { query: question, fallback: true };
  }

  // 任何错误（超时/worker 崩溃/取消）都静默回退原问题，不阻断问答
  // —— condenseQuery 是辅助任务，失败时上层仍用原问题检索
  let raw: string;
  try {
    raw = await callWorker(
      { type: "condense", text: question, history: recent },
      onLoadProgress,
      CONDENSE_TIMEOUT_MS,
      "low", // 改写是检索前的辅助任务，不抢占用户等待的推理
    );
  } catch (err) {
    console.warn(
      "[condense] 改写失败，回退原问题：",
      err instanceof Error ? err.message : err,
    );
    return { query: question, fallback: true };
  }
  return parseCondensedQuery(raw, question);
}

/**
 * 纯函数：清洗模型输出的改写问题，不可用时回退原问题。
 * export 供单元测试。
 */
export function parseCondensedQuery(
  raw: string,
  original: string,
): CondenseResult {
  let text = stripThinkTags(raw ?? "").trim();
  if (!text) return { query: original, fallback: true };

  // 去掉引号/前缀（模型常自作主张加「改后问题：」或引号）
  text = text.replace(/^["“'「【]|["”'」】]$/g, "").trim();
  text = text.replace(/^(改写后的问题|改写问题|问题|检索问题)\s*[:：]\s*/, "").trim();
  // 只保留第一行（模型可能附解释）
  text = (text.split(/\n/)[0] ?? "").trim();
  text = text.replace(/^["“'「【]|["”'」】]$/g, "").trim();

  if (!text) return { query: original, fallback: true };
  if (text.length > MAX_CONDENSED_CHARS) {
    return { query: original, fallback: true };
  }
  // 含拒答/寒暄语说明模型没按指令输出
  if (/(无法|不能|抱歉|没有足够|请提供)/.test(text)) {
    return { query: original, fallback: true };
  }
  return { query: text, fallback: false };
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
