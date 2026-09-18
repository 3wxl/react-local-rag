import { modelLoadError } from "./errors";
import type { HistoryTurn } from "./history";
import {
  callWorker,
  callWorkerStreaming,
} from "../agent/agentPlanner";

export interface GenerateCallbacks {
  /** 模型加载进度 0~1 */
  onLoadProgress?: (progress: number) => void;
  /** 模型开始生成 */
  onGenerating?: () => void;
  /** 思考过程增量文本 */
  onThinking?: (delta: string) => void;
  /** 答案增量文本 */
  onToken?: (text: string) => void;
}

export interface GenerateHandle {
  promise: Promise<string>;
  cancel: () => void;
}

/**
 * 流式解析  @{@"think"}...@{"/think"}  标签，将原始 token 流拆为思考流与回答流。
 * 由于标签可能被分到多个 token 中，需要保留缓冲进行最长前缀匹配。
 */
function createThinkTagSplitter(
  onThinking: (delta: string) => void,
  onAnswer: (delta: string) => void,
) {
  const OPEN = "<think>";
  const CLOSE = "</think>";

  let mode: "pre" | "thinking" | "answering" = "pre";
  // 用于跨 token 拼接的不完整标签前缀
  let pending = "";

  return (raw: string) => {
    if (!raw) return;
    pending += raw;
    let buffer = pending;

    while (buffer.length > 0) {
      if (mode === "pre") {
        // 在 buffer 中寻找 OPEN 标签
        const idx = buffer.indexOf(OPEN);
        if (idx === -1) {
          // 没找到：检查 buffer 末尾是否可能是不完整的 OPEN 前缀
          let overlap = 0;
          for (let i = Math.min(OPEN.length - 1, buffer.length); i > 0; i--) {
            if (OPEN.startsWith(buffer.slice(buffer.length - i))) {
              overlap = i;
              break;
            }
          }
          if (overlap > 0) {
            // 把 OPEN 前缀之前的内容作为普通回答输出，保留 overlap 等下一轮
            const safe = buffer.slice(0, buffer.length - overlap);
            if (safe) onAnswer(safe);
            pending = buffer.slice(buffer.length - overlap);
            return;
          } else {
            // 没有重叠前缀，全部作为回答
            onAnswer(buffer);
            pending = "";
            return;
          }
        } else {
          // 找到 OPEN：之前的内容作为回答
          if (idx > 0) onAnswer(buffer.slice(0, idx));
          buffer = buffer.slice(idx + OPEN.length);
          mode = "thinking";
          pending = buffer;
          // 继续 thinking 解析
        }
      }

      if (mode === "thinking") {
        const idx = buffer.indexOf(CLOSE);
        if (idx === -1) {
          // 没找到 CLOSE：检查不完整前缀
          let overlap = 0;
          for (let i = Math.min(CLOSE.length - 1, buffer.length); i > 0; i--) {
            if (CLOSE.startsWith(buffer.slice(buffer.length - i))) {
              overlap = i;
              break;
            }
          }
          if (overlap > 0) {
            const safe = buffer.slice(0, buffer.length - overlap);
            if (safe) onThinking(safe);
            pending = buffer.slice(buffer.length - overlap);
            return;
          } else {
            onThinking(buffer);
            pending = "";
            return;
          }
        } else {
          // 找到 CLOSE：之前是思考
          if (idx > 0) onThinking(buffer.slice(0, idx));
          buffer = buffer.slice(idx + CLOSE.length);
          mode = "answering";
          pending = buffer;
          // 继续 answering 解析
        }
      }

      if (mode === "answering") {
        // 已经分到回答阶段，剩余全部作为回答
        onAnswer(buffer);
        pending = "";
        return;
      }
    }
    // buffer 全部消费完
    pending = "";
  };
}

export interface GenerateOptions {
  /** 最近对话历史（旧对话应已由调用方压缩为摘要） */
  history?: HistoryTurn[];
  /** 更早对话的滚动摘要 */
  historySummary?: string;
}

/**
 * 流式调用 LLM 生成回答，复用共享常驻 Worker 实例。
 * 模型只加载一次：split/evaluate/summarize/generate 全部走同一 worker。
 * cancel 不 terminate worker，只标记取消 + reject promise，worker 继续服务其他任务。
 */
export function generateAnswer(
  question: string,
  contextChunks: string[],
  callbacks: GenerateCallbacks = {},
  options: GenerateOptions = {},
): GenerateHandle {
  const splitter = createThinkTagSplitter(
    (delta) => callbacks.onThinking?.(delta),
    (delta) => callbacks.onToken?.(delta),
  );

  const handle = callWorkerStreaming(
    {
      question,
      contextChunks,
      history: options.history,
      historySummary: options.historySummary,
    },
    {
      onLoadProgress: callbacks.onLoadProgress,
      onGenerating: callbacks.onGenerating,
      onToken: (text: string) => splitter(text),
    },
  );

  // 错误重分类：共享 worker 统一抛 modelInferenceError，
  // 这里根据原始错误消息重新判断是加载失败（内存/网络/文件）还是推理失败
  const promise = handle.promise.catch((err: Error) => {
    // 从 AppError 的 cause 中取原始错误消息
    const causeMsg =
      err.name === "AppError" && err.cause instanceof Error
        ? err.cause.message
        : err.message;
    const lower = causeMsg.toLowerCase();

    if (
      lower.includes("memory") ||
      lower.includes("allocation") ||
      lower.includes("oom") ||
      lower.includes("fetch") ||
      lower.includes("not found") ||
      lower.includes("404") ||
      lower.includes("network")
    ) {
      throw modelLoadError(new Error(causeMsg));
    }
    // 已经是 AppError（model-inference / worker-timeout / worker-crash）直接透传
    throw err;
  });

  return { promise, cancel: handle.cancel };
}

/**
 * 调用本地模型生成滚动摘要（非流式）。
 * 复用共享常驻 LLM Worker（split/evaluate/summarize/generate 同一实例，模型只加载一次）。
 * 失败时抛出异常，由调用方决定是否放弃本轮压缩（不阻断问答）。
 */
export function summarizeHistory(
  text: string,
  onLoadProgress?: (progress: number) => void,
): Promise<string> {
  return callWorker({ type: "summarize", text }, onLoadProgress);
}
