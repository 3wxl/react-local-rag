import LlmWorker from "../worker/llm.worker?worker";
import {
  modelInferenceError,
  modelLoadError,
  workerCrashError,
  workerTimeoutError,
} from "./errors";
//导入后 LlmWorker 是构造函数，new LlmWorker() 创建实例
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

/** LLM 推理超时（含模型加载 5 分钟） */
const LLM_TIMEOUT = 300_000;

export function generateAnswer(
  question: string,
  contextChunks: string[],
  callbacks: GenerateCallbacks = {},
): GenerateHandle {
  const worker = new LlmWorker();

  let resolve: (text: string) => void;
  let reject: (err: Error) => void;

  const promise = new Promise<string>((res, rej) => {
    resolve = res;
    reject = rej;
  });

  // 超时兜底：模型加载卡死/推理无响应
  const timeoutTimer = setTimeout(() => {
    worker.terminate();
    reject(workerTimeoutError("大模型", LLM_TIMEOUT));
  }, LLM_TIMEOUT);

  const splitter = createThinkTagSplitter(
    (delta) => callbacks.onThinking?.(delta),
    (delta) => callbacks.onToken?.(delta),
  );

  worker.onmessage = (e: MessageEvent) => {
    const msg = e.data;
    switch (msg.type) {
      case "load-progress":
        callbacks.onLoadProgress?.(msg.progress);
        break;
      case "generating":
        callbacks.onGenerating?.();
        break;
      case "token":
        splitter(msg.text as string);
        break;
      case "done":
        clearTimeout(timeoutTimer);
        resolve(msg.text);
        worker.terminate();
        break;
      case "error": {
        clearTimeout(timeoutTimer);
        const msgStr = msg.error || "模型推理失败";
        // 加载阶段失败 vs 生成阶段失败，分类不同
        const err = msgStr.match(/加载|load|memory|内存|not found|404|fetch/i)
          ? modelLoadError(new Error(msgStr))
          : modelInferenceError(new Error(msgStr));
        reject(err);
        worker.terminate();
        break;
      }
    }
  };

  worker.onerror = (err) => {
    clearTimeout(timeoutTimer);
    reject(workerCrashError("大模型", err));
    worker.terminate();
  };

  worker.postMessage({ question, contextChunks });

  return {
    promise,
    cancel: () => {
      clearTimeout(timeoutTimer);
      worker.terminate();
    },
  };
}
