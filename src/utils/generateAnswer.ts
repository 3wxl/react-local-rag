import LlmWorker from "../worker/llm.worker?worker";
//?worker 是 Vite 语法，把该文件打包成 Web Worker
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

export function generateAnswer(
  question: string, //用户提问文本
  contextChunks: string[], //RAG 检索出来的文档切片数组，作为上下文塞给大模型
  callbacks: GenerateCallbacks = {}, //回调函数集合（加载进度、开始生成、思考增量、答案增量）
): GenerateHandle {
  const worker = new LlmWorker(); //新建一个 WebWorker 实例，单独开一个线程跑大模型推理

  let resolve: (text: string) => void;
  let reject: (err: Error) => void;

  const promise = new Promise<string>((res, rej) => {
    resolve = res;
    reject = rej;
  });

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
        resolve(msg.text);
        worker.terminate(); //销毁当前 worker 线程，释放内存
        break;
      case "error":
        reject(new Error(msg.error));
        worker.terminate();
        break;
    }
  };

  worker.onerror = (err) => {
    reject(new Error(err.message));
    worker.terminate();
  };

  worker.postMessage({ question, contextChunks });

  return {
    promise,
    cancel: () => {
      worker.terminate();
    },
  };
}
