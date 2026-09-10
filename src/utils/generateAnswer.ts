import LlmWorker from "./llm.worker.ts?worker";

export interface GenerateCallbacks {
  onLoadProgress?: (progress: number) => void;
  onToken?: (text: string) => void;
  onGenerating?: () => void;
}

export interface GenerateHandle {
  promise: Promise<string>;
  cancel: () => void;
}

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
        callbacks.onToken?.(msg.text);
        break;
      case "done":
        resolve(msg.text);
        worker.terminate();
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
