import EmbeddingWorker from "../worker/embedding.worker?worker";
import type { VectorChunk } from "../types/doc";
import { embeddingError, workerCrashError, workerTimeoutError } from "./errors";

/**
 * embedding worker 的主线程客户端（Promise 化请求/响应）：
 * - worker 全局单例：向量模型只加载一次，索引缓存在 worker 内存
 * - 文档向量化：批量在 worker 内完成，向量只回传一次用于 IndexedDB 持久化
 * - 检索：只传 query + topK，worker 内做余弦相似度，仅返回 topK 文本
 */

interface PendingEntry {
  resolve: (value: any) => void;
  reject: (err: Error) => void;
  onProgress?: (done: number, total: number) => void;
}

export interface EmbedOptions {
  /** 批量向量化进度（done/total） */
  onBatchProgress?: (done: number, total: number) => void;
  /** 向量模型加载进度 0~100（首次使用时触发） */
  onLoadProgress?: (progress: number) => void;
}

export interface SearchHit {
  content: string;
  score: number;
}

/** 幻觉校验：单个句子在文档索引中的最大余弦依据分 */
export interface SentenceScore {
  text: string;
  score: number;
}

/* ---------- 全局单例 worker ---------- */

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, PendingEntry>();

/** 请求超时定时器映射（id -> timer） */
const timeouts = new Map<number, ReturnType<typeof setTimeout>>();

/** 默认超时：向量化/检索/校验共用，模型首次加载时允许更久 */
const DEFAULT_TIMEOUT = 120_000;
/** 仅模型加载阶段的长超时 */
const LOAD_TIMEOUT = 300_000;

/** 模型加载进度监听者 */
const loadListeners = new Set<(progress: number) => void>();

/** Worker 是否已崩溃（崩溃后不再接受新请求，需刷新页面） */
let crashed = false;

function getWorker(): Worker {
  if (!worker) {
    worker = new EmbeddingWorker();

    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data as any;
      if (msg?.type === "load-progress") {
        loadListeners.forEach((fn) => fn(msg.progress));
        return;
      }
      const entry = msg?.id != null ? pending.get(msg.id) : undefined;

      // 无 id 的 error 是 unhandledrejection 兜底，广播给所有 pending
      if (msg?.type === "error" && msg.id == null) {
        const error = embeddingError(new Error(msg.error || "向量模型线程发生未捕获异常"));
        pending.forEach((p) => p.reject(error));
        pending.clear();
        timeouts.forEach((t) => clearTimeout(t));
        timeouts.clear();
        crashed = true;
        return;
      }

      if (!entry) return;
      // 收到任意响应都清除超时
      const timer = timeouts.get(msg.id);
      if (timer) {
        clearTimeout(timer);
        timeouts.delete(msg.id);
      }

      switch (msg.type) {
        case "embed-progress":
          entry.onProgress?.(msg.done, msg.total);
          break;
        case "embed-result":
          entry.resolve(
            (msg.vectors as Float32Array[]).map((v) => Array.from(v)),
          );
          pending.delete(msg.id);
          break;
        case "search-result":
          entry.resolve(msg.results as SearchHit[]);
          pending.delete(msg.id);
          break;
        case "verify-result":
          entry.resolve(msg.results as SentenceScore[]);
          pending.delete(msg.id);
          break;
        case "error":
          entry.reject(embeddingError(new Error(msg.error || "向量模型线程出错")));
          pending.delete(msg.id);
          break;
      }
    };

    worker.onerror = (err) => {
      crashed = true;
      const error = workerCrashError("向量模型", err);
      pending.forEach((p) => p.reject(error));
      pending.clear();
      timeouts.forEach((t) => clearTimeout(t));
      timeouts.clear();
    };
  }
  return worker;
}

function request<T>(
  msg: Record<string, unknown>,
  entry?: Partial<PendingEntry>,
  timeoutMs = DEFAULT_TIMEOUT,
): Promise<T> {
  if (crashed) {
    return Promise.reject(
      workerCrashError("向量模型", new Error("线程已崩溃，请刷新页面后重试")),
    );
  }
  const w = getWorker();
  const id = nextId++;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve, reject, ...entry });
    // 超时兜底：模型加载卡死/消息丢失
    const timer = setTimeout(() => {
      if (pending.has(id)) {
        pending.delete(id);
        timeouts.delete(id);
        reject(workerTimeoutError("向量模型", timeoutMs));
      }
    }, timeoutMs);
    timeouts.set(id, timer);
    w.postMessage({ ...msg, id });
  });
}

/** 挂载/卸载模型加载进度监听 */
async function withLoadListener<T>(
  onLoadProgress: ((progress: number) => void) | undefined,
  run: () => Promise<T>,
): Promise<T> {
  if (onLoadProgress) loadListeners.add(onLoadProgress);
  try {
    return await run();
  } finally {
    if (onLoadProgress) loadListeners.delete(onLoadProgress);
  }
}

/* ---------- 对外 API ---------- */

/**
 * 批量文档 chunk 向量化（passage 前缀），worker 内缓存索引。
 * 返回的向量仅用于 IndexedDB 持久化，检索时不再依赖主线程持有向量。
 */
export function embedPassages(
  indexId: string,
  texts: string[],
  options: EmbedOptions = {},
): Promise<number[][]> {
  if (texts.length === 0) return Promise.resolve([]);
  return withLoadListener(options.onLoadProgress, () =>
    request<number[][]>(
      { type: "embed-passages", indexId, texts },
      { onProgress: options.onBatchProgress },
      LOAD_TIMEOUT, // 首次加载模型可能很久
    ),
  );
}

/**
 * 把已有向量索引同步到 worker（页面刷新恢复 / 导入备份后调用）。
 * 向量转为 Float32Array 并 transfer buffer，零拷贝交给 worker。
 */
export function syncVectorIndex(indexId: string, chunks: VectorChunk[]): void {
  if (chunks.length === 0) return;
  const vectors = chunks.map((c) => new Float32Array(c.vector));
  getWorker().postMessage(
    {
      type: "set-index",
      indexId,
      chunks: chunks.map((c, i) => ({
        content: c.content,
        vector: vectors[i],
      })),
    },
    vectors.map((v) => v.buffer),
  );
}

/** 删除会话时清理 worker 端索引缓存 */
export function removeVectorIndex(indexId: string): void {
  if (worker) worker.postMessage({ type: "remove-index", indexId });
}

/**
 * 检索：worker 内生成 query 向量并计算余弦相似度，只返回 topK 文本+分数。
 * 请求体不含任何向量，检索结果也不回传向量。
 */
export function searchTopK(
  indexId: string,
  query: string,
  topK = 3,
  options: EmbedOptions = {},
): Promise<SearchHit[]> {
  return withLoadListener(options.onLoadProgress, () =>
    request<SearchHit[]>({ type: "search", indexId, query, topK }),
  );
}

/**
 * 幻觉校验：worker 内逐句生成向量，与全索引算最大余弦相似度。
 * 纯向量数学判定，不调用大模型、不传输向量，只返回每句的依据分。
 */
export function verifySentences(
  indexId: string,
  sentences: string[],
): Promise<SentenceScore[]> {
  if (sentences.length === 0) return Promise.resolve([]);
  return request<SentenceScore[]>({ type: "verify", indexId, sentences });
}
