import EmbeddingWorker from "../worker/embedding.worker?worker";
import type { VectorChunk } from "../types/doc";
import { embeddingError, workerCrashError, workerTimeoutError } from "./errors";
import { acquireInferenceLock } from "./inferenceLock";

/** 文档向量化的客户端切批大小：每批独立持锁，使实时问答能在批次间插入 */
const EMBED_BATCH_SIZE = 8;

/**
 * embedding worker 的主线程客户端（Promise 化请求/响应）：
 * - worker 全局单例：向量模型只加载一次，索引缓存在 worker 内存
 * - 文档向量化：批量在 worker 内完成，向量只回传一次用于 IndexedDB 持久化
 * - 检索：只传 query + topK，worker 内做余弦相似度，仅返回 topK 文本
 */

interface PendingEntry {
  resolve: (value: unknown) => void;
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
  /** RRF 融合分（最终排序依据，越大越相关） */
  score: number;
  /** 向量路排名（1=最相关，null=向量降级或未上榜） */
  vecRank?: number | null;
  /** BM25 路排名（1=最相关，0=未上榜/零分） */
  bm25Rank?: number;
}

/** 检索模式：hybrid=向量+BM25 RRF 融合；bm25=向量模型不可用时的降级模式 */
export type SearchMode = "hybrid" | "bm25";

export interface SearchResult {
  mode: SearchMode;
  hits: SearchHit[];
}

/** 幻觉校验：单个句子在文档索引中的最大余弦依据分 */
export interface SentenceScore {
  text: string;
  score: number;
}

/** embedding worker 回传消息的统一形态（按 type 判别，字段可选） */
interface WorkerMessage {
  type: string;
  id?: number;
  progress?: number;
  error?: string;
  done?: number;
  total?: number;
  vectors?: Float32Array[];
  /** search-result 时为 SearchHit[]，verify-result 时为 SentenceScore[] */
  results?: unknown;
  mode?: string;
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
      const msg = e.data as WorkerMessage;
      if (msg?.type === "load-progress") {
        loadListeners.forEach((fn) => fn(msg.progress ?? 0));
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

      // entry 存在 ⇒ 必有 id（pending 一律带 id 注册）
      if (!entry || msg.id == null) return;
      const id = msg.id;
      // 收到任意响应都清除超时
      const timer = timeouts.get(id);
      if (timer) {
        clearTimeout(timer);
        timeouts.delete(id);
      }

      switch (msg.type) {
        case "embed-progress":
          entry.onProgress?.(msg.done ?? 0, msg.total ?? 0);
          break;
        case "embed-result":
          entry.resolve((msg.vectors ?? []).map((v) => Array.from(v)));
          pending.delete(id);
          break;
        case "search-result":
          entry.resolve({
            mode: (msg.mode as SearchMode) ?? "hybrid",
            hits: (msg.results ?? []) as SearchHit[],
          } as SearchResult);
          pending.delete(id);
          break;
        case "verify-result":
          entry.resolve((msg.results ?? []) as SentenceScore[]);
          pending.delete(id);
          break;
        case "unloaded":
          entry.resolve(undefined);
          pending.delete(id);
          break;
        case "error":
          entry.reject(embeddingError(new Error(msg.error || "向量模型线程出错")));
          pending.delete(id);
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
    pending.set(id, {
      resolve: (v: unknown) => resolve(v as T),
      reject,
      ...entry,
    });
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
 *
 * 优先级与分批让出：文档向量化是后台任务（low），客户端按 EMBED_BATCH_SIZE(8)
 * 切批，每批独立 acquire/release 推理锁；用户实时问答（high）可在任意两批之间
 * 插入优先执行，不必等整篇文档向量化完成。worker 端以 append 模式累积索引。
 * 单批失败会留下部分索引，但文档不会挂载；下次重新上传首批 append=false 自动覆盖。
 */
export function embedPassages(
  indexId: string,
  texts: string[],
  options: EmbedOptions = {},
): Promise<number[][]> {
  if (texts.length === 0) return Promise.resolve([]);

  return (async (): Promise<number[][]> => {
    const all: number[][] = new Array(texts.length);

    await withLoadListener(options.onLoadProgress, async () => {
      for (let start = 0; start < texts.length; start += EMBED_BATCH_SIZE) {
        const slice = texts.slice(start, start + EMBED_BATCH_SIZE);
        const batchEnd = Math.min(start + EMBED_BATCH_SIZE, texts.length);
        const release = await acquireInferenceLock("bge:embed-passages", "low");
        try {
          // 首批覆盖（顺带承担模型首次加载，给长超时）；后续批 append 累积
          const batchVectors = await request<number[][]>(
            {
              type: "embed-passages",
              indexId,
              texts: slice,
              append: start > 0,
            },
            undefined,
            start === 0 ? LOAD_TIMEOUT : DEFAULT_TIMEOUT,
          );
          for (let j = 0; j < batchVectors.length; j++) {
            all[start + j] = batchVectors[j];
          }
          options.onBatchProgress?.(batchEnd, texts.length);
        } finally {
          release();
        }
      }
    });

    return all;
  })();
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
 * 混合检索：worker 内分别对向量余弦和 BM25 关键词分排名，用 RRF 倒数排名融合后返回 topK。
 * 向量模型不可用时自动降级为纯 BM25（mode="bm25"），不阻断问答。
 * 请求体不含任何向量，检索结果也不回传向量。
 *
 * 推理锁：query 向量化是 ORT 推理，与本地 LLM 互斥，防止 WASM 堆冲突。
 */
export function searchTopK(
  indexId: string,
  query: string,
  topK = 3,
  options: EmbedOptions = {},
): Promise<SearchResult> {
  return (async (): Promise<SearchResult> => {
    const release = await acquireInferenceLock("bge:search");
    try {
      return await withLoadListener(options.onLoadProgress, () =>
        request<SearchResult>({ type: "search", indexId, query, topK }),
      );
    } finally {
      release();
    }
  })();
}

/**
 * 幻觉校验：worker 内逐句生成向量，与全索引算最大余弦相似度。
 * 纯向量数学判定，不调用大模型、不传输向量，只返回每句的依据分。
 *
 * 推理锁：逐句向量化是 ORT 推理，与本地 LLM 互斥。
 */
export function verifySentences(
  indexId: string,
  sentences: string[],
): Promise<SentenceScore[]> {
  if (sentences.length === 0) return Promise.resolve([]);
  return (async (): Promise<SentenceScore[]> => {
    const release = await acquireInferenceLock("bge:verify");
    try {
      return await request<SentenceScore[]>({
        type: "verify",
        indexId,
        sentences,
      });
    } finally {
      release();
    }
  })();
}

/**
 * 手动/空闲释放向量模型，减少浏览器内存占用。
 * - clearIndex=false：只释放模型权重，索引缓存保留（下次检索/校验无需重新向量化）
 * - clearIndex=true：同时清空全部向量索引缓存（释放更多内存，但下次需要重新从 IndexedDB 同步）
 * 释放后下次请求会自动重新加载模型。
 */
export function unloadEmbeddingModel(clearIndex = false): Promise<void> {
  if (!worker || crashed) return Promise.resolve();
  return request<void>(
    { type: "unload-model", clearIndex },
    undefined,
    30_000,
  );
}
