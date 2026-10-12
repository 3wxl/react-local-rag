import EmbeddingWorker from "../worker/embedding.worker?worker";
import type { TextChunk, VectorChunk } from "../types/doc";
import {
  embeddingError,
  indexMissingError,
  isIndexMissingError,
  workerCrashError,
  workerTimeoutError,
  INDEX_MISSING_CODE,
} from "./errors";
import { loadVectorIndex } from "./db";
import { acquireInferenceLock } from "./inferenceLock";
import { recordBench, reportDevice } from "./perf";
import type { DeviceKind } from "./deviceAccel";
import type { RetrievalOptions, SearchHit } from "./retrieval";

export type { SearchHit } from "./retrieval";

/** 文档向量化的客户端切批大小：每批独立持锁，使实时问答能在批次间插入 */
const EMBED_BATCH_SIZE = 8;

/**
 * embedding worker 的主线程客户端（Promise 化请求/响应）：
 * - worker 全局单例：向量模型只加载一次，索引缓存在 worker 内存
 * - 文档向量化：分批在 worker 内完成，向量只回传一次用于 IndexedDB 持久化
 * - 检索：只传 query + topK，worker 内完成 RRF 融合与后处理，仅返回命中的身份与分数
 * - 校验：逐句向量化，只在与「被引用段落」的范围内比对，返回逐句依据分
 */

/* ---------------- worker -> 主线程消息（判别联合，无 any） ---------------- */

/** 向量模型加载进度（全局广播，无请求 id） */
export interface LoadProgressMsg {
  type: "load-progress";
  progress?: number;
}
export interface EmbedProgressMsg {
  type: "embed-progress";
  id: number;
  done?: number;
  total?: number;
}
export interface EmbedResultMsg {
  type: "embed-result";
  id: number;
  vectors?: Float32Array[];
}
export interface SearchResultMsg {
  type: "search-result";
  id: number;
  mode?: string;
  reranked?: boolean;
  /** rerank 前后 top-K 集合重合度 0~1（未精排为 null） */
  rerankOverlap?: number | null;
  /** rerank 后 top1 是否换人（未精排为 null） */
  rerankTop1Changed?: boolean | null;
  /** 精排耗时（毫秒，未精排为 null） */
  rerankMs?: number | null;
  /** 请求了精排但没生效时的原因（生效或未请求时为 null） */
  rerankSkippedReason?: string | null;
  /** 本次索引的分片数 */
  indexSize?: number;
  /** 被哪一路检索侧信号拦下（无拒答为 null）；精排只排序不拒答，所以只有 idf */
  refusedBySignal?: "idf" | null;
  noRelevantHit?: boolean;
  droppedCount?: number;
  results?: SearchHit[];
}
export interface VerifyResultMsg {
  type: "verify-result";
  id: number;
  results?: SentenceScore[];
  scopedChunkCount?: number;
}
export interface UnloadedMsg {
  type: "unloaded" | "reranker-unloaded";
  id: number;
}
/** 设备回报（无 id 广播）：模型真正加载 + 预热成功后通告本通道实际生效设备 */
export interface DeviceMsg {
  type: "device";
  channel: "embed";
  device: DeviceKind;
  /** 期望 WebGPU 但被回退的原因；未回退为 null */
  fallbackReason?: string | null;
}
/** 加速对比测试结果：WASM / WebGPU 两侧各一条（GPU 侧失败时只有 WASM 一条） */
export interface BenchResultMsg {
  type: "bench-result";
  id: number;
  measurements?: {
    device: DeviceKind;
    avgMs: number;
    minMs: number;
    runs: number;
  }[];
}
export interface ErrorMsg {
  type: "error";
  /** 无 id = worker 内未捕获异常，广播给全部 pending */
  id?: number;
  error?: string;
  /** 机器可读错误码（如 index-missing：索引被 LRU 淘汰，可从 IDB 自愈） */
  code?: string;
}

export type WorkerResponse =
  | LoadProgressMsg
  | EmbedProgressMsg
  | EmbedResultMsg
  | SearchResultMsg
  | VerifyResultMsg
  | UnloadedMsg
  | DeviceMsg
  | BenchResultMsg
  | ErrorMsg;

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

/** 检索模式：hybrid=向量+BM25 RRF 融合；bm25=向量模型不可用时的降级模式 */
export type SearchMode = "hybrid" | "bm25";

export interface SearchResult {
  mode: SearchMode;
  /** 检索命中（每条含 chunkId/ord/页码/章节 + 三路分数，可直接做引用展示） */
  hits: SearchHit[];
  /** 相关度门限过滤后一条都不剩：调用方应走「文档未提及」拒答分支 */
  noRelevantHit: boolean;
  /** 被门限/去重丢弃的候选数（可观测性） */
  droppedCount: number;
  /** 是否经过 reranker 精排（false = 模型不可用，保持 RRF 顺序） */
  reranked: boolean;
  /** 精排前后 top-K 集合重合度 0~1（未精排为 null） */
  rerankOverlap: number | null;
  /** 精排后 top1 是否换人（未精排为 null） */
  rerankTop1Changed: boolean | null;
  /** 精排耗时（毫秒，未精排为 null） */
  rerankMs: number | null;
  /** 请求了精排但没生效时的原因（生效或未请求时为 null） */
  rerankSkippedReason: string | null;
  /** 本次索引的分片数 */
  indexSize: number;
  /** 被哪一路检索侧信号拦下（无拒答为 null）：目前仅 idf=覆盖率不足；精排不参与拒答 */
  refusedBySignal: "idf" | null;
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

/** 设备偏好：worker 可能尚未创建，先记下，创建时立刻下发 */
let pendingDevice: DeviceKind | null = null;

function getWorker(): Worker {
  if (!worker) {
    worker = new EmbeddingWorker();
    // 新实例按当前设备偏好加载（worker 内模型是惰性的，这里只下发偏好）
    if (pendingDevice) {
      worker.postMessage({ type: "set-device", device: pendingDevice });
    }

    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data as WorkerResponse;
      if (msg.type === "load-progress") {
        loadListeners.forEach((fn) => fn(msg.progress ?? 0));
        return;
      }

      // 设备回报（无 id 广播）：模型真正加载 + 预热成功后通告实际生效设备
      if (msg.type === "device") {
        reportDevice("embed", msg.device, msg.fallbackReason ?? null);
        return;
      }

      // 无 id 的 error 是 unhandledrejection 兜底，广播给所有 pending
      if (msg.type === "error" && msg.id == null) {
        const error = embeddingError(
          new Error(msg.error || "向量模型线程发生未捕获异常"),
        );
        pending.forEach((p) => p.reject(error));
        pending.clear();
        timeouts.forEach((t) => clearTimeout(t));
        timeouts.clear();
        crashed = true;
        return;
      }

      // 走到这里 id 必存在（无 id 的 error 已在上面广播返回）
      if (msg.id == null) return;
      const id = msg.id;
      const entry = pending.get(id);
      if (!entry) return;
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
            hits: msg.results ?? [],
            noRelevantHit: msg.noRelevantHit ?? false,
            droppedCount: msg.droppedCount ?? 0,
            reranked: msg.reranked ?? false,
            rerankOverlap: msg.rerankOverlap ?? null,
            rerankTop1Changed: msg.rerankTop1Changed ?? null,
            rerankMs: msg.rerankMs ?? null,
            rerankSkippedReason: msg.rerankSkippedReason ?? null,
            indexSize: msg.indexSize ?? 0,
            refusedBySignal: msg.refusedBySignal ?? null,
          } satisfies SearchResult);
          pending.delete(id);
          break;
        case "verify-result":
          entry.resolve(msg.results ?? []);
          pending.delete(id);
          break;
        case "unloaded":
        case "reranker-unloaded":
          entry.resolve(undefined);
          pending.delete(id);
          break;
        case "bench-result":
          entry.resolve(msg.measurements ?? []);
          pending.delete(id);
          break;
        case "error":
          entry.reject(
            msg.code === INDEX_MISSING_CODE
              ? indexMissingError(new Error(msg.error || "索引不在检索线程"))
              : embeddingError(new Error(msg.error || "向量模型线程出错")),
          );
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

/* ---------- 索引自愈：LRU 淘汰后从 IndexedDB 重同步 ---------- */

/**
 * 正在重同步的索引（indexId -> 任务）。并发请求同时撞上淘汰时，
 * 只拉一次 IDB，其余等同一个 Promise。
 */
const resyncInFlight = new Map<string, Promise<number>>();

/**
 * 从 IndexedDB 重新拉该会话的分块并同步给 worker。
 * @returns 同步的块数；0 = 本地也没有（无法自愈，错误应上抛）
 */
function resyncIndexFromDb(indexId: string): Promise<number> {
  const existing = resyncInFlight.get(indexId);
  if (existing) return existing;

  const task = (async (): Promise<number> => {
    const chunks = await loadVectorIndex(indexId);
    if (chunks.length === 0) return 0;
    syncVectorIndex(indexId, chunks);
    return chunks.length;
  })();

  resyncInFlight.set(indexId, task);
  // 无论成败都要清理，否则失败会永久卡住后续重试
  void task.finally(() => resyncInFlight.delete(indexId));
  return task;
}

/**
 * 包一层「索引被 LRU 淘汰 → 从 IDB 重同步 → 重试一次」的恢复逻辑。
 *
 * 为什么必需：worker 内索引缓存有 LRU 上限，会话切多了会淘汰最久未用的那份；
 * 而主线程内存里 vectorChunks 仍然非空，useConversations 的懒加载因此不会重新同步
 * ——结果就是切回旧会话提问必报错，只能重新上传或刷新页面。
 *
 * 重试只做一次：再失败说明不是淘汰（例如 IDB 里也没有），上抛给用户。
 * 注意锁的边界：重同步在锁**外**做，避免拉 IDB 期间占着推理锁挡住 LLM 推理。
 */
async function withIndexRecovery<T>(
  indexId: string,
  attempt: () => Promise<T>,
): Promise<T> {
  try {
    return await attempt();
  } catch (err) {
    if (!isIndexMissingError(err)) throw err;
    const synced = await resyncIndexFromDb(indexId);
    if (synced === 0) {
      // 本地也没有这份索引（例如文档已被删/DB 不可用），无法自愈
      throw err;
    }
    console.info(
      `[索引自愈] 会话 ${indexId} 的索引不在检索线程，已从本地存储重新同步 ${synced} 块，重试本次请求`,
    );
    return await attempt();
  }
}

/* ---------- 对外 API ---------- */

/**
 * 批量文档 chunk 向量化（passage 前缀），worker 内缓存索引 + 增量维护 BM25。
 * 返回的向量仅用于 IndexedDB 持久化，检索时不再依赖主线程持有向量。
 *
 * 优先级与分批让出：文档向量化是后台任务（low），客户端按 EMBED_BATCH_SIZE(8)
 * 切批，每批独立 acquire/release 推理锁；用户实时问答（high）可在任意两批之间
 * 插入优先执行，不必等整篇文档向量化完成。worker 端以 append 模式累积索引。
 * 单批失败会留下部分索引，但文档不会挂载；下次重新上传首批 append=false 自动覆盖。
 *
 * 分块身份元数据（docName/ord/页码/章节）随向量一起进索引，检索结果才能做来源引用。
 */
export function embedPassages(
  indexId: string,
  chunks: TextChunk[],
  options: EmbedOptions = {},
): Promise<number[][]> {
  if (chunks.length === 0) return Promise.resolve([]);

  return (async (): Promise<number[][]> => {
    const all: number[][] = new Array(chunks.length);

    await withLoadListener(options.onLoadProgress, async () => {
      for (let start = 0; start < chunks.length; start += EMBED_BATCH_SIZE) {
        const slice = chunks.slice(start, start + EMBED_BATCH_SIZE);
        const batchEnd = Math.min(start + EMBED_BATCH_SIZE, chunks.length);
        const release = await acquireInferenceLock("bge:embed-passages", "low");
        try {
          // 首批覆盖（顺带承担模型首次加载，给长超时）；后续批 append 累积
          const batchVectors = await request<number[][]>(
            {
              type: "embed-passages",
              indexId,
              chunks: slice,
              append: start > 0,
            },
            undefined,
            start === 0 ? LOAD_TIMEOUT : DEFAULT_TIMEOUT,
          );
          for (let j = 0; j < batchVectors.length; j++) {
            all[start + j] = batchVectors[j];
          }
          options.onBatchProgress?.(batchEnd, chunks.length);
        } finally {
          release();
        }
      }
    });

    return all;
  })();
}

/**
 * 原样向量化一批文本并返回向量（不写任何索引）。
 *
 * 用途：离线工具（评测向量固化）需要按指定前缀算向量，
 * 且结果要在浏览器里导出成文件，不能走「挂索引 → 检索」那条路。
 */
export function embedTexts(
  texts: string[],
  kind: "passage" | "query",
  options: EmbedOptions = {},
): Promise<number[][]> {
  if (texts.length === 0) return Promise.resolve([]);
  return (async (): Promise<number[][]> => {
    const all: number[][] = new Array(texts.length);
    await withLoadListener(options.onLoadProgress, async () => {
      for (let start = 0; start < texts.length; start += EMBED_BATCH_SIZE) {
        const slice = texts.slice(start, start + EMBED_BATCH_SIZE);
        const release = await acquireInferenceLock("bge:embed-texts", "low");
        try {
          const batchVectors = await request<number[][]>(
            { type: "embed-texts", texts: slice, kind },
            undefined,
            start === 0 ? LOAD_TIMEOUT : DEFAULT_TIMEOUT,
          );
          for (let j = 0; j < batchVectors.length; j++) all[start + j] = batchVectors[j];
          options.onBatchProgress?.(
            Math.min(start + EMBED_BATCH_SIZE, texts.length),
            texts.length,
          );
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
 * 分块身份元数据一并传入，检索结果才能做来源引用。
 * 向量转 Float32Array 并 transfer buffer，零拷贝交给 worker。
 */
export function syncVectorIndex(indexId: string, chunks: VectorChunk[]): void {
  if (chunks.length === 0) return;
  const vectors = chunks.map((c) => new Float32Array(c.vector));
  getWorker().postMessage(
    {
      type: "set-index",
      indexId,
      chunks: chunks.map((c, i) => ({ ...c, vector: vectors[i] })),
    },
    vectors.map((v) => v.buffer),
  );
}

/** 删除会话时清理 worker 端索引缓存 */
export function removeVectorIndex(indexId: string): void {
  if (worker) worker.postMessage({ type: "remove-index", indexId });
}

/**
 * 下发向量模型的推理设备偏好（WebGPU / WASM）。
 *
 * worker 尚未创建时只记下偏好（设置变更不提前拉起 worker），已存在则直接下发；
 * worker 内若设备真的变了会释放当前模型，下次请求按新设备重新加载
 * （索引缓存与 BM25 保留，不需要重新向量化文档）。
 */
export function setEmbedDevice(device: DeviceKind): void {
  pendingDevice = device;
  if (worker && !crashed) {
    worker.postMessage({ type: "set-device", device });
  }
}

/** 加速对比测试的单台设备测量结果 */
export interface EmbedBenchMeasurement {
  device: DeviceKind;
  /** 平均耗时（毫秒，已剔除预热轮） */
  avgMs: number;
  /** 最快一轮耗时（毫秒） */
  minMs: number;
  /** 计时轮数 */
  runs: number;
}

/**
 * 加速对比测试：用同一批固定文本（worker 内 BENCH_TEXTS）分别在 WASM 与 WebGPU 上
 * 各跑 runs 轮，结果写入性能埋点（stage=bench），面板直接展示「加速前后」耗时与加速比。
 *
 * 代价：会先释放当前向量模型，测试期间独占推理锁；结束后缓存为空，
 * 下一次真实请求按偏好设备重新加载（约 1 秒）。因此只在用户点按钮时调用。
 */
export async function benchEmbeddingDevices(
  runs = 3,
): Promise<EmbedBenchMeasurement[]> {
  // low 优先级：对比测试是诊断任务，排队时给实时问答让路
  const release = await acquireInferenceLock("bge:bench", "low");
  try {
    const measurements = await request<EmbedBenchMeasurement[]>(
      { type: "bench", runs },
      undefined,
      LOAD_TIMEOUT, // 两侧模型加载 + 预热，用加载级长超时
    );
    if (measurements.length > 0) recordBench("embed", measurements);
    return measurements;
  } finally {
    release();
  }
}

/**
 * 混合检索：worker 内做 RRF 融合 → 邻近合并 → 去重 → MMR → 门限/自适应 K，
 * （可选）再用 reranker 精排。只返回命中的身份元数据与三路分数，不回传向量。
 *
 * 推理锁：query 向量化是 ORT 推理，与本地 LLM 互斥，防止 WASM 堆冲突。
 */
export function searchTopK(
  indexId: string,
  query: string,
  topK = 3,
  options: EmbedOptions & {
    /** 门限/自适应 K/MMR/窗口扩展参数覆盖 */
    retrieval?: Partial<RetrievalOptions>;
    /** 是否启用 reranker 精排（默认关闭；模型不可用或内存不足时 worker 内自动跳过） */
    rerank?: boolean;
  } = {},
): Promise<SearchResult> {
  const attempt = async (): Promise<SearchResult> => {
    const release = await acquireInferenceLock("bge:search");
    try {
      return await withLoadListener(options.onLoadProgress, () =>
        request<SearchResult>({
          type: "search",
          indexId,
          query,
          topK,
          retrieval: options.retrieval,
          rerank: options.rerank,
        }),
      );
    } finally {
      release();
    }
  };
  // 索引被 LRU 淘汰时自动从 IDB 重同步并重试一次
  return withIndexRecovery(indexId, attempt);
}

/**
 * 幻觉校验：worker 内逐句生成向量，与索引算最大余弦相似度。
 * 纯向量数学判定，不调用大模型、不传输向量，只返回每句的依据分。
 *
 * 推理锁：逐句向量化是 ORT 推理，与本地 LLM 互斥。
 */
export function verifySentences(
  indexId: string,
  sentences: string[],
  /**
   * 校验范围（块 id 列表）：只在这些块里找最大余弦。
   * 不传 = 全索引（旧行为）；传了但一块都匹配不上时 worker 自动退回全索引，
   * 避免因索引不同步把正常回答误判成无依据。
   */
  scopeChunkIds?: string[],
): Promise<SentenceScore[]> {
  if (sentences.length === 0) return Promise.resolve([]);
  const attempt = async (): Promise<SentenceScore[]> => {
    const release = await acquireInferenceLock("bge:verify");
    try {
      return await request<SentenceScore[]>({
        type: "verify",
        indexId,
        sentences,
        scopeChunkIds,
      });
    } finally {
      release();
    }
  };
  // 同 searchTopK：索引被淘汰时自愈，否则校验会莫名其妙失败
  return withIndexRecovery(indexId, attempt);
}

/**
 * 手动/空闲释放向量模型，减少浏览器内存占用。
 * - clearIndex=false：只释放模型权重，索引缓存保留（下次检索/校验无需重新向量化）
 * - clearIndex=true：同时清空全部向量索引缓存（释放更多内存，但下次需要重新从 IndexedDB 同步）
 * 释放后下次请求会自动重新加载模型。精排模型也一并释放。
 */
export function unloadEmbeddingModel(clearIndex = false): Promise<void> {
  if (!worker || crashed) return Promise.resolve();
  return request<void>(
    { type: "unload-model", clearIndex },
    undefined,
    30_000,
  );
}

/**
 * 只释放精排模型（保留向量模型与索引缓存）。
 *
 * 用途：检索已完成、马上要加载本地 Qwen 生成答案时调用。
 * 精排模型常驻约 300~400MB，与 Qwen 同时驻留会触发 OrtRun std::bad_alloc；
 * 释放后下次开启精排会重新加载（约 1 秒），换来生成阶段的堆空间安全。
 *
 * 走推理锁：避免在精排推理进行中把模型 dispose 掉。
 */
export function releaseReranker(): Promise<void> {
  if (!worker || crashed) return Promise.resolve();
  return (async (): Promise<void> => {
    const release = await acquireInferenceLock("bge:unload-rerank");
    try {
      await request<void>({ type: "unload-rerank" }, undefined, 30_000);
    } finally {
      release();
    }
  })();
}
