/*pipeline：HuggingFace Transformers.js 的高层 API，一行完成「加载模型 + 推理」。
env：全局配置对象，控制模型从哪加载。*/
import { pipeline, env } from "@huggingface/transformers";
import type {
  FeatureExtractionPipeline,
  PretrainedModelOptions,
  ProgressInfo,
  Tensor,
} from "@huggingface/transformers";
import { BM25 } from "../utils/bm25";

//模型从本地 /models 加载,禁止远程，避免走 HF 官网
env.localModelPath = "/models";
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = false; //不使用浏览器 CacheStorage 缓存模型（因为已本地托管，不需要）

/* ================= 请求/响应协议 ================= */

/** 主线程 -> worker 请求 */
interface EmbedPassagesReq {
  type: "embed-passages"; // 批量文档 chunk 向量化
  id: number;
  indexId: string; // 索引标识（会话 id），worker 内缓存键
  texts: string[];
  /**
   * 是否追加到同 indexId 的已有索引（客户端分批让出推理锁时使用）：
   * - false/缺省：覆盖语义（单次全量上传，保持旧行为）
   * - true：追加到旧 entry 末尾，BM25 基于全量文本重建
   */
  append?: boolean;
}
interface SetIndexReq {
  type: "set-index"; // 主线程上传已有向量索引（页面刷新/导入备份后恢复 worker 缓存）
  id: number;
  indexId: string;
  chunks: { content: string; vector: Float32Array }[];
}
interface SearchReq {
  type: "search"; // 检索：worker 内生成 query 向量 + 余弦相似度 + topK
  id: number;
  indexId: string;
  query: string;
  topK: number;
}
interface VerifyReq {
  type: "verify"; // 幻觉校验：每个句子向量与全索引算最大余弦，返回逐句依据分
  id: number;
  indexId: string;
  sentences: string[];
}
interface RemoveIndexReq {
  type: "remove-index"; // 删除会话时清理 worker 缓存
  indexId: string;
}
interface UnloadModelReq {
  type: "unload-model"; // 手动/空闲释放向量模型 + 可选清空索引缓存
  id: number;
  clearIndex: boolean; // 是否同时清空全部向量索引缓存
}
type MainRequest =
  | EmbedPassagesReq
  | SetIndexReq
  | SearchReq
  | VerifyReq
  | UnloadModelReq;

/** worker -> 主线程响应 */
// { type: "load-progress", progress }      向量模型加载进度（全局，无 id）
// { type: "embed-progress", id, done, total } 批量向量化进度
// { type: "embed-result", id, vectors }    批量向量化结果（Float32Array，transfer 零拷贝）
// { type: "search-result", id, mode, results }  混合检索 topK：mode=hybrid|bm25，results 含混合分/向量分/BM25分，不传向量
// { type: "verify-result", id, results }   逐句依据分 {text, score}[]，不传向量
// { type: "error", id, error }

/* ================= 向量模型：惰性加载 + Promise 缓存（单例） ================= */

/** Chrome 非标准 performance.memory 字段（其他浏览器无此字段，访问前先探测） */
interface ChromeMemoryInfo {
  jsHeapSizeLimit: number; //浏览器给当前页面分配的 JS 堆**最大总上限**
  usedJSHeapSize: number; //当前已经占用的 JS 堆内存
}

let embedderPromise: Promise<FeatureExtractionPipeline> | null = null; //缓存加载模型的 Promise

function getEmbedder(): Promise<FeatureExtractionPipeline> {
  if (!embedderPromise) {
    // 内存预算检查（bge-small 约 100~150MB）
    if (!checkMemoryBudget()) {
      return Promise.reject(
        new Error("浏览器内存不足，无法加载向量模型。请关闭其他标签页后重试。"),
      );
    }
    console.log("开始加载本地模型 bge-small-zh-v1.5");
    embedderPromise = pipeline(
      "feature-extraction",
      "Xenova/bge-small-zh-v1.5",
      {
        quantized: true,
        progress_callback: (info: ProgressInfo) => {
          if (info.status === "progress" && typeof info.progress === "number") {
            self.postMessage({
              type: "load-progress",
              progress: info.progress,
            });
          }
        },
      } as unknown as PretrainedModelOptions,
    ).catch((err) => {
      embedderPromise = null;
      throw err;
    });
  }
  return embedderPromise;
}

/** 检查浏览器内存预算（Chrome 非标准 API，无则放行） */
function checkMemoryBudget(): boolean {
  const perf = performance as unknown as { memory?: ChromeMemoryInfo };
  const mem = perf.memory;
  if (!mem) return true;
  const available = mem.jsHeapSizeLimit - mem.usedJSHeapSize;
  return available > 200 * 1024 * 1024;
}

/**
 * 释放向量模型 + 可选清空索引缓存，减少浏览器内存占用。
 * 立即置空 embedderPromise（新请求会触发重新加载），
 * 后台异步 dispose 旧实例 + 显式 release ONNX Runtime session，
 * 彻底释放 WASM 堆上的模型权重，避免内存越跑越高。
 */
async function unloadEmbedder(clearIndex: boolean): Promise<void> {
  // 立即标记为已卸载，并发新请求会重新加载
  const oldPromise = embedderPromise;
  embedderPromise = null;
  if (oldPromise) {
    try {
      const embedder = await oldPromise;
      // 优先调用 pipeline.dispose（内部会 dispose model + tokenizer）
      if (embedder && typeof embedder.dispose === "function") {
        await embedder.dispose(); //优先调用`pipeline.dispose()`：transformers.js 官方 API，释放模型、分词器
      }
      // 显式 release ONNX Runtime session，释放 WASM 堆上的模型权重
      // （session 是 ORT 私有字段，不在 transformers.js 公开类型里，探测式访问）
      const ortModel = embedder?.model as unknown as
        | { session?: { release?: () => void } }
        | undefined;
      const session = ortModel?.session;
      if (session && typeof session.release === "function") {
        try {
          session.release(); //直接释放 ONNX Runtime WASM 堆上的张量权重（最关键）
        } catch {
          // release 失败忽略
        }
      } else if (
        embedder?.model &&
        typeof embedder.model.dispose === "function"
      ) {
        // 旧版本 transformers.js 没有 session.release，退而用 model.dispose
        await embedder.model.dispose();
      }
    } catch {
      // 旧实例加载中失败或 dispose 失败，忽略
    }
  }
  if (clearIndex) {
    indexCache.clear();
    bm25Cache.clear();
    console.log("向量索引与 BM25 索引缓存已清空");
  }
  console.log("向量模型已卸载，session 已 release，内存释放");
}

/* ================= 向量索引缓存：存在 worker 内存，检索不搬运向量 ================= */

/** indexId -> 该会话全部分块（文本 + 向量）。normalize:true，模长恒为 1 */
const indexCache = new Map<
  string,
  { content: string; vector: Float32Array }[]
>();

/**
 * indexId -> BM25 关键词索引。
 * 纯文本派生（词频统计），从 chunk 文本即可重建，无需持久化到 IndexedDB。
 */
const bm25Cache = new Map<string, BM25>(); //存储 BM25 实例，纯文本词频索引，**不需要向量，只靠文本构建**

/** 由 chunk 文本构建 BM25 索引并缓存 */
function buildBm25(indexId: string, contents: string[]): BM25 {
  const bm25 = new BM25(contents);
  bm25Cache.set(indexId, bm25);
  return bm25;
}

/* ---------------- 混合检索：向量余弦 + BM25 倒数排名融合（RRF） ---------------- */

/**
 * RRF（Reciprocal Rank Fusion）常数 k：
 * 平滑排名影响，k 越大对排名差距越不敏感，工业经验默认 60。
 * 公式：rrf_score = Σ 1 / (k + rank)，rank 从 1 开始（0 表示未上榜）。
 */
const RRF_K = 60;

interface HybridHit {
  content: string;
  /** RRF 融合分（最终排序依据，0~1 区间，越大越相关） */
  score: number;
  /** 向量路排名（1 = 最相关，null = 向量降级或未上榜） */
  vecRank: number | null;
  /** BM25 路排名（1 = 最相关，0 = 未上榜/零分） */
  bm25Rank: number;
}

/**
 * 把原始分数数组转成按 docIndex 的排名（1-based，0 表示零分未上榜）。
 * 返回 ranks：ranks[i] 表示第 i 篇文档在该路检索中的名次。
 * 同分文档并列同排名（标准竞赛排名法）。
 */
function scoresToRanks(scores: number[], includeZero: boolean): number[] {
  //scores: number[] 每个元素是文档分数（BM25 分数 / 向量相似度）；includeZero 是否把分数 = 0 的文档也纳入排名
  const n = scores.length;
  if (n === 0) return [];
  const ranks = new Array<number>(n).fill(0);
  // 只对分数 > 0（或 includeZero 时所有）的文档参与排名
  const order = scores
    .map((s, i) => ({ s, i }))
    .filter((x) => (includeZero ? true : x.s > 0))
    .sort((a, b) => b.s - a.s);
  /*.map((s,i) => ({s,i}))：把分数和它原始下标绑定。记住原始位置，排序之后还能写回正确的 ranks 位置。
.filter：过滤规则
includeZero=true：全部保留，0 分文档也参与排名
includeZero=false：只保留 s>0，0 分 / 负分文档直接剔除，不参与排名
.sort((a,b)=>b.s-a.s)：降序排序，分数越高越靠前 */
  for (let r = 0; r < order.length; r++) {
    // 处理并列：跳过同分
    if (r > 0 && order[r].s === order[r - 1].s) {
      ranks[order[r].i] = ranks[order[r - 1].i];
    } else {
      ranks[order[r].i] = r + 1;
    }
  }
  return ranks;
}

/**
 * RRF 混合检索：
 * - 向量路：余弦相似度排名（语义召回）
 * - BM25 路：关键词分数排名（精确匹配补强）
 * 两路独立排名后用 1/(k+rank) 融合，**不做任何归一化**，
 * 彻底规避两路分数量纲不一致的坑。
 * 向量模型不可用时自动降级为纯 BM25（仅按 BM25 排名取 topK）。
 */
function hybridSearch(
  index: { content: string; vector: Float32Array }[], //index：文档库，数组。每一项存原文 content + 文档向量 vector（Float32）
  bm25: BM25, //bm25：预构建好的 BM25 实例（已经把所有文档分词建索引）
  query: string, //query：用户查询文本
  qvec: Float32Array | null,
): HybridHit[] {
  // 路 1：BM25 原始分 -> 排名（零分文档不上榜，rank=0）
  const bm25Raw = bm25.scoreAll(query); //对索引里全部文档，一次性算出每篇文档的 BM25 分数
  const bm25Ranks = scoresToRanks(bm25Raw, false);

  // 路 2：向量余弦 -> 排名（所有文档都参与，余弦恒 > 0）
  let vecRanks: number[] | null = null;
  if (qvec) {
    const cosine = index.map((item) => dotProduct(qvec, item.vector));
    vecRanks = scoresToRanks(cosine, true);
  }

  return index.map((item, i) => {
    const vr = vecRanks ? vecRanks[i] : null;
    let rrf = 0;
    if (vr && vr > 0) rrf += 1 / (RRF_K + vr);
    if (bm25Ranks[i] > 0) rrf += 1 / (RRF_K + bm25Ranks[i]);
    return {
      content: item.content,
      score: rrf,
      vecRank: vr && vr > 0 ? vr : null,
      bm25Rank: bm25Ranks[i],
    };
  });
}

/** 点积：归一化向量的余弦相似度等于点积 */
function dotProduct(a: Float32Array, b: Float32Array): number {
  const len = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < len; i++) sum += a[i] * b[i];
  return sum;
}

/** 查询向量与整个索引的最大余弦相似度（幻觉校验用：找最相似的一块文档） */
function maxSimilarity(
  qvec: Float32Array,
  index: { content: string; vector: Float32Array }[],
): number {
  let max = -Infinity; //max === -Infinity 是防御（空索引时返回 0）
  for (const item of index) {
    const s = dotProduct(qvec, item.vector); //用 dotProduct（归一化向量 = 余弦）
    if (s > max) max = s;
  }
  return max === -Infinity ? 0 : max;
}

/**
 * 批量向量化：分批（每批最多 8 条），每批内部一次性传数组给 pipeline，
 * 内部 batch padding 优化。兼顾速度 + 内存安全。
 * @param prefix BGE 约定前缀：文档用 "passage:"，查询用 "query:"
 */
async function embedBatch(
  texts: string[],
  prefix: "passage" | "query",
  onBatch?: (done: number, total: number) => void,
): Promise<Float32Array[]> {
  const embedder = await getEmbedder();
  const out: Float32Array[] = [];
  // 每批 8 条：既能利用 batch padding 减少调用开销，
  // 又能控制单批内存峰值，避免 WASM 堆溢出
  const batchSize = 8;

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    // BGE 前缀约定：passage 给文档，query 给查询
    const prefixed = batch.map((t) => `${prefix}: ${t}`);

    // 一次性传数组给 pipeline，内部 batch padding 后批量推理
    const result: Tensor = await embedder(prefixed, {
      pooling: "mean",
      normalize: true,
    });

    // result.data 是展平的 Float32Array：[batch_size × hidden_size]
    // 按行切分，每行 dim 个浮点数
    const data = result.data as Float32Array;
    const dim = data.length / batch.length;

    for (let j = 0; j < batch.length; j++) {
      // slice() 复制出独立 buffer，避免张量底层内存被复用
      out.push(data.slice(j * dim, (j + 1) * dim));
    }

    // 关键：本批次推理结束后立即 dispose 原始 ort.Tensor，
    // 释放 ONNX Runtime WASM 堆上的中间张量内存，
    // 否则每轮推理累积的 tensor 会让内存越跑越高直至 std::bad_alloc
    try {
      if (result && typeof result.dispose === "function") {
        result.dispose();
      }
    } catch {
      // dispose 失败忽略，已复制出独立 buffer 不影响后续
    }

    onBatch?.(Math.min(i + batchSize, texts.length), texts.length);
  }
  return out;
}

/* ================= 消息处理 ================= */

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data as MainRequest | RemoveIndexReq;
  const id = "id" in msg ? msg.id : undefined;

  try {
    switch (msg.type) {
      /* 批量向量化文档 chunk：worker 内缓存向量索引 + BM25 索引，同时把向量返回主线程持久化。
         append=true 时追加到已有索引（客户端按 8 条/批分次持锁，给实时问答让路） */
      case "embed-passages": {
        const { indexId, texts, append } = msg as EmbedPassagesReq;
        const vectors = await embedBatch(texts, "passage", (done, total) => {
          self.postMessage({ type: "embed-progress", id, done, total });
        });
        const newEntries = texts.map((content, i) => ({
          content,
          vector: vectors[i],
        }));
        if (append) {
          // 追加语义：取旧索引拼接（首批 append=false 已写入，旧索引缺失时退化为覆盖）
          const prev = indexCache.get(indexId);
          indexCache.set(indexId, prev ? [...prev, ...newEntries] : newEntries);
        } else {
          indexCache.set(indexId, newEntries);
        }
        // BM25 不支持增量，基于全量文本重建（纯词频统计，开销远小于向量推理）
        const all = indexCache.get(indexId)!;
        buildBm25(
          indexId,
          all.map((e) => e.content),
        );
        // 发送副本并 transfer buffer（零拷贝），worker 缓存里的原件不受影响
        // （tsconfig 未引入 WebWorker lib，self 按 Window 解析，transfer 重载需显式转换）
        const copies = vectors.map((v) => v.slice());
        const workerSelf = self as unknown as {
          postMessage(message: unknown, transfer: Transferable[]): void;
        };
        workerSelf.postMessage(
          { type: "embed-result", id, vectors: copies },
          copies.map((v) => v.buffer),
        );
        break;
      }

      /* 恢复 worker 端索引缓存（IndexedDB 恢复 / 导入备份后调用），同时重建 BM25 */
      case "set-index": {
        const { indexId, chunks } = msg as SetIndexReq;
        const contents = chunks.map((c) => c.content);
        indexCache.set(
          indexId,
          chunks.map((c) => ({ content: c.content, vector: c.vector })),
        );
        buildBm25(indexId, contents);
        break;
      }

      /* 混合检索：query 向量化（可失败降级）+ BM25 -> 归一化加权 -> topK；只返回文本+分数 */
      case "search": {
        const { indexId, query, topK } = msg as SearchReq;
        const index = indexCache.get(indexId);
        if (!index || index.length === 0) {
          self.postMessage({
            type: "error",
            id,
            error: "文档索引未同步到检索线程，请重新上传文档或刷新页面",
          });
          break;
        }
        // BM25 索引惰性补建（旧缓存兼容）
        const bm25 =
          bm25Cache.get(indexId) ??
          buildBm25(
            indexId,
            index.map((i) => i.content),
          );

        // 向量路径失败（模型加载失败/内存不足）时降级为纯 BM25，不阻断问答
        let qvec: Float32Array | null = null;
        let mode: "hybrid" | "bm25" = "hybrid";
        try {
          [qvec] = await embedBatch([query], "query");
        } catch (err) {
          console.warn("向量检索不可用，本次降级为纯 BM25 关键词检索", err);
          mode = "bm25";
        }

        const k = Math.max(1, Math.min(topK, index.length));
        const scored = hybridSearch(index, bm25, query, qvec);
        scored.sort((a, b) => b.score - a.score);
        self.postMessage({
          type: "search-result",
          id,
          mode,
          results: scored.slice(0, k),
        });
        break;
      }

      /* 幻觉校验：逐句向量化，与全索引算最大余弦，只返回句子+依据分（不传任何向量） */
      case "verify": {
        const { indexId, sentences } = msg as VerifyReq;
        const index = indexCache.get(indexId);
        if (!index || index.length === 0) {
          self.postMessage({
            type: "error",
            id,
            error: "向量索引未同步到检索线程，无法校验答案依据",
          });
          break;
        }
        if (sentences.length === 0) {
          self.postMessage({ type: "verify-result", id, results: [] });
          break;
        }
        const svecs = await embedBatch(sentences, "query"); //用 "query:" 前缀：因为待校验的句子相当于"查询"
        const results = sentences.map((text, i) => ({
          text,
          score: maxSimilarity(svecs[i], index),
        }));
        self.postMessage({ type: "verify-result", id, results });
        break;
      }

      /* 删除会话时清理对应缓存（向量 + BM25） */
      case "remove-index": {
        const { indexId } = msg as RemoveIndexReq;
        indexCache.delete(indexId);
        bm25Cache.delete(indexId);
        break;
      }

      /* 手动/空闲释放向量模型，减少内存占用 */
      case "unload-model": {
        const { clearIndex } = msg as UnloadModelReq;
        await unloadEmbedder(clearIndex);
        self.postMessage({ type: "unloaded", id });
        break;
      }
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    self.postMessage({ type: "error", id, error: message });
  }
};

// 兜底：Worker 内未捕获的 Promise rejection
self.addEventListener("unhandledrejection", (event) => {
  const msg =
    event.reason instanceof Error ? event.reason.message : String(event.reason);
  self.postMessage({
    type: "error",
    id: undefined,
    error: `向量模型线程发生未捕获异常：${msg}`,
  });
  event.preventDefault();
});
