/*pipeline：HuggingFace Transformers.js 的高层 API，一行完成「加载模型 + 推理」。
env：全局配置对象，控制模型从哪加载。*/
import { pipeline, env } from "@huggingface/transformers";
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

let embedderPromise: Promise<any> | null = null;

function getEmbedder(): Promise<any> {
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
        progress_callback: (info: any) => {
          if (info.status === "progress" && typeof info.progress === "number") {
            self.postMessage({
              type: "load-progress",
              progress: info.progress,
            });
          }
        },
      } as any,
    ).catch((err) => {
      embedderPromise = null;
      throw err;
    });
  }
  return embedderPromise;
}

/** 检查浏览器内存预算（Chrome 非标准 API，无则放行） */
function checkMemoryBudget(): boolean {
  const mem = (performance as any).memory;
  if (!mem) return true;
  const available = mem.jsHeapSizeLimit - mem.usedJSHeapSize;
  return available > 200 * 1024 * 1024;
}

/**
 * 释放向量模型 + 可选清空索引缓存，减少浏览器内存占用。
 * 立即置空 embedderPromise（新请求会触发重新加载），
 * 后台异步 dispose 旧实例（transformers.js pipeline 可能有 dispose 方法）。
 */
async function unloadEmbedder(clearIndex: boolean): Promise<void> {
  // 立即标记为已卸载，并发新请求会重新加载
  const oldPromise = embedderPromise;
  embedderPromise = null;
  if (oldPromise) {
    try {
      const embedder = await oldPromise;
      // transformers.js pipeline 实例可能暴露 dispose
      if (embedder && typeof embedder.dispose === "function") {
        await embedder.dispose();
      } else if (
        embedder?.model &&
        typeof embedder.model.dispose === "function"
      ) {
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
  console.log("向量模型已卸载，内存释放");
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
const bm25Cache = new Map<string, BM25>();

/** 由 chunk 文本构建 BM25 索引并缓存 */
function buildBm25(indexId: string, contents: string[]): BM25 {
  const bm25 = new BM25(contents);
  bm25Cache.set(indexId, bm25);
  return bm25;
}

/* ---------------- 混合检索：向量余弦 + BM25 加权融合 ---------------- */

/** 语义（向量）权重 / 关键词（BM25）权重：语义稍重，关键词做精确匹配补强 */
const VECTOR_WEIGHT = 0.6;
const BM25_WEIGHT = 0.4;

interface HybridHit {
  content: string;
  /** 混合分（最终排序依据，0~1） */
  score: number;
  /** 归一化后的向量相似度 0~1（向量不可用时为 null） */
  vecScore: number | null;
  /** 归一化后的 BM25 分 0~1 */
  bm25Score: number;
}

/** min-max 归一化到 0~1（所有值相同时退化为 0，避免除零） */
function minMax(values: number[]): number[] {
  if (values.length === 0) return [];
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (max - min < 1e-9) return values.map(() => 0);
  return values.map((v) => (v - min) / (max - min));
}

/**
 * 混合检索打分：
 * - 向量相似度：语义召回（同义、近义、跨表述）
 * - BM25：精确关键词匹配（专有名词、数字、型号等向量容易丢失的硬信息）
 * 两路分别归一化后加权求和。向量模型不可用时自动降级为纯 BM25。
 */
function hybridSearch(
  index: { content: string; vector: Float32Array }[], //index：文档库，数组。每一项存原文 content + 文档向量 vector（Float32）
  bm25: BM25, //bm25：预构建好的 BM25 实例（已经把所有文档分词建索引）
  query: string, //query：用户查询文本
  qvec: Float32Array | null,
): HybridHit[] {
  // 路 1：BM25 原始分 -> 按全集最大值归一化
  const bm25Raw = bm25.scoreAll(query);
  let bm25Max = 0;
  for (const s of bm25Raw) if (s > bm25Max) bm25Max = s;
  const bm25Norm = bm25Max > 0 ? bm25Raw.map((s) => s / bm25Max) : bm25Raw;

  // 路 2：向量余弦 -> min-max 归一化（BGE 余弦通常挤在 0.2~0.8，拉开相对差距）
  let vecNorm: number[] | null = null;
  if (qvec) {
    const cosine = index.map((item) => dotProduct(qvec, item.vector));
    vecNorm = minMax(cosine);
  }

  return index.map((item, i) => {
    const v = vecNorm ? vecNorm[i] : null;
    const score =
      v !== null ? VECTOR_WEIGHT * v + BM25_WEIGHT * bm25Norm[i] : bm25Norm[i]; // 向量降级：纯 BM25
    return {
      content: item.content,
      score,
      vecScore: v,
      bm25Score: bm25Norm[i],
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
    const result = (await embedder(prefixed, {
      pooling: "mean",
      normalize: true,
    })) as any;

    // result.data 是展平的 Float32Array：[batch_size × hidden_size]
    // 按行切分，每行 dim 个浮点数
    const data = result.data as Float32Array;
    const dim = data.length / batch.length;

    for (let j = 0; j < batch.length; j++) {
      // slice() 复制出独立 buffer，避免张量底层内存被复用
      out.push(data.slice(j * dim, (j + 1) * dim));
    }

    onBatch?.(Math.min(i + batchSize, texts.length), texts.length);
  }
  return out;
}

/* ================= 消息处理 ================= */

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data as MainRequest | RemoveIndexReq;
  const id = (msg as any).id as number | undefined;

  try {
    switch (msg.type) {
      /* 批量向量化文档 chunk：worker 内缓存向量索引 + BM25 索引，同时把向量返回主线程持久化 */
      case "embed-passages": {
        const { indexId, texts } = msg as EmbedPassagesReq;
        const vectors = await embedBatch(texts, "passage", (done, total) => {
          self.postMessage({ type: "embed-progress", id, done, total });
        });
        indexCache.set(
          indexId,
          texts.map((content, i) => ({ content, vector: vectors[i] })),
        );
        // 同步构建 BM25 关键词索引（纯文本词频统计，开销很小）
        buildBm25(indexId, texts);
        // 发送副本并 transfer buffer（零拷贝），worker 缓存里的原件不受影响
        const copies = vectors.map((v) => v.slice());
        (self as any).postMessage(
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
  } catch (err: any) {
    self.postMessage({ type: "error", id, error: err?.message || String(err) });
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
