/*pipeline：HuggingFace Transformers.js 的高层 API，一行完成「加载模型 + 推理」。
env：全局配置对象，控制模型从哪加载。*/
import { pipeline, env } from "@huggingface/transformers";

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
type MainRequest = EmbedPassagesReq | SetIndexReq | SearchReq | VerifyReq;

/** worker -> 主线程响应 */
// { type: "load-progress", progress }      向量模型加载进度（全局，无 id）
// { type: "embed-progress", id, done, total } 批量向量化进度
// { type: "embed-result", id, vectors }    批量向量化结果（Float32Array，transfer 零拷贝）
// { type: "search-result", id, results }   topK 检索结果：只含文本+分数，不传向量
// { type: "verify-result", id, results }   逐句依据分 {text, score}[]，不传向量
// { type: "error", id, error }

/* ================= 向量模型：惰性加载 + Promise 缓存（单例） ================= */

let embedderPromise: Promise<any> | null = null;

function getEmbedder(): Promise<any> {
  if (!embedderPromise) {
    console.log("开始加载本地模型 bge-small-zh-v1.5");
    embedderPromise = pipeline(
      "feature-extraction", // 任务类型：特征提取（即 embedding）
      "Xenova/bge-small-zh-v1.5", // 模型名（中文优化的 BGE 小模型）
      {
        quantized: true, // 量化：体积更小、推理更快
        progress_callback: (info: any) => {
          if (info.status === "progress" && typeof info.progress === "number") {
            self.postMessage({
              type: "load-progress",
              progress: info.progress,
            });
          }
        },
      } as any,
      // 加载失败时清空缓存，允许下次请求重新尝试，避免永久卡死在失败的 Promise 上
    ).catch((err) => {
      embedderPromise = null;
      throw err;
    });
  }
  return embedderPromise;
}

/* ================= 向量索引缓存：存在 worker 内存，检索不搬运向量 ================= */

/** indexId -> 该会话全部分块（文本 + 向量）。normalize:true，模长恒为 1 */
const indexCache = new Map<
  string,
  { content: string; vector: Float32Array }[]
>();

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
  let max = -Infinity;
  for (const item of index) {
    const s = dotProduct(qvec, item.vector);
    if (s > max) max = s;
  }
  return max === -Infinity ? 0 : max;
}

/**
 * 批量向量化：分批并发（每批 4 条），避免一次塞满 CPU
 * @param prefix BGE 约定前缀：文档用 "passage:"，查询用 "query:"
 */
async function embedBatch(
  texts: string[],
  prefix: "passage" | "query",
  onBatch?: (done: number, total: number) => void,
): Promise<Float32Array[]> {
  const embedder = await getEmbedder();
  const out: Float32Array[] = [];
  /*分批并发（每批 4 条），避免一次塞满 CPU

WASM 推理是 CPU 密集的，一次并发太多会：

阻塞 Worker 事件循环太久（无法响应进度/新请求）。

内存峰值过高。

反而因调度开销降低总吞吐。*/
  const batchSize = 4;

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const results = await Promise.all(
      batch.map(
        (text) =>
          embedder(`${prefix}: ${text}`, {
            //：BGE 模型的前缀约定——文档要加 "passage: "，查询要加 "query: "。这是模型训练时的格式，加错会严重影响效果。
            pooling: "mean", //把 token 级向量按平均池化成句向量
            normalize: true, //输出即归一化（模长=1），这是前面点积能代替余弦的前提。
          }) as Promise<any>,
      ),
    );
    // slice() 复制出独立 buffer：张量底层数据可能被复用，不能直接持有
    for (const r of results) out.push((r.data as Float32Array).slice());
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
      /* 批量向量化文档 chunk：worker 内缓存索引，同时把向量返回主线程持久化 */
      case "embed-passages": {
        const { indexId, texts } = msg as EmbedPassagesReq;
        const vectors = await embedBatch(texts, "passage", (done, total) => {
          self.postMessage({ type: "embed-progress", id, done, total });
        });
        indexCache.set(
          indexId,
          texts.map((content, i) => ({ content, vector: vectors[i] })),
        );
        // 发送副本并 transfer buffer（零拷贝），worker 缓存里的原件不受影响
        const copies = vectors.map((v) => v.slice());
        (self as any).postMessage(
          { type: "embed-result", id, vectors: copies },
          copies.map((v) => v.buffer),
        );
        break;
      }

      /* 恢复 worker 端索引缓存（IndexedDB 恢复 / 导入备份后调用） */
      case "set-index": {
        const { indexId, chunks } = msg as SetIndexReq;
        indexCache.set(
          indexId,
          chunks.map((c) => ({ content: c.content, vector: c.vector })),
        );
        break;
      }

      /* 检索：query 向量化 -> 全量点积 -> topK；只返回文本+分数 */
      case "search": {
        const { indexId, query, topK } = msg as SearchReq;
        const index = indexCache.get(indexId);
        if (!index || index.length === 0) {
          self.postMessage({
            type: "error",
            id,
            error: "向量索引未同步到检索线程，请重新上传文档或刷新页面",
          });
          break;
        }
        const [qvec] = await embedBatch([query], "query");
        const k = Math.max(1, Math.min(topK, index.length));
        const scored = index.map((item) => ({
          content: item.content,
          score: dotProduct(qvec, item.vector),
        }));
        scored.sort((a, b) => b.score - a.score);
        self.postMessage({
          type: "search-result",
          id,
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
        const svecs = await embedBatch(sentences, "query");
        const results = sentences.map((text, i) => ({
          text,
          score: maxSimilarity(svecs[i], index),
        }));
        self.postMessage({ type: "verify-result", id, results });
        break;
      }

      /* 删除会话时清理对应缓存 */
      case "remove-index": {
        indexCache.delete((msg as RemoveIndexReq).indexId);
        break;
      }
    }
  } catch (err: any) {
    self.postMessage({ type: "error", id, error: err?.message || String(err) });
  }
};
