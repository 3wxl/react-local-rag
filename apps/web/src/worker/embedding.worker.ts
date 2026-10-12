/*pipeline：HuggingFace Transformers.js 的高层 API，一行完成「加载模型 + 推理」。
env：全局配置对象，控制模型从哪加载。*/
import { pipeline, env } from "@huggingface/transformers";
import type {
  FeatureExtractionPipeline,
  PretrainedModelOptions,
  ProgressInfo,
  Tensor,
  TextClassificationPipeline,
} from "@huggingface/transformers";
import { BM25 } from "../utils/bm25";
import { INDEX_MISSING_CODE } from "../utils/errors";
import { applyEmbedPrefix, type EmbedKind } from "../utils/embedPrefix";
import type { DeviceKind } from "../utils/deviceAccel";
import type { TextChunk } from "../types/doc";
import {
  dotProduct,
  isDegenerateScores,
  postProcessRetrieval,
  DEFAULT_RETRIEVAL_OPTIONS,
  type RetrievalDoc,
  type RetrievalOptions,
  type SearchHit,
  type SearchHitWithVector,
} from "../utils/retrieval";

//模型从本地 models/ 加载,禁止远程，避免走 HF 官网。
//用 BASE_URL 拼相对路径：部署到子路径（如 /app/）时不会 404
env.localModelPath = `${import.meta.env.BASE_URL}models`;
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = false; //不使用浏览器 CacheStorage 缓存模型（因为已本地托管，不需要）
// 单线程推理：BGE/reranker/Qwen 同时多线程抢 WASM 线程池会触发 std::bad_alloc，
// 强制单线程让推理串行化，内存分配可预测
if (env.backends?.onnx?.wasm) env.backends.onnx.wasm.numThreads = 1;

/* ================= 请求/响应协议 ================= */

/** 主线程 -> worker 请求 */
interface EmbedPassagesReq {
  type: "embed-passages"; // 批量文档 chunk 向量化
  id: number;
  indexId: string; // 索引标识（会话 id），worker 内缓存键
  /** 带身份元数据的分块（文本从 content 取，元数据随向量一起进索引） */
  chunks: TextChunk[];
  /**
   * 是否追加到同 indexId 的已有索引（客户端分批让出推理锁时使用）：
   * - false/缺省：覆盖语义（单次全量上传，保持旧行为）
   * - true：追加到旧 entry 末尾，BM25 只增量更新本批词频
   */
  append?: boolean;
}
interface SetIndexReq {
  type: "set-index"; // 主线程上传已有向量索引（页面刷新/导入备份后恢复 worker 缓存）
  id: number;
  indexId: string;
  chunks: (TextChunk & { vector: Float32Array })[];
}
interface SearchReq {
  type: "search"; // 检索：worker 内生成 query 向量 + 余弦相似度 + topK
  id: number;
  indexId: string;
  query: string;
  topK: number;
  /** 检索后处理参数覆盖（门限/自适应 K/MMR，缺省用 DEFAULT_RETRIEVAL_OPTIONS） */
  retrieval?: Partial<RetrievalOptions>;
  /** 是否启用两阶段精排（reranker 模型不可用时自动跳过） */
  rerank?: boolean;
}
interface VerifyReq {
  type: "verify"; // 幻觉校验：每个句子向量与索引算最大余弦，返回逐句依据分
  id: number;
  indexId: string;
  sentences: string[];
  /**
   * 校验范围（块 id）：只在这些块里找 max，实现「答案只被自己引用的片段证明」。
   * 缺省或一块都匹配不上 → 退回全索引（避免索引不同步导致误判无依据）。
   */
  scopeChunkIds?: string[];
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
interface UnloadRerankReq {
  type: "unload-rerank"; // 只释放精排模型（检索已完成、准备加载本地 LLM 时调用，腾出堆给 Qwen）
  id: number;
}
interface EmbedTextsReq {
  type: "embed-texts"; // 原样向量化一批文本并返回向量（不动索引）：评测向量固化等离线工具用
  id: number;
  texts: string[];
  /** 前缀口径：passage=文档；query=查询 */
  kind: "passage" | "query";
}
/** 切换推理设备（WebGPU / WASM）：设备变更时释放旧实例，下次请求按新设备重新加载 */
interface SetDeviceReq {
  type: "set-device";
  device: DeviceKind;
}
/** 加速对比测试：固定文本分别在 WASM / WebGPU 上各跑 N 轮，返回两侧耗时 */
interface BenchReq {
  type: "bench";
  id: number;
  runs: number;
}
type MainRequest =
  | EmbedPassagesReq
  | SetIndexReq
  | SearchReq
  | VerifyReq
  | UnloadModelReq
  | UnloadRerankReq
  | EmbedTextsReq
  | SetDeviceReq
  | BenchReq;

/** worker -> 主线程响应 */
// { type: "load-progress", progress }      向量模型加载进度（全局，无 id）
// { type: "embed-progress", id, done, total } 批量向量化进度
// { type: "embed-result", id, vectors }    批量向量化结果（Float32Array，transfer 零拷贝）
// { type: "search-result", id, mode, results, noRelevantHit, reranked }  检索结果（含 chunkId/ord/页码/三路分数，不传向量）
// { type: "verify-result", id, results }   逐句依据分 {text, score}[]，不传向量
// { type: "error", id, error }

/* ================= BGE 前缀方案 ================= */

/**
 * bge-small-zh-v1.5 官方用法：
 * - 文档（passage）：**不加任何前缀**
 * - 查询（query）：加中文 instruction（BGE_QUERY_INSTRUCTION）
 * - 校验（verify）：不加前缀 —— 「答案句子 vs 文档块」是对称语义比较，与文档同处一个空间
 *
 * 旧实现给两路都加了 "passage:"/"query:" 英文前缀，与模型训练分布不符，相似度整体偏低。
 * 指令常量来自 chunk.ts，参与索引版本 hash：改指令 → 索引过期 → UI 提示重建。
 */

/* ================= 向量模型：惰性加载 + Promise 缓存（单例） ================= */

/** Chrome 非标准 performance.memory 字段（其他浏览器无此字段，访问前先探测） */
interface ChromeMemoryInfo {
  jsHeapSizeLimit: number; //浏览器给当前页面分配的 JS 堆**最大总上限**
  usedJSHeapSize: number; //当前已经占用的 JS 堆内存
}

let embedderPromise: Promise<FeatureExtractionPipeline> | null = null; //缓存加载模型的 Promise

/** 期望设备（主线程 set-device 下发；实际生效设备以加载结果为准） */
let preferredDevice: DeviceKind = "wasm";
/**
 * WebGPU 不可用原因。非 null 时不再尝试 GPU（每次检索都重试会反复报错、
 * 拖慢问答），仅由 set-device（用户重新开关）或模型卸载复位。
 */
let webgpuBlockedReason: string | null = null;

/** 向主线程回报「本通道当前生效设备」（埋点按设备分桶、面板显示回退原因） */
function reportDevice(device: DeviceKind, fallbackReason: string | null): void {
  self.postMessage({ type: "device", channel: "embed", device, fallbackReason });
}

/**
 * 按指定设备创建向量模型 pipeline。
 *
 * 权重的选择必须显式给 dtype：transformers.js v4 里 wasm 默认 q8、**webgpu 默认
 * fp32**，而本仓库只托管了量化权重（model_quantized.onnx / model_q4.onnx），
 * WebGPU 不给 dtype 会去找不存在的 model.onnx。
 * - wasm：q8（model_quantized.onnx），与既有行为一致
 * - webgpu：q4（model_q4.onnx，int4 MatMulNBits，WebGPU 后端支持的量化路径）
 */
function createEmbedder(device: DeviceKind): Promise<FeatureExtractionPipeline> {
  const options: Record<string, unknown> = {
    dtype: device === "webgpu" ? "q4" : "q8",
    progress_callback: (info: ProgressInfo) => {
      if (info.status === "progress" && typeof info.progress === "number") {
        self.postMessage({
          type: "load-progress",
          progress: info.progress,
        });
      }
    },
  };
  if (device === "webgpu") options.device = "webgpu";
  console.log(`开始加载本地模型 bge-small-zh-v1.5（${device}）`);
  return pipeline(
    "feature-extraction",
    "Xenova/bge-small-zh-v1.5",
    options as unknown as PretrainedModelOptions,
  );
}

/** 单条短文本跑一次前向：WebGPU 用它触发着色器编译，暴露「能建会话但跑不动」的问题 */
async function warmupEmbedder(embedder: FeatureExtractionPipeline): Promise<void> {
  const out = await embedder(["预热"], { pooling: "mean", normalize: true });
  if (out && typeof (out as { dispose?: () => void }).dispose === "function") {
    (out as unknown as { dispose: () => void }).dispose();
  }
}

/**
 * 按偏好设备加载向量模型：WebGPU 失败（不支持/显存不足/着色器编译失败）时
 * 自动回退 WASM，并把原因回报给主线程 —— 回退必须可见，不能静默降级。
 */
async function loadEmbedder(): Promise<FeatureExtractionPipeline> {
  // 换设备时摘下的旧实例在这里释放（此刻没有任何推理在跑）
  await releaseStaleInstances();
  if (preferredDevice === "webgpu" && !webgpuBlockedReason) {
    let attempt: FeatureExtractionPipeline | null = null;
    try {
      attempt = await createEmbedder("webgpu");
      await warmupEmbedder(attempt);
      reportDevice("webgpu", null);
      console.log("[device] 向量模型运行在 WebGPU（dtype=q4）");
      return attempt;
    } catch (err) {
      webgpuBlockedReason = err instanceof Error ? err.message : String(err);
      console.warn(
        "WebGPU 加载/预热失败，回退 WASM 向量模型：",
        webgpuBlockedReason,
      );
      // 失败实例必须释放：WebGPU 侧可能已占住显存/session，留着会拖垮随后的 WASM 加载
      await releaseEmbedderInstance(attempt ?? undefined);
    }
  }
  const embedder = await createEmbedder("wasm");
  reportDevice("wasm", preferredDevice === "webgpu" ? webgpuBlockedReason : null);
  return embedder;
}

function getEmbedder(): Promise<FeatureExtractionPipeline> {
  if (!embedderPromise) {
    // 内存预算检查（bge-small 约 100~150MB）
    if (!checkMemoryBudget()) {
      return Promise.reject(
        new Error("浏览器内存不足，无法加载向量模型。请关闭其他标签页后重试。"),
      );
    }
    embedderPromise = loadEmbedder().catch((err) => {
      // 加载失败清空缓存，允许下次重试（不回退到上一台设备的旧实例）
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

/** 释放一个已加载的向量模型实例（dispose pipeline + 显式 release ORT session） */
async function releaseEmbedderInstance(
  embedder: FeatureExtractionPipeline | undefined,
): Promise<void> {
  try {
    if (!embedder) return;
    // 优先调用 pipeline.dispose（内部会 dispose model + tokenizer）
    if (typeof embedder.dispose === "function") {
      await embedder.dispose(); //优先调用`pipeline.dispose()`：transformers.js 官方 API，释放模型、分词器
    }
    // 显式 release ONNX Runtime session，释放 WASM 堆上的模型权重
    // （session 是 ORT 私有字段，不在 transformers.js 公开类型里，探测式访问）
    const ortModel = embedder.model as unknown as
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
      embedder.model &&
      typeof embedder.model.dispose === "function"
    ) {
      // 旧版本 transformers.js 没有 session.release，退而用 model.dispose
      await embedder.model.dispose();
    }
  } catch {
    // 旧实例加载中失败或 dispose 失败，忽略
  }
}

/**
 * 切换设备时被摘下的旧实例：**延后到下一次加载时才释放**。
 *
 * 不能立即 dispose —— set-device 消息可能在一次检索/向量化推理中途到达，
 * 直接释放正在使用的 session 会让进行中的推理报错。推迟到下一次加载时，
 * 客户端推理锁已保证没有在跑的推理（锁在下一次请求前必然空闲）。
 */
let staleInstances: Promise<FeatureExtractionPipeline>[] = [];

/** 释放所有被摘下的旧实例（在新模型加载前调用） */
async function releaseStaleInstances(): Promise<void> {
  const olds = staleInstances;
  staleInstances = [];
  for (const oldPromise of olds) {
    try {
      await releaseEmbedderInstance(await oldPromise);
    } catch {
      // 旧实例加载中失败，忽略
    }
  }
}

/**
 * 只释放向量模型实例（立即置空缓存，并发新请求会按当前设备重新加载）。
 * 切换推理设备时用：设备变了必须换一份 session，索引缓存保留不动。
 */
async function disposeEmbedder(): Promise<void> {
  const oldPromise = embedderPromise;
  embedderPromise = null;
  if (oldPromise) {
    try {
      await releaseEmbedderInstance(await oldPromise);
    } catch {
      // 旧实例加载中失败，忽略
    }
  }
  await releaseStaleInstances();
}

/**
 * 释放向量模型 + 可选清空索引缓存，减少浏览器内存占用。
 * 立即置空 embedderPromise（新请求会触发重新加载），
 * 后台异步 dispose 旧实例 + 显式 release ONNX Runtime session，
 * 彻底释放 WASM 堆上的模型权重，避免内存越跑越高。
 */
async function unloadEmbedder(clearIndex: boolean): Promise<void> {
  await disposeEmbedder();
  // 精排模型比向量模型更占内存（约 300~400MB）：卸载向量模型时一并释放，
  // 给随后加载的本地 Qwen 让出堆空间；下次开启精排会重新加载。
  await unloadReranker();
  if (clearIndex) {
    indexCache.clear();
    bm25Cache.clear();
    console.log("向量索引与 BM25 索引缓存已清空");
  }
  console.log("向量模型已卸载，session 已 release，内存释放");
}

/* ================= 两阶段检索：bge-reranker 精排（可选，模型缺失自动跳过） ================= */

/** 精排模型名（需与 bge-small 一起放在 public/models/Xenova 下） */
const RERANKER_MODEL = "Xenova/bge-reranker-base";
/** 精排召回阶段取多少条候选 */
const RERANK_RECALL = 10;
/**
 * 精排分批大小：每批送 N 条 (query, passage) 对进模型，跑完一批再跑下一批。
 * 一次送 10 条会让单次推理的中间 tensor 堆积过高触发 std::bad_alloc；
 * 分批 5 条把峰值压住，批与批之间 transformers.js 内部会释放中间 tensor。
 */
const RERANK_BATCH_SIZE = 5;

/** 加载中/已加载的精排模型 Promise；null = 未加载或已卸载 */
let rerankerPromise: Promise<TextClassificationPipeline | null> | null = null;
/** 精排不可用（模型缺失/加载失败）后置位，避免每次检索都重试加载 */
let rerankerUnavailable = false;
/** 精排不可用原因（供日志与 UI 提示） */
let rerankerUnavailableReason = "";

/**
 * 惰性加载精排模型；不可用时返回 null（不抛错，检索照常走 RRF 结果）。
 *
 * 不做 performance.memory 内存预算检查：WASM 堆不计入 usedJSHeapSize，
 * 检查对 WASM 模型基本无效（见项目 memory 记录的教训）；
 * 加载失败由 catch 兜底，置 unavailable 后跳过，绝不拖垮检索链路。
 */
async function getReranker(): Promise<TextClassificationPipeline | null> {
  if (rerankerUnavailable) return null;
  if (!rerankerPromise) {
    rerankerPromise = pipeline("text-classification", RERANKER_MODEL, {
      dtype: "q8",
    } as unknown as PretrainedModelOptions).then(
      (p: TextClassificationPipeline) => {
        console.log(
          `[rerank] 精排模型 ${RERANKER_MODEL} 已加载（dtype=q8 量化版）`,
        );
        return p;
      },
      (err: unknown) => {
        rerankerUnavailable = true;
        rerankerUnavailableReason =
          err instanceof Error ? err.message : String(err);
        // 清理 promise 引用：允许下次调用重新尝试加载
        // （但 rerankerUnavailable=true 会拦住，除非 unloadReranker 复位）
        rerankerPromise = null;
        console.warn(
          `[rerank] 精排模型 ${RERANKER_MODEL} 不可用，改用 RRF 排序：`,
          rerankerUnavailableReason,
        );
        return null;
      },
    );
  }
  return rerankerPromise;
}

/**
 * 释放精排模型（内存回收；下次开启精排会重新加载）。
 *
 * 只释放一次：优先 pipeline.dispose()（transformers.js v3 会清理底层 ORT session）；
 * dispose 不存在时才手动 release ORT session 兜底。
 * 之前的实现 dispose 之后又 release session 是重复释放，可能触发已释放对象报错。
 */
async function unloadReranker(): Promise<void> {
  const old = rerankerPromise;
  rerankerPromise = null;
  // 复位 unavailable：让用户下次开启精排时能重新尝试加载
  rerankerUnavailable = false;
  rerankerUnavailableReason = "";
  if (!old) return;
  try {
    const reranker = await old;
    if (!reranker) return;
    // 优先用 pipeline.dispose()：transformers.js v3 提供，会清理底层 ORT session
    if (typeof reranker.dispose === "function") {
      await reranker.dispose();
      console.log("[rerank] 精排模型已释放（pipeline.dispose）");
      return;
    }
    // 兜底：dispose 不存在时手动 release ORT session
    const ortModel = (reranker as unknown as {
      model?: { session?: { release?: () => void } };
    }).model;
    if (ortModel?.session && typeof ortModel.session.release === "function") {
      ortModel.session.release();
      console.log("[rerank] 精排模型已释放（session.release）");
    }
  } catch {
    // 释放失败忽略（可能仍在加载中，或对象已被释放）
  }
}

/**
 * 交叉编码器精排：把 (query, passage) 成对送入模型打相关性分，按分降序。
 *
 * 只做**相对排序**，不做绝对分过滤 —— cross-encoder 的 logit 跨查询不可比
 * （同一分数对 A 查询是强相关、对 B 查询可能只是弱相关），拿它做全局拒答门限
 * 必然误拒。「文档未提及」的置信度裁决在答案侧 verifyAnswer 双层证据校验完成。
 *
 * 分批推理：每批 RERANK_BATCH_SIZE 条，避免单次送 10 条导致内存峰值过高；
 * 跑完一批再跑下一批，批与批之间 transformers.js 内部释放中间 tensor。
 * 任一步失败返回 null，调用方回退 RRF 顺序。
 */
async function rerankHits(
  query: string,
  hits: SearchHit[],
  limit: number,
): Promise<SearchHit[] | null> {
  if (hits.length <= 1) return hits;
  try {
    const reranker = await getReranker();
    if (!reranker) return null;

    // 分批推理：每批 RERANK_BATCH_SIZE 条 (query, passage) 对
    const pairs = hits.map((h) => `${query}</s></s>${h.content}`);
    const scores: number[] = new Array(pairs.length).fill(0);
    for (let i = 0; i < pairs.length; i += RERANK_BATCH_SIZE) {
      const batch = pairs.slice(i, i + RERANK_BATCH_SIZE);
      // bge-reranker-base 上限 512 token：长文档扩展窗口（最长 1200 字）会超，
      // 不截断 ONNX 直接报 invalid expand shape。query 在序列前部，右截只丢段尾。
      // truncation/max_length 未写进 TextClassificationPipelineOptions 的公开类型，
      // 但会透传给内部分词器；这里显式断言，避免用 as any 把整个调用变成无类型。
      const output = (await reranker(
        batch,
        { truncation: true, max_length: 512 } as unknown as Parameters<
          typeof reranker
        >[1],
      )) as unknown;
      const list = Array.isArray(output) ? output : [output];
      for (let j = 0; j < list.length && i + j < scores.length; j++) {
        const raw = list[j] as { score?: number } | undefined;
        scores[i + j] = typeof raw?.score === "number" ? raw.score : 0;
      }
    }

    // 退化检测：所有分数完全相同 = 模型没在做区分，只是把同一个常数吐了 N 遍。
    // 典型成因是 config.json 缺 problem_type，transformers.js 对单 logit 走了
    // softmax（单元素 softmax 恒等于 1），于是全体 1.0、排序原样不动 —— 静默失效。
    // 这种「跑通了但没生效」比直接报错更危险（用户以为在用两阶段检索），必须显式拒绝。
    const degenerate = isDegenerateScores(scores);
    if (degenerate) {
      rerankerUnavailable = true;
      rerankerUnavailableReason = degenerate;
      console.warn(
        `[rerank] 精排输出无区分度（分数恒为 ${scores[0]}），判定为不可用并停用精排：${degenerate}`,
      );
      return null;
    }

    const scored = hits.map((hit, i) => ({ hit, score: scores[i] }));
    scored.sort((a, b) => b.score - a.score);
    // 只返回重排后的命中，分数不外泄：sigmoid 分只在本次排序内有相对意义
    return scored.slice(0, limit).map((s) => s.hit);
  } catch (err) {
    console.warn("[rerank] 精排失败，回退 RRF 排序", err);
    return null;
  }
}


/* ================= 向量索引缓存：存在 worker 内存，检索不搬运向量 ================= */

/** indexId -> 该会话全部分块（身份元数据 + 文本 + 向量）。normalize:true，模长恒为 1 */
const indexCache = new Map<string, RetrievalDoc[]>();

/**
 * indexId -> BM25 关键词索引（**增量维护**）。
 * 纯文本派生（词频统计），从 chunk 文本即可重建，无需持久化到 IndexedDB。
 * 分批上传时只 add 本批文本，不再每批全量重建。
 */
const bm25Cache = new Map<string, BM25>();

/* ---------------- LRU 淘汰：会话切得多不撑爆 worker 内存 ---------------- */

/** 索引缓存上限（会话数）。超出后按最近使用淘汰最旧会话（向量 + BM25 一起清） */
const INDEX_CACHE_LIMIT = 4;
/** LRU 访问序（尾部 = 最近使用）。元素去重后 push 到尾部，队首 = 最久未用 */
const lruOrder: string[] = [];

/** 记录一次 indexId 访问（读取时调用），并淘汰超限的最旧会话 */
function touchIndex(indexId: string): void {
  const idx = lruOrder.indexOf(indexId);
  if (idx >= 0) lruOrder.splice(idx, 1);
  lruOrder.push(indexId);
  while (lruOrder.length > INDEX_CACHE_LIMIT) {
    const evicted = lruOrder.shift();
    if (!evicted) break;
    indexCache.delete(evicted);
    bm25Cache.delete(evicted);
    console.log(
      `[LRU] 向量索引缓存淘汰会话 ${evicted}（缓存上限 ${INDEX_CACHE_LIMIT}）`,
    );
  }
}

/** 取该 indexId 的 BM25 实例（不存在则建空实例并登记） */
function getBm25(indexId: string): BM25 {
  let bm25 = bm25Cache.get(indexId);
  if (!bm25) {
    bm25 = new BM25();
    bm25Cache.set(indexId, bm25);
  }
  return bm25;
}

/**
 * 批量向量化：分批（每批最多 8 条），每批内部一次性传数组给 pipeline，
 * 内部 batch padding 优化。兼顾速度 + 内存安全。
 */
async function embedBatch(
  texts: string[],
  kind: EmbedKind,
  onBatch?: (done: number, total: number) => void,
): Promise<Float32Array[]> {
  const embedder = await getEmbedder();
  const out: Float32Array[] = [];
  // 每批 8 条：既能利用 batch padding 减少调用开销，
  // 又能控制单批内存峰值，避免 WASM 堆溢出
  const batchSize = 8;

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const prefixed = batch.map((t) => applyEmbedPrefix(t, kind));

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

/** 查询向量与整个索引的最大余弦相似度（幻觉校验用：找最相似的一块文档） */
function maxSimilarity(qvec: Float32Array, index: RetrievalDoc[]): number {
  let max = -Infinity; //max === -Infinity 是防御（空索引时返回 0）
  for (const item of index) {
    if (!item.vector) continue;
    const s = dotProduct(qvec, item.vector); //用 dotProduct（归一化向量 = 余弦）
    if (s > max) max = s;
  }
  return max === -Infinity ? 0 : max;
}

/* ================= 加速对比测试（WASM vs WebGPU） ================= */

/**
 * 固定测试语料：短句 + 长句混合，贴近真实分块规模。
 * 两侧跑的是同一批文本、同一套 pooling/normalize 参数，结果才可比。
 */
const BENCH_TEXTS = [
  "向量检索把问题和文档块映射到同一向量空间，用余弦相似度排序。",
  "混合检索把向量排名与 BM25 关键词排名做 RRF 倒数排名融合，兼顾语义与词面。",
  "文档分块后写入本地索引，命中的小块作为引用锚点，喂给大模型的是前后邻块拼出的窗口。",
  "幻觉校验逐句比对原文依据：语义证据取最大余弦，词法证据算 bigram 覆盖率，数字必须在原文出现。",
];

export interface BenchMeasurementRaw {
  device: DeviceKind;
  avgMs: number;
  minMs: number;
  runs: number;
}

/** 在指定设备上创建模型 → 预热一轮（不计时）→ 计时 runs 轮 → 释放实例 */
async function benchOneDevice(
  device: DeviceKind,
  runs: number,
): Promise<BenchMeasurementRaw> {
  const embedder = await createEmbedder(device);
  try {
    // 预热轮：触发 WebGPU 着色器编译 / WASM 线程池就绪，避免首轮耗时虚高
    await warmupEmbedder(embedder);
    const times: number[] = [];
    for (let i = 0; i < runs; i++) {
      const t0 = performance.now();
      const out = await embedder(BENCH_TEXTS, {
        pooling: "mean",
        normalize: true,
      });
      times.push(performance.now() - t0);
      const disposable = out as unknown as { dispose?: () => void };
      if (typeof disposable.dispose === "function") disposable.dispose();
    }
    return {
      device,
      avgMs: times.reduce((a, b) => a + b, 0) / times.length,
      minMs: Math.min(...times),
      runs,
    };
  } finally {
    await releaseEmbedderInstance(embedder);
  }
}

/**
 * 依次在两台设备上跑同一批文本。
 *
 * 顺序执行 + 每个设备跑完立即释放：任何时刻只驻留一份模型权重
 * （WASM 与 WebGPU 两套权重同时驻留极易 std::bad_alloc）。
 * 测试前先释放当前实例腾出内存；结束后缓存保持为空，
 * 下一次真实请求按用户偏好设备重新加载（约 1 秒，可接受）。
 */
async function benchDevices(runs: number): Promise<BenchMeasurementRaw[]> {
  await disposeEmbedder();
  const devices: DeviceKind[] = webgpuBlockedReason
    ? ["wasm"]
    : ["wasm", "webgpu"];
  const results: BenchMeasurementRaw[] = [];
  for (const device of devices) {
    try {
      results.push(await benchOneDevice(device, runs));
    } catch (err) {
      // WebGPU 侧失败：记下原因后跳过（面板显示回退/失败原因），WASM 侧结果照常返回
      if (device !== "webgpu") throw err;
      webgpuBlockedReason = err instanceof Error ? err.message : String(err);
      console.warn("加速对比测试：WebGPU 侧不可用，仅返回 WASM 结果", err);
    }
  }
  return results;
}

/* ================= 消息处理 ================= */

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data as MainRequest | RemoveIndexReq;
  const id = "id" in msg ? msg.id : undefined;

  try {
    switch (msg.type) {
      /* 批量向量化文档 chunk：worker 内缓存索引 + 增量维护 BM25，同时把向量返回主线程持久化。
         append=true 时追加到已有索引（客户端按 8 条/批分次持锁，给实时问答让路） */
      case "embed-passages": {
        const {
          indexId,
          chunks: batchChunks,
          append,
        } = msg as EmbedPassagesReq;
        const texts = batchChunks.map((c) => c.content);
        const vectors = await embedBatch(texts, "passage", (done, total) => {
          self.postMessage({ type: "embed-progress", id, done, total });
        });
        const newEntries: RetrievalDoc[] = batchChunks.map((chunk, i) => ({
          ...chunk,
          vector: vectors[i],
        }));

        // 追加语义：取旧索引拼接；首批或旧索引已被 LRU 淘汰时退化为覆盖
        const prev = append ? indexCache.get(indexId) : undefined;
        const merged = prev ? [...prev, ...newEntries] : newEntries;
        indexCache.set(indexId, merged);
        touchIndex(indexId);

        // BM25 增量维护：append 且旧索引在 → 只 add 本批；否则（覆盖/索引缺失）重建
        const bm25 = getBm25(indexId);
        if (!prev) bm25.reset();
        bm25.add(texts);

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

      /* 恢复 worker 端索引缓存（IndexedDB 恢复 / 导入备份后调用），BM25 重建 */
      case "set-index": {
        const { indexId, chunks } = msg as SetIndexReq;
        indexCache.set(
          indexId,
          chunks.map((c) => ({ ...c, vector: c.vector })),
        );
        touchIndex(indexId);
        const bm25 = getBm25(indexId);
        bm25.reset();
        bm25.add(chunks.map((c) => c.content));
        break;
      }

      /* 检索：query 向量化（可失败降级）+ BM25 → RRF → 邻近合并/去重/MMR/门限 →（可选）精排 */
      case "search": {
        const { indexId, query, topK, retrieval, rerank } = msg as SearchReq;
        const index = indexCache.get(indexId);
        if (!index || index.length === 0) {
          // 带机器可读错误码：索引被 LRU 淘汰属于可自愈状态，
          // 客户端据此从 IndexedDB 重同步后自动重试，而不是让用户重新上传
          self.postMessage({
            type: "error",
            id,
            code: INDEX_MISSING_CODE,
            error: "会话索引不在检索线程（可能已被 LRU 淘汰）",
          });
          break;
        }
        touchIndex(indexId); // LRU：本次检索记为最近使用
        const bm25 = getBm25(indexId);

        // 向量路径失败（模型加载失败/内存不足）时降级为纯 BM25，不阻断问答
        let qvec: Float32Array | null = null;
        let mode: "hybrid" | "bm25" = "hybrid";
        try {
          [qvec] = await embedBatch([query], "query");
        } catch (err) {
          console.warn("向量检索不可用，本次降级为纯 BM25 关键词检索", err);
          mode = "bm25";
        }

        const bm25Scores = bm25.scoreAll(query);
        const vecScores = qvec
          ? index.map((item) =>
              item.vector ? dotProduct(qvec, item.vector) : 0,
            )
          : null;

        const options: RetrievalOptions = {
          topK: Math.max(1, Math.min(topK, index.length)),
          ...DEFAULT_RETRIEVAL_OPTIONS,
          ...retrieval,
        };
        // 拒答信号 1：IDF 加权查询覆盖率。纯词表统计、不需要向量模型，
        // 所以「问的主题库里根本没有」（股票代码/银行账号这类）在纯 BM25 模式下也能识别
        const idfCoverage = bm25.idfWeightedCoverage(query);
        const processing = postProcessRetrieval(index, bm25Scores, vecScores, {
          ...options,
          idfCoverage,
        });

        // 检索侧粗滤结果：IDF 覆盖率（信号 1）+ 余弦门限（信号 2）在此已判定。
        // 随后的精排只重排候选，不会再改 noRelevantHit。
        const processed = processing;

        // 两阶段检索：候选池先精排再截断（模型不可用/内存不足时静默跳过，保持 RRF 顺序）
        let hits = processed.hits;
        let reranked = false;
        let rerankOverlap: number | null = null;
        let rerankTop1Changed: boolean | null = null;
        let rerankMs: number | null = null;
        /** 被哪一路检索侧信号拦下（可观测性）；精排不参与拒答，所以只有 idf */
        const refusedBySignal: "idf" | null = processed.noRelevantHit
          ? "idf"
          : null;
        if (rerank === true && hits.length > 1) {
          const candidates = processed.hits.slice(0, RERANK_RECALL);
          const tRerank = performance.now();
          const refined = await rerankHits(query, candidates, options.topK);
          if (refined) {
            // 精排前后对比：top1 是否换人、top-K 集合重合度（埋点用，衡量精排实际改变了多少）
            const before = candidates.slice(0, options.topK).map((h) => h.chunkIds[0]);
            const after = refined.map((h) => h.chunkIds[0]);
            const overlapCount = after.filter((id) => before.includes(id)).length;
            rerankOverlap = after.length === 0 ? 0 : overlapCount / after.length;
            rerankTop1Changed = before[0] !== after[0];
            rerankMs = performance.now() - tRerank;
            // 精排只重排、不过滤：命中集合与拒答判定不变，「文档未提及」由
            // 答案侧 verifyAnswer（语义 + 词法证据）裁决
            hits = refined;
            reranked = true;
          }
        }

        self.postMessage({
          type: "search-result",
          id,
          mode,
          reranked,
          rerankOverlap,
          rerankTop1Changed,
          rerankMs,
          // 请求了精排却没生效时，把原因带回去（面板显式标注，不再静默）
          rerankSkippedReason:
            rerank === true && !reranked
              ? (rerankerUnavailableReason || "精排模型不可用")
              : null,
          indexSize: index.length,
          noRelevantHit: processed.noRelevantHit,
          refusedBySignal,
          droppedCount: processed.droppedCount,
          // 向量只留在 worker：去掉 vector 字段再回传（结构化克隆会连 Float32Array 一起搬）
          results: hits.map((hit) => {
            const copy: Partial<SearchHitWithVector> = { ...hit };
            delete copy.vector;
            return copy as SearchHit;
          }),
        });
        break;
      }

      /* 幻觉校验：逐句向量化，与全索引算最大余弦，只返回句子+依据分（不传任何向量） */
      case "verify": {
        const { indexId, sentences, scopeChunkIds } = msg as VerifyReq;
        const index = indexCache.get(indexId);
        if (!index || index.length === 0) {
          self.postMessage({
            type: "error",
            id,
            code: INDEX_MISSING_CODE,
            error: "会话索引不在检索线程，无法校验答案依据（可能已被 LRU 淘汰）",
          });
          break;
        }
        touchIndex(indexId); // LRU：本次校验记为最近使用
        if (sentences.length === 0) {
          self.postMessage({ type: "verify-result", id, results: [] });
          break;
        }
        // 校验范围收敛：只在「该句引用的段落 + 其窗口」内找 max。
        // 范围为空（索引不同步 / 引用了别的文档）时退回全索引，宁可宽松也不误杀。
        let scoped = index;
        if (scopeChunkIds && scopeChunkIds.length > 0) {
          const wanted = new Set(scopeChunkIds);
          const filtered = index.filter((d) => wanted.has(d.chunkId));
          if (filtered.length > 0) scoped = filtered;
          else
            console.warn(
              "[verify] 校验范围一块都匹配不上，退回全索引（可能索引不同步）",
            );
        }
        // verify 与 passage 同前缀（无）：校验是「答案句子 vs 文档块」的对称语义比较
        const svecs = await embedBatch(sentences, "verify");
        const results = sentences.map((text, i) => ({
          text,
          score: maxSimilarity(svecs[i], scoped),
        }));
        self.postMessage({
          type: "verify-result",
          id,
          results,
          scopedChunkCount: scoped.length,
        });
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

      /* 只释放精排模型：检索完成后、加载本地 LLM 前调用，把 300~400MB 堆让给 Qwen */
      case "unload-rerank": {
        await unloadReranker();
        self.postMessage({ type: "reranker-unloaded", id });
        break;
      }

      /* 切换推理设备：摘掉当前实例（延后释放），下次请求按新设备重新加载（索引缓存保留） */
      case "set-device": {
        const { device } = msg as SetDeviceReq;
        if (device === preferredDevice) break;
        preferredDevice = device;
        // 重新开关 = 再给 WebGPU 一次机会（上次可能只是显存被别的标签页占着）
        webgpuBlockedReason = null;
        const old = embedderPromise;
        embedderPromise = null;
        if (old) staleInstances.push(old);
        console.log(
          `[device] 向量模型设备偏好切换为 ${device}，下次请求时重新加载`,
        );
        break;
      }

      /* 加速对比测试：WASM / WebGPU 各跑 N 轮，结果交回主线程记入性能埋点 */
      case "bench": {
        const { runs } = msg as BenchReq;
        const measurements = await benchDevices(
          Math.max(1, Math.min(runs, 10)),
        );
        self.postMessage({ type: "bench-result", id, measurements });
        break;
      }

      /* 原样向量化一批文本（不动索引）：评测向量固化等离线工具用 */
      case "embed-texts": {
        const { texts, kind } = msg as EmbedTextsReq;
        const vectors = await embedBatch(texts, kind, (done, total) => {
          self.postMessage({ type: "embed-progress", id, done, total });
        });
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
