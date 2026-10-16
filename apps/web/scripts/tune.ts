#!/usr/bin/env node
/**
 * 检索参数扫描：把「拍脑袋的默认值」换成有曲线支撑的选择。
 *
 * 用法：npm run tune:retrieval        （可选 --json 输出完整结果）
 *
 * 扫描顺序，每段给出该看的那条曲线：
 *
 * 1. maxChunkTokens —— 影响**分块**，所以每换一个值都要重新切块建索引。
 *    看 recall@3 / recall@5 / MRR 与块数。块太小语义不完整，太大稀释主题且顶到模型上限。
 * 2. RRF 权重 × k —— w_vec / w_bm25 / rrfK 三维扫描，外加**语料无关的相对门槛**
 *    hybrid ≥ BM25-only × 0.95（防止融合配置退化到不如纯关键词）。
 *    **需要 packages/eval/embeddings.json**（无向量时无法对比 hybrid 与 BM25-only，本段跳过）。
 * 3. minVecScore —— 相关度弱门限（只粗滤、不承担最终拒答），输出真正的 **P/R 曲线**。
 * 4. mmrLambda —— 多样性权重。λ=1 关闭 MMR。看 recall@3 与 top3 冗余占位率。
 * 5. minIdfCoverage —— IDF 覆盖率门限扫描（默认关闭）。
 * 6. 精排排序增益 —— rerank 只做相对排序、不做绝对分拒答；比较 RRF vs logit 重排指标。
 * 7. 拒答信号可用性诊断 —— 四路候选信号在「零误拒」前提下各能拒掉多少负样本。
 * 8. 向量路准入审计 —— hybrid recall@3 是被余弦准入线还是 MMR 重排拖低的
 *    （三种准入对比 + eval 口径下 λ 对 recall@3 的 q/m 分组影响）。
 * 9. 查询扩展权重 × mmrLambda 联合扫描（M2）—— 扩展与 MMR 都在做"多样性"，
 *    必须一起扫：λ=1.0 + 扩展开时 s 组 0.50，λ=0.9 时扩展的收益被 MMR 吃掉（s 0.36）。
 *    当前默认值（mmrLambda=1.0、扩展权重 0.7、语义粗拒）就是这样定下来的。
 *
 * 设计原则（2026-10-02，方案 1）：检索侧只做粗滤，「文档未提及」的最终置信度裁决
 * 在答案侧由 verifyAnswer 的语义证据 + 词法证据 + 数字核查负责。
 *
 * **调参人群（2026-10-02）**：1~6 段所有选优指标只统计 q* 关键词正样本；
 * golden.json 中 `semantic: true` 的 s* 口语化样本是刻意构造的难例，不参与选参 ——
 * 让门限/块大小去拟合它们会把面向主流关键词查询的最优值带偏。
 * s* 仅在第 7 段以「保守口径」单列做压力观察（eval.ts 仍按 q/s 分组全量报告）。
 *
 * 所有实现都 import src/，与线上同源（见 scripts/eval.ts 顶部说明）。
 */
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { BM25 } from "../src/utils/bm25";
import { createTextChunks, DEFAULT_CHUNK_OPTIONS } from "../src/utils/chunk";
import { normalizeMarkdownForEval } from "../src/utils/evalCorpus";
import {
  evaluateFusion,
  evaluateSignal,
  formatFusionReport,
  formatSignalReports,
  groupOf,
  type SignalName,
  type SignalValues,
} from "./signalDiagnostics";
import { loadFixedRerank, logitToScore } from "./fixedRerank";
import {
  dotProduct,
  postProcessRetrieval,
  computeRefusalSignals,
  DEFAULT_RETRIEVAL_OPTIONS,
  type QueryVariantScores,
  type RetrievalDoc,
} from "../src/utils/retrieval";
import {
  loadFixedEmbeddings,
  loadFixedExpansions,
  findMissingChunkHashes,
  findMissingQueryIds,
} from "./fixedEmbeddings";
import { AutoTokenizer, AutoModelForSequenceClassification, env, pipeline } from "@huggingface/transformers";
import { applyEmbedPrefix } from "../src/utils/embedPrefix";

import { EVAL_DIR } from "./evalDir";

// 与 gen-embeddings.ts 同配置：用本地模型，不走远程下载
env.localModelPath = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "models");
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = false;

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const CORPUS_DIR = join(EVAL_DIR, "corpus");

interface GoldenQuery {
  id: string;
  question: string;
  relevant?: string[];
  keyPoints?: string[];
  negative?: boolean;
  /**
   * 语义（口语化改写）样本标记：查询措辞与原文零重合，是刻意构造的难例。
   * 不参与调参选值（避免把门限/块大小往「兼容难例」的方向带偏），
   * 只在 eval 分组报告与第 7 段信号诊断中作为压力观察。
   */
  semantic?: boolean;
  /**
   * 查询类型（只用于分组统计与选优，不参与命中判定）：
   * - single_topic（缺省）：答案集中在单一块
   * - multi_subject_single_doc（m*）：答案分散在同一文档的多个块/多个窗口
   * 两类的命中口径完全相同：top-k 窗口文本包含 relevant 原句即算命中。
   * 不维护 contentHash —— 改分块参数或改文档不会让标注静默失效。
   */
  type?: "single_topic" | "multi_subject_single_doc";
}
interface Golden {
  queries: GoldenQuery[];
}

const golden = JSON.parse(
  readFileSync(join(EVAL_DIR, "golden.json"), "utf8"),
) as Golden;
const negQueries = golden.queries.filter((q) => q.negative);
/** 语义难例（s*）：只用于观察，不参与任何调参指标 */
const semanticQueries = golden.queries.filter((q) => q.semantic);
/**
 * 调参用正样本：**仅关键词样本（q*）**。
 * s* 口语化样本是难度探针 —— 让门限参数去拟合它们，会把面向真实主流查询
 * （术语一致的关键词提问）的最优值带偏；难例表现交给 eval 分组指标与
 * 第 7 段信号诊断单独观察。
 */
const posQueries = golden.queries.filter((q) => !q.negative && !q.semantic);

/* ---------------- 语料与索引 ---------------- */

/** 按给定分块参数切块（maxChunkTokens 变化时必须重切） */
function buildCorpus(maxChunkTokens: number): RetrievalDoc[] {
  const files = readdirSync(CORPUS_DIR)
    .filter((f) => f.endsWith(".md"))
    .sort();
  // 与 eval.ts loadCorpus 对齐：给每个 chunk 挂上固化向量。
  // 这一步不能省 —— MMR 的多样性判定读的是 doc.vector（retrieval.ts similarity()），
  // 不挂向量会让 MMR 退化为纯相关度排序（λ 失效），
  // 于是 tune 扫出来的 rrfK/vecWeight/λ 结论全部基于一条与 eval/线上不同的管线。
  const fixed = loadFixedEmbeddings();
  const chunks: RetrievalDoc[] = [];
  for (const file of files) {
    const raw = readFileSync(join(CORPUS_DIR, file), "utf8");
    // 与 eval.ts / 浏览器向量固化共用同一份预处理（src/utils/evalCorpus）
    const text = normalizeMarkdownForEval(raw);
    for (const chunk of createTextChunks(file, file, text, {
      ...DEFAULT_CHUNK_OPTIONS,
      maxChunkTokens,
    })) {
      const v = fixed?.chunks[chunk.contentHash];
      chunks.push({ ...chunk, vector: v ? Float32Array.from(v) : null });
    }
  }
  return chunks;
}

/**
 * 实时编码：对当前切块结果在线跑 bge-small-zh，得到与线上一致的向量。
 *
 * 口径对齐（与 gen-embeddings.ts 同链路）：
 * - applyEmbedPrefix：passage 无前缀，query 加中文 instruction
 * - pooling: "mean", normalize: true
 * - dtype q8（与浏览器 WASM 默认一致）
 */
async function encodeLive(
  embedder: Awaited<ReturnType<typeof pipeline<"feature-extraction">>>,
  texts: string[],
  kind: "passage" | "query",
): Promise<number[][]> {
  const out: number[][] = [];
  const BATCH = 16;
  for (let i = 0; i < texts.length; i += BATCH) {
    const batch = texts.slice(i, i + BATCH).map((t) => applyEmbedPrefix(t, kind));
    const result = await embedder(batch, { pooling: "mean", normalize: true });
    const data = result.data as Float32Array;
    const dim = data.length / batch.length;
    for (let j = 0; j < batch.length; j++) {
      const row = Array.from(
        data.subarray(j * dim, (j + 1) * dim),
        (v) => Number(v.toFixed(3)),
      );
      out.push(row);
    }
  }
  return out;
}

/* ---------------- 单次评测 ---------------- */

interface RunOptions {
  minVecScore?: number;
  mmrLambda?: number;
  minIdfCoverage?: number;
  topK?: number;
  /** RRF 融合常数 k（不传走 DEFAULT_RETRIEVAL_OPTIONS.rrfK=60） */
  rrfK?: number;
  /** 向量路 RRF 权重（默认 1） */
  vecWeight?: number;
  /** BM25 路 RRF 权重（默认 1） */
  bm25Weight?: number;
  /** 强制纯 BM25（忽略向量），用于 hybrid vs BM25-only 基线对比 */
  forceBm25?: boolean;
  /** 查询扩展变体的 RRF 权重：undefined = 不开扩展，0 = 只抬拒答信号不参与排序 */
  expansionWeight?: number;
  /**
   * 实时编码向量（绕过 fixedEmbeddings.json 固化文件）。
   * 用于 tune 扫 maxChunkTokens 时，对每个候选块大小动态生成对应向量，
   * 避免「非默认块大小退化为纯 BM25」的口径漂移。
   */
  liveEmbeddings?: {
    chunkVectors: Record<string, number[]>;
    queryVectors: Record<string, number[]>;
  };
}

interface RunMetrics {
  recallAt3: number;
  recallAt5: number;
  mrr: number;
  keyPointRate: number;
  /** 门限二分类：正样本被判「命中」= TP，负样本被判「命中」= FP */
  precision: number;
  recall: number;
  f1: number;
  negativeRefusalRate: number;
  duplicateRate: number;
  avgHits: number;
  /**
   * 文档级 top-1 命中率：top-1 是否来自「包含答案的那篇文档」。
   *
   * 为什么必须看这个：句级 recall 对块大小天生有偏 —— 块越大，一个窗口越可能
   * 顺带盖住关键句，"recall 上升"未必是检索变准了。文档级指标与块大小无关，
   * 只有它才能判断「是不是真的把对的文档排到了第一」。
   */
  docTop1Rate: number;
  /**
   * 本轮是否真的走了向量路。
   * 固化向量只按 DEFAULT_CHUNK_OPTIONS.maxChunkTokens 导出；扫别的块大小时
   * contentHash 全部对不上，此时必须显式按纯 BM25 评估，不能让缺失向量
   * 静默取 0 分（那样 hybrid 分数被污染，曲线看起来「平稳」却是假的）。
   */
  vectorsUsable: boolean;
  /** 分组指标：single（单主题）vs multi（单文档多主题），用于 MMR 扫描对比 */
  byGroup: {
    single: { count: number; r3: number; r5: number; mrr: number; docTop1: number };
    multi: { count: number; r3: number; r5: number; mrr: number; docTop1: number };
  };
}

/** 语料全文缓存：判断某个关键句属于哪篇文档（答案文档） */
const docTextCache = new Map<string, string>();
function docText(docId: string): string {
  let t = docTextCache.get(docId);
  if (t === undefined) {
    t = readFileSync(join(CORPUS_DIR, docId), "utf8");
    docTextCache.set(docId, t);
  }
  return t;
}

/** 该查询的答案文档集合（含任一关键句的文档） */
function goldDocs(q: GoldenQuery, allDocs: string[]): Set<string> {
  const keys = q.relevant ?? [];
  const out = new Set<string>();
  for (const doc of allDocs) {
    const text = docText(doc);
    if (keys.some((k) => text.includes(k))) out.add(doc);
  }
  return out;
}

/**
 * 构造查询扩展变体（与 eval.ts 的 expansionRunsFor 同口径，读同一份固化产物）。
 * weight === undefined 时返回 undefined（不开扩展）。
 */
function expansionRuns(
  q: GoldenQuery,
  bm25: BM25,
  chunks: RetrievalDoc[],
  fixed: ReturnType<typeof loadFixedEmbeddings>,
  live: { chunkVectors: Record<string, number[]>; queryVectors: Record<string, number[]> } | undefined,
  weight: number | undefined,
): QueryVariantScores[] | undefined {
  if (weight === undefined) return undefined;
  const texts = loadFixedExpansions()?.[q.id] ?? [];
  const vectors = fixed?.expansions?.[q.id] ?? [];
  if (texts.length === 0 || vectors.length === 0) return undefined;
  return texts.slice(0, vectors.length).map((text, i) => {
    const qvec = vectors[i];
    const vecScores =
      chunks.length > 0
        ? chunks.map((d) => {
            const v = live?.chunkVectors[d.contentHash] ?? fixed!.chunks[d.contentHash];
            return v ? dotProduct(Float32Array.from(qvec), Float32Array.from(v)) : 0;
          })
        : null;
    return {
      bm25Scores: bm25.scoreAll(text),
      vecScores,
      idfCoverage: bm25.idfWeightedCoverage(text),
      weight,
    };
  });
}

function run(
  chunks: RetrievalDoc[],
  opts: RunOptions,
  bm25 = new BM25(chunks.map((c) => c.content)),
): RunMetrics {
  const topK = opts.topK ?? 5;
  // 优先使用实时编码向量（绕过 fixedEmbeddings.json 固化文件）；
  // 否则回退到固化文件，仅当哈希完全匹配时走向量路。
  const live = opts.liveEmbeddings;
  const fixed = loadFixedEmbeddings();
  const vectorsUsable = !opts.forceBm25 && (!!live ||
    (!!fixed &&
      findMissingChunkHashes(
        fixed,
        chunks.map((c) => c.contentHash),
      ).length === 0 &&
      findMissingQueryIds(
        fixed,
        golden.queries.map((q) => q.id),
      ).length === 0));

  const retrievalOverrides: Partial<typeof DEFAULT_RETRIEVAL_OPTIONS> = {
    ...(opts.minVecScore !== undefined ? { minVecScore: opts.minVecScore } : {}),
    ...(opts.mmrLambda !== undefined ? { mmrLambda: opts.mmrLambda } : {}),
    ...(opts.minIdfCoverage !== undefined
      ? { minIdfCoverage: opts.minIdfCoverage }
      : {}),
    ...(opts.rrfK !== undefined ? { rrfK: opts.rrfK } : {}),
    ...(opts.vecWeight !== undefined ? { vecWeight: opts.vecWeight } : {}),
    ...(opts.bm25Weight !== undefined ? { bm25Weight: opts.bm25Weight } : {}),
  };

  const retrieve = (q: GoldenQuery) => {
    const qvec = live?.queryVectors[q.id] ?? (vectorsUsable ? fixed!.queries[q.id] : undefined);
    const vecScores =
      vectorsUsable && qvec
        ? chunks.map((d) => {
            const v = live?.chunkVectors[d.contentHash] ?? fixed!.chunks[d.contentHash];
            return v ? dotProduct(Float32Array.from(qvec), Float32Array.from(v)) : 0;
          })
        : null;
    const bm25Scores = bm25.scoreAll(q.question);
    // 与 eval.ts / worker 同口径：拒答信号 1 由同一份 BM25 统计算出
    const idfCoverage = bm25.idfWeightedCoverage(q.question);
    const variants = expansionRuns(q, bm25, chunks, fixed, live, opts.expansionWeight);
    const processed = postProcessRetrieval(
      chunks,
      bm25Scores,
      vecScores,
      { topK, ...DEFAULT_RETRIEVAL_OPTIONS, ...retrievalOverrides, idfCoverage },
      variants,
    );
    // MRR 与 eval.ts 同口径：同一条（可能降级的）管线、宽候选 top10 找首个相关块
    const wide = postProcessRetrieval(
      chunks,
      bm25Scores,
      vecScores,
      { topK: 10, ...DEFAULT_RETRIEVAL_OPTIONS, ...retrievalOverrides, idfCoverage },
      variants,
    );
    return { processed, wide };
  };

  const allDocs = readdirSync(CORPUS_DIR).filter((f) => f.endsWith(".md"));

  // 正样本：该命中
  let tp = 0;
  let fn = 0;
  let r3 = 0;
  let r5 = 0;
  let mrr = 0;
  let pointHits = 0;
  let pointTotal = 0;
  let dup = 0;
  let hitSum = 0;
  let docTop1 = 0;
  // 分组累计：single（单主题）vs multi（单文档多主题）
  const gSingle = { count: 0, r3: 0, r5: 0, mrr: 0, docTop1: 0 };
  const gMulti = { count: 0, r3: 0, r5: 0, mrr: 0, docTop1: 0 };

  for (const q of posQueries) {
    const { processed: result, wide } = retrieve(q);
    const relevant = q.relevant ?? [];
    const isMulti = q.type === "multi_subject_single_doc";

    const answered = !result.noRelevantHit;
    if (answered) tp++;
    else fn++;
    hitSum += result.hits.length;

    // 文档级：top-1 是否落在答案文档里
    const gold = goldDocs(q, allDocs);
    const top1InGold = !!(result.hits[0] && gold.has(result.hits[0].docId));
    if (top1InGold) docTop1++;

    // recall@k（q/s/m 统一口径）：top-k 窗口文本包含某条 relevant 原句，
    // 该要点即算命中。多主题查询的多条 relevant 分散在不同块/窗口，
    // recall = 被任一 top-k 窗口覆盖的要点比例。
    const covered = (k: number) =>
      relevant.length === 0
        ? 0
        : relevant.filter((key) =>
            result.hits.slice(0, k).some((h) => h.content.includes(key)),
          ).length / relevant.length;
    const c3 = covered(3);
    const c5 = covered(5);
    r3 += c3;
    r5 += c5;

    // MRR@10：与 eval.ts 同口径，在同管线宽候选里取首个相关窗口排名
    const rank = wide.hits.findIndex((h) =>
      relevant.some((key) => h.content.includes(key)),
    );
    const mrrScore = rank >= 0 ? 1 / (rank + 1) : 0;
    mrr += mrrScore;

    // 分组累计
    const bucket = isMulti ? gMulti : gSingle;
    bucket.count++;
    bucket.r3 += c3;
    bucket.r5 += c5;
    bucket.mrr += mrrScore;
    if (top1InGold) bucket.docTop1++;

    const topText = result.hits.slice(0, topK).map((h) => h.content).join("");
    const points = q.keyPoints ?? [];
    pointHits += points.filter((p) => topText.includes(p)).length;
    pointTotal += points.length;

    // 冗余占位：top3 内同一篇文档的两个「窗口」区间重叠或紧邻（中间不隔块）。
    // 相邻命中块会被 mergeNeighbors 熔成一个窗口（熔掉就不计数——说明合并生效）；
    // 长文档上窗口常因 maxWindowChars 字符上限切成 [0-3]/[4-7] 这样首尾相接的
    // 两个窗口、各占一个 top3 位，此时计 1。远隔窗口（中间隔着 ≥2 个整块）不算，
    // 那是多主题查询需要的正常多窗口覆盖。
    // 这是 mergeNeighbors 是否真正被测到的传感器：合并失效时该值会明显升高。
    const t3 = result.hits.slice(0, 3);
    for (let i = 0; i < t3.length; i++) {
      for (let j = i + 1; j < t3.length; j++) {
        if (
          t3[i].docId === t3[j].docId &&
          t3[i].ordEnd + 1 >= t3[j].ordStart &&
          t3[j].ordEnd + 1 >= t3[i].ordStart
        ) {
          dup++;
        }
      }
    }
  }

  // 负样本：该拒答
  let negativeRefused = 0;
  for (const q of negQueries) {
    if (retrieve(q).processed.noRelevantHit) negativeRefused++;
  }
  const fp = negQueries.length - negativeRefused;

  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const n = posQueries.length;
  return {
    recallAt3: r3 / n,
    recallAt5: r5 / n,
    mrr: mrr / n,
    keyPointRate: pointTotal === 0 ? 0 : pointHits / pointTotal,
    precision,
    recall,
    f1: precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall),
    negativeRefusalRate: negQueries.length === 0 ? 0 : negativeRefused / negQueries.length,
    duplicateRate: dup / n,
    avgHits: hitSum / n,
    docTop1Rate: docTop1 / n,
    vectorsUsable,
    byGroup: {
      single: {
        count: gSingle.count,
        r3: gSingle.count === 0 ? 0 : gSingle.r3 / gSingle.count,
        r5: gSingle.count === 0 ? 0 : gSingle.r5 / gSingle.count,
        mrr: gSingle.count === 0 ? 0 : gSingle.mrr / gSingle.count,
        docTop1: gSingle.count === 0 ? 0 : gSingle.docTop1 / gSingle.count,
      },
      multi: {
        count: gMulti.count,
        r3: gMulti.count === 0 ? 0 : gMulti.r3 / gMulti.count,
        r5: gMulti.count === 0 ? 0 : gMulti.r5 / gMulti.count,
        mrr: gMulti.count === 0 ? 0 : gMulti.mrr / gMulti.count,
        docTop1: gMulti.count === 0 ? 0 : gMulti.docTop1 / gMulti.count,
      },
    },
  };
}

/* ---------------- 输出 ---------------- */

const pct = (v: number) => (v * 100).toFixed(1).padStart(5);
const pad = (v: string | number, n = 8) => String(v).padStart(n);

function printRow(cols: (string | number)[], widths: number[]): void {
  console.log(cols.map((c, i) => pad(c, widths[i])).join(" "));
}

async function sweepChunkTokens(
  embedder?: Awaited<ReturnType<typeof pipeline<"feature-extraction">>>,
): Promise<void> {
  console.log("\n===== 1. maxChunkTokens（分块预算，单位 token）=====");
  console.log(`说明：bge-small-zh 上限 512，当前默认 ${DEFAULT_CHUNK_OPTIONS.maxChunkTokens}。`);
  console.log(
    "┌ 句级 recall 对块大小天生有偏（块越大越容易顺带盖住关键句）；\n" +
      "└ 判优劣请以「文档级 top-1」为准，它与块大小无关。",
  );
  if (embedder) {
    console.log(
      "  方案 A：在线实时编码 → 所有候选块大小均为真实 hybrid 指标，可横向对比。",
    );
  } else {
    console.log(
      "  ⚠ embedding 模型未加载：非默认块大小退化为 bm25（无固化向量），跨块大小对比失真。",
    );
  }
  const widths = [14, 8, 9, 9, 8, 8, 10, 8];
  printRow(
    ["maxChunkTokens", "块数", "文档top1", "recall@3", "MRR", "要点率", "平均命中", "模式"],
    widths,
  );
  const rows: { t: number; m: RunMetrics }[] = [];
  for (const t of [128, 160, 200, 256, 320, 384, 448]) {
    const chunks = buildCorpus(t);
    let liveOpts: RunOptions = {};
    if (embedder) {
      const chunkVecs = await encodeLive(
        embedder,
        chunks.map((c) => c.content),
        "passage",
      );
      const queryVecs = await encodeLive(
        embedder,
        golden.queries.map((q) => q.question),
        "query",
      );
      liveOpts = {
        liveEmbeddings: {
          chunkVectors: Object.fromEntries(
            chunks.map((c, i) => [c.contentHash, chunkVecs[i]]),
          ),
          queryVectors: Object.fromEntries(
            golden.queries.map((q, i) => [q.id, queryVecs[i]]),
          ),
        },
      };
    }
    const m = run(chunks, liveOpts);
    rows.push({ t, m });
    printRow(
      [
        t,
        chunks.length,
        pct(m.docTop1Rate),
        pct(m.recallAt3),
        pct(m.mrr),
        pct(m.keyPointRate),
        m.avgHits.toFixed(1),
        m.vectorsUsable ? "hybrid" : "bm25",
      ],
      widths,
    );
  }
  // 最优判断：全部 hybrid 时可跨块大小直接比较
  const current = DEFAULT_CHUNK_OPTIONS.maxChunkTokens;
  const curRow = rows.find((r) => r.t === current);
  const allHybrid = rows.every((r) => r.m.vectorsUsable);
  if (allHybrid) {
    const bestDoc = rows.reduce(
      (a, b) => (b.m.docTop1Rate > a.m.docTop1Rate ? b : a),
      rows[0],
    );
    console.log(
      `→ 文档级 top-1 最优 maxChunkTokens=${bestDoc.t}（${pct(bestDoc.m.docTop1Rate)}）；` +
        `当前默认 ${current} 为 ${pct(curRow?.m.docTop1Rate ?? 0)}`,
    );
  } else {
    // 部分行是 bm25（无向量），必须限定同一模式内比较
    const curMode = curRow?.m.vectorsUsable ? "hybrid" : "bm25";
    const sameModeRows = rows.filter((r) =>
      curMode === "hybrid" ? r.m.vectorsUsable : !r.m.vectorsUsable,
    );
    const bestDoc = sameModeRows.reduce(
      (a, b) => (b.m.docTop1Rate > a.m.docTop1Rate ? b : a),
      sameModeRows[0],
    );
    console.log(
      `→ 文档级 top-1 最优 maxChunkTokens=${bestDoc.t}（${pct(bestDoc.m.docTop1Rate)}）` +
        `【${curMode} 模式内比较】；` +
        `当前默认 ${current} 为 ${pct(curRow?.m.docTop1Rate ?? 0)}（${curMode}）`,
    );
    const hybridRows = rows.filter((r) => r.m.vectorsUsable);
    if (hybridRows.length <= 2 && curMode === "hybrid") {
      console.log(
        `  ⚠ 注意：hybrid 模式只有 ${hybridRows.length} 个数据点（${hybridRows.map((r) => r.t).join("/")}），` +
          `且它们的指标完全相同 —— 结论"最优"只是当前模式内唯一选择，不是跨模式最优。`,
      );
    }
    if (rows.some((r) => !r.m.vectorsUsable)) {
      console.log(
        "  注：标 bm25 的行没有对应固化向量，只能比较纯关键词路径；跨块大小比较 hybrid 请按目标值重新导出向量。",
      );
    }
    const bm25Rows = rows.filter((r) => !r.m.vectorsUsable);
    if (bm25Rows.length >= 3) {
      const bestBm25 = bm25Rows.reduce((a, b) =>
        b.m.docTop1Rate > a.m.docTop1Rate ? b : a,
      );
      const curIsHybrid = rows.find((r) => r.t === current)?.m.vectorsUsable ?? false;
      console.log(
        `→ 跨块大小比较（同口径，仅 BM25 行可比）：文档级 top-1 最优 maxChunkTokens=${bestBm25.t}` +
          `（${pct(bestBm25.m.docTop1Rate)}）` +
          (curIsHybrid
            ? `；当前默认 ${current} 走的是 hybrid 口径，无法与这些行直接比大小`
            : `；当前默认 ${current} 为 ${pct(rows.find((r) => r.t === current)?.m.docTop1Rate ?? 0)}`),
      );
      console.log(
        "  若要把默认值改成这里的更优值，必须先按该值重新导出 embeddings.json 再验证 hybrid，" +
          "否则等于拿 BM25 的结论去改 hybrid 的配置。",
      );
    }
  }
}

/**
 * RRF 权重 × k 扫描 + 语料无关的相对门槛。
 *
 * 为什么需要相对门槛：绝对指标（recall 0.8 还是 0.6）随语料难度漂移，
 * 无法判断「加了向量路到底有没有用」。但「hybrid 不应显著差于纯 BM25」
 * 是语料无关的不变量 —— 向量路再差也不该把关键词路的结果拖低 5% 以上。
 * 门槛：hybrid recall@3 ≥ BM25-only recall@3 × 0.95。
 *
 * @returns default-pass 默认配置达标 / alternative-pass 默认不达标但有替代配置 / all-fail 全部不达标
 */
function sweepRrfWeights(): "default-pass" | "alternative-pass" | "all-fail" {
  console.log("\n===== 2. RRF 权重 × k（hybrid 不允许显著低于 BM25-only）=====");
  console.log(
    "score = w_vec/(k+vecRank) + w_bm25/(k+bm25Rank)；等权（1/1, k=60）= 原始 RRF。\n" +
      "相对门槛（语料无关）：hybrid recall@3 ≥ BM25-only recall@3 × 0.95，否则标 ✗。",
  );
  const chunks = buildCorpus(DEFAULT_CHUNK_OPTIONS.maxChunkTokens);
  const bm25 = new BM25(chunks.map((c) => c.content));
  const fixed = loadFixedEmbeddings();
  if (!fixed) {
    console.log("⚠ 缺 embeddings.json：无法对比 hybrid 与 BM25-only，本段跳过。");
    return "default-pass";
  }

  // BM25-only 基线（强制不走向量路，即线上向量模型加载失败时的降级路径）
  const baseline = run(chunks, { forceBm25: true }, bm25);
  console.log(
    `BM25-only 基线：文档top1=${pct(baseline.docTop1Rate)} recall@3=${pct(baseline.recallAt3)} MRR=${pct(baseline.mrr)}`,
  );
  const gate = baseline.recallAt3 * 0.95;
  console.log(
    `门槛值：hybrid recall@3 ≥ ${pct(gate)}（基线 ${pct(baseline.recallAt3)} × 0.95）`,
  );

  const widths = [10, 6, 9, 9, 8, 10, 8];
  printRow(["vecWeight", "k", "文档top1", "recall@3", "MRR", "r3/基线", "门槛"], widths);
  const DEF_W = DEFAULT_RETRIEVAL_OPTIONS.vecWeight;
  const DEF_K = DEFAULT_RETRIEVAL_OPTIONS.rrfK;
  const rows: { w: number; k: number; m: RunMetrics; pass: boolean; isDefault: boolean }[] = [];
  for (const w of [0.5, 1, 1.5, 2]) {
    for (const k of [20, 40, 60, 80]) {
      const m = run(chunks, { vecWeight: w, rrfK: k }, bm25);
      const pass = m.recallAt3 >= gate;
      const isDefault = w === DEF_W && k === DEF_K;
      rows.push({ w, k, m, pass, isDefault });
      printRow(
        [
          w.toFixed(1) + (isDefault ? " *" : ""),
          k,
          pct(m.docTop1Rate),
          pct(m.recallAt3),
          pct(m.mrr),
          `${Math.round((m.recallAt3 / Math.max(baseline.recallAt3, 1e-9)) * 100)}%`,
          pass ? "✓" : "✗",
        ],
        widths,
      );
    }
  }
  console.log("  * = 当前默认配置；bm25Weight 固定为 1（只扫两路相对权重比 vecWeight）。");

  const def = rows.find((r) => r.isDefault)!;
  const passing = rows.filter((r) => r.pass);
  if (def.pass) {
    console.log(
      `→ 默认配置（w_vec=${DEF_W}, k=${DEF_K}）通过相对门槛：hybrid 没有拖低于 BM25-only × 0.95。`,
    );
    // 即使达标，也提示是否存在更优配置（仅提示，不建议为微小差异改动）
    const better = passing.filter(
      (r) => !r.isDefault && (r.m.recallAt3 > def.m.recallAt3 || r.m.mrr > def.m.mrr),
    );
    if (better.length > 0) {
      const best = better.reduce((a, b) =>
        b.m.recallAt3 > a.m.recallAt3 ||
        (b.m.recallAt3 === a.m.recallAt3 && b.m.mrr > a.m.mrr)
          ? b
          : a,
      );
      console.log(
        `  （参考）达标配置中最高 recall@3=${pct(best.m.recallAt3)}（w_vec=${best.w}, k=${best.k}），` +
          `默认 recall@3=${pct(def.m.recallAt3)}；默认已达标，无需为该差异改动。`,
      );
    }
    return "default-pass";
  }
  if (passing.length > 0) {
    const best = passing.reduce((a, b) =>
      b.m.recallAt3 > a.m.recallAt3 ||
      (b.m.recallAt3 === a.m.recallAt3 && b.m.mrr > a.m.mrr)
        ? b
        : a,
    );
    console.log(
      `⚠ 默认配置（w_vec=${DEF_W}, k=${DEF_K}）未通过门槛（recall@3=${pct(def.m.recallAt3)} < ${pct(gate)}）。\n` +
        `  有 ${passing.length} 个配置达标，最优 w_vec=${best.w}, k=${best.k}：` +
        `recall@3=${pct(best.m.recallAt3)} / MRR=${pct(best.m.mrr)}。\n` +
        `  二选一：① 把 DEFAULT_RETRIEVAL_OPTIONS 改成该配置；② 保留默认并在 docs/eval/README.md\n` +
        `  显式记录「hybrid 相对 BM25-only 的偏差及取舍」。不处理则下条门槛会失败。`,
    );
    return "alternative-pass";
  }
  console.log(
    `✗ 全部 ${rows.length} 个权重/k 配置都未通过门槛 —— 向量路在本语料上系统性有害，\n` +
      `  问题不在融合参数，应检查向量口径（query instruction / 归一化 / 模型选型）。`,
  );
  return "all-fail";
}

function sweepMinVecScore(): void {
  console.log("\n===== 3. minVecScore（相关度门限）=====");
  const fixed = loadFixedEmbeddings();
  if (!fixed) {
    console.log(
      "⚠ 缺 packages/eval/embeddings.json：没有向量分，该参数在纯 BM25 下不起作用，曲线会是一条直线。",
    );
    console.log(
      "  生成方法：npm run gen:embeddings（见 docs/eval/README.md）。",
    );
  }
  console.log("看 P/R：precision = 判为命中的里有多少是该命中的；recall = 该命中的里有多少被判为命中。");
  const widths = [12, 10, 10, 8, 8, 8, 12];
  const chunks = buildCorpus(DEFAULT_CHUNK_OPTIONS.maxChunkTokens);
  const bm25 = new BM25(chunks.map((c) => c.content));

  // 先打印 top1 vecScore 分布，解释为什么曲线会平
  const vecDistribution = { pos: [] as number[], neg: [] as number[] };
  const vectorsUsable =
    !!fixed &&
    findMissingChunkHashes(fixed, chunks.map((c) => c.contentHash)).length === 0 &&
    findMissingQueryIds(fixed, golden.queries.map((q) => q.id)).length === 0;
  for (const q of [...posQueries, ...negQueries]) {
    const qvec = vectorsUsable ? fixed!.queries[q.id] : undefined;
    const vecScores =
      vectorsUsable && qvec
        ? chunks.map((d) =>
            dotProduct(
              Float32Array.from(qvec),
              Float32Array.from(fixed!.chunks[d.contentHash]),
            ),
          )
        : null;
    const bm25Scores = bm25.scoreAll(q.question);
    const wide = postProcessRetrieval(chunks, bm25Scores, vecScores, {
      topK: 10,
      ...DEFAULT_RETRIEVAL_OPTIONS,
      minIdfCoverage: 0,
    });
    const topScore = wide.hits[0]?.relevance ?? 0;
    if (q.negative) vecDistribution.neg.push(topScore);
    else vecDistribution.pos.push(topScore);
  }
  console.log(
    `top1 vecScore 分布：q 关键词正样本均值 ${(vecDistribution.pos.reduce((a, b) => a + b, 0) / vecDistribution.pos.length).toFixed(3)}` +
      `（最低 ${Math.min(...vecDistribution.pos).toFixed(3)}）` +
      ` / 负样本均值 ${(vecDistribution.neg.reduce((a, b) => a + b, 0) / vecDistribution.neg.length).toFixed(3)}` +
      `（最高 ${Math.max(...vecDistribution.neg).toFixed(3)}）；s 语义样本不参与本段）`,
  );
  if (Math.min(...vecDistribution.pos) <= Math.max(...vecDistribution.neg)) {
    console.log(
      "  ⚠ 两类分布重叠（负样本最高分 ≥ 正样本最低分），minVecScore 无法同时零误拒。",
    );
  }
  printRow(
    ["minVecScore", "precision", "recall", "F1", "漏检数", "拒答率", "平均命中"],
    widths,
  );

  let best: { v: number; f1: number } | null = null;
  for (let v = 0.2; v <= 0.6001; v += 0.05) {
    const t = Number(v.toFixed(2));
    // 扫 minVecScore 时关掉 IDF 覆盖率门限，只看余弦门限的边界
    const m = run(chunks, { minVecScore: t, minIdfCoverage: 0 }, bm25);
    printRow(
      [
        t.toFixed(2),
        pct(m.precision),
        pct(m.recall),
        pct(m.f1),
        posQueries.length - Math.round(m.recall * posQueries.length),
        pct(m.negativeRefusalRate),
        m.avgHits.toFixed(1),
      ],
      widths,
    );
    // 只有本轮确实走了向量路（固化向量与默认切块完全对得上），F1 最优点才有意义；
    // 文件缺失/内容陈旧导致的纯 BM25 行不能参与 minVecScore 选优
    if (m.vectorsUsable && (!best || m.f1 > best.f1)) best = { v: t, f1: m.f1 };
  }
  if (best) {
    const current = DEFAULT_RETRIEVAL_OPTIONS.minVecScore;
    console.log(
      `→ F1 最优 minVecScore=${best.v.toFixed(2)}（当前默认 ${current}），F1=${pct(best.f1)}`,
    );
  } else if (fixed) {
    console.log(
      "⚠ embeddings.json 存在但与当前默认切块对不上（vectorsUsable=false），" +
        "minVecScore 曲线实际是纯 BM25。请重新执行 npm run gen:embeddings 导出。",
    );
  }
}

/**
 * 向量路准入审计：回答「hybrid recall@3 被什么拖低」。
 *
 * 由来（2026-10-15）：hybrid 曾比 BM25-only 低 13 个点（81.1 vs 94.6），一度怀疑向量口径。
 * 本段把嫌疑逐个排除：RRF 顺序全量 top3 与 BM25-only 齐平（94.6），准入审计显示只有极少数
 * 正样本的答案被余弦准入线过滤 —— 真正的差额来自 MMR 的多样性重排（见第 4 段 λ 曲线）。
 *
 * hybrid 模式下 applyThreshold 的准入判据是**只看余弦**（minVecScore=0.5，
 * 不给 BM25 豁免）；而 BM25-only 基线的准入是 bm25Rank≤3。两者口径不对称 ——
 * 一个 BM25 排第 1、但余弦 <0.5 的正确块，在 hybrid 里会被直接丢弃。
 *
 * 本段对每条受保护正样本，把窗口按 RRF 顺序全量列出（关掉准入/相对切/MMR），
 * 找出「内容含答案」的 oracle 窗口，看它的最高余弦与 BM25 名次，并对比三种准入：
 *   全量（不过滤） / 余弦准入（cos≥0.5） / 并集准入（cos≥0.5 或 bm25Rank≤3）；
 * 另附 eval 口径下 mmrLambda 对 hybrid recall@3 的影响（含 q/m 分组）。
 */
function auditVectorAdmission(): void {
  console.log("\n===== 8. 向量路准入审计 =====");
  const chunks = buildCorpus(DEFAULT_CHUNK_OPTIONS.maxChunkTokens);
  const bm25 = new BM25(chunks.map((c) => c.content));
  const fixed = loadFixedEmbeddings();
  const MINVEC = DEFAULT_RETRIEVAL_OPTIONS.minVecScore;
  const base = {
    topK: 30,
    ...DEFAULT_RETRIEVAL_OPTIONS,
    mmrLambda: 1, // 关 MMR：保留纯 RRF 顺序
    minVecScore: -1, // 准入全放行
    relativeCutRatio: 0,
    minIdfCoverage: 0,
    minMaxCosine: 0,
    minCosGap: 0,
    minBm25: 0,
  };

  let r3All = 0;
  let r3Cos = 0;
  let r3Union = 0;
  let failCos = 0;
  let notAdmitted = 0;
  let bm25Rescuable = 0;
  const detail: string[] = [];

  for (const q of posQueries) {
    const relevant = q.relevant ?? [];
    const qvec = fixed?.queries[q.id];
    const vecScores =
      qvec && fixed
        ? chunks.map((d) => {
            const v = fixed.chunks[d.contentHash];
            return v
              ? dotProduct(Float32Array.from(qvec), Float32Array.from(v))
              : 0;
          })
        : null;
    const bm25Scores = bm25.scoreAll(q.question);
    const audit = postProcessRetrieval(chunks, bm25Scores, vecScores, {
      ...base,
      idfCoverage: bm25.idfWeightedCoverage(q.question),
    });
    const hits = audit.hits;
    const oracle = hits.filter((h) =>
      relevant.some((key) => h.content.includes(key)),
    );
    const ansBestCos = oracle.length
      ? Math.max(...oracle.map((h) => h.vecScore ?? 0))
      : -1;
    const ansBestBm25Rank = oracle.length
      ? Math.min(...oracle.map((h) => h.bm25Rank))
      : 999;

    const cov = (list: typeof hits, k: number) =>
      relevant.length === 0
        ? 0
        : relevant.filter((key) =>
            list.slice(0, k).some((h) => h.content.includes(key)),
          ).length / relevant.length;

    const r3c = cov(
      hits.filter((h) => (h.vecScore ?? 0) >= MINVEC),
      3,
    );
    const r3u = cov(
      hits.filter((h) => (h.vecScore ?? 0) >= MINVEC || h.bm25Rank <= 3),
      3,
    );
    r3All += cov(hits, 3);
    r3Cos += r3c;
    r3Union += r3u;

    if (r3c < 1) {
      failCos++;
      if (ansBestCos < MINVEC) notAdmitted++;
      if (ansBestBm25Rank <= 3) bm25Rescuable++;
      detail.push(
        `  ${q.id} r3(余弦准入)=${r3c.toFixed(2)} r3(并集)=${r3u.toFixed(2)} ` +
          `答案最高余弦=${ansBestCos.toFixed(3)} 答案BM25名次=${ansBestBm25Rank === 999 ? "-" : ansBestBm25Rank} ` +
          `oracle窗口=${oracle.length} 总窗口=${hits.length}`,
      );
    }
  }

  console.log(detail.join("\n"));
  const n = posQueries.length;
  console.log(
    `汇总（受保护 ${n} 条）：RRF 顺序全量 top3=${pct(r3All / n)} | ` +
      `余弦准入=${pct(r3Cos / n)} | 并集准入=${pct(r3Union / n)}`,
  );
  console.log(
    `  hybrid 漏检 ${failCos} 条中：答案最高余弦<${MINVEC}（被准入线过滤）${notAdmitted} 条；` +
      `答案 BM25 名次≤3（加 BM25 豁免即可救回）${bm25Rescuable} 条。`,
  );

  // 第二步：eval 口径（minVec=0.5 / ratio=0.4 / topK=5）下，看 MMR λ 对 recall@3 的影响
  console.log("\n  —— eval 口径下 mmrLambda 对 hybrid recall@3 的影响 ——");
  for (const lambda of [0.7, 0.8, 0.9, 1.0]) {
    let r3 = 0;
    let q3 = 0;
    let qN = 0;
    let m3 = 0;
    let mN = 0;
    for (const q of posQueries) {
      const relevant = q.relevant ?? [];
      const qvec = fixed?.queries[q.id];
      const vecScores =
        qvec && fixed
          ? chunks.map((d) => {
              const v = fixed.chunks[d.contentHash];
              return v
                ? dotProduct(Float32Array.from(qvec), Float32Array.from(v))
                : 0;
            })
          : null;
      const bm25Scores = bm25.scoreAll(q.question);
      const res = postProcessRetrieval(chunks, bm25Scores, vecScores, {
        topK: 5,
        ...DEFAULT_RETRIEVAL_OPTIONS,
        mmrLambda: lambda,
        idfCoverage: bm25.idfWeightedCoverage(q.question),
      });
      const c =
        relevant.length === 0
          ? 0
          : relevant.filter((key) =>
              res.hits.slice(0, 3).some((h) => h.content.includes(key)),
            ).length / relevant.length;
      r3 += c;
      if (q.type === "multi_subject_single_doc") {
        m3 += c;
        mN++;
      } else {
        q3 += c;
        qN++;
      }
    }
    console.log(
      `  mmrLambda=${lambda.toFixed(1)}  hybrid recall@3=${pct(r3 / posQueries.length)}` +
        `（q 单主题 ${pct(q3 / qN)} / m 多主题 ${pct(m3 / mN)}）`,
    );
  }
}

function sweepMmrLambda(): void {
  console.log("\n===== 4. mmrLambda（MMR 多样性权重，1 = 关闭 MMR）=====");
  console.log(
    "看 recall@3 / MRR / 冗余占位率。λ 太小会为了多样性牺牲相关性；\n" +
      "  λ=1 关闭 MMR，多样性=0 但相关性最强。单主题查询期望 λ 偏高，\n" +
      "  多主题查询期望 λ 偏低（让不同主题的块都进 topK）。",
  );
  console.log(
    `评估集：单主题 ${posQueries.filter((q) => q.type !== "multi_subject_single_doc").length} 条 + ` +
      `多主题 ${posQueries.filter((q) => q.type === "multi_subject_single_doc").length} 条（m*）`,
  );
  const chunks = buildCorpus(DEFAULT_CHUNK_OPTIONS.maxChunkTokens);
  const bm25 = new BM25(chunks.map((c) => c.content));

  // 总表（聚合）
  const wAgg = [10, 10, 10, 8, 14, 10];
  console.log("\n【聚合指标】");
  printRow(["mmrLambda", "recall@3", "recall@5", "MRR", "冗余占位率", "平均命中"], wAgg);
  const rows: { l: number; m: RunMetrics }[] = [];
  for (const l of [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) {
    const m = run(chunks, { mmrLambda: l }, bm25);
    rows.push({ l, m });
    printRow(
      [
        l.toFixed(1),
        pct(m.recallAt3),
        pct(m.recallAt5),
        pct(m.mrr),
        m.duplicateRate.toFixed(2),
        m.avgHits.toFixed(1),
      ],
      wAgg,
    );
  }

  // 分组对比表：single vs multi 的 recall@3 / MRR
  console.log("\n【分组对比：单主题 vs 多主题】");
  const wGrp = [10, 10, 10, 10, 10, 10];
  printRow(["mmrLambda", "single r3", "single MRR", "multi r3", "multi MRR", "multi docTop1"], wGrp);
  for (const r of rows) {
    printRow(
      [
        r.l.toFixed(1),
        pct(r.m.byGroup.single.r3),
        pct(r.m.byGroup.single.mrr),
        pct(r.m.byGroup.multi.r3),
        pct(r.m.byGroup.multi.mrr),
        pct(r.m.byGroup.multi.docTop1),
      ],
      wGrp,
    );
  }

  // 选优建议：
  // - 若 multi 组 recall@3 在所有 λ 下相同（窗口已覆盖全部答案块），MMR 对多主题无影响，
  //   直接取 single 组最优 λ（多主题不吃亏，单主题收益最大）。
  // - 若 multi 组随 λ 变化，取 multi 最优，但不能让 single 掉超过 5%。
  const multiScores = rows.map((r) => r.m.byGroup.multi.r3);
  const multiFlat = multiScores.every((s) => s === multiScores[0]);
  // 选优：先比 recall@3，并列时用 MRR 打破平局（r3 全平时 reduce 会误取首个 λ）
  const bestByGroup = (g: "single" | "multi") =>
    rows.reduce((a, b) => {
      if (b.m.byGroup[g].r3 !== a.m.byGroup[g].r3) {
        return b.m.byGroup[g].r3 > a.m.byGroup[g].r3 ? b : a;
      }
      return b.m.byGroup[g].mrr > a.m.byGroup[g].mrr ? b : a;
    });
  const multiBest = bestByGroup("multi");
  const singleBest = bestByGroup("single");
  console.log(
    `\n→ 单主题最优 λ=${singleBest.l.toFixed(1)}（recall@3=${pct(singleBest.m.byGroup.single.r3)}）；` +
      `多主题最优 λ=${multiBest.l.toFixed(1)}（recall@3=${pct(multiBest.m.byGroup.multi.r3)}）`,
  );
  if (multiFlat) {
    console.log(
      `  多主题 recall@3 在所有 λ 下均为 ${pct(multiScores[0])}（窗口已覆盖全部答案块，MMR 对多主题无影响），` +
        `取单主题最优 λ=${singleBest.l.toFixed(1)}。`,
    );
  } else if (singleBest.l === multiBest.l) {
    console.log(`  两组最优一致，推荐 mmrLambda=${multiBest.l.toFixed(1)}。`);
  } else {
    // 两组最优不同时：在「保持多主题最优」的 λ 集合里，取单主题最大者。
    //
    // 不再取「中点」——中点只是位置上的折中，不最优化任何指标。实测（2026-10-15）：
    // multi 在 λ≤0.9 全为 100，single 随 λ 单调增（0.7→72 / 0.8→84 / 0.9→88 / 1.0→96）；
    // 中点会落到 0.8，其合计 recall@3=89.2 恰好低于相对门槛（BM25-only×0.95=89.9），
    // 于是「工具推荐值」与「CI 门槛」互相打架。改取「multi 不降前提下 single 最大」= 0.9
    // （合计 91.9，过门槛），两目标同时最优，不再需要人为折中。
    const keepMulti = rows.filter((r) => r.m.byGroup.multi.r3 === multiBest.m.byGroup.multi.r3);
    const pick = keepMulti.reduce((a, b) =>
      b.m.byGroup.single.r3 > a.m.byGroup.single.r3 ? b : a,
    );
    console.log(
      `  多主题最优 λ=${multiBest.l.toFixed(1)}；在「多主题不掉分」的 λ 里取单主题最大者，` +
        `推荐 mmrLambda=${pick.l.toFixed(1)}` +
        `（单主题 recall@3=${pct(pick.m.byGroup.single.r3)}，多主题=${pct(pick.m.byGroup.multi.r3)}，` +
        `合计=${pct(pick.m.recallAt3)}）。`,
    );
  }

  // MMR 是否真正触发：若所有 λ 的冗余占位率都是 0，说明 top3 没有同文档重叠窗口
  if (rows.every((r) => r.m.duplicateRate === 0)) {
    console.log(
      "  注：所有 λ 的冗余占位率均为 0 —— MMR 未触发重排（top3 无同文档重叠窗口），" +
        "分组差异来自 MMR 对跨文档多样性的影响。",
    );
  }
}

/**
 * 三信号拒答的门限扫描。
 *
 * 拒答是二分类：负样本 = 该拒答，正样本 = 不该拒答。
 * 因此 precision = 拒答里有多少确实该拒，recall = 该拒的里拒掉多少。
 * 单独看 negativeRefusalRate 会把系统调成哑巴 —— 什么都拒答时它是 100%，
 * 所以必须与 falseRefusal（正样本被误拒）成对看。
 */
function sweepRefusalSignals(): void {
  console.log("\n===== 5. 拒答信号（minIdfCoverage 扫描）=====");
  console.log(
    "本段正样本只含 q 关键词样本（调参口径）；是否启用请看第 7 段信号诊断的两口径对比 ——\n" +
      "  调参口径决定默认值，保守口径（含 s 难例）只作压力观察。",
  );
  console.log(
    "信号 1 = IDF 加权查询覆盖率：查询词的信息量有多少能在语料词表里找到。\n" +
      "纯词表统计，不需要向量模型 —— 问一个库里没有的主题（股票代码/银行账号）时它会明显偏低。",
  );
  const chunks = buildCorpus(DEFAULT_CHUNK_OPTIONS.maxChunkTokens);
  const bm25 = new BM25(chunks.map((c) => c.content));

  // 先看两类的覆盖率分布，理解可分性
  const posCov = posQueries.map((q) => bm25.idfWeightedCoverage(q.question));
  const negCov = negQueries.map((q) => bm25.idfWeightedCoverage(q.question));
  const avg = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
  console.log(
    `覆盖率分布：q 关键词正样本均值 ${avg(posCov).toFixed(3)}（最低 ${Math.min(...posCov).toFixed(3)}）` +
      ` / 负样本均值 ${avg(negCov).toFixed(3)}（最高 ${Math.max(...negCov).toFixed(3)}）`,
  );
  // s 语义样本不参与选参，但把它们的覆盖率摆在这里做对照观察：
  // 若大量 s* 覆盖率为 0，说明该信号一旦启用就会系统性误伤口语化提问
  if (semanticQueries.length > 0) {
    const semCov = semanticQueries.map((q) => bm25.idfWeightedCoverage(q.question));
    const zeroSem = semanticQueries.filter(
      (q) => bm25.idfWeightedCoverage(q.question) === 0,
    );
    console.log(
      `  （观察）s 语义样本覆盖率均值 ${avg(semCov).toFixed(3)}（最低 ${Math.min(...semCov).toFixed(3)}），` +
        `其中 ${zeroSem.length} 条为 0.000${zeroSem.length > 0 ? `：${zeroSem.map((q) => q.id).join("、")}` : ""} —— ` +
        `这些难例不参与本段选参，只提示该信号对口语化提问的代价。`,
    );
  }

  const widths = [14, 10, 10, 9, 8, 8, 10];
  printRow(
    ["minIdfCoverage", "precision", "recall", "F1", "拒答率", "误拒数", "残留命中"],
    widths,
  );
  let best: { t: number; f1: number; fp: number } | null = null;
  for (let v = 0.05; v <= 0.6001; v += 0.05) {
    const t = Number(v.toFixed(2));
    // 扫 minIdfCoverage 时关掉余弦门限，只看词表覆盖率的边界
    const m = run(chunks, { minIdfCoverage: t, minVecScore: -1 }, bm25);
    const refused = Math.round(m.negativeRefusalRate * negQueries.length);
    const falseRefusal = Math.round((1 - m.recall) * posQueries.length);
    printRow(
      [
        t.toFixed(2),
        pct(m.precision),
        pct(m.recall),
        pct(m.f1),
        pct(m.negativeRefusalRate),
        falseRefusal,
        refused === negQueries.length ? 0 : negQueries.length - refused,
      ],
      widths,
    );
    // 只在「零误拒」里挑 F1 最高：误拒正样本是直接可感的体验损伤
    if (falseRefusal === 0 && (!best || m.f1 > best.f1)) best = { t, f1: m.f1, fp: falseRefusal };
  }
  if (best) {
    console.log(
      `→ 零误拒前提下 F1 最优 minIdfCoverage=${best.t.toFixed(2)}（F1=${pct(best.f1)}）` +
        `；当前默认 ${DEFAULT_RETRIEVAL_OPTIONS.minIdfCoverage}`,
    );
    // 与 s 难例的冲突预警：调参口径选出的门限可能系统性误伤口语化样本。
    // eval.ts 的门槛仍要求全部正样本（含 s）零误拒，不处理这个冲突直接写回默认值会 FAIL。
    if (best.t > 0 && semanticQueries.length > 0) {
      const semCov = semanticQueries.map((q) => bm25.idfWeightedCoverage(q.question));
      const semHit = semCov.filter((c) => c < best.t).length;
      if (semHit > 0) {
        console.log(
          `  ⚠ 写回前注意：该门限在调参口径（q）下零误拒，但会误伤 ${semHit}/${semanticQueries.length} 条 s 语义样本。\n` +
            `    eval.ts 门槛仍要求含 s 的全部正样本零误拒 —— 直接写回会 FAIL。\n` +
            `    要启用此门限，需先决定 eval 门槛是否也按 q 口径放宽（否则保持 0，难例交给答案侧证据校验）。`,
        );
      }
    }
  } else {
    console.log("→ 没有任何门限能同时做到零误拒：检索侧信号到此为止，拒答置信度改由答案侧证据校验负责。");
  }
}

/**
 * 精排**相对排序增益**评估（第 6 段）。
 *
 * 设计决策（2026-10-02，方案 1）：cross-encoder 擅长相对排序、不擅长输出跨查询
 * 可比的全局绝对置信度。bge-reranker-base 原始 logit 实测：正样本均值 0.78 但存在
 * -10.18 离群值，负样本最高 -1.13，两类分布重叠 —— 任何绝对门限都会误拒正样本。
 * 所以线上精排**只重排候选、不做过滤拒答**，「文档未提及」交给答案侧
 * verifyAnswer（语义证据 + 词法证据 + 数字核查）裁决。
 *
 * 本段要回答的问题因此变成：**精排的排序到底有没有用？**
 * 用同管线 top10 宽候选，比较两种顺序：
 *   - 基线：postProcessRetrieval 给出的 RRF 顺序
 *   - 精排：按原始 logit 降序重排（输入格式与 worker 一致：query</s></s>passage）
 * 指标：文档级 top-1 / recall@3 / recall@5 / MRR@10，以及单查询 MRR 改善/退步数。
 *
 * 负样本不参与此表（精排不做拒答；负样本在检索侧仍会返回候选）。
 *
 * 前置条件：public/models/Xenova/bge-reranker-base 必须存在（npm run gen:embeddings 会下载）。
 * 模型不存在时输出警告并跳过。
 */
async function evaluateRerankGain(): Promise<void> {
  console.log("\n===== 6. 精排排序增益（rerank 只重排、不做拒答）=====");
  console.log(
    "cross-encoder 的原始 logit 跨查询不可比（正样本存在 -10 离群值，与负样本重叠），\n" +
      "绝对门限必然误拒 → 精排只做相对排序；「文档未提及」由答案侧 verifyAnswer 裁决。\n" +
      "本段比较同候选池下「RRF 基线顺序」与「logit 重排顺序」的排序指标。",
  );

  // 检查模型是否可用
  const modelPath = join(ROOT, "public", "models", "Xenova", "bge-reranker-base");
  if (!existsSync(modelPath)) {
    console.log(
      "⚠ 精排模型未下载：public/models/Xenova/bge-reranker-base 不存在。",
    );
    console.log("  请先执行 npm run gen:embeddings（会自动下载模型）。");
    return;
  }

  console.log("加载精排模型（dtype=q8，原始 logits）...");
  const tokenizer = await AutoTokenizer.from_pretrained("Xenova/bge-reranker-base");
  const model = await AutoModelForSequenceClassification.from_pretrained("Xenova/bge-reranker-base", {
    dtype: "q8",
  });
  console.log("模型加载完成。\n");

  const chunks = buildCorpus(DEFAULT_CHUNK_OPTIONS.maxChunkTokens);
  const bm25 = new BM25(chunks.map((c) => c.content));
  const fixed = loadFixedEmbeddings();
  const vectorsUsable =
    !!fixed &&
    findMissingChunkHashes(fixed, chunks.map((c) => c.contentHash)).length === 0 &&
    findMissingQueryIds(fixed, golden.queries.map((q) => q.id)).length === 0;
  const allDocs = readdirSync(CORPUS_DIR).filter((f) => f.endsWith(".md"));

  interface OrderMetrics {
    docTop1: number;
    r3: number;
    r5: number;
    mrr: number;
  }
  const base: OrderMetrics = { docTop1: 0, r3: 0, r5: 0, mrr: 0 };
  const ranked: OrderMetrics = { docTop1: 0, r3: 0, r5: 0, mrr: 0 };
  let top1Changed = 0;
  let mrrImproved = 0;
  let mrrRegressed = 0;

  for (const q of posQueries) {
    const qvec = vectorsUsable ? fixed!.queries[q.id] : undefined;
    const vecScores =
      vectorsUsable && qvec
        ? chunks.map((d) =>
            dotProduct(
              Float32Array.from(qvec),
              Float32Array.from(fixed!.chunks[d.contentHash]),
            ),
          )
        : null;
    const bm25Scores = bm25.scoreAll(q.question);
    const wide = postProcessRetrieval(chunks, bm25Scores, vecScores, {
      topK: 10,
      ...DEFAULT_RETRIEVAL_OPTIONS,
    });
    const candidates = wide.hits;

    const relevant = q.relevant ?? [];
    const gold = goldDocs(q, allDocs);
    const covered = (ordered: typeof candidates, k: number) =>
      relevant.length === 0
        ? 0
        : relevant.filter((key) =>
            ordered.slice(0, k).some((h) => h.content.includes(key)),
          ).length / relevant.length;
    const mrrOf = (ordered: typeof candidates) => {
      const idx = ordered.findIndex((h) =>
        relevant.some((key) => h.content.includes(key)),
      );
      return idx >= 0 ? 1 / (idx + 1) : 0;
    };
    const score = (ordered: typeof candidates, acc: OrderMetrics): void => {
      if (ordered[0] && gold.has(ordered[0].docId)) acc.docTop1++;
      acc.r3 += covered(ordered, 3);
      acc.r5 += covered(ordered, 5);
      acc.mrr += mrrOf(ordered);
    };

    score(candidates, base);
    if (candidates.length === 0) continue;

    // 原始 logits 重排（与 worker 同输入格式；sigmoid 是单调变换，排序等价）
    const scored: { hit: (typeof candidates)[number]; logit: number }[] = [];
    for (const h of candidates) {
      // 与 worker 同口径截断到 512 token：长文档扩展窗口超限时 ONNX 会直接崩
      const inputs = await tokenizer(`${q.question}</s></s>${h.content}`, {
        truncation: true,
        max_length: 512,
      });
      const { logits } = await model(inputs);
      scored.push({ hit: h, logit: logits.data[0] });
    }
    const rerankedHits = scored.sort((a, b) => b.logit - a.logit).map((s) => s.hit);
    score(rerankedHits, ranked);

    const baseMrr = mrrOf(candidates);
    const rerankMrr = mrrOf(rerankedHits);
    if (rerankMrr > baseMrr) mrrImproved++;
    else if (rerankMrr < baseMrr) mrrRegressed++;
    if (candidates[0]?.chunkIds[0] !== rerankedHits[0]?.chunkIds[0]) top1Changed++;
  }

  const n = posQueries.length;
  const row = (label: string, m: OrderMetrics) => [
    label,
    pct(m.docTop1 / n),
    pct(m.r3 / n),
    pct(m.r5 / n),
    pct(m.mrr / n),
  ];
  const widths = [16, 10, 10, 10, 10];
  printRow(["排序方式", "文档top1", "recall@3", "recall@5", "MRR@10"], widths);
  printRow(row("RRF 基线", base), widths);
  printRow(row("精排重排", ranked), widths);
  console.log(
    `Δ（精排 − 基线）：文档top1 ${pct((ranked.docTop1 - base.docTop1) / n)}  ` +
      `recall@3 ${pct((ranked.r3 - base.r3) / n)}  ` +
      `recall@5 ${pct((ranked.r5 - base.r5) / n)}  ` +
      `MRR@10 ${pct((ranked.mrr - base.mrr) / n)}`,
  );
  console.log(
    `单查询 MRR：改善 ${mrrImproved} 条 / 退步 ${mrrRegressed} 条 / 持平 ${n - mrrImproved - mrrRegressed} 条；` +
      `top1 换人 ${top1Changed} 条。`,
  );
  console.log(
    "判读：Δ 持续为正 → 精排值得保留（仅作重排）；Δ≈0 或为负 → 两阶段检索纯属开销，应在设置里默认关闭。",
  );
  console.log(
    "注：本表只统计 q 关键词正样本（调参口径，n=" + n + "）；s 语义难例与负样本均不在此表 ——\n" +
      "精排不做拒答；检索侧弱门限放行后，由答案侧证据校验决定是否「文档未提及」。",
  );

  // 释放模型内存
  if (typeof model.dispose === "function") {
    await model.dispose();
  }
}

/**
 * 信号可用性诊断：回答「哪个信号真的能把该拒的拒掉、又不误伤正样本」。
 *
 * 输出两个口径，用途不同：
 *   - 调参口径（q + n）：与 1~5 段同一人群，零误拒只保护 q 关键词正样本，
 *     这才是门限参数实际面对的分布；
 *   - 保守口径（q + s + n）：再加入 s* 口语化难例做压力观察，
 *     看同一门限在难例上会误伤多少 —— 只观察，不回改参数。
 * 一个信号在调参口径下可用、保守口径下不可用，说明它对关键词查询有效、
 * 对口语化提问无效，难例的最终置信度仍要靠答案侧证据校验。
 */
async function sweepSignalUsability(): Promise<void> {
  console.log("\n===== 7. 拒答信号可用性诊断 =====");
  const chunks = buildCorpus(DEFAULT_CHUNK_OPTIONS.maxChunkTokens);
  const bm25 = new BM25(chunks.map((c) => c.content));
  const fixed = loadFixedEmbeddings();

  const fixedRerank = loadFixedRerank();
  const values: SignalValues[] = [];
  for (const q of golden.queries) {
    const bs = bm25.scoreAll(q.question);
    const qv = fixed?.queries[q.id];
    // 与检索管线共用同一实现（retrieval.ts computeRefusalSignals）：
    // 保证诊断扫出的阈值与线上门限看到的取值逐位一致，避免口径分叉。
    const simsRaw: number[] | null = qv
      ? chunks.map((d) => {
          const v = fixed!.chunks[d.contentHash];
          return v ? dotProduct(Float32Array.from(qv), Float32Array.from(v)) : 0;
        })
      : null;
    const sig = computeRefusalSignals(bs, simsRaw);
    // 精排信号与 eval 同口径：取「该查询对所有块的最大 logit」
    // （候选窗口由多个块拼成，取块内最高分代表窗口分）
    let maxRerankLogit: number | null = null;
    if (fixedRerank) {
      const perChunk = fixedRerank.scores[q.id];
      if (perChunk) {
        const vs = Object.values(perChunk).filter((x) => typeof x === "number");
        if (vs.length > 0) maxRerankLogit = Math.max(...vs.map(logitToScore));
      }
    }
    values.push({
      id: q.id,
      group: groupOf(q.id),
      question: q.question,
      idfCoverage: bm25.idfWeightedCoverage(q.question),
      maxCosine: sig.maxCosine ?? 0,
      cosineGap: sig.cosGap ?? 0,
      maxBm25: sig.maxBm25 ?? 0,
      maxRerankLogit,
    });
  }

  if (!fixed) {
    console.log("⚠ 无向量路（缺 embeddings.json）：余弦类信号无法评估，只测词表/BM25 信号。");
  }
  const signals: SignalName[] = ["idfCoverage", "maxBm25"];
  if (fixed) signals.push("maxCosine", "cosineGap");
  // 精排信号只需要 rerank-scores.json，不需要向量
  if (fixedRerank) signals.push("maxRerankLogit");

  console.log(
    "信号可用性（每行 = 该信号单独用来判拒答时的最好成绩）\n" +
      "  safestRefusal = 零误拒前提下能拒掉的负样本比例 ← 决定能不能设门槛\n" +
      "  bestF1 那三列是「允许误拒」时的最优，用来看清代价",
  );
  // 调参口径：剔除 s* 难例，零误拒只保护 q 关键词正样本（与 1~5 段同一人群）
  const tuningValues = values.filter((v) => v.group !== "s");
  console.log(`\n【调参口径】q 关键词正样本 ${posQueries.length} + 负样本 ${negQueries.length}：`);
  console.log(
    formatSignalReports(signals.map((s) => evaluateSignal(s, tuningValues)))
      .split("\n")
      .slice(3)
      .join("\n"),
  );
  // 保守口径：加入 s* 难例的压力观察（不回改参数）
  console.log(
    `\n【保守口径·仅观察】再加 ${semanticQueries.length} 条 s 语义难例（零误拒同时保护 q + s）：`,
  );
  console.log(
    formatSignalReports(signals.map((s) => evaluateSignal(s, values)))
      .split("\n")
      .slice(3)
      .join("\n"),
  );
  console.log(
    "\n【调参口径】多信号融合（q+m 零误拒）：",
  );
  console.log(formatFusionReport(evaluateFusion(tuningValues)));
  console.log("\n【保守口径·仅观察】多信号融合（q+s+m 零误拒）：");
  console.log(formatFusionReport(evaluateFusion(values)));
  // 仅检索侧（不含精排）：精排在 applyThreshold 之后才执行，线上无法在门限处使用，
  // 因此要单独看「只用检索侧 4 信号」能到多少 —— 这才是能直接写回默认值的口径。
  const retrievalSignals: SignalName[] = [
    "idfCoverage",
    "maxCosine",
    "cosineGap",
    "maxBm25",
  ];
  console.log("\n【调参口径】融合·仅检索侧（不含精排；q+m 零误拒）：");
  console.log(formatFusionReport(evaluateFusion(tuningValues, 0.6, retrievalSignals)));
  console.log("\n【保守口径·仅观察】融合·仅检索侧（不含精排）：");
  console.log(formatFusionReport(evaluateFusion(values, 0.6, retrievalSignals)));
  console.log(
    "\n判读：调参口径决定门槛写多少；保守口径只告诉你难例代价。" +
      "\n两个口径结论分叉时，以调参口径选参，难例的「文档未提及」交给答案侧证据校验。" +
      "\n融合（并集）只会扩大覆盖、不引入误拒：单信号各自不达标时，看并集能否到 60%。",
  );
}


/**
 * 9. 查询扩展权重 × mmrLambda 联合扫描（M2，2026-10-15）。
 *
 * 为什么这两件事必须一起扫：查询扩展与 MMR 做的是**同一件事** ——
 * 让 top-k 别挤在一个主题上（前者靠"多问几种说法"，后者靠"惩罚相似结果"）。
 * 两者同时开就是双重惩罚：实测 λ=0.9 时扩展把 s 组从 0.214 压回 0.143，
 * 而 λ=1.0（关掉 MMR）时扩展的收益才释放出来（s 0.143 → 0.50）。
 *
 * 报告口径：s 组单独列出。run() 的调参口径只含 q+m（见文件头说明），
 * 而这一段的结论恰恰是为 s 组服务的 —— 所以这里对 s 组单独跑同一条管线。
 */
function sGroupStats(
  chunks: RetrievalDoc[],
  bm25: BM25,
  opts: RunOptions,
): { r3: number; refused: number; n: number } {
  const fixed = loadFixedEmbeddings();
  let r3 = 0;
  let refused = 0;
  for (const q of semanticQueries) {
    const qvec = fixed?.queries[q.id];
    const vecScores =
      qvec && fixed
        ? chunks.map((d) => {
            const v = fixed.chunks[d.contentHash];
            return v ? dotProduct(Float32Array.from(qvec), Float32Array.from(v)) : 0;
          })
        : null;
    const variants = expansionRuns(q, bm25, chunks, fixed, undefined, opts.expansionWeight);
    const result = postProcessRetrieval(
      chunks,
      bm25.scoreAll(q.question),
      vecScores,
      {
        topK: 5,
        ...DEFAULT_RETRIEVAL_OPTIONS,
        ...(opts.mmrLambda !== undefined ? { mmrLambda: opts.mmrLambda } : {}),
        idfCoverage: bm25.idfWeightedCoverage(q.question),
      },
      variants,
    );
    if (result.noRelevantHit) refused++;
    const relevant = q.relevant ?? [];
    if (relevant.length > 0) {
      r3 +=
        relevant.filter((key) => result.hits.slice(0, 3).some((h) => h.content.includes(key)))
          .length / relevant.length;
    }
  }
  return { r3: semanticQueries.length === 0 ? 0 : r3 / semanticQueries.length, refused, n: semanticQueries.length };
}

function sweepExpansionWeight(): void {
  console.log("\n===== 9. 查询扩展权重 × mmrLambda（M2）=====");
  console.log(
    "看分组 recall@3（q/s/m）与负样本拒答数。权重 0 = 只抬拒答信号、不参与排序（相当于只做否决权的扩展）。",
  );
  if (!loadFixedExpansions()) {
    console.log("  [跳过] 没有 packages/eval/expansions.json（先跑 npm run gen:expansions）");
    return;
  }
  const chunks = buildCorpus(DEFAULT_CHUNK_OPTIONS.maxChunkTokens);
  const bm25 = new BM25(chunks.map((c) => c.content));
  const w = [12, 6, 8, 8, 8, 10, 10];
  printRow(["扩展权重", "λ", "q r3", "s r3", "s 被拒", "m r3", "负样本拒", "合计 r3"], w);

  for (const lambda of [0.9, 1.0]) {
    for (const weight of [undefined, 0, 0.5, 0.7, 1.0]) {
      const m = run(chunks, { expansionWeight: weight, mmrLambda: lambda }, bm25);
      const sStats = sGroupStats(chunks, bm25, { expansionWeight: weight, mmrLambda: lambda });
      printRow(
        [
          weight === undefined ? "不开" : weight.toFixed(1),
          lambda.toFixed(1),
          pct(m.byGroup.single.r3),
          pct(sStats.r3),
          `${sStats.refused}/${sStats.n}`,
          pct(m.byGroup.multi.r3),
          pct(m.negativeRefusalRate),
          pct(m.recallAt3),
        ],
        w,
      );
    }
  }
  console.log(
    "  读法：λ=1.0（关 MMR）+ 扩展开 才同时拿到 q 高、s 高、m 不掉；" +
      "λ=0.9 时扩展几乎全被 MMR 的多样性惩罚吃掉。",
  );
}

async function main(): Promise<void> {
  const qCount = posQueries.filter((q) => q.type !== "multi_subject_single_doc").length;
  const mCount = posQueries.filter((q) => q.type === "multi_subject_single_doc").length;
  console.log("========== 检索参数扫描 ==========");
  console.log(
    `语料 ${readdirSync(CORPUS_DIR).filter((f) => f.endsWith(".md")).length} 篇 | ` +
      `调参正样本 ${posQueries.length}（q 单主题 ${qCount} + m 多主题 ${mCount}）| ` +
      `语义难例 ${semanticQueries.length}（s*，仅观察不调参）| ` +
      `负样本 ${negQueries.length} | ` +
      `向量路 ${loadFixedEmbeddings() ? "有" : "无（退化为纯 BM25）"}`,
  );
  // 口径说明：1~6 段所有选优指标只统计调参正样本（q 单主题 + m 多主题，不含 s 口语难例）；
  // s* 是刻意构造的难例（措辞与原文零重合），让门限/权重去拟合它们会把面向主流查询的
  // 最优值带偏。s* 的表现只在 eval 分组报告与第 7 段信号诊断中观察。
  console.log(
    `调参口径：1~6 段只用 ${posQueries.length} 条正样本选参（q 单主题 + m 多主题）；` +
      `${semanticQueries.length} 条 s 语义难例不参与选参，仅在第 7 段做压力观察。`,
  );

  // 快速开关：--only=N 只跑第 N 段（都不加载模型，秒级）：
  // 2 = RRF 权重×k 扫描，4 = mmrLambda 扫描，7 = 拒答信号诊断，8 = 向量路准入审计，
  // 9 = 查询扩展权重 × mmrLambda 联合扫描。
  const only =
    process.argv.find((a) => a.startsWith("--only="))?.slice("--only=".length) ?? null;
  if (only === "7") {
    await sweepSignalUsability();
    return;
  }
  if (only === "2") {
    sweepRrfWeights();
    return;
  }
  if (only === "9") {
    sweepExpansionWeight();
    return;
  }
  if (only === "4") {
    sweepMmrLambda();
    return;
  }
  if (only === "8") {
    auditVectorAdmission();
    return;
  }

  // 加载 embedding 模型用于实时编码（方案 A：不依赖 embeddings.json）
  let embedder: Awaited<ReturnType<typeof pipeline<"feature-extraction">>> | undefined;
  try {
    console.log("加载 bge-small-zh 模型用于实时编码…");
    embedder = await pipeline("feature-extraction", "Xenova/bge-small-zh-v1.5", {
      dtype: "q8",
    });
    console.log("  模型加载完成。");
  } catch (err) {
    console.warn(
      "  ⚠ 模型加载失败，将退化为固化向量模式（embeddings.json）：",
      err instanceof Error ? err.message : err,
    );
  }

  await sweepChunkTokens(embedder);
  // 语料无关的硬门槛：hybrid 不允许显著低于 BM25-only。全部配置都不达标时 exit 1。
  const rrfGate = sweepRrfWeights();
  sweepMinVecScore();
  sweepMmrLambda();
  sweepRefusalSignals();
  await evaluateRerankGain();
  await sweepSignalUsability();

  // 释放模型内存
  if (embedder && typeof embedder.dispose === "function") {
    await embedder.dispose();
  }
  console.log(
    "\n结论要落到具体值上：把选定的参数写回 src/utils/retrieval.ts 的 DEFAULT_RETRIEVAL_OPTIONS" +
      "（或 chunk.ts 的 DEFAULT_CHUNK_OPTIONS）。",
  );
  console.log(
    "注意：改 DEFAULT_CHUNK_OPTIONS 会让 CURRENT_INDEX_VERSION 变化，用户端会提示「索引已过期，请重新上传文档」。",
  );
  if (rrfGate === "all-fail") {
    console.error(
      "\n[FAIL] 第 2 段相对门槛未通过：没有任何 RRF 权重/k 配置能让 hybrid ≥ BM25-only×0.95。",
    );
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("tune 执行失败：", err);
  process.exit(1);
});
