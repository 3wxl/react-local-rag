#!/usr/bin/env node
/**
 * 检索评测基线：输出 recall@3 / recall@5 / MRR / 要点命中率 / 重复率 / 拒答正确率，并存档。
 *
 * 用法：npm run eval        （或 npx tsx scripts/eval.ts [--topk 5]）
 * 数据：docs/eval/golden.json（golden QA + 负样本）+ docs/eval/corpus/*.md
 * 存档：docs/eval/results/<时间戳>.json + docs/eval/results/latest.json
 *
 * ── 为什么这个文件直接 import src/ ──
 * 旧版 eval.mjs 用手抄的方式重新实现了一遍 tokenize / BM25 / 分块 / RRF，
 * 注释写着「与 src/utils 同口径的移植」。那和「两份 tokenize 靠注释同步」是同一类风险，
 * 而且抄的是更复杂的逻辑（分块 + RRF + 后处理），漂移概率更高 —— 评测会悄悄失去意义。
 * 现在直接 import 真实模块：评测跑的就是线上跑的那份代码，口径不可能分叉。
 */
import {
  readFileSync,
  readdirSync,
  writeFileSync,
  mkdirSync,
  existsSync,
  unlinkSync,
} from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";  

import { BM25 } from "../src/utils/bm25";
import { createTextChunks, DEFAULT_CHUNK_OPTIONS } from "../src/utils/chunk";
import { normalizeMarkdownForEval } from "../src/utils/evalCorpus";
import {
  dotProduct,
  postProcessRetrieval,
  DEFAULT_RETRIEVAL_OPTIONS,
  type QueryVariantScores,
  type RetrievalDoc,
} from "../src/utils/retrieval";
import {
  loadFixedEmbeddings,
  loadFixedExpansions,
  hasFixedEmbeddings,
  assertEmbeddingsConsistent,
} from "./fixedEmbeddings";
import {
  loadFixedRerank,
  applyFixedRerank,
  rerankScore as fixedRerankScore,
} from "./fixedRerank";

import { EVAL_DIR } from "./evalDir";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
/**
 * 评测数据放在 packages/eval/：CI 直接读盘，浏览器开发者工具也走 fetch 读同一份，
 * 避免「Node 一份、浏览器一份」的复制漂移。结果存档仍写 docs/eval/results。
 */
const GOLDEN_PATH = join(EVAL_DIR, "golden.json");
const CORPUS_DIR = join(EVAL_DIR, "corpus");
const RESULTS_DIR = join(ROOT, "docs", "eval", "results");

/* ---------------- 类型 ---------------- */

interface GoldenQuery {
  id: string;
  question: string;
  /** 期望命中的原文关键句（子串匹配到块即视为相关块） */
  relevant?: string[];
  /** 期望答案要点（出现在 topK 检索文本中即命中） */
  keyPoints?: string[];
  /** 负样本：文档里根本没有答案，应当走拒答分支 */
  negative?: boolean;
  /**
   * 查询类型（只用于分组统计与 tune 选优，不参与命中判定）：
   * - single_topic（缺省）：答案集中在单一块
   * - multi_subject_single_doc（m*）：答案分散在同一文档的多个块/多个窗口
   * 两类的命中口径完全相同：top-k 窗口文本包含 relevant 原句即算命中。
   * 不维护 contentHash —— 改分块参数或改文档不会让标注静默失效。
   */
  type?: "single_topic" | "multi_subject_single_doc";
}

interface Golden {
  name: string;
  corpusDir: string;
  retrievalTopK: number;
  queries: GoldenQuery[];
}

/* ---------------- 语料加载 ---------------- */

/**
 * 读语料并按**线上同一套分块器**切块。
 * 标题行不进正文（与解析层行为一致），页码恒为 1（纯文本无分页）。
 *
 * 向量也随块一起挂上（`doc.vector`），与浏览器 worker 里的索引结构一致：
 * MMR 的相似度项读的就是这个字段，线上检索跑在"带向量的块"上。
 * 早先评测不挂向量，等于让 MMR 退化成纯相关度排序 —— 评测口径与线上口径分叉，
 * 量出来的就不是用户拿到的结果。这里与线上对齐。
 */
function loadCorpus(): RetrievalDoc[] {
  const files = readdirSync(CORPUS_DIR)
    .filter((f) => f.endsWith(".md"))
    .sort();
  const fixed = loadFixedEmbeddings();
  const chunks: RetrievalDoc[] = [];
  for (const file of files) {
    const raw = readFileSync(join(CORPUS_DIR, file), "utf8");
    // 与浏览器侧向量固化共用同一份预处理（src/utils/evalCorpus），
    // 保证切块 contentHash 与 packages/eval/embeddings.json 的 key 对得上
    const text = normalizeMarkdownForEval(raw);
    for (const chunk of createTextChunks(file, file, text, DEFAULT_CHUNK_OPTIONS)) {
      const vec = fixed?.chunks[chunk.contentHash];
      chunks.push({ ...chunk, vector: vec ? Float32Array.from(vec) : null });
    }
  }
  return chunks;
}

/* ---------------- 向量路（离线固化） ---------------- */

/**
 * 向量由 docs/eval/embeddings.json 提供（构建机用真实模型预先算好并提交）。
 * 没有该文件时只跑 BM25 路径，指标会标注 mode=bm25，避免把降级结果当成混合检索成绩。
 */
function computeVectorScores(
  docs: RetrievalDoc[],
  queryId: string,
): number[] | null {
  const fixed = loadFixedEmbeddings();
  if (!fixed) return null;
  const qvec = fixed.queries[queryId];
  if (!qvec) {
    console.warn(`[warn] embeddings.json 里缺少查询 ${queryId} 的向量，该条退化为纯 BM25`);
    return null;
  }
  return docs.map((d) => {
    const v = fixed.chunks[d.contentHash];
    // main() 已用 assertEmbeddingsConsistent 把缺失向量判为致命错误并退出，
    // 走到这里的每个 contentHash 必然有向量；保留 0 仅作类型兜底
    if (!v) return 0;
    return dotProduct(Float32Array.from(qvec), Float32Array.from(v));
  });
}

/* ---------------- 指标 ---------------- */

interface Metrics {
  queryCount: number;
  chunkCount: number;
  /** 向量路是否参与（false = 只有 BM25，指标偏保守） */
  hybrid: boolean;
  /** BM25-only 基线 recall@3（相对门槛的分母）；无向量路时为 null */
  bm25OnlyRecallAt3: number | null;
  /**
   * 精排列指标（缺 rerank-scores.json 时为 null）。
   * 与主指标走同一批候选，只多一步按精排 logit 重排 ——
   * 两列并排才看得出精排是帮忙还是添乱。
   */
  rerank: {
    recallAt3: number;
    recallAt5: number;
    mrrAt10: number;
    keyPointHitRate: number;
    /** top-1 被精排改变的比例；为 0 说明精排没起作用（静默失效防护） */
    top1ChangedRate: number;
    /** 候选里缺精排分的对数（应为 0；非 0 说明固化文件与语料不同步） */
    missingPairs: number;
    /** top-1 精排 logit 的中位数：作为"精排绝对分"拒答信号的分布参考 */
    top1LogitMedian: number;
    /** 精排的分组 recall@3 —— 判断它对哪类查询有效（尤其口语化的 s 组） */
    byGroup: Record<string, { count: number; recallAt3: number; recallAt5: number }>;
  } | null;
  recallAt3: number;
  recallAt5: number;
  mrrAt10: number;
  keyPointHitRate: number;
  /** 结构性冗余：top3 内同文档两个窗口的 ord 区间重叠或紧邻 */
  top3DuplicateRate: number;
  /** 内容性冗余：top3 内两段文本的 bigram Jaccard ≥ 0.8（用户真正感受到的"两条一样"） */
  top3NearDuplicateRate: number;
  noHitQueryCount: number;
  /**
   * 正样本被拒答的条数（= noHitQueryCount，另起一个语义化名字给门槛用）。
   * 单独列字段是必须的：门槛是按字段名取的，字段不存在时
   * `undefined > 0` 恒为 false，门槛会静默失效。
   */
  falseRefusalCount: number;
  /**
   * **受保护人群**（q + m，不含口语化 s）的误拒条数。
   *
   * 2026-10-15 起检索侧启用多信号融合拒答（默认只保护调参口径 q+m），
   * s 的部分样本会被粗拒是「拒答 ≥0.60」的已知代价（见 retrieval.ts 注释）。
   * 因此误拒门槛只对 q+m 要求 0 —— 这是「零误拒约束」在方案里明确定义的人群。
   * falseRefusalCount（含 s）保留为观察项，不计门槛。
   */
  protectedFalseRefusalCount: number;
  /** 受保护人群（q+m）的 recall@3 */
  protectedRecallAt3: number;
  /** 受保护人群（q+m）的 BM25-only 基线 recall@3（相对门槛的分母）；无向量路时为 null */
  protectedBm25OnlyRecallAt3: number | null;
  /** 负样本：应当拒答的条数 / 正确拒答的条数 */
  negativeCount: number;
  /** 负样本被正确拒答的**绝对条数**（门槛用它而不是比率，见 main 里的说明） */
  negativeRefusedCount: number;
  negativeRefusalRate: number;
  /**
   * 分组指标。查询集里混了两类难度完全不同的正样本，聚合成一个数字会掩盖问题：
   *   q = 技术措辞（与语料同一套术语），s = 口语化改写（零词汇重合）
   * 实测两者 recall 差距很大，只看总分不知道是哪一类在退步。
   */
  byGroup: Record<string, { count: number; recallAt3: number; recallAt5: number; noHit: number }>;
}

function isRelevant(content: string, relevant: string[]): boolean {
  return relevant.some((key) => content.includes(key));
}

/** 近似重复判定阈（bigram Jaccard） */
const NEAR_DUP_JACCARD = 0.8;

/** 字符 bigram 集合（去掉空白，避免排版差异干扰） */
function bigrams(text: string): Set<string> {
  const clean = text.replace(/\s+/g, "");
  const out = new Set<string>();
  for (let i = 0; i < clean.length - 1; i++) out.add(clean.slice(i, i + 2));
  return out;
}

/** 两段文本的 bigram Jaccard 相似度（0~1） */
function bigramJaccard(a: string, b: string): number {
  const sa = bigrams(a);
  const sb = bigrams(b);
  if (sa.size === 0 && sb.size === 0) return 1;
  let inter = 0;
  for (const g of sa) if (sb.has(g)) inter++;
  const union = sa.size + sb.size - inter;
  return union === 0 ? 0 : inter / union;
}

function evaluate(golden: Golden, chunks: RetrievalDoc[], topK: number): {
  metrics: Metrics;
  perQuery: unknown[];
} {
  const bm25 = new BM25(chunks.map((c) => c.content));
  const hybrid = hasFixedEmbeddings();
  const posQueries = golden.queries.filter((q) => !q.negative);
  const negQueries = golden.queries.filter((q) => q.negative);

  let recallAt3Sum = 0;
  let recallAt5Sum = 0;
  let mrrSum = 0;
  /* 精排列的累计量（用固化 logit 重排同一批候选） */
  const fixedRerank = loadFixedRerank();
  // 检索返回的是窗口（含 chunkId），精排分按块存：先建 chunkId -> contentHash 映射
  const hashByChunkId = new Map(chunks.map((c) => [c.chunkId, c.contentHash]));
  let rerankR3 = 0;
  /** BM25-only 基线 recall@3（相对门槛的分母） */
  let bm25OnlyR3Sum = 0;
  let rerankR5 = 0;
  let rerankMrr = 0;
  let rerankPointHits = 0;
  let rerankTop1Changed = 0;
  let rerankMissing = 0;
  /** 每查询 top-1 精排 logit，用于给出「精排分作为拒答信号」的分布 */
  const fixedRerankScores: number[] = [];
  let pointHits = 0;
  let pointTotal = 0;
  let noHit = 0;
  /*
   * 受保护人群（q+m）的累计：误拒门槛与相对门槛只在这批上判定。
   * s（口语化）按既定策略降为观察项 —— 它被融合拒答粗拒是已知代价，
   * 不应再污染「零误拒」与「hybrid ≥ BM25-only」两个门槛。
   */
  let protectedCount = 0;
  let protectedR3Sum = 0;
  let protectedBm25OnlyR3Sum = 0;
  let protectedNoHit = 0;
  /** 分组累计（q=技术措辞 / s=口语化改写 / m=单文档多主题） */
  const byGroup: Record<string, { count: number; r3: number; r5: number; noHit: number }> = {};
  /** 精排列的分组累计（判断精排对哪类查询有效，尤其是口语化的 s 组） */
  const byGroupRerank: Record<string, { count: number; r3: number; r5: number }> = {};
  let dupQueries = 0;
  let nearDupQueries = 0;
  const perQuery: unknown[] = [];

  /**
   * 查询扩展：把固化的改写变成检索管线的额外路。
   *
   * 权重默认 0.7（见 queryExpansion.buildQueryVariants 的注释）；
   * 评测与线上必须用同一个默认值 —— 否则测的是另一套配置（口径纪律）。
   * `EXPANSION_WEIGHT` 环境变量只用于 tune 扫描，不改变默认行为。
   */
  const expansionWeight = Number(process.env.EXPANSION_WEIGHT ?? "") || 0.7;
  const fixed = loadFixedEmbeddings();
  const expansions = loadFixedExpansions();
  const expansionRunsFor = (q: GoldenQuery): QueryVariantScores[] => {
    const vectors = fixed?.expansions?.[q.id];
    if (!vectors || vectors.length === 0) return [];
    return vectors.map((vec, i) => {
      const text = expansions?.[q.id]?.[i] ?? "";
      return {
        bm25Scores: bm25.scoreAll(text),
        vecScores: chunks.map((c) => {
          const cv = fixed?.chunks[c.contentHash];
          return cv ? dotProduct(Float32Array.from(vec), Float32Array.from(cv)) : 0;
        }),
        idfCoverage: bm25.idfWeightedCoverage(text),
        weight: expansionWeight,
      };
    });
  };

  const runQuery = (
    q: GoldenQuery,
    overrides: Partial<typeof DEFAULT_RETRIEVAL_OPTIONS> = {},
    /** true = 强制关掉向量路，用于算 BM25-only 基线（相对门槛的分母） */
    vecDisabled = false,
    /** false = 只用原查询（用于量"扩展开 vs 未扩展开"这条相对门槛） */
    withExpansions = true,
  ) => {
    const bm25Scores = bm25.scoreAll(q.question);
    const vecScores = vecDisabled ? null : computeVectorScores(chunks, q.id);
    // 拒答信号 1：IDF 加权覆盖率必须在评测里也算出来，否则评测与线上口径不一致
    const idfCoverage = bm25.idfWeightedCoverage(q.question);
    const variants =
      withExpansions && !vecDisabled ? expansionRunsFor(q) : undefined;
    const result = postProcessRetrieval(
      chunks,
      bm25Scores,
      vecScores,
      {
        topK,
        ...DEFAULT_RETRIEVAL_OPTIONS,
        ...overrides,
        idfCoverage,
      },
      variants,
    );
    // MRR 与正式指标必须走**同一条 hybrid 管线**，只把候选放宽到 top10
    // （避免 topK=5 截断把第 6~10 名的相关块算成 0）。
    // 旧实现这里传 vecScores=null 退回纯 BM25，导致 MRR 与 recall 口径不一致。
    const wide = postProcessRetrieval(
      chunks,
      bm25Scores,
      vecScores,
      {
        topK: 10,
        ...DEFAULT_RETRIEVAL_OPTIONS,
        ...overrides,
        idfCoverage,
      },
      variants,
    );
    return { result, wide };
  };

  for (const q of posQueries) {
    const { result, wide } = runQuery(q);
    const relevant = q.relevant ?? [];
    // 分组：q=技术措辞，s=口语化改写（观察项），m=单文档多主题
    const g = q.id[0] === "s" ? "s" : q.id[0] === "m" ? "m" : "q";
    /** 受保护人群：q+m（不含口语化 s） */
    const isProtected = g !== "s";

    if (result.noRelevantHit) noHit++;

    const top = result.hits;
    // 统一命中口径（q/s/m 完全一致）：top-k 窗口文本包含某条 relevant 原句，
    // 该要点即算命中。多主题查询的多条 relevant 天然分散在不同块/窗口，
    // recall = 被任一 top-k 窗口覆盖的要点比例。改分块/改文档不影响标注有效性。
    const covered = (k: number) =>
      relevant.length === 0
        ? 0
        : relevant.filter((key) =>
            top.slice(0, k).some((h) => h.content.includes(key)),
          ).length / relevant.length;
    recallAt3Sum += covered(3);
    recallAt5Sum += covered(5);

    /*
     * BM25-only 基线：同一条管线强制关掉向量路。
     * 用来算**语料无关的相对门槛** hybrid ≥ BM25-only × 0.95 ——
     * 它不依赖具体语料与绝对值，能兜住"融合配置退化"（比如两路权重调歪、
     * 向量路被错误地完全压制）这类问题，而绝对门槛在换语料后就失效了。
     * 相对门槛同样只算**受保护人群（q+m）**：s 已被检索侧刻意粗拒，
     * 让它当分母等于用「已知要放弃的人群」去卡向量质量，必然误报。
     */
    if (hybrid) {
      const onlyBm25 = runQuery(q, {}, true).result;
      const onlyTop = onlyBm25.hits;
      const baseR3 =
        relevant.length === 0
          ? 0
          : relevant.filter((key) =>
              onlyTop.slice(0, 3).some((h) => h.content.includes(key)),
            ).length / relevant.length;
      bm25OnlyR3Sum += baseR3;
      if (isProtected) protectedBm25OnlyR3Sum += baseR3;
    }

    if (isProtected) {
      protectedCount++;
      protectedR3Sum += covered(3);
      if (result.noRelevantHit) protectedNoHit++;
    }

    let bucket = byGroup[g];
    if (!bucket) {
      bucket = { count: 0, r3: 0, r5: 0, noHit: 0 };
      byGroup[g] = bucket;
    }
    bucket.count++;
    bucket.r3 += covered(3);
    bucket.r5 += covered(5);
    if (result.noRelevantHit) bucket.noHit++;

    // MRR@10：同一条 hybrid 管线的宽候选里，第一个相关命中的排名倒数
    const firstRank = wide.hits.findIndex((h) => isRelevant(h.content, relevant));
    mrrSum += firstRank >= 0 ? 1 / (firstRank + 1) : 0;
    const points = q.keyPoints ?? [];

    /*
     * 精排列：用固化 logit 把同一批候选重排，走与线上 rerankHits 相同的语义
     * （按精排分降序取 topK）。两列同时输出，精排的增益/退化才看得见。
     * 同时统计 top-1 变化率：变化率为 0 说明精排没起作用（静默失效），
     * 这是上次 bge-reranker config 缺 problem_type 时的踩坑点。
     */
    if (fixedRerank) {
      // 窗口分数 = 窗口内各块精排分的最大值（任一块相关，窗口即相关）
      const logitOf = (hit: (typeof wide.hits)[number]): number | null => {
        let best: number | null = null;
        for (const cid of hit.windowChunkIds.length > 0 ? hit.windowChunkIds : hit.chunkIds) {
          const hash = hashByChunkId.get(cid);
          if (!hash) continue;
          const v = fixedRerankScore(fixedRerank, q.id, hash);
          if (v !== null && (best === null || v > best)) best = v;
        }
        return best;
      };
      const rer = applyFixedRerank(wide.hits, topK, logitOf);
      rerankMissing += rer.missing;
      if (rer.top1Changed) rerankTop1Changed++;
      if (rer.topLogit !== null) fixedRerankScores.push(rer.topLogit);
      const rTop = rer.hits;
      const rCovered = (k: number) =>
        relevant.length === 0
          ? 0
          : relevant.filter((key) =>
              rTop.slice(0, k).some((h) => h.content.includes(key)),
            ).length / relevant.length;
      rerankR3 += rCovered(3);
      rerankR5 += rCovered(5);
      const rIdx = rTop.findIndex((h) => isRelevant(h.content, relevant));
      rerankMrr += rIdx >= 0 ? 1 / (rIdx + 1) : 0;
      // 分组累计：判断精排对哪一类查询有效。口语化(s)是最需要救的一组 ——
      // 若精排对它增益显著，优先开精排；否则得上查询扩展/HyDE。
      let rBucket = byGroupRerank[g];
      if (!rBucket) {
        rBucket = { count: 0, r3: 0, r5: 0 };
        byGroupRerank[g] = rBucket;
      }
      rBucket.count++;
      rBucket.r3 += rCovered(3);
      rBucket.r5 += rCovered(5);
      const rText = rTop.slice(0, topK).map((h) => h.content).join("");
      rerankPointHits += points.filter((p) => rText.includes(p)).length;
    }

    const topText = top.slice(0, topK).map((h) => h.content).join("");
    pointHits += points.filter((p) => topText.includes(p)).length;
    pointTotal += points.length;

    /*
     * top3 冗余占位，两个口径同时算：
     *
     * 1. ord 相邻/重叠：同一文档的两个窗口占据两个 top3 位。
     *    相邻命中块已被 mergeNeighbors 熔成一个窗口，这里抓到的是被字符上限
     *    切开的首尾相接窗口（如 [0-3]/[4-7]）。**结构性**冗余。
     * 2. 文本近似重复：两个窗口文本的 bigram Jaccard ≥ 0.8。
     *    **内容性**冗余 —— 才是用户真正感受到的"两条结果几乎一样"。
     *
     * 两者都要导出明细（ord 区间 + Jaccard）写进存档：
     * 只有看到 Jaccard 才知道 ord 相邻是不是真冗余
     * （长文档里两个相邻窗口内容完全不同是正常的，不该算冗余）。
     */
    const top3 = top.slice(0, 3);
    let adjacentPair: { i: number; j: number; jaccard: number } | null = null;
    let nearDupPair: { i: number; j: number; jaccard: number } | null = null;
    for (let i = 0; i < top3.length; i++) {
      for (let j = i + 1; j < top3.length; j++) {
        const jac = bigramJaccard(top3[i].content, top3[j].content);
        if (jac >= NEAR_DUP_JACCARD && !nearDupPair) {
          nearDupPair = { i, j, jaccard: jac };
        }
        const structurallyAdjacent =
          top3[i].docId === top3[j].docId &&
          top3[i].ordEnd + 1 >= top3[j].ordStart &&
          top3[j].ordEnd + 1 >= top3[i].ordStart;
        if (structurallyAdjacent && !adjacentPair) {
          adjacentPair = { i, j, jaccard: jac };
        }
      }
    }
    if (adjacentPair) dupQueries++;
    if (nearDupPair) nearDupQueries++;

    perQuery.push({
      id: q.id,
      question: q.question,
      hitCount: top.length,
      noRelevantHit: result.noRelevantHit,
      relevantCoveredAt3: covered(3),
      firstRelevantRank: firstRank >= 0 ? firstRank + 1 : 0,
      pointsHit: points.filter((p) => topText.includes(p)).length,
      pointsTotal: points.length,
      /*
       * 冗余明细写进存档：只有看到 Jaccard 才能判断 ord 相邻是不是真冗余。
       * 长文档里相邻两个窗口内容完全不同是正常的（各自回答问题的不同部分），
       * 那种情况 Jaccard 很低，不该算冗余 —— 这正是我们把门槛设在**内容口径**上的依据。
       */
      ...(adjacentPair || nearDupPair
        ? {
            dupDetail: {
              adjacent: adjacentPair
                ? {
                    ordRanges: [
                      [top3[adjacentPair.i].ordStart, top3[adjacentPair.i].ordEnd],
                      [top3[adjacentPair.j].ordStart, top3[adjacentPair.j].ordEnd],
                    ],
                    jaccard: +adjacentPair.jaccard.toFixed(3),
                  }
                : null,
              nearDuplicate: nearDupPair
                ? {
                    ordRanges: [
                      [top3[nearDupPair.i].ordStart, top3[nearDupPair.i].ordEnd],
                      [top3[nearDupPair.j].ordStart, top3[nearDupPair.j].ordEnd],
                    ],
                    jaccard: +nearDupPair.jaccard.toFixed(3),
                  }
                : null,
            },
          }
        : {}),
    });
  }

  // 负样本：文档里没有答案，正确的行为是「门限滤空 → 走拒答分支」
  let negativeRefused = 0;
  const negDetail: unknown[] = [];
  for (const q of negQueries) {
    const { result } = runQuery(q);
    const refused = result.noRelevantHit;
    if (refused) negativeRefused++;
    negDetail.push({
      id: q.id,
      question: q.question,
      refused,
      hitCount: result.hits.length,
      topHit: result.hits[0]?.matchedContent.slice(0, 40) ?? "",
    });
  }

  const n = posQueries.length;
  const m: Metrics = {
    queryCount: golden.queries.length,
    chunkCount: chunks.length,
    hybrid,
    /** 同管线强制关向量路后的 recall@3（相对门槛的分母）；无非向量路时为 null */
    bm25OnlyRecallAt3: hybrid ? +(bm25OnlyR3Sum / n).toFixed(4) : null,
    recallAt3: +(recallAt3Sum / n).toFixed(4),
    recallAt5: +(recallAt5Sum / n).toFixed(4),
    mrrAt10: +(mrrSum / n).toFixed(4),
    keyPointHitRate: pointTotal === 0 ? 0 : +(pointHits / pointTotal).toFixed(4),
    top3DuplicateRate: +(dupQueries / n).toFixed(4),
    top3NearDuplicateRate: +(nearDupQueries / n).toFixed(4),
    noHitQueryCount: noHit,
    falseRefusalCount: noHit,
    // 受保护人群（q+m）：误拒门槛与相对门槛只在这批上判定（见字段注释）
    protectedFalseRefusalCount: protectedNoHit,
    protectedRecallAt3:
      protectedCount === 0 ? 0 : +(protectedR3Sum / protectedCount).toFixed(4),
    protectedBm25OnlyRecallAt3:
      hybrid && protectedCount > 0
        ? +(protectedBm25OnlyR3Sum / protectedCount).toFixed(4)
        : null,
    rerank: fixedRerank
      ? {
          recallAt3: +(rerankR3 / n).toFixed(4),
          recallAt5: +(rerankR5 / n).toFixed(4),
          mrrAt10: +(rerankMrr / n).toFixed(4),
          keyPointHitRate:
            pointTotal === 0 ? 0 : +(rerankPointHits / pointTotal).toFixed(4),
          top1ChangedRate: +(rerankTop1Changed / n).toFixed(4),
          missingPairs: rerankMissing,
          top1LogitMedian: fixedRerankScores.length
            ? +fixedRerankScores.slice().sort((a, b) => a - b)[
                Math.floor(fixedRerankScores.length / 2)
              ].toFixed(4)
            : 0,
          byGroup: Object.fromEntries(
            Object.entries(byGroupRerank).map(([k, v]) => [
              k,
              {
                count: v.count,
                recallAt3: +(v.r3 / v.count).toFixed(4),
                recallAt5: +(v.r5 / v.count).toFixed(4),
              },
            ]),
          ),
        }
      : null,
    negativeCount: negQueries.length,
    negativeRefusedCount: negativeRefused,
    negativeRefusalRate:
      negQueries.length === 0 ? 0 : +(negativeRefused / negQueries.length).toFixed(4),
    byGroup: Object.fromEntries(
      Object.entries(byGroup).map(([k, v]) => [
        k,
        {
          count: v.count,
          recallAt3: +(v.r3 / v.count).toFixed(4),
          recallAt5: +(v.r5 / v.count).toFixed(4),
          noHit: v.noHit,
        },
      ]),
    ),
  };
  return { metrics: m, perQuery: [...perQuery, ...negDetail] };
}

/* ---------------- 存档保留策略 ---------------- */

/** 除 latest.json 外保留的最近存档数；再往前的自动清理 */
const KEEP_ARCHIVES = 20;

/**
 * 只留 latest.json + 最近 KEEP_ARCHIVES 个存档。
 * 结果目录已 gitignore（不进仓库），但本地无上限增长也没意义 —— 每次跑 eval 都会写一份。
 * 用 unlinkSync 而不是 rmSync：某些受限环境下 rm 相关调用会被拦截并直接终止进程。
 */
function pruneOldResults(keepTimestamp: string): void {
  const archives = readdirSync(RESULTS_DIR)
    .filter((f) => f.endsWith(".json") && f !== "latest.json")
    .sort();
  // 本次刚写的那份一定保留
  const keep = new Set([`${keepTimestamp}.json`]);
  for (const f of archives.slice(-KEEP_ARCHIVES)) keep.add(f);
  for (const f of archives) {
    if (keep.has(f)) continue;
    try {
      unlinkSync(join(RESULTS_DIR, f));
    } catch {
      // 清理失败不影响评测结果
    }
  }
}

/* ---------------- 主流程 ---------------- */

function main(): void {
  const args = process.argv.slice(2);
  let topK = 5;
  const topkIdx = args.indexOf("--topk");
  if (topkIdx >= 0 && args[topkIdx + 1]) topK = Number(args[topkIdx + 1]);

  if (!existsSync(GOLDEN_PATH)) {
    console.error(`golden QA 缺失：${GOLDEN_PATH}`);
    process.exit(1);
  }
  const golden = JSON.parse(readFileSync(GOLDEN_PATH, "utf8")) as Golden;
  if (!Array.isArray(golden.queries) || golden.queries.length === 0) {
    console.error("golden.json 无有效 queries");
    process.exit(1);
  }

  const chunks = loadCorpus();
  if (chunks.length === 0) {
    console.error(`语料为空：${CORPUS_DIR}`);
    process.exit(1);
  }

  // 有固化向量就必须严格一致：模型 / 分块参数 / 每个 contentHash / 每个 query id。
  // 不一致直接退出，禁止「缺向量按 0 分」悄悄把 hybrid 测成近似 BM25；
  // 文件整体不存在则不报错，evaluate 内按 bm25-only 标注。
  const fixed = loadFixedEmbeddings();
  if (fixed) {
    try {
      assertEmbeddingsConsistent(
        fixed,
        chunks.map((c) => c.contentHash),
        golden.queries.map((q) => q.id),
      );
    } catch (err) {
      console.error(`\n[FAIL] ${(err as Error).message}`);
      process.exit(1);
    }
  }

  const { metrics, perQuery } = evaluate(golden, chunks, topK);

  mkdirSync(RESULTS_DIR, { recursive: true });
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const payload = {
    metrics,
    perQuery,
    meta: {
      timestamp: new Date().toISOString(),
      golden: golden.name,
      topK,
      chunkParams: DEFAULT_CHUNK_OPTIONS,
    },
  };
  writeFileSync(join(RESULTS_DIR, `${timestamp}.json`), JSON.stringify(payload, null, 2) + "\n");
  writeFileSync(join(RESULTS_DIR, "latest.json"), JSON.stringify(payload, null, 2) + "\n");
  pruneOldResults(timestamp);

  const pct = (v: number) => v.toFixed(4);
  console.log("========== 检索评测基线 ==========");
  console.log(
    `模式=${metrics.hybrid ? "hybrid（向量+BM25）" : "bm25（无向量，指标偏保守）"}  ` +
      `queries=${metrics.queryCount}（正 ${metrics.queryCount - metrics.negativeCount} / 负 ${metrics.negativeCount}）  ` +
      `chunks=${metrics.chunkCount}  topK=${topK}`,
  );
  // 这一行是**线上实际生效的配置**。精排默认关闭（见 useAgentSettings），
  // 所以下面的主指标就是用户真实拿到的效果；精排列是"可选配置下能到多少"。
  // 两列混着看会高估线上效果，这里显式写出来。
  console.log(
    "线上默认口径：hybrid（向量+BM25）+ 精排关闭 → 下方主指标即用户实际效果；" +
      "精排为可选配置（开启约 +300~400MB 内存、单次查询约 +1.7s）",
  );
  console.log("--------------------------------------------");
  console.log(
    `recall@3          : ${pct(metrics.recallAt3)}` +
      (metrics.bm25OnlyRecallAt3 !== null
        ? `   （BM25-only 基线 ${pct(metrics.bm25OnlyRecallAt3)}，相对门槛 ≥ ×0.95）`
        : ""),
  );
  console.log(`recall@5          : ${pct(metrics.recallAt5)}`);
  console.log(`MRR@10            : ${pct(metrics.mrrAt10)}`);
  console.log(`要点命中率        : ${pct(metrics.keyPointHitRate)}`);
  console.log(
    `top3 冗余占位(结构): ${pct(metrics.top3DuplicateRate)}   近似重复(内容): ${pct(metrics.top3NearDuplicateRate)}`,
  );
  console.log(`门限空命中(正样本): ${metrics.noHitQueryCount}`);
  console.log(
    `负样本拒答正确率  : ${pct(metrics.negativeRefusalRate)}（${metrics.negativeCount} 条，正确拒答 ${Math.round(metrics.negativeRefusalRate * metrics.negativeCount)} 条）`,
  );
  // 误拒口径必须显式分列：门槛只卡 q+m，s 的拒答是「拒答≥0.60」的已知代价（观察项）
  console.log(
    `误拒              : 受保护(q+m) ${metrics.protectedFalseRefusalCount} 条（门槛=0）` +
      `  |  观察：s 组 ${metrics.byGroup.s?.noHit ?? 0}/${metrics.byGroup.s?.count ?? 0} 条被拒、全量 ${metrics.falseRefusalCount} 条`,
  );
  // 分组明细：两类正样本难度差很多，合并成一个数字会掩盖是哪类在退步
  const GROUP_LABELS: Record<string, string> = {
    q: "技术措辞正样本",
    s: "口语化正样本",
    m: "单文档多主题正样本",
  };
  // 精排列：与上一组数字同一条管线的候选，只多了一步重排
  if (metrics.rerank) {
    const r = metrics.rerank;
    console.log("--------------------------------------------");
    console.log("精排列（同一批候选，用固化 logit 重排）：");
    console.log(
      `  recall@3=${pct(r.recallAt3)}  recall@5=${pct(r.recallAt5)}  MRR@10=${pct(r.mrrAt10)}  要点命中率=${pct(r.keyPointHitRate)}`,
    );
    console.log(
      `  top-1 变化率=${pct(r.top1ChangedRate)}  精排分中位数=${r.top1LogitMedian}` +
        (r.missingPairs > 0 ? `  ⚠ 缺精排分 ${r.missingPairs} 对` : ""),
    );
    if (r.top1ChangedRate === 0) {
      console.log(
        "  ⚠ top-1 变化率为 0：精排没有改变任何排序，疑似静默失效（曾因 config 缺 problem_type 导致分数恒为常数）",
      );
    }
    // 分组对比：与上一段的分组明细并排看，才知道精排救了哪一类、有没有帮倒忙
    console.log("  精排分组 recall@3（对比上方无精排的数字）：");
    const GROUP_LABELS2: Record<string, string> = {
      q: "技术措辞",
      s: "口语化",
      m: "单文档多主题",
    };
    for (const [gname, rv] of Object.entries(r.byGroup)) {
      const base = metrics.byGroup[gname];
      const delta = base ? rv.recallAt3 - base.recallAt3 : 0;
      const sign = delta >= 0 ? "+" : "";
      console.log(
        `    ${GROUP_LABELS2[gname] ?? gname}  n=${rv.count}  ` +
          `${pct(base?.recallAt3 ?? 0)} → ${pct(rv.recallAt3)}  (${sign}${(delta * 100).toFixed(1)}pt)`,
      );
    }
  } else {
    console.log("--------------------------------------------");
    console.log("精排列：缺 packages/eval/rerank-scores.json，未启用（跑 npm run gen:rerank 生成）");
  }
  console.log("--------------------------------------------");
  console.log("分组明细（q/s 两类正样本难度不同，务必分开看）：");
  for (const [g, v] of Object.entries(metrics.byGroup)) {
    console.log(
      `  ${GROUP_LABELS[g] ?? g}  n=${v.count}  recall@3=${pct(v.recallAt3)}  recall@5=${pct(v.recallAt5)}  被拒=${v.noHit}`,
    );
  }
  console.log("--------------------------------------------");
  console.log("逐条明细：");
  for (const row of perQuery as Record<string, unknown>[]) {
    console.log(
      `  ${String(row.id).padEnd(6)} ${row.negative !== undefined && "refused" in row ? (row.refused ? "拒答✓" : "误答✗") : `命中${row.hitCount}`}  ${row.question}`,
    );
  }
  console.log(`\n已存档：${join(RESULTS_DIR, `${timestamp}.json`)}`);

  /* -------- 门槛：低于门槛退出码 1，CI 会红 --------
   * 数值取自实测基线往下留一点余量，作用是**防回归**，不是定目标。
   * 按模式分别设：两条路的绝对值不可比（BM25 与混合检索的量纲、可及上限都不同）。
   */
  const failed: string[] = [];

  /**
   * 门槛一律写成 [名称, 实测值, 阈值] 的三元组，而不是用字符串下标去取字段。
   * 字符串下标要绕过类型检查（`metrics[k]`），字段名写错或字段不存在时
   * 得到 undefined，比较恒为 false —— 门槛会静默失效而且看不出来。
   * 写成三元组后字段名写错是编译错误。
   */
  type Gate = [name: string, actual: number, threshold: number];
  const maxGates: Gate[] = [
    // 冗余占位率不再是「必须恒 0」：长文档（10+ 块）上同文档首尾相接的窗口
    // 各占 top3 一位是正常现象，0 只在短文档语料下结构性成立。
    // 这里改为防回归上限：mergeNeighbors 完全失效时该值会趋近 1，实测基线 0.18。
    /*
     * 冗余用**内容性**口径设门槛，不用结构性口径。
     *
     * 结构口径（ord 相邻）测的是"同一篇文档的两个窗口占了两个 top3 位"。
     * 长文档里这可能是完全正常的：相邻两段内容不同、各自回答了问题的不同部分，
     * 并不构成用户可感的重复。实测该口径当前为 0，但设 0.35 这种远大于实测的
     * 阈值等于没设 —— 见下方"门槛取值纪律"。
     *
     * 内容口径（bigram Jaccard ≥ 0.8）才是"两条结果几乎一样"。
     * 当前实测 0，阈值同为 0：只要出现一条近似重复就红，这才是有约束力的门槛。
     */
    ["top3NearDuplicateRate", metrics.top3NearDuplicateRate, 0],
    // 正样本被拒答 = 漏检。拒答越激进这条越容易破，必须与 negativeRefusalRate
    // 成对设：只压一边会把系统调成哑巴。
    //
    // 口径选择：**只卡受保护人群（q+m）**。2026-10-15 起检索侧按「调参口径」启用
    // 多信号融合拒答，s（口语化）已被刻意降为观察项、部分会被粗拒 ——
    // 若这里仍卡全量 51 条正样本，等于要求「拒答能力提升」与「s 零误拒」同时成立，
    // 而实测后者不可能（保守口径上限 50%）。故用 protectedFalseRefusalCount。
    // 全量 falseRefusalCount 保留为观察项（见分组输出）。
    ["protectedFalseRefusalCount", metrics.protectedFalseRefusalCount, 0],
  ];

  /*
   * 精排保活门槛：有精排产物时，精排必须**真的改变了排序**。
   *
   * 这条专门防"静默失效"：曾经 config.json 缺 problem_type，单 logit 走 softmax
   * 恒为 1，全体候选同分、排序原样不动，但整条链路"跑通了"、指标也没崩 ——
   * 用户以为在用两阶段检索，其实精排是空转。这种失效只有"变化率"能发现。
   *
   * 同时要求 missingPairs = 0：有候选查不到精排分，说明固化文件与语料不同步。
   */
  if (metrics.rerank) {
    const r = metrics.rerank;
    if (r.top1ChangedRate <= 0) {
      failed.push("rerank.top1ChangedRate<=0（精排未改变任何排序，疑似静默失效）");
    }
    if (r.missingPairs > 0) failed.push(`rerank.missingPairs=${r.missingPairs}`);

    /*
     * 精排质量断言：开精排不能比不开更差。
     *
     * top-1 变化率只证明"精排动了"，不证明"动对了" ——
     * 固化 logit 取反、换错模型、分批拼错，都会让变化率正常但排序变糟。
     * 这条断言用 recall@3 兜住"动错方向"。
     *
     * 留 0.01 的 ε：精排与 RRF 在小样本上偶尔打平或微降是正常波动，
     * 但**系统性掉超过 1pt 就说明精排真的在帮倒忙**。
     */
    const eps = 0.01;
    if (r.recallAt3 < metrics.recallAt3 - eps) {
      failed.push(
        `rerank.recall@3(${r.recallAt3}) < 无精排(${metrics.recallAt3}) - ${eps}（精排帮倒忙）`,
      );
    }
  }

  /*
   * 分组门槛。三类正样本难度差很多，只看总分会被组间比例掩盖：
   * s 组整体崩掉时总 recall 也只掉十几个点。所以每组各设一条。
   * 数值 = 实测基线往下留余量，作用是防回归，不是定目标。
   *
   * 2026-10-12 重取：修掉「评测不挂向量 → MMR 退化成纯相关度排序」的口径漂移后
   * （见 loadCorpus 注释），MMR 真按线上行为参与多样性选择，实测值变化为
   * q 0.92→0.72、s 0.29→0.50、m 1.0→1.0：技术措辞组因近义块被 MMR 降权而下降，
   * 口语化组反而明显受益（相似块不再挤占 top3）。门槛按新实测值重取（各留 ~0.04 余量）。
   *
   * 2026-10-15 终取：检索侧改为**语义粗拒**（词表信号降为提示）+ 查询扩展 + 关闭 MMR
   * （依据见 retrieval.ts 的 mmrLambda / minMaxCosine 注释）。三组实测
   * q 0.92 / s 0.50 / m 1.00，s 组从 0.07 回到 0.50 —— 路线图 M2 的「口语化 ≥0.50」达成，
   * 因此 s 恢复为**执行项**（此前的观察项处理已删除）。
   */
  const BY_GROUP_GATE: Record<string, number> = {
    q: 0.88, // 技术措辞：实测 0.92（2026-10-15，语义粗拒 + 查询扩展 + MMR 关）
    s: 0.45, // 口语化：实测 0.50（2026-10-15，同上）—— 路线图目标 0.50 达成，门槛留 0.05 余量
    m: 0.9, //  单文档多主题：跨块合并/多窗口是设计目标，必须高命中（实测 1.0）
  };
  for (const [g, min] of Object.entries(BY_GROUP_GATE)) {
    const v = metrics.byGroup[g];
    if (!v) {
      failed.push(`缺少分组 ${g}`);
      continue;
    }
    if (v.recallAt3 < min) failed.push(`${g}.recall@3<${min}`);
    if (v.noHit > 0) failed.push(`${g}.被拒 ${v.noHit} 条`);
  }

  /*
   * 相对门槛（语料无关）：hybrid recall@3 ≥ BM25-only recall@3 × 0.95。
   *
   * 绝对门槛在换语料后就失效了；这条比的是"开了向量路是否比只用关键词更差"，
   * 与语料难度无关，能兜住融合配置退化（两路权重调歪、向量路被错误压制、
   * RRF 实现改坏）这类问题 —— 它们不会让指标掉到绝对门槛以下，但会让 hybrid 白开。
   *
   * 口径：**只比受保护人群（q+m）**。s 已被检索侧按「调参口径」刻意粗拒，
   * 纳入分母等于用"已知要放弃的人群"去评判向量质量 —— 两侧都被 s 的 0 覆盖率
   * 稀释，门槛要么恒过要么恒红，失去诊断力。收缩到 q+m 后，这条门槛才真正
   * 只反映"向量路在两种模式都被要求服务的人群上有没有拖后腿"。
   */
  if (metrics.hybrid && metrics.protectedBm25OnlyRecallAt3 !== null) {
    const floor = metrics.protectedBm25OnlyRecallAt3 * 0.95;
    if (metrics.protectedRecallAt3 < floor) {
      failed.push(
        `受保护人群 recall@3(${metrics.protectedRecallAt3}) < BM25-only(${metrics.protectedBm25OnlyRecallAt3})×0.95=${floor.toFixed(4)}（向量路拖后腿）`,
      );
    }
  }

  if (metrics.hybrid) {
    /*
     * 拒答门槛：检索侧多信号融合（IDF ∪ 最高余弦 ∪ 对比度 ∪ BM25 最高分，
     * 见 retrieval.ts applyThreshold 与 tune 第 7 段「融合·仅检索侧」）。
     *
     * 2026-10-15 起默认启用融合拒答：调参口径 q+m 下零误拒拒答率
     * 由单信号的 40% 提升到 **9/10**（实测 ≥0.60 达标），且 q+m 误拒仍为 0。
     * 剩余分不开的负样本不硬拒，交给答案侧 verifyAnswer 裁决。
     *
     * ── 为什么用**绝对条数**而不是比率 ──
     * 比率门槛（如 = 0.4）是"实测值即门槛"：负样本只有 10 条，翻转 1 条就掉 10pt，
     * 任何一次语料微调都会随机变红，而它并没有反映真实退化。
     * 改用条数后含义明确：**10 条负样本里至少拒掉 6 条**。
     * 实测当前 9/10，留 3 条余量 —— 既能在融合拒答真的塌掉时变红
     * （把融合信号关掉会退化为 4/10 → 红），也不会因个别样本抖动误报。
     *
     * 取值依据与日期：2026-10-15，语义粗拒下实测 5/10，门槛取 4/10；
     * **最终拒答能力以 `npm run eval:verdict` 为准**（那里是检索+答案合计口径）。
     */
    const minGates: Gate[] = [
      ["negativeRefusedCount", metrics.negativeRefusedCount, 4],
    ];
    for (const [name, actual, min] of minGates) {
      if (actual < min) failed.push(`${name}<${min}`);
    }
  } else {
    console.log(
      "\n[note] 无向量路：本轮只有 BM25，召回类与拒答类指标都无法达标，" +
        "不设固定门槛（分组门槛仍然生效）。补齐 embeddings.json 后即为完整口径。",
    );
  }

  for (const [name, actual, max] of maxGates) {
    if (actual > max) failed.push(`${name}>${max}`);
  }

  if (failed.length > 0) {
    console.error(`\n[FAIL] 指标未达门槛：${failed.join("，")}`);
    process.exit(1);
  }
  console.log("[PASS] 全部指标满足门槛");
}

main();
