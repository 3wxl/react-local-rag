/**
 * 检索后处理（纯函数，不碰 worker / IndexedDB，可单测）：
 *
 * 1. RRF 融合：向量余弦排名 + BM25 排名用 1/(k+rank) 合并，不做分数归一化
 * 2. 邻近合并：同文档 ord 连续的命中合并成一个窗口（overlap 片段不再各占一个坑）
 * 3. contentHash 去重：合并后仍完全相同的内容只留一条
 * 4. MMR 保多样：按最大边际相关性挑最终 K 条，抑制近似重复占位
 * 5. 门限 + 自适应 K：低于相关度门限直接丢弃（调用方据此走「文档未提及」拒答）
 */

import type { TextChunk } from "../types/doc";

/** RRF 常数 k：平滑排名影响（与 worker 内取值一致） */
export const RRF_K = 60; //`RetrievalDoc`：参与检索的单个分块，继承`TextChunk`，可以附带向量，`vector?:Float32Array|null`，`null`代表向量检索降级，只用 BM25。

/** 参与融合的检索文档：分块身份 + 文本 + 向量（向量可为空 = 向量路降级） */
export interface RetrievalDoc extends TextChunk {
  vector?: Float32Array | null;
}

/** 融合后的原始命中（合并/去重前） */
export interface FusedHit {
  index: number;
  doc: RetrievalDoc;
  /** 向量余弦相似度（向量路降级时为 null） */
  vecScore: number | null;
  /** 向量路排名（1 起，null = 未上榜） */
  vecRank: number | null;
  /** BM25 原始分 */
  bm25Score: number;
  /** BM25 路排名（1 起，0 = 零分未上榜） */
  bm25Rank: number;
  /** RRF 融合分（最终排序依据） */
  rrfScore: number;
}

/**
 * 最终返回给调用方的检索命中。
 *
 * 小块检索 / 大块喂 LLM：
 * - `chunkIds` / `matchedContent` 是**精确命中**的块（向量定位出来的），
 *   作为引用锚点与高亮范围；
 * - `windowChunkIds` / `content` 是命中块 + 前后邻块拼出的**窗口**，
 *   送进 prompt 给模型足够上下文（窗口内去重叠拼接，不重复文本）。
 */
export interface SearchHit {
  /** 精确命中的块 id（按 ord 升序），引用锚点 */
  chunkIds: string[];
  /** 送进 prompt 的窗口块 id（命中块 + 邻块，按 ord 升序） */
  windowChunkIds: string[];
  docId: string;
  docName: string;
  /** 窗口覆盖的 ord 区间（闭区间） */
  ordStart: number;
  ordEnd: number;
  /** 标题路径（引用展示「在哪个章节下」） */
  headingPath: string[];
  pageStart: number;
  pageEnd: number;
  /** 窗口全文（喂 LLM 的上下文） */
  content: string;
  /** 精确命中的文本（引用卡片预览 / 高亮用，不含邻块扩展） */
  matchedContent: string;
  /** 窗口内最高 RRF 分 */
  rrfScore: number;
  /** 窗口内最高向量余弦（无向量时为 null） */
  vecScore: number | null;
  /** 窗口内最高 BM25 分 */
  bm25Score: number;
  vecRank: number | null;
  bm25Rank: number;
  /** 归一化相关度 0~1（门限判定 + UI 展示） */
  relevance: number;
}

export interface RetrievalOptions {
  /** 返回条数上限（自适应 K 不会超过它） */
  topK: number;
  /** 向量余弦相关度门限：低于此值且 BM25 未命中前列的命中被丢弃 */
  minVecScore?: number;
  /** BM25 豁免排名上限：BM25 排进前 N 名视为强关键词命中，豁免向量门限 */
  bm25ExemptRank?: number;
  /**
   * 拒答信号 1：IDF 加权查询覆盖率下限。
   * 查询词的信息量有多少能在语料词表里找到；低于此值判定「文档未提及该主题」。
   * 不需要向量模型，纯词表统计，因此纯 BM25 模式下也生效。
   */
  minIdfCoverage?: number;
  /** 相对门限：RRF 分低于最高分 * ratio 的命中被丢弃 */
  relativeCutRatio?: number;
  /** 邻近合并窗口的字符上限（超过则不再合并下一块） */
  maxWindowChars?: number;
  /** 上下文扩展半径：命中块前后各取多少块拼进喂 LLM 的窗口（默认 1，0 = 不扩展） */
  windowRadius?: number;
  /** MMR 多样性权重（1 = 只看相关度，0 = 只看多样性），1 表示关闭 MMR */
  mmrLambda?: number;
  /** MMR 候选池大小（先按 RRF 取前 N 再挑） */
  mmrPoolSize?: number;
  /**
   * RRF 融合常数 k：score = Σ w/(k+rank)。
   * k 越大，头部排名优势越弱（排名 1 和 10 的得分差距被摊平），融合越"民主"；
   * k 越小，越偏向各自的头部命中。默认 60（RRF 论文推荐值）。
   */
  rrfK?: number;
  /**
   * 向量路 RRF 权重（w_vec）。默认 1 = 等权（原始 RRF）。
   * 调参见 tune.ts「RRF 权重/k 扫描」：若向量路在本语料上系统性拖后腿，
   * 可下调（0.5）或上调（1.5/2）——但必须通过 hybrid ≥ BM25-only×0.95 的相对门槛。
   */
  vecWeight?: number;
  /** BM25 路 RRF 权重（w_bm25）。默认 1 = 等权。 */
  bm25Weight?: number;
  /**
   * 查询级「最高余弦」门限：**全语料**最高余弦低于此值 → 直接拒答。
   * 与 minVecScore（逐命中门限）不同，这是**查询级**信号，参与多信号融合拒答。
   * 阈值来源：tune 第 7 段「融合·仅检索侧」在调参口径（q+m）下扫出的零误拒最宽点。
   */
  minMaxCosine?: number;
  /** 查询级「余弦对比度」（最高余弦 − 全语料余弦均值）门限：低于此值 → 拒答 */
  minCosGap?: number;
  /** 查询级「BM25 最高分」（全语料）门限——**已不参与检索侧粗拒**（同上，降为提示信号） */
  minBm25?: number;
}

export const DEFAULT_RETRIEVAL_OPTIONS: Omit<
  Required<RetrievalOptions>,
  "topK"
> = {
  /**
   * 余弦相关度门限（检索侧弱门限，粗滤明显无关候选）。
   *
   * 2026-10-02 实测（45 块旧语料 / 10 负样本，`npm run tune:retrieval` 第 3、7 段）：
   *   - 调参口径（25 条 q 关键词正样本）：top1 余弦最低 0.593，零误拒门限可放到 0.534、
   *     拒掉 80% 负样本；
   *   - 保守口径（再加 14 条 s 口语化难例）：最低余弦降到 0.499，零误拒门限只能取 0.499、
   *     拒答 60%。
   * 默认取 0.5 —— 对两类人群都零误拒的最宽点；不用 0.534 贪那 20%，因为线上真实
   * 查询本来就混有口语化提问。两类分布尾部仍重叠（负样本最高 0.617），分不开的负样本
   * **不靠提高门限硬拒**，而是放行给答案侧 verifyAnswer
   * （语义证据 + 词法证据 + 数字核查）做最终置信度裁决。
   */
  minVecScore: 0.5,
  /**
   * BM25 豁免排名：**仅在向量降级（vecAvailable=false）时生效**。
   * 向量可用时一律只看余弦 —— 实测豁免一旦生效，负样本靠「公司」「产品」
   * 这类通用词的 bigram 命中就能排进前三，把余弦门限的粗滤效果架空。
   */
  bm25ExemptRank: 3,
  relativeCutRatio: 0.4,
  /**
   * IDF 覆盖率门限——**已不参与检索侧粗拒**（2026-10-15 起降为提示信号）。
   * 保留字段是为了：① 响应里的 refusal_signals 要如实报告取值；
   * ② 答案侧与排障需要一条"查询词在语料里能不能找到"的参考线。
   * 详见 applyThreshold 顶部注释：它与 maxBm25 都无法区分口语化提问与否样本。
   *
   * 2026-10-15 重取（119 块语料 / 37 正调参样本 q+m / 10 负样本，tune 第 7 段「融合·仅检索侧」）：
   * 调参口径 q+m 下零误拒最宽切点 = **0.227795**（拒 6/10 负样本）。
   * 这里取 0.2277（**略微向下**，不向上取整）：向上取整会把取值恰好等于最小值的
   * 正样本压到门限之下而误拒 —— 首次落地正是踩了这个坑（q/m 各误拒 1 条）。
   *
   * 取舍（已知代价）：口语化难例 s* 的 IDF 覆盖率与负样本重叠，纳入保护则本信号拒答率为 0；
   * 按「调参口径」（只保护 q+m）可换取拒答能力，代价是部分 s 会被误拒。
   * 这是刻意选择：s 的「文档未提及」由答案侧 verifyAnswer 兜底，检索侧不再为它让路。
   */
  minIdfCoverage: 0.2277,
  /**
   * 查询级最高余弦门限（检索侧粗拒的语义路）。
   *
   * 2026-10-15 取值 **0.53**：受保护人群（q+m）的实测最低余弦是 0.5427，
   * 因此 0.53 对 q+m 仍是零误拒（留了 0.013 的余量），同时把两条口语化难例
   * （s006 0.535 / s011 0.541，恰好在旧值 0.5426 之下）放进来 ——
   * 这两条正是旧阈值下 s 组唯二的被拒样本。
   * 取舍：对**未见过的语料**，0.53 比 0.5426 更宽松一点，
   * 代价是可能多放行极少数与语料弱相关的查询 —— 由答案侧 verifyAnswer 兜底（这正是分工）。
   */
  minMaxCosine: 0.53,
  /** 查询级余弦对比度门限（粗拒的语义路）：零误拒切点 0.077628，取 0.0776（略向下） */
  minCosGap: 0.0776,
  /** 查询级 BM25 最高分（提示信号，不参与粗拒。取值 = 当时扫出的零误拒切点，留档用） */
  minBm25: 8.2246,
  // 注意：没有 minRerankScore。精排（cross-encoder）只做**相对排序**，不做绝对分拒答 ——
  // bge-reranker-base 的原始 logit 对「跨查询的全局置信度」不可靠（正样本里存在 -10 离群值，
  // 与负样本分布重叠，任何绝对门限都会误拒）。「文档未提及」统一由答案侧
  // verifyAnswer（语义证据 + 词法证据 + 数字核查）裁决。
  maxWindowChars: 1200,
  windowRadius: 1,
  /**
   * MMR 多样性权重 λ。**1.0 = 关闭 MMR**（2026-10-15 重测后确定）。
   *
   * 旧默认 0.7 出自一次失效的扫描（tune 的 buildCorpus 漏挂 doc.vector → 多样性判定恒为 0，
   * 曲线是平的，0.7 没有依据）；修好后重扫一度取 0.9。**引入查询扩展后结论反转**：
   * 四种配置实测（119 块语料，语义粗拒 + 扩展权重 0.7，分组 recall@3）：
   *   λ=0.9 扩展关 → q 0.920 / s 0.214 / m 1.000
   *   λ=0.9 扩展开 → q 0.920 / s 0.143 / m 1.000
   *   λ=1.0 扩展关 → q 0.960 / s 0.143 / m 0.917
   *   λ=1.0 扩展开 → q 0.920 / **s 0.429~0.500** / m 1.000   ← 本值
   *
   * 读法：**MMR 的结果侧多样性会抵消多查询融合的收益**。两者做的是同一件事 ——
   * 让 top-k 别挤在一个主题上：MMR 靠"惩罚与已选相似的结果"，
   * 多查询融合靠"多问几种说法再融合排名"。本语料上前者的代价更大
   * （把正确答案当成"重复"压下去），后者更便宜（改写本身就带来了问法多样性）。
   * 故关掉 MMR（λ=1.0），把多样性交给查询扩展。
   */
  mmrLambda: 1,
  mmrPoolSize: 20,
  /*
   * RRF 融合参数：**维持 RRF 论文默认值**（等权 + k=60）。
   *
   * 依据（tune 第 2 段，2026-10-15 重取，119 块语料，hybrid 全实时编码）：
   * 先修了一处工具分叉 —— tune 的 buildCorpus 曾漏挂 doc.vector，使 MMR 的
   * 多样性判定（读 doc.vector）退化为恒 0，λ 失效。修好后再扫
   * vecWeight ∈ {0.5, 1.0, 1.5, 2.0} × k ∈ {20, 40, 60, 80} 共 16 组：
   * hybrid recall@3 落在 81.1~83.8，**全部低于 BM25-only 94.6 × 0.95 = 89.9**。
   * 也就是说本语料上向量路系统性拖后腿，**问题不在融合参数**（换权重/k 抬不上去），
   * 而在向量口径本身（query instruction / 归一化 / 模型选型），需要单独排查。
   *
   * 既然没有任何一组配置能过相对门槛，就没有理由偏离论文默认（Cormack et al. 2009
   * 原始 RRF 即等权 + k=60）；改成别的值等于凭空引入无依据的超参。
   *
   * 相对门槛（语料无关）在 CI：hybrid recall@3 ≥ BM25-only × 0.95 ——
   * 它不依赖具体语料，能兜住"融合配置退化"这类问题。见 scripts/eval.ts 的 gate。
   */
  rrfK: 60,
  vecWeight: 1,
  bm25Weight: 1,
};

/* ---------------- 1. RRF 融合 ---------------- */

/**
 * 把原始分数数组转成排名（1-based，0 = 零分未上榜；同分并列）。
 */
export function scoresToRanks(
  scores: number[],
  includeZero: boolean,
): number[] {
  const n = scores.length;
  if (n === 0) return [];
  const ranks = new Array<number>(n).fill(0);
  const order = scores
    .map((s, i) => ({ s, i }))
    .filter((x) => (includeZero ? true : x.s > 0))
    .sort((a, b) => b.s - a.s);
  for (let r = 0; r < order.length; r++) {
    if (r > 0 && order[r].s === order[r - 1].s) {
      ranks[order[r].i] = ranks[order[r - 1].i];
    } else {
      ranks[order[r].i] = r + 1;
    }
  }
  return ranks;
}

/** 点积：归一化向量的余弦相似度等于点积 */
export function dotProduct(a: Float32Array, b: Float32Array): number {
  const len = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < len; i++) sum += a[i] * b[i];
  return sum;
}

/**
 * RRF 融合：两路各自排名后用 Σ w/(k+rank) 相加。
 * 不做任何分数归一化，彻底规避两路分数量纲不一致。
 *
 * @param vecScores 向量余弦分（null = 向量路降级，只用 BM25）
 * @param opts.k 融合常数（默认 RRF_K=60）
 * @param opts.vecWeight 向量路权重（默认 1，等权即原始 RRF）
 * @param opts.bm25Weight BM25 路权重（默认 1）
 */
export interface RrfFuseOptions {
  k?: number;
  vecWeight?: number;
  bm25Weight?: number;
}

export function rrfFuse(
  docs: RetrievalDoc[],
  bm25Scores: number[],
  vecScores: number[] | null,
  opts: RrfFuseOptions = {},
): FusedHit[] {
  return rrfFuseRuns(docs, [{ bm25Scores, vecScores }], opts);
}

/**
 * 多路 RRF 融合（查询扩展用）：每个查询变体算一路，Σ w/(k+rank) 相加。
 *
 * 与单查询路径的关系：`rrfFuse` 就是"只有一路"的特例 ——
 * 传一路时结果与重构前**逐位相同**（parity dump 钉住这一点），
 * 所以查询扩展是纯增量：不开扩展时行为不变。
 *
 * **只有第一路（原查询）决定命中上的分数与排名字段**，后续路只累加 RRF 分：
 * `vecScore` / `bm25Rank` 既是展示值，也是**接纳判据**（`applyThreshold` 里
 * `vecScore >= minVecScore`）与 **MMR 的相关度**输入。若让改写路也能抬高这些值，
 * 等于让「我们猜的问法」替用户的问题接纳候选 —— 实测代价明确：
 * 改写参与接纳时 q 组 recall@3 从 0.88 掉到 0.84（改写带来的次优块挤掉了正确答案），
 * 而排名贡献几乎为零（权重 0.2~1.0 扫下来主指标不变）。
 * 所以改写只做一件事：**把"文档里有没有这件事"的信号抬高**（见 bestRefusalSignals），
 * 不干预排序与接纳。
 */
export function rrfFuseRuns(
  docs: RetrievalDoc[],
  runs: { bm25Scores: number[]; vecScores: number[] | null; weight?: number }[],
  opts: RrfFuseOptions = {},
): FusedHit[] {
  const k = opts.k ?? RRF_K;
  const vecWeight = opts.vecWeight ?? 1;
  const bm25Weight = opts.bm25Weight ?? 1;

  const prepared = runs.map((run) => ({
    bm25Ranks: scoresToRanks(run.bm25Scores, false),
    vecRanks: run.vecScores ? scoresToRanks(run.vecScores, true) : null,
    bm25Scores: run.bm25Scores,
    vecScores: run.vecScores,
    weight: run.weight ?? 1,
  }));
  const primary = prepared[0];

  return docs.map((doc, i) => {
    let rrfScore = 0;
    for (const run of prepared) {
      const vr = run.vecRanks ? run.vecRanks[i] : null;
      if (vr && vr > 0) rrfScore += (vecWeight * run.weight) / (k + vr);
      if (run.bm25Ranks[i] > 0) rrfScore += (bm25Weight * run.weight) / (k + run.bm25Ranks[i]);
    }
    const vr = primary.vecRanks ? primary.vecRanks[i] : null;
    return {
      index: i,
      doc,
      vecScore: primary.vecScores ? (primary.vecScores[i] ?? 0) : null,
      vecRank: vr && vr > 0 ? vr : null,
      bm25Score: primary.bm25Scores[i] ?? 0,
      bm25Rank: primary.bm25Ranks[i] ?? 0,
      rrfScore,
    };
  });
}

/** 一个查询变体的两路原始分（查询扩展的多路输入） */
export interface QueryVariantScores {
  bm25Scores: number[];
  vecScores: number[] | null;
  /** IDF 加权覆盖率（BM25 侧统计量，调用方算好） */
  idfCoverage?: number;
  /** 该变体在融合里的相对权重（原查询 = 1，扩展变体默认 < 1） */
  weight?: number;
}

/**
 * 拒答信号在多变体下的取值：**逐信号取最优**（最高余弦取最大、对比度取最大、
 * BM25 最高分取最大、IDF 覆盖率取最大）。
 *
 * 判据因此变成：**原查询与所有改写都看不出相关性时才拒答**。
 * 这不是"放松"，而是把信号用在它们真正该在的位置上 ——
 * 词表类信号（IDF 覆盖率 / BM25 最高分）在**语义化提问**上天然失效：
 * 「这台东西在严寒酷暑下还能正常干活吗」与语料「工作温度 -20℃~70℃」零词重合，
 * 与真正的负样本（问语料里根本没有的东西）在该信号上无法区分。
 * 只看原查询会把这类问题一律粗拒；而如果某个**改写**（术语化后）能命中语料，
 * 就说明文档里确实有这件事 —— 不该拒。
 */
export function bestRefusalSignals(
  variants: QueryVariantScores[],
): { maxCosine?: number; cosGap?: number; maxBm25?: number; idfCoverage?: number } {
  let maxCosine: number | undefined;
  let cosGap: number | undefined;
  let maxBm25: number | undefined;
  let idfCoverage: number | undefined;
  const takeMax = (cur: number | undefined, next: number | undefined) =>
    next === undefined ? cur : cur === undefined ? next : Math.max(cur, next);
  for (const variant of variants) {
    const signals = computeRefusalSignals(variant.bm25Scores, variant.vecScores);
    maxCosine = takeMax(maxCosine, signals.maxCosine);
    cosGap = takeMax(cosGap, signals.cosGap);
    maxBm25 = takeMax(maxBm25, signals.maxBm25);
    idfCoverage = takeMax(idfCoverage, variant.idfCoverage);
  }
  return { maxCosine, cosGap, maxBm25, idfCoverage };
}

/* ---------------- 2. 邻近合并 ---------------- */

/** a 的末尾与 b 的开头的最长重叠长度（分块自带整句重叠，拼接时去重） */
export function overlapLength(a: string, b: string): number {
  const max = Math.min(a.length, b.length);
  for (let len = max; len > 0; len--) {
    if (a.endsWith(b.slice(0, len))) return len;
  }
  return 0;
}

/** 去重叠拼接：去掉 b 开头与 a 结尾重复的部分 */
export function appendWithoutOverlap(a: string, b: string): string {
  return a + b.slice(overlapLength(a, b));
}

/**
 * 邻近合并：同一 docId 且 ord 连续的命中合并成一个窗口。
 * 合并后的窗口只占一个结果位，避免 overlap 造成的近似重复片段挤占 topK。
 */
export function mergeNeighbors(
  hits: FusedHit[],
  maxWindowChars = DEFAULT_RETRIEVAL_OPTIONS.maxWindowChars,
): FusedHit[][] {
  const byDoc = new Map<string, FusedHit[]>();
  for (const hit of hits) {
    const list = byDoc.get(hit.doc.docId);
    if (list) list.push(hit);
    else byDoc.set(hit.doc.docId, [hit]);
  }

  const windows: FusedHit[][] = [];
  for (const list of byDoc.values()) {
    list.sort((a, b) => a.doc.ord - b.doc.ord);
    let current: FusedHit[] = [];
    let currentChars = 0;
    for (const hit of list) {
      if (current.length === 0) {
        current = [hit];
        currentChars = hit.doc.content.length;
        continue;
      }
      const prev = current[current.length - 1];
      const contiguous = hit.doc.ord === prev.doc.ord + 1;
      const fits = currentChars + hit.doc.content.length <= maxWindowChars;
      if (contiguous && fits) {
        current.push(hit);
        currentChars += hit.doc.content.length;
      } else {
        windows.push(current);
        current = [hit];
        currentChars = hit.doc.content.length;
      }
    }
    if (current.length > 0) windows.push(current);
  }
  return windows;
}

/** 一个窗口内的全部信息合成一条 SearchHit（核心=精确命中；窗口扩张见 expandHit） */
export function windowToHit(window: FusedHit[]): SearchHit {
  const sorted = [...window].sort((a, b) => a.doc.ord - b.doc.ord);
  const best = window.reduce(
    (m, h) => (h.rrfScore > m.rrfScore ? h : m),
    window[0],
  );
  const matchedContent = joinChunks(sorted.map((h) => h.doc.content));
  const vecScore = window.reduce<number | null>(
    (m, h) =>
      h.vecScore === null
        ? m
        : m === null
          ? h.vecScore
          : Math.max(m, h.vecScore),
    null,
  );
  const first = sorted[0].doc;
  const last = sorted[sorted.length - 1].doc;
  const chunkIds = sorted.map((h) => h.doc.chunkId);
  return {
    chunkIds,
    // 未扩展前，窗口 = 命中块自身（expandHit 负责向两侧扩邻块）
    windowChunkIds: chunkIds,
    docId: first.docId,
    docName: first.docName,
    ordStart: first.ord,
    ordEnd: last.ord,
    headingPath: first.headingPath,
    pageStart: first.pageStart,
    pageEnd: last.pageEnd,
    content: matchedContent,
    matchedContent,
    rrfScore: best.rrfScore,
    vecScore,
    bm25Score: window.reduce((m, h) => Math.max(m, h.bm25Score), 0),
    vecRank: best.vecRank,
    bm25Rank: window.reduce(
      (m, h) =>
        h.bm25Rank > 0 ? Math.min(m === 0 ? Infinity : m, h.bm25Rank) : m,
      0,
    ),
    relevance: 0, // 由 applyThreshold 统一计算
  };
}

/** 按 ord 顺序去重叠拼接多块文本 */
function joinChunks(contents: string[]): string {
  if (contents.length === 0) return "";
  let out = contents[0];
  for (let i = 1; i < contents.length; i++) {
    out = appendWithoutOverlap(out, contents[i]);
  }
  return out;
}

/**
 * 上下文扩展（小块检索 / 大块喂 LLM）：
 * 把精确命中的块向两侧各扩 radius 块，拼成送进 prompt 的窗口。
 * 检索精度由命中块保证（chunkIds 仍是引用锚点），窗口只负责给 LLM 足够上下文。
 * 窗口字符总量受 maxChars 限制，超限时向命中区收拢（先丢远端邻块，绝不丢命中块）。
 */
export function expandHit(
  hit: SearchHit, //windowToHit 生成的 SearchHit（原始命中窗口)
  docChunks: RetrievalDoc[], //`docChunks`：**该文档全部 chunk 数组**（整篇文档所有分片，按 ord 顺序排好）
  radius = DEFAULT_RETRIEVAL_OPTIONS.windowRadius, //`radius`：向左右最多向外扩展多少个 chunk
  maxChars = DEFAULT_RETRIEVAL_OPTIONS.maxWindowChars, //`maxChars`：扩展后窗口总字符上限
): SearchHit {
  if (radius <= 0 || docChunks.length === 0) return hit;
  const matched = new Set(hit.chunkIds);
  const matchedIdx = docChunks
    .map((d, i) => (matched.has(d.chunkId) ? i : -1))
    .filter((i) => i >= 0);
  if (matchedIdx.length === 0) return hit;

  const coreLo = matchedIdx[0];
  const coreHi = matchedIdx[matchedIdx.length - 1];
  let lo = Math.max(0, coreLo - radius);
  let hi = Math.min(docChunks.length - 1, coreHi + radius);

  const size = (l: number, h: number) =>
    docChunks.slice(l, h + 1).reduce((n, d) => n + d.content.length, 0);
  while (lo < coreLo && size(lo, hi) > maxChars) lo++;
  while (hi > coreHi && size(lo, hi) > maxChars) hi--;

  const slice = docChunks.slice(lo, hi + 1);
  return {
    ...hit,
    windowChunkIds: slice.map((d) => d.chunkId),
    content: joinChunks(slice.map((d) => d.content)),
    ordStart: slice[0].ord,
    ordEnd: slice[slice.length - 1].ord,
    pageStart: slice[0].pageStart,
    pageEnd: slice[slice.length - 1].pageEnd,
  };
}

/* ---------------- 3. 去重 ---------------- */

/** contentHash 去重：完全相同内容只保留 RRF 分最高的一条 */
export function dedupeByHash(hits: SearchHit[]): SearchHit[] {
  const seen = new Map<string, SearchHit>(); //哈希 → 命中 的映射，记录"每个哈希对应的最佳命中"
  const order: string[] = []; //哈希的出现顺序，用于最后按顺序输出
  for (const hit of hits) {
    const key = hashOf(hit.content);
    const prev = seen.get(key);
    if (!prev) {
      seen.set(key, hit);
      order.push(key);
    } else if (hit.rrfScore > prev.rrfScore) {
      seen.set(key, hit);
    }
  }
  return order.map((k) => seen.get(k)!);
}

/** 文本哈希（FNV-1a 32 位；与 chunk.ts 的 contentHash 同算法） */
export function hashOf(text: string): string {
  let hash = 0x811c9dc5; //0x811c9dc5 是 FNV-1a 的 32 位标准偏移基数
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

/* ---------------- 4. MMR 保多样 ---------------- */

/** 两条命中的语义相似度（任一条无向量时按 0 处理，退化为纯相关度排序） */
function similarity(a: SearchHitWithVector, b: SearchHitWithVector): number {
  if (!a.vector || !b.vector) return 0;
  return dotProduct(a.vector, b.vector);
}

/**
 * MMR（最大边际相关性）：每轮挑「λ·相关度 − (1−λ)·与已选集合的最大相似度」最高的一条。
 * λ=1 退化为纯相关度排序；λ<1 时抑制语义重复片段挤占结果。
 * 相关度用归一化后的 relevance（0~1），与余弦相似度同量纲。
 */
export function mmrSelect(
  hits: SearchHitWithVector[], //经过前面所有步骤、**带向量**的候选列表
  limit: number, //最终想要选出多少条 top 结果
  lambda = DEFAULT_RETRIEVAL_OPTIONS.mmrLambda, //`lambda`：平衡「相关性」和「多样性」的系数
): SearchHitWithVector[] {
  if (lambda >= 1 || hits.length <= 1) return hits.slice(0, limit);
  const pool = [...hits];
  const selected: SearchHitWithVector[] = [];
  while (pool.length > 0 && selected.length < limit) {
    let bestIdx = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < pool.length; i++) {
      const cand = pool[i];
      const maxSim = selected.reduce(
        (m, s) => Math.max(m, similarity(cand, s)),
        0,
      );
      const score = lambda * cand.relevance - (1 - lambda) * maxSim;
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }
    selected.push(pool[bestIdx]);
    pool.splice(bestIdx, 1);
  }
  return selected;
}

/* ---------------- 5. 门限 + 自适应 K ---------------- */

/** SearchHit 内部扩展：保留向量供 MMR 使用（不进入 postMessage，序列化时丢弃） */
export interface SearchHitWithVector extends SearchHit {
  vector?: Float32Array | null;
}

/**
 * 归一化相关度：
 * - 向量路可用：余弦值（bge 归一化向量，落在 0~1）
 * - 向量路降级：按 BM25 排名换算的经验值（rank1≈0.6 递减），仅用于排序展示
 */
function computeRelevance(hit: SearchHit, vecAvailable: boolean): number {
  if (vecAvailable && hit.vecScore !== null) {
    // 有 BM25 精确命中时给一点加成，避免纯关键词命中被语义门限误杀
    const keywordBonus = hit.bm25Rank > 0 ? 0.05 : 0;
    return Math.min(1, hit.vecScore + keywordBonus);
  }
  if (hit.bm25Rank > 0) {
    return Math.max(0, 0.6 - (hit.bm25Rank - 1) * 0.05);
  }
  return 0;
}

export interface ThresholdResult {
  hits: SearchHitWithVector[];
  /** 是否因相关度不足而全部丢弃（调用方据此走「文档未提及」拒答分支） */
  noRelevantHit: boolean;
  /** 丢弃的候选数（可观测性） */
  droppedCount: number;
}

/** 拒答信号快照：检索侧只做粗滤，最终置信度在答案侧裁决 */
export interface RefusalSignals {
  /** 信号 1：IDF 加权查询覆盖率 0~1（缺省 1 = 不参与判定） */
  idfCoverage?: number;
  /** 信号 2：最高归一化相关度（由 hits 内部算出，不必外部传） */
  topRelevance?: number;
  /** 信号 3：全语料最高余弦（查询级；由 postProcessRetrieval 从 vecScores 算出） */
  maxCosine?: number;
  /** 信号 4：余弦对比度 = 全语料最高余弦 − 全语料余弦均值（查询级） */
  cosGap?: number;
  /** 信号 5：全语料 BM25 最高分（查询级） */
  maxBm25?: number;
}

/**
 * 从**全量**两路原始分算出查询级拒答信号（最高余弦 / 对比度 / BM25 最高分）。
 *
 * 单一实现：tune 第 7 段信号诊断与检索管线共用本函数，保证「诊断扫出的阈值」
 * 与「线上门限看到的取值」逐位一致 —— 否则会重演「诊断说零误拒、线上却误拒」的分叉。
 * 遍历按数组原序累加（不排序），避免浮点加法结合律带来的均值漂移。
 */
export function computeRefusalSignals(
  bm25Scores: number[],
  vecScores: number[] | null,
): { maxCosine?: number; cosGap?: number; maxBm25?: number } {
  const maxBm25 = bm25Scores.length > 0 ? Math.max(...bm25Scores) : undefined;
  if (!vecScores || vecScores.length === 0) return { maxBm25 };
  let max = -Infinity;
  let sum = 0;
  for (const v of vecScores) {
    if (v > max) max = v;
    sum += v;
  }
  return { maxCosine: max, cosGap: max - sum / vecScores.length, maxBm25 };
}

/**
 * 门限 + 自适应 K + 检索侧拒答信号：
 *
 * 检索侧只保留两路**粗滤**信号（任一触发 → noRelevantHit，走「文档中未提及」分支）：
 *   1. IDF 加权查询覆盖率：库里有这个词吗（纯词表，不需要向量）
 *   2. 最高归一化相关度：命中的分够高吗（向量模式下的余弦，或 BM25 排名的经验分）
 *
 * 精排（cross-encoder）**不参与拒答**：它只在 worker 里对候选做相对排序。
 * cross-encoder 的 logit 跨查询不可比，绝对门限必然误拒（2026-10-02 tune 实测：
 * 正样本存在 -10 离群值，与负样本最高区间重叠）。最终「答得有没有依据」由答案侧
 * verifyAnswer 的语义证据 + 词法证据 + 数字核查双层校验裁决。
 *
 * 之后再做自适应 K（相对门限）与 topK 封顶。
 */
export function applyThreshold(
  hits: SearchHitWithVector[],
  options: RetrievalOptions & { vecAvailable: boolean } & RefusalSignals,
): ThresholdResult {
  const minVec = options.minVecScore ?? DEFAULT_RETRIEVAL_OPTIONS.minVecScore;
  const ratio =
    options.relativeCutRatio ?? DEFAULT_RETRIEVAL_OPTIONS.relativeCutRatio;
  const exemptRank =
    options.bm25ExemptRank ?? DEFAULT_RETRIEVAL_OPTIONS.bm25ExemptRank;

  /*
   * 检索侧**粗拒**：只看语义信号（最高余弦 ∪ 余弦对比度）。
   *
   * 为什么词表信号（IDF 覆盖率 / BM25 最高分）不再是**一票否决**：
   * 它们无法区分「口语化提问」与「文档里没有」——
   * 2026-10-15 逐条诊断（119 块语料 / 61 条 golden，本地 0.5B 关键词式改写）：
   * s 组的 maxBm25 落在 4.46~7.51、IDF 覆盖率 0.137~0.162，
   * 负样本分别是 4.77~7.06 / 0.152~0.197 —— **两条信号上完全重叠**，
   * 任何切点都会同时误拒 s、放行负样本。实测代价：把这两条放进并集时，
   * s 组 recall@3 从 0.50 掉到 0.07（14 条里 13 条被粗拒），
   * 而负样本拒答只从 4/10 升到 9/10。
   *
   * 所以职责按路线图拆开：**检索侧只做粗拒（语义），细拒交给答案侧 verifyAnswer**。
   * 词表信号仍然计算并随响应返回（`refusal_signals.idf_coverage / max_bm25`），
   * 供答案侧与排障使用，但不在这里杀查询。
   *
   * 语义侧的两条阈值取「q+m 正样本最小值」（零误拒最宽点），因此并集构造上零误拒。
   * **只在向量可用时启用**：降级纯 BM25 时保持原行为（否则 BM25-only 基线自己也会拒正样本，
   * 相对门槛 hybrid ≥ BM25-only × 0.95 失去可比性）。
   * 精排分不参与（它在 applyThreshold 之后才算，且 logit 跨查询不可比）。
   */
  if (options.vecAvailable) {
    const minMaxCos =
      options.minMaxCosine ?? DEFAULT_RETRIEVAL_OPTIONS.minMaxCosine;
    const minGap = options.minCosGap ?? DEFAULT_RETRIEVAL_OPTIONS.minCosGap;
    if (
      (options.maxCosine !== undefined && options.maxCosine < minMaxCos) ||
      (options.cosGap !== undefined && options.cosGap < minGap)
    ) {
      return { hits: [], noRelevantHit: true, droppedCount: hits.length };
    }
  }

  for (const hit of hits) {
    hit.relevance = computeRelevance(hit, options.vecAvailable);
  }

  /*
   * 相关度门限。两条路径的判据不同，不能混用：
   *
   * - 向量可用时：**只看余弦**，不给 BM25 豁免。
   *   实测（2026-10-02，45 块语料 + 39 正 + 10 负）：最高余弦在 0.499 处
   *   零误拒拒答 60%，是检索侧唯一可用信号（剩余 40% 交给答案侧证据校验）；
   *   而 IDF 覆盖率 / 对比度 / BM25 最高分在零误拒约束下拒答率都是 0%。
   *   BM25 豁免一旦生效，负样本靠「公司」「产品」
   *   这类通用词的 bigram 命中就能排进前三，把这条粗滤能力废掉 ——
   *   实测端到端拒答率从 60% 掉到 0%。
   * - 向量不可用（降级纯 BM25）时：没有余弦可判，只能退回到
   *   「BM25 排进前列即算相关」，此时豁免是唯一可用的判据。
   */
  const passed = hits.filter((h) => {
    if (!options.vecAvailable) {
      return h.bm25Rank > 0 && h.bm25Rank <= exemptRank;
    }
    return (h.vecScore ?? 0) >= minVec;
  });

  if (passed.length === 0) {
    return { hits: [], noRelevantHit: true, droppedCount: hits.length };
  }

  const topRrf = passed.reduce((m, h) => Math.max(m, h.rrfScore), 0);
  const cut = topRrf * ratio;
  const kept = passed.filter((h) => h.rrfScore >= cut);
  const capped = kept.slice(0, options.topK);

  return {
    hits: capped,
    noRelevantHit: false,
    droppedCount: hits.length - capped.length,
  };
}

/* ---------------- 精排输出防御 ---------------- */

/** 分数完全相同的判定容差（浮点输出理论上不会精确相等，除非输出的是常数） */
export const RERANK_DEGENERATE_EPSILON = 1e-6;

/**
 * 检测「精排输出无区分度」：两条以上候选的分数极差小于容差。
 *
 * 典型成因：模型 config.json 缺 `problem_type: multi_label_classification`，
 * transformers.js 对单 logit 走 softmax —— 单元素 softmax 恒等于 1，
 * 于是全体候选得分都是 1.0，排序原样不动。**跑通了但完全没生效**，
 * 比直接报错更危险（用户以为在用两阶段检索）。必须显式识别为不可用。
 *
 * @returns null 表示正常；否则返回可展示的原因说明
 */
export function isDegenerateScores(scores: number[]): string | null {
  if (scores.length <= 1) return null;
  let min = Infinity;
  let max = -Infinity;
  for (const s of scores) {
    if (!Number.isFinite(s)) {
      return `精排输出含非有限值（${s}），模型输出异常`;
    }
    if (s < min) min = s;
    if (s > max) max = s;
  }
  if (max - min < RERANK_DEGENERATE_EPSILON) {
    return `全部 ${scores.length} 条候选得分相同（${max}），模型未产生区分度；请检查模型 config.json 是否声明 problem_type: multi_label_classification`;
  }
  return null;
}

/* ---------------- 主流程 ---------------- */

/**
 * 检索后处理全流程：RRF 融合 → 邻近合并 → 去重 → MMR → 门限/自适应 K。
 * 输入是「全量文档 + 两路原始分」，输出是最终 topK 命中（可能为空 = 无相关内容）。
 */
export function postProcessRetrieval(
  docs: RetrievalDoc[], //`docs: RetrievalDoc[]`：原始检索到的所有分块文档
  bm25Scores: number[], //`bm25Scores: number[]`：BM25 关键词检索分数，和 docs 一一对应
  vecScores: number[] | null, //`vecScores: number[] | null`：向量相似度分数；`null`代表本次没有做向量检索（只跑 BM25）
  options: RetrievalOptions & RefusalSignals, //`options: RetrievalOptions`：检索配置 + 拒答信号
  /**
   * 查询扩展的额外变体（可选）：每个变体作为独立一路参与 RRF，
   * 拒答信号逐信号取各变体的**最优值**（见 bestRefusalSignals）。
   * 不传 = 单查询路径，行为与引入查询扩展之前逐位相同。
   */
  variants?: QueryVariantScores[],
): ThresholdResult & { fused: FusedHit[] } {
  const primary: QueryVariantScores = { bm25Scores, vecScores, idfCoverage: options.idfCoverage };
  const runs = variants && variants.length > 0 ? [primary, ...variants] : [primary];
  const fused = rrfFuseRuns(docs, runs, {
    k: options.rrfK ?? DEFAULT_RETRIEVAL_OPTIONS.rrfK,
    vecWeight: options.vecWeight ?? DEFAULT_RETRIEVAL_OPTIONS.vecWeight,
    bm25Weight: options.bm25Weight ?? DEFAULT_RETRIEVAL_OPTIONS.bm25Weight,
  });
  // 只对「有相关性信号」的命中做合并，零分文档直接排除，减少无谓计算
  const candidates = fused.filter((h) => h.rrfScore > 0);
  const sorted = candidates.sort((a, b) => b.rrfScore - a.rrfScore);

  // 邻近合并 → 窗口级命中
  const windows = mergeNeighbors(sorted, options.maxWindowChars);
  const hits = windows
    .map((w) => {
      const hit = windowToHit(w) as SearchHitWithVector;
      // 窗口的向量取窗口内 RRF 最高那条（MMR 多样性判定用）
      const best = w.reduce((m, h) => (h.rrfScore > m.rrfScore ? h : m), w[0]);
      hit.vector = best.doc.vector ?? null;
      return hit;
    })
    .sort((a, b) => b.rrfScore - a.rrfScore);

  const unique = dedupeByHash(hits);
  const poolSize = options.mmrPoolSize ?? DEFAULT_RETRIEVAL_OPTIONS.mmrPoolSize;
  const pool = unique.slice(0, poolSize);

  // 门限先算，给出 relevance 供 MMR 使用
  // 拒答信号透传：idfCoverage 由调用方（worker/eval）从同一份 BM25 统计算出；
  // 余弦类信号用 computeRefusalSignals（与 tune 第 7 段同实现、同口径）。
  // 有扩展变体时逐信号取最优 —— 「原查询与所有改写都看不出相关」才拒答。
  const signals =
    runs.length > 1
      ? bestRefusalSignals(runs)
      : { ...computeRefusalSignals(bm25Scores, vecScores), idfCoverage: options.idfCoverage };
  const gated = applyThreshold(pool, {
    ...options,
    topK: poolSize,
    vecAvailable: runs.some((r) => r.vecScores !== null),
    idfCoverage: signals.idfCoverage ?? options.idfCoverage,
    ...signals,
  });

  if (gated.hits.length === 0) {
    return { hits: [], noRelevantHit: true, droppedCount: hits.length, fused };
  }

  const selected = mmrSelect(
    gated.hits,
    options.topK,
    options.mmrLambda ?? DEFAULT_RETRIEVAL_OPTIONS.mmrLambda,
  );

  // 最后一步才做上下文扩展：只给最终选中的 ≤ topK 条拼邻块窗口，省无用拼接
  const radius = options.windowRadius ?? DEFAULT_RETRIEVAL_OPTIONS.windowRadius;
  const maxChars =
    options.maxWindowChars ?? DEFAULT_RETRIEVAL_OPTIONS.maxWindowChars;
  const byDoc = new Map<string, RetrievalDoc[]>();
  for (const d of docs) {
    const list = byDoc.get(d.docId);
    if (list) list.push(d);
    else byDoc.set(d.docId, [d]);
  }
  for (const list of byDoc.values()) list.sort((a, b) => a.ord - b.ord);

  return {
    hits: selected.map((h) =>
      expandHit(h, byDoc.get(h.docId) ?? [], radius, maxChars),
    ),
    noRelevantHit: false,
    droppedCount: hits.length - selected.length,
    fused,
  };
}
