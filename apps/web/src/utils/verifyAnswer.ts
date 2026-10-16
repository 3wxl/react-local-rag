import { tokenizeSet } from "./tokenize";

/**
 * 幻觉后处理（纯 JS 确定性校验，不调用大模型、不让模型自我判断）
 *
 * 两路证据：
 * 1. 语义证据：embedding worker 把答案逐句向量化，与**实际引用的片段及其窗口**
 *    算最大余弦相似度（不再对全篇索引取 max —— 那样长文档上几乎句句都能找到
 *    某块"像"的文本，导致整体虚高）
 * 2. 词法证据：中文二元组/单词覆盖率 + 数字事实核查（答案中的数字必须能在原文找到）
 *
 * 两路证据加权后给每句裁决：supported / weak / unsupported，再汇总整体裁决。
 * worker 不可用时降级为纯词法判定。
 *
 * 分词走 utils/tokenize.ts 的公共实现，与 BM25 检索口径天然一致。
 */

export type SentenceLevel = "supported" | "weak" | "unsupported";
export type VerifyVerdict = "supported" | "partial" | "unsupported" | "refused";

export interface VerifiedSentence {
  /** 句子原文（与 splitSentences 的切分顺序一一对应，供 UI 逐段渲染） */
  text: string;
  level: SentenceLevel;
  /** 综合依据分 0~1 */
  score: number;
  /** 语义依据分（worker 不可用时为 null） */
  semantic: number | null;
  /** 词法覆盖率 0~1 */
  lexical: number;
  /** 该句中在原文找不到的数字（硬事实捏造信号） */
  unknownNumbers: string[];
  /** 该句声明的引用编号（[1] → 1），空数组 = 未标注引用 */
  citations: number[];
  /** 越界引用（引用了不存在的编号），是编造引用的强信号 */
  invalidCitations: number[];
  /**
   * 主张句但没有**有效**角标（引用强制闭环的第三只眼）。
   *
   * 与 invalidCitations 的区别：那是"引错了"，这是"根本没引"。
   * 只在提供了编号段落表（passages）时才有意义 —— 没有编号可引的旧调用不算未标注。
   */
  uncited: boolean;
  /** 过短的寒暄/语气句，不参与裁决 */
  trivial: boolean;
}

/** 引用闭环的收敛情况（逐句角标 → 越界检测 → 证据收敛的汇总） */
export interface CitationConvergence {
  /** 主张句里没有有效角标的条数 */
  uncitedClaims: number;
  /** 主张句里引用了不存在编号的条数 */
  outOfRangeClaims: number;
  /** 所有有效角标是否都落在编号段落表内（false = 有越界） */
  citationsSubsetOfPassages: boolean;
}

export interface VerificationResult {
  verdict: VerifyVerdict;
  sentences: VerifiedSentence[];
  checkedAt: number;
  /** 校验依据来源：semantic=语义+词法，lexical=仅词法（worker 不可用降级） */
  source: "semantic" | "lexical";
  /** 实际参与语义比对的块数（范围收敛后；0 表示退化为纯词法） */
  scopedChunkCount: number;
  /** 引用闭环收敛情况（逐句角标 → 越界 → 收敛） */
  convergence: CitationConvergence;
}

/* ---------------- 阈值 ---------------- */

export interface VerifyThresholds {
  /** 综合分 ≥ 此值：有文档依据 */
  supported: number;
  /** 综合分 ≥ 此值：依据较弱 */
  weak: number;
  /** 语义证据权重 */
  semWeight: number;
  /** 词法证据权重 */
  lexWeight: number;
}

/**
 * 默认阈值：由 docs/eval/verify-labels.json 的人工标注集校准得到
 * （见 scripts/calibrate-verify.mjs，F1 扫描 selected 阈值区间）。
 * bge-small-zh 上「句子 vs 其真实依据块」余弦普遍 > 0.6，与主题无关块 < 0.4，
 * 加权后 0.55 作为 supported 分界能同时压住假阳与假阴。
 */
export const DEFAULT_VERIFY_THRESHOLDS: VerifyThresholds = {
  supported: 0.55,
  weak: 0.38,
  semWeight: 0.75,
  lexWeight: 0.25,
};

/** 句子至少包含多少个有效字符才算"主张句"，否则视为寒暄句跳过 */
const MIN_CLAIM_CHARS = 4;

/**
 * 单句依据分（校验主流程与阈值校准脚本共用同一公式，避免两处口径漂移）。
 *
 * @param semantic 语义依据分（worker 不可用传 null）
 * @param lexical 词法覆盖率
 * @param flags fabricatedNumber=句子含原文查无此数（数字捏造）；
 *              danglingCitation=引用了不存在的编号（编造引用）
 *
 * 两个硬信号的力度不同：
 * - 数字捏造且语义依据不强（<0.6）→ 压到 weak 之下（判 unsupported，红色告警）
 * - 编造引用 → 一律封顶到 weak（黄色告警）：句子本身可能是真的，只是角标对不上原文，
 *   不足以直接判为编造，但必须让用户看到「这条引用不可信」
 */
export function computeSentenceScore(
  semantic: number | null,
  lexical: number,
  thresholds: VerifyThresholds,
  flags: {
    fabricatedNumber?: boolean;
    danglingCitation?: boolean;
    /** 主张句没有任何有效角标（引用强制闭环：无角标不予采信） */
    uncitedClaim?: boolean;
  } = {},
): number {
  let score =
    semantic !== null
      ? semantic * thresholds.semWeight + lexical * thresholds.lexWeight
      : lexical;
  if (flags.fabricatedNumber && (semantic === null || semantic < 0.6)) {
    score = Math.min(score, thresholds.weak - 0.01);
  }
  if (flags.danglingCitation) {
    score = Math.min(score, thresholds.weak);
  }
  /*
   * 无角标主张句同样封顶到 weak —— 与「引错编号」同一档力度，理由也相同：
   * 句子本身可能完全为真，只是**无法回跳核对**，所以不给绿（supported），
   * 但也不判红（unsupported，那是"判定为编造"）。
   * 强弱模型都适用：弱模型正是因为不产出逐句角标才需要这条兜底，
   * 强制闭环的意义就是让"没引"和"引错"一样可见，而不是静默放行。
   */
  if (flags.uncitedClaim) {
    score = Math.min(score, thresholds.weak);
  }
  return score;
}

/** 依据分 → 裁决等级 */
export function levelOf(
  score: number,
  thresholds: VerifyThresholds,
): SentenceLevel {
  if (score >= thresholds.supported) return "supported";
  if (score >= thresholds.weak) return "weak";
  return "unsupported";
}

/** 模型按指令拒答时的标志性短语（说明模型没有编造，应判为合规拒答） */
export const REFUSAL_RE =
  /(没有找到|未找到|找不到|无法回答|未能找到|不含相关|没有相关)/;

/** 引用角标：[1] / [1,2] / [1，2] / [1][2] */
const CITATION_RE = /\[(\d+(?:\s*[,，]\s*\d+)*)\]/g;

/** 去除模型思考标签（worker 最终文本可能残留 think 块）。标签用拼接构造，避免转写问题。 */
export function stripThinkTags(text: string): string {
  const open = "<" + "think" + ">";
  const close = "<" + "/think" + ">";
  let out = text;
  const i = out.indexOf(open);
  if (i !== -1) {
    const j = out.indexOf(close, i + open.length);
    // 有完整闭合：删掉整块；只有开标签：删掉其后全部残留
    out =
      j !== -1
        ? out.slice(0, i) + out.slice(j + close.length)
        : out.slice(0, i);
  }
  return out.split(open).join("").split(close).join("").trim();
}

/** 整段只由引用角标组成（模型把 [1] 写在了句号之后） */
const ONLY_CITATION_RE = /^(?:\s*\[\d+(?:\s*[,，]\s*\d+)*\])+\s*$/;
/** 段首引用角标（模型把 [1] 写在了下一句开头） */
const LEADING_CITATION_RE =
  /^((?:\s*\[\d+(?:\s*[,，]\s*\d+)*\])+)\s*([\s\S]*)$/;

/**
 * 句子切分（中英文标点 + 换行），保留分隔符。
 * 校验与 UI 渲染共用此函数，保证逐句结果可按顺序回填高亮。
 *
 * 角标归一：模型可能把角标写在句号之后（「…元。[1]」）或下一句开头（「[1]周期…」），
 * 直接切会切出孤立的 "[1]" 段，引用就挂不到它证明的那句话上。
 * 这里把游离角标归并到前一句，保证「引用 ↔ 句子」始终绑定。
 */
export function splitSentences(text: string): string[] {
  const parts = text.match(/[^。！？!?；;\n]+[。！？!?；;\n]?|\n+/g);
  if (!parts) return [];
  const merged: string[] = [];
  for (const part of parts) {
    if (part.length === 0) continue;
    // 整段都是角标 → 并回前一句
    if (merged.length > 0 && ONLY_CITATION_RE.test(part)) {
      merged[merged.length - 1] += part;
      continue;
    }
    const lead = merged.length > 0 ? part.match(LEADING_CITATION_RE) : null;
    if (lead) {
      // 段首角标属于前一句，摘出来归位；剩余正文另起一句
      merged[merged.length - 1] += lead[1];
      if (lead[2].trim()) merged.push(lead[2]);
      continue;
    }
    merged.push(part);
  }
  return merged;
}

/** 扫描角标编号：min=1 只取合法编号；min=0 连 [0] 这类非法编号一起取（越界检测用） */
function scanCitations(text: string, min: number): number[] {
  const out = new Set<number>();
  CITATION_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = CITATION_RE.exec(text))) {
    for (const part of m[1].split(/[,，]/)) {
      const n = Number(part.trim());
      if (Number.isInteger(n) && n >= min) out.add(n);
    }
  }
  return [...out].sort((a, b) => a - b);
}

/** 解析句子中的引用编号（去重升序） */
export function parseCitations(text: string): number[] {
  return scanCitations(text, 1);
}

/**
 * 解析句子中的**全部**角标编号，含 [0] 这类永远不合法的编号。
 *
 * 为什么必须单独有这个函数：旧实现用 `n > 0` 过滤，`[0]` 被静默丢弃当作"无引用"，
 * 随后走 fallback 范围（= 全部段落）比语义分 —— 反而最容易判成 supported。
 * 「引用了一个根本不存在的编号」是编造引用的强信号，不能因为编号难看就装作没看见。
 */
export function parseAllCitations(text: string): number[] {
  return scanCitations(text, 0);
}

/* ---------------- 词法层 ---------------- */

// 分词走公共实现：与 BM25 检索完全同口径（见 utils/tokenize.ts）。
// 重新导出同一个函数引用，单测可断言「检索与校验分词口径一致」这件事本身。
export { tokenizeSet };
const tokenize = tokenizeSet;

/**
 * 抽取句子中的数字事实（阿拉伯数字，可带小数/百分号/常见单位）。
 * export 供阈值校准（verifyCalibrationRunner）以**线上同口径**自动识别数字捏造，
 * 避免校准时漏算 fabricatedNumber 硬信号导致扫出的阈值偏松。
 */
export function extractNumbers(input: string): string[] {
  const re =
    /\d+(?:\.\d+)?\s*(?:%|％|元|块|天|小时|分钟|个|折|倍|年|月|周|页|万|亿)?/g;
  const out: string[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(input))) out.push(m[0].replace(/\s+/g, ""));
  return out;
}

/**
 * 上下文里是否出现某个数字核（带数字边界，避免子串误匹配）。
 *
 * 为什么不用 String#includes：纯子串匹配会把 "100" 命中到 "1000" 里，
 * 把句子「k 取 100」误判为上下文「k 取 1000」有依据——数字捏造假阴。
 * 这里用 (?<!\d)core(?!\d) 锁住前后非数字，"100" 不会再匹配 "1000"。
 *
 * export 供校准复用，与线上同口径，避免两处实现漂移。
 */
/** 从带单位的数字串里取出纯数字核（"15天" → "15"） */
export function numberCore(matched: string): string {
  return matched.match(/\d+(?:\.\d+)?/)?.[0] ?? matched;
}

export function contextContainsNumber(core: string, contextText: string): boolean {
  const escaped = core.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`(?<!\\d)${escaped}(?!\\d)`);
  return re.test(contextText);
}

/**
 * 找出句子中「原文里查无出处」的数字（数字捏造信号）。
 *
 * 抽成导出函数是必须的：校准脚本要复用同一逻辑，而这段有两处易错点 ——
 * 1. 必须先去掉单位再比对核（"15天" 的核是 "15"），直接拿带单位的串去匹配会全部落空
 * 2. 比对要用带数字边界的正则，否则 "100" 会命中 "1000" 造成假阴
 * 之前校准脚本就是漏了第 1 点，凭空多出 6 条"误伤"。放在这里两边共用就不会再错。
 *
 * @param contextText 原文全文
 * @param contextNumbers 原文全文抽出的数字集合（调用方预计算，避免每句重扫）
 */
export function findUnknownNumbers(
  sentence: string,
  contextText: string,
  contextNumbers?: Set<string>,
): string[] {
  const nums = contextNumbers ?? new Set(extractNumbers(contextText));
  return extractNumbers(sentence).filter(
    (n) => !nums.has(n) && !contextContainsNumber(numberCore(n), contextText),
  );
}

/**
 * 词法覆盖率：句子的有效 token 中有多少能在上下文 token 集合中找到。
 * contextTokens 由调用方预先 tokenize 一次后传入（export 供校准复用：
 * 一个文档的上下文只 tokenize 一次，多句共享，避免逐句重复分词）。
 */
export function lexicalCoverage(sentence: string, contextTokens: Set<string>): number {
  const tokens = tokenize(sentence);
  if (tokens.size === 0) return 1;
  let hit = 0;
  for (const t of tokens) if (contextTokens.has(t)) hit++;
  return hit / tokens.size;
}

/**
 * 判断句子是否为主张句（含足够实质内容）。
 * 导出：引用闭环的可观测指标（角标率/未标注数）要在评测脚本里按**同一判据**统计，
 * 否则"哪些句子需要角标"这个口径会两处漂移。
 */
export function isClaimSentence(s: string): boolean {
  const core = s.replace(
    /[\s\p{P}\p{S}。！？!?；;，,、：:""'（）()【】\-—…·]/gu,
    "",
  );
  return core.length >= MIN_CLAIM_CHARS;
}

/* ---------------- 校验范围 ---------------- */

/**
 * 回答用到的引用段落。语义比对只在「该句引用的段落（+其窗口）」内找最大余弦，
 * 而不是对全篇索引取 max —— 后者会让长文档上任何句子都能蹭到高相似度块。
 */
export interface VerifyPassage {
  /** 引用编号 [n] */
  index: number;
  /** 该段落覆盖的块 id（窗口内全部块），语义比对范围 */
  chunkIds: string[];
}

export interface VerifyOptions {
  /** 编号段落表；提供时启用范围收敛，缺省则对全文上下文校验（旧行为） */
  passages?: VerifyPassage[];
  /** 阈值覆盖（校准脚本用） */
  thresholds?: Partial<VerifyThresholds>;
  /**
   * 是否强制逐句角标（默认 true，且仅在提供 passages 时生效）。
   * 关掉只用于**对照测量**（"有角标要求 vs 没有"差多少），不作为线上默认。
   */
  requireCitation?: boolean;
}

/** 按引用编号取校验范围（块 id 集合）；无引用则用全部段落（退化为上下文校验） */
function scopeFor(citations: number[], passages: VerifyPassage[]): string[] {
  const all = passages.flatMap((p) => p.chunkIds);
  if (citations.length === 0) return all;
  const byIndex = new Map(passages.map((p) => [p.index, p]));
  const scoped = citations.flatMap((c) => byIndex.get(c)?.chunkIds ?? []);
  return scoped.length > 0 ? scoped : all;
}

/**
 * 引用编号合法性判据（**集合判定**，不是"≤ 最大值"）。
 *
 * 旧实现用 `c > maxPassageIndex` 判越界，只要编号段是 1..n 连续就等价；
 * 但它有两个洞：`[0]` 被 parseCitations 的 `n > 0` 提前吞掉（永远检测不到），
 * 以及编号段不连续时（1,3）引用中间的 [2] 会被当成合法。
 * 现在统一为：编号必须**存在**于段落表里。
 *
 * passages 为空（旧调用：没有编号可引）时沿用旧口径 —— 正编号一律视为越界，
 * 行为不变，避免改动历史路径的裁决。
 */
export function isInvalidCitation(
  n: number,
  passageIndexes: Set<number>,
  hasPassages: boolean,
): boolean {
  if (!hasPassages) return n > 0;
  return n <= 0 || !passageIndexes.has(n);
}

/* ---------------- 拒答（检索无相关命中时由调用方直接使用） ---------------- */

/** 检索门限过滤后一条都不剩时的标准拒答文本 */
export const NO_RELEVANT_ANSWER = "文档中没有找到相关内容。";

/** 构造拒答的校验结果（全部句子 trivial，不得触发依赖告警） */
export function buildRefusalResult(answer: string): VerificationResult {
  return {
    verdict: "refused",
    source: "lexical",
    checkedAt: Date.now(),
    scopedChunkCount: 0,
    convergence: {
      uncitedClaims: 0,
      outOfRangeClaims: 0,
      citationsSubsetOfPassages: true,
    },
    sentences: splitSentences(answer).map((text) => ({
      text,
      level: "supported",
      score: 1,
      semantic: null,
      lexical: 1,
      unknownNumbers: [],
      citations: [],
      invalidCitations: [],
      uncited: false,
      trivial: true,
    })),
  };
}

/**
 * 整体裁决（只统计主张句）。
 *
 * 抽成导出纯函数：服务端 parity/verify.py 必须用**同一套**聚合规则，
 * 否则「两端 verdict 相同」只能靠人肉保证；评测脚本也要直接调用它复算。
 */
export function summarizeVerdict(sentences: VerifiedSentence[]): VerifyVerdict {
  const claims = sentences.filter((s) => !s.trivial);
  if (claims.length === 0) return "supported";
  const unsupportedCount = claims.filter((s) => s.level === "unsupported").length;
  const weakCount = claims.filter((s) => s.level === "weak").length;
  if (unsupportedCount === claims.length) return "unsupported";
  if (unsupportedCount > 0 || weakCount > 0) return "partial";
  return "supported";
}

/* ---------------- 主入口 ---------------- */

/**
 * 拒答裁决（答案侧细拒，M2）。**纯函数**，两端同一套规则。
 *
 * 职责划分：检索侧只做**粗拒**（明显的无关查询，见 retrieval.applyThreshold），
 * 「文档里有没有这件事」的细判归这里 —— 依据是逐句证据（语义 + 词法 + 数字核查）。
 *
 * 三条判据：
 * 1. `refused`：模型自己按指令拒答了（短答案 + 拒答短语），合规，不算幻觉
 * 2. `unsupported`：全部主张句都无依据 —— 答不出有依据的内容
 * 3. 语义路径不可用（source="lexical"）且 verdict 非 supported：
 *    只剩词法证据时宁严勿宽（覆盖率高的句子未必真实，只是用词重合）
 *
 * 刻意**不**因为 `partial` 就拒答：部分句子弱依据（含"无角标"这种格式问题）
 * 不等于答案无价值，那属于 UI 告警层，不属于拒答。拒答越激进，正样本误拒越多。
 */
export function shouldRefuse(result: VerificationResult): boolean {
  if (result.verdict === "refused") return true;
  if (result.verdict === "unsupported") return true;
  if (result.source === "lexical" && result.verdict !== "supported") return true;
  return false;
}

/** 纯内核的输入：句子切分与角标分类的产物（含语义分来源） */
interface VerifyCoreInput {
  answer: string;
  contextText: string;
  passages: VerifyPassage[];
  requireCitation: boolean;
  thresholds: VerifyThresholds;
  /** 句子 → 语义分；null = 语义不可用（降级纯词法） */
  semanticMap: Map<string, number> | null;
  scopedChunkCount: number;
  checkedAt: number;
}

/**
 * 校验内核（**纯函数**：不碰 worker、不碰模型）。
 *
 * 抽出来的理由不是"代码整洁"，而是**评测可回放**：CI 没有模型，
 * `scripts/eval-verdict.ts --check` 要拿固化好的语义分复算一遍拒答裁决，
 * 若它与线上走两份实现，"CI 说没问题"就证明不了任何事。
 */
export function verifyCore(input: VerifyCoreInput): VerificationResult {
  const { answer, contextText, passages, thresholds, checkedAt } = input;
  const hasPassages = passages.length > 0;
  const passageIndexes = new Set(passages.map((p) => p.index));
  const segments = splitSentences(answer);

  // 模型按指令拒答：答案短且含拒答短语，判为合规，不算幻觉
  const compact = answer.replace(/\s+/g, "");
  if (compact.length <= 80 && REFUSAL_RE.test(answer)) {
    return {
      verdict: "refused",
      source: "lexical",
      checkedAt,
      scopedChunkCount: 0,
      convergence: {
        uncitedClaims: 0,
        outOfRangeClaims: 0,
        citationsSubsetOfPassages: true,
      },
      sentences: segments.map((text) => ({
        text,
        level: "supported",
        score: 1,
        semantic: null,
        lexical: 1,
        unknownNumbers: [],
        citations: [],
        invalidCitations: [],
        uncited: false,
        trivial: true,
      })),
    };
  }

  // 只把主张句送进语义比对，保持顺序对齐
  const claimFlags = segments.map(isClaimSentence);
  // citations 保持"正编号"语义（UI/契约不变）；越界检测另用含 [0] 的全量扫描
  const citedPerSentence = segments.map(parseCitations);
  const invalidPerSentence = segments.map((_, idx) =>
    parseAllCitations(segments[idx]).filter((c) =>
      isInvalidCitation(c, passageIndexes, hasPassages),
    ),
  );
  /** 有编号段落表时，主张句必须带**有效**角标（引用强制闭环） */
  const requireCitation = hasPassages && input.requireCitation;
  const uncitedPerSentence = segments.map(
    (_, idx) =>
      requireCitation &&
      claimFlags[idx] &&
      citedPerSentence[idx].every((c) => !passageIndexes.has(c)),
  );

  const contextTokens = tokenize(contextText);
  const contextNumbers = new Set(extractNumbers(contextText));
  const semanticMap = input.semanticMap;
  const source: "semantic" | "lexical" = semanticMap ? "semantic" : "lexical";

  const sentences: VerifiedSentence[] = segments.map((text, idx) => {
    const lexical = lexicalCoverage(text, contextTokens);
    const unknownNumbers = findUnknownNumbers(text, contextText, contextNumbers);

    // 寒暄/语气句不参与裁决
    if (!claimFlags[idx]) {
      return {
        text,
        level: "supported",
        score: 1,
        semantic: null,
        lexical,
        unknownNumbers: [],
        citations: citedPerSentence[idx],
        invalidCitations: invalidPerSentence[idx],
        uncited: false,
        trivial: true,
      };
    }

    const semantic =
      source === "semantic" && semanticMap?.has(text)
        ? semanticMap.get(text)!
        : null;
    const score = computeSentenceScore(semantic, lexical, thresholds, {
      fabricatedNumber: unknownNumbers.length > 0,
      danglingCitation: invalidPerSentence[idx].length > 0,
      uncitedClaim: uncitedPerSentence[idx],
    });
    const level = levelOf(score, thresholds);

    return {
      text,
      level,
      score,
      semantic,
      lexical,
      unknownNumbers,
      citations: citedPerSentence[idx],
      invalidCitations: invalidPerSentence[idx],
      uncited: uncitedPerSentence[idx],
      trivial: false,
    };
  });

  const claims = sentences.filter((s) => !s.trivial);
  const outOfRangeClaims = claims.filter((s) => s.invalidCitations.length > 0).length;
  const convergence: CitationConvergence = {
    uncitedClaims: claims.filter((s) => s.uncited).length,
    outOfRangeClaims,
    // 收敛判据：不存在任何越界角标 —— 即"答案引用的每一个编号都能在编号段落表里找到"
    citationsSubsetOfPassages: outOfRangeClaims === 0,
  };

  return {
    verdict: summarizeVerdict(sentences),
    sentences,
    checkedAt,
    source,
    scopedChunkCount: input.scopedChunkCount,
    convergence,
  };
}

/**
 * 用**已算好的语义分**跑一遍校验（评测脚本在无模型环境下的回放入口）。
 * 语义分缺失（传 null）等价于 worker 不可用 → 走纯词法降级。
 */
export function verifyAnswerWithScores(
  answer: string,
  contextText: string,
  passages: VerifyPassage[],
  semanticMap: Map<string, number> | null,
  options: VerifyOptions = {},
): VerificationResult {
  return verifyCore({
    answer,
    contextText,
    passages,
    requireCitation: options.requireCitation ?? true,
    thresholds: { ...DEFAULT_VERIFY_THRESHOLDS, ...options.thresholds },
    semanticMap,
    scopedChunkCount: 0,
    checkedAt: Date.now(),
  });
}

/**
 * 对模型最终答案做事后校验
 * @param indexId 会话 id（worker 内向量索引键）
 * @param answer 模型最终答案（建议先 stripThinkTags）
 * @param contextText 喂给模型的上下文文本（词法覆盖率的分母）
 * @param options passages=编号段落表（语义范围收敛）；thresholds=阈值覆盖
 */
export async function verifyAnswer(
  indexId: string,
  answer: string,
  contextText: string,
  options: VerifyOptions = {},
): Promise<VerificationResult> {
  const checkedAt = Date.now();
  const thresholds: VerifyThresholds = {
    ...DEFAULT_VERIFY_THRESHOLDS,
    ...options.thresholds,
  };
  const passages = options.passages ?? [];
  const hasPassages = passages.length > 0;
  const passageIndexes = new Set(passages.map((p) => p.index));
  const segments = splitSentences(answer);

  // 提前短路的拒答分支不必进分组/worker，直接交给内核（内核里同一判据）
  const compact = answer.replace(/\s+/g, "");
  if (compact.length <= 80 && REFUSAL_RE.test(answer)) {
    return verifyCore({
      answer,
      contextText,
      passages,
      requireCitation: options.requireCitation ?? true,
      thresholds,
      semanticMap: null,
      scopedChunkCount: 0,
      checkedAt,
    });
  }

  const claimFlags = segments.map(isClaimSentence);
  const citedPerSentence = segments.map(parseCitations);

  // 按「引用范围」分组：同一范围的句子合并成一次 worker 调用（通常只有 1~2 组）
  const scopeKey = (citations: number[]) =>
    !hasPassages
      ? "all"
      : citations.length === 0
        ? "fallback"
        : [...citations].sort((a, b) => a - b).join(",");
  const groups = new Map<string, { texts: string[]; scope: string[] }>();
  segments.forEach((text, idx) => {
    if (!claimFlags[idx]) return;
    // 越界引用不参与范围计算（视为无引用，走 fallback 范围）
    const valid = citedPerSentence[idx].filter((c) => passageIndexes.has(c));
    const key = scopeKey(valid);
    const scope = scopeFor(valid, passages);
    const group = groups.get(key);
    if (group) group.texts.push(text);
    else groups.set(key, { texts: [text], scope });
  });

  const semanticMap = new Map<string, number>();
  let source: "semantic" | "lexical" = "semantic";
  /** 本次实际参与比对的块数（取最大的一组，用于可观测性） */
  let scopedChunkCount = 0;
  try {
    /*
     * 惰性 import embeddingClient：它是唯一依赖 Worker 的模块
     *（`import ... ?worker` 是 Vite 专有语法，Node 里无法解析）。
     * 本文件其余部分是**纯函数**，评测/校准脚本要在 Node 里直接 import 它们
     * （scripts/calibrate-verify.ts 量 P/R 时就走这条路）——
     * 静态 import 会让整个模块在 Node 下一加载就炸。
     * 动态 import 只在真正要调 worker 时才解析，且不影响打包（App 已静态引用它）。
     */
    const { verifySentences } = await import("./embeddingClient");
    for (const group of groups.values()) {
      scopedChunkCount = Math.max(scopedChunkCount, group.scope.length);
      const scores = await verifySentences(
        indexId,
        group.texts,
        passages.length > 0 ? group.scope : undefined,
      );
      // embedBatch 保序，按索引对齐
      scores.forEach((s, i) => semanticMap.set(group.texts[i], s.score));
    }
  } catch (err) {
    // worker/模型不可用：降级为纯词法校验，不阻塞回答展示
    console.warn("语义校验不可用，降级为词法校验", err);
    source = "lexical";
  }

  // 句级裁决/收敛/整体裁决统一走纯内核（与评测回放同一份实现，杜绝两处口径漂移）
  return verifyCore({
    answer,
    contextText,
    passages,
    requireCitation: options.requireCitation ?? true,
    thresholds,
    semanticMap: source === "semantic" ? semanticMap : null,
    scopedChunkCount,
    checkedAt,
  });
}
