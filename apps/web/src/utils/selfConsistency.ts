/**
 * 自一致性度量（纯函数，不调模型）。
 *
 * 定位：弱模型（0.5B）在同题多次采样下的**稳定性**是它的可信度下界 ——
 * 同一问题问三遍给出三套说法（数字、角标、结论都不同），
 * 说明模型在"编"而不是在"读"，这时不该把答案直接交给用户。
 *
 * 四个维度分别抓一类漂移：
 * - verdict 一致：结论级漂移（答 / 拒答 / 弱依据之间翻转）—— 最严重，出现即判不稳定
 * - bigram Jaccard：措辞级漂移（同义改写属正常，句式大改则说明生成不受证据约束）
 * - 数字集合 Jaccard：**事实级漂移**（数字变了就是编造，不是措辞问题）
 * - 角标集合 Jaccard：证据级漂移（同一答案引用了不同原文）
 *
 * 度量与阈值分离：这里只算数，门槛在 scripts/eval-consistency.ts（CI 口径）与
 * answerConfidence.ts（运行时回退）里，两处共用同一份计算。
 */

export interface SampleView {
  /** 该次采样的整体裁决（supported / partial / unsupported / refused） */
  verdict: string;
  /** 答案正文（已 stripThinkTags） */
  answer: string;
  /** 逐句角标集合（去重） */
  citations: number[];
}

/** 字符 bigram 集合（去空白，与检索侧近重复判定同一口径） */
function bigrams(text: string): Set<string> {
  const clean = text.replace(/\s+/g, "");
  const out = new Set<string>();
  for (let i = 0; i + 2 <= clean.length; i++) out.add(clean.slice(i, i + 2));
  return out;
}

export function jaccard(a: Set<unknown>, b: Set<unknown>): number {
  if (a.size === 0 && b.size === 0) return 1;
  let inter = 0;
  for (const v of a) if (b.has(v)) inter++;
  const union = a.size + b.size - inter;
  return union === 0 ? 1 : inter / union;
}

/** 从答案文本里抽数字事实（与校验层同一口径：带单位的数字串） */
export function numbersOf(answer: string): Set<string> {
  const out = new Set<string>();
  const re = /\d+(?:\.\d+)?\s*(?:%|％|元|块|天|小时|分钟|个|折|倍|年|月|周|页|万|亿)?/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(answer))) out.add(m[0].replace(/\s+/g, ""));
  return out;
}

export interface ConsistencyReport {
  /** 采样数 */
  samples: number;
  /** 所有采样裁决一致（含全为 refused） */
  verdictAgreement: number;
  /** 两两 bigram Jaccard 的均值 */
  answerJaccard: number;
  /** 两两数字集合 Jaccard 的均值（都无数字时为 1） */
  numberJaccard: number;
  /** 两两角标集合 Jaccard 的均值 */
  citationJaccard: number;
  /** 是否出现「拒答 ↔ 作答」翻转 —— 最该被拦住的一种不稳定 */
  refusalFlip: boolean;
}

/** 两两组合（i < j），采样数很小（k=3），O(k²) 足够 */
function pairs<T>(items: T[]): [T, T][] {
  const out: [T, T][] = [];
  for (let i = 0; i < items.length; i++) {
    for (let j = i + 1; j < items.length; j++) out.push([items[i], items[j]]);
  }
  return out;
}

export function measureConsistency(views: SampleView[]): ConsistencyReport {
  if (views.length <= 1) {
    return {
      samples: views.length,
      verdictAgreement: 1,
      answerJaccard: 1,
      numberJaccard: 1,
      citationJaccard: 1,
      refusalFlip: false,
    };
  }
  const combos = pairs(views);
  const mean = (values: number[]) => values.reduce((s, v) => s + v, 0) / values.length;

  const verdicts = new Set(views.map((v) => v.verdict));
  const refused = new Set(views.map((v) => v.verdict === "refused"));
  return {
    samples: views.length,
    verdictAgreement: verdicts.size === 1 ? 1 : 0,
    answerJaccard: mean(combos.map(([a, b]) => jaccard(bigrams(a.answer), bigrams(b.answer)))),
    numberJaccard: mean(
      combos.map(([a, b]) => jaccard(numbersOf(a.answer), numbersOf(b.answer))),
    ),
    citationJaccard: mean(
      combos.map(([a, b]) => jaccard(new Set(a.citations), new Set(b.citations))),
    ),
    refusalFlip: refused.size > 1,
  };
}
