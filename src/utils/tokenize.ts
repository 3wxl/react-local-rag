/**
 * 统一分词口径（BM25 关键词检索 与 词法幻觉校验 共用同一实现）。
 *
 * 之前 bm25.ts 与 verifyAnswer.ts 各写一份分词，靠注释约定「保持同一套口径」，
 * 一旦有人只改一处，检索命中与词法校验的信号就会悄悄分叉。抽到这里后，
 * 两处天然一致，不可能再漂移。
 *
 * 策略（适配中文 + 英文混合文档）：
 * - 连续中文段 → bigram（二元组），避免引入分词词典的体积
 * - 単字中文段 → 保留该字（给短句兜底）
 * - 英文/数字串 → 小写单词（含小数、百分号）
 */

/** 英文单词、数字（含小数/百分号） */
const WORD_RE = /[a-z0-9]+(?:\.\d+)?%?/gi;
/** 中文连续段（一-鿿 = CJK 统一表意文字基本区） */
const CJK_RE = /[一-鿿]+/g;

/**
 * 分词（保留重复，供 BM25 统计词频）。
 * BM25 的 TF 依赖重复次数，因此这里不能去重。
 */
export function tokenize(input: string): string[] {
  const tokens: string[] = [];
  let m: RegExpExecArray | null;

  WORD_RE.lastIndex = 0;
  while ((m = WORD_RE.exec(input))) tokens.push(m[0].toLowerCase());

  CJK_RE.lastIndex = 0;
  while ((m = CJK_RE.exec(input))) {
    const seg = m[0];
    for (let i = 0; i < seg.length - 1; i++) {
      tokens.push(seg.slice(i, i + 2));
    }
    if (seg.length === 1) tokens.push(seg);
  }
  return tokens;
}

/** 分词（去重，供覆盖率计算：句子 token 有多少能在上下文里找到） */
export function tokenizeSet(input: string): Set<string> {
  return new Set(tokenize(input));
}
