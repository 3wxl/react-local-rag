/**
 * BM25 关键词检索（Okapi BM25），纯 JS 实现，无第三方依赖。
 *
 * 分词策略（适配中文 + 英文混合文档）：
 * - 连续中文段 → bigram（二元组），避免引入分词词典的体积
 * - 英文/数字串 → 小写单词
 * - 单字保留（给短句兜底）
 *
 * 与 verifyAnswer.ts 的 tokenize 保持同一套分词口径，
 * 使「检索到的片段」与「词法校验」信号一致。
 */

/** BM25 调参：k1 控制词频饱和度，b 控制文档长度归一化强度（行业经验默认值） */
const BM25_K1 = 1.5;
const BM25_B = 0.75;

/**
 * 分词：连续中文 -> 二元组；英文/数字串 -> 小写单词；单字也保留。
 */
export function tokenize(input: string): string[] {
  const tokens: string[] = [];
  // 英文单词、数字（含小数/百分号）
  const wordRe = /[a-z0-9]+(?:\.\d+)?%?/gi;
  let m: RegExpExecArray | null;
  while ((m = wordRe.exec(input))) tokens.push(m[0].toLowerCase());

  // 中文连续段 -> bigram
  const cjkRe = /[一-鿿]+/g;
  while ((m = cjkRe.exec(input))) {
    const seg = m[0];
    for (let i = 0; i < seg.length - 1; i++) {
      tokens.push(seg.slice(i, i + 2));
    }
    if (seg.length === 1) tokens.push(seg);
  }
  return tokens;
}

/** 词项在单篇文档中的词频 */
type TermFreq = Map<string, number>;

export class BM25 {
  /** 文档总数 N */
  private readonly n: number;
  /** 每篇文档的词频表 */
  private readonly docFreqs: TermFreq[];
  /** 每篇文档长度（token 数） */
  private readonly docLen: number[];
  /** 平均文档长度 avgdl */
  private readonly avgDocLen: number;
  /** 词项 -> 包含该词项的文档数 n(qi) */
  private readonly df: Map<string, number>;

  constructor(documents: string[]) {
    this.n = documents.length; //文档总数
    this.docFreqs = []; //第 i 篇文档的 Map<词, 词频>
    this.docLen = new Array(this.n).fill(0); //第 i 篇文档的 token 数
    this.df = new Map();

    let totalLen = 0;
    for (let i = 0; i < this.n; i++) {
      const tokens = tokenize(documents[i]); //`tokenize(documents[i])`：文档分词，得到 token 数组（英文单词、中文 bigram）
      const tf: TermFreq = new Map();
      for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1); //内层循环统计**本篇文档内词频 TF**：这个词在这篇文档出现多少次
      this.docFreqs.push(tf); //把 TF Map 放进`docFreqs`
      this.docLen[i] = tokens.length; //记录文档 token 长度累加到总长度`totalLen`
      totalLen += tokens.length;
      // 统计文档频率（每篇只计一次）
      for (const term of tf.keys()) {
        //`tf.keys()`：遍历本篇**出现过的所有唯一词**。
        this.df.set(term, (this.df.get(term) ?? 0) + 1); //✅重点：`tf.keys()` 而不是 tokens，**同一篇文档重复出现的词，df 只 + 1**。df 的定义：包含这个词的**文档数量**，不是词出现总次数。
      }
    }
    this.avgDocLen = this.n > 0 ? totalLen / this.n : 0; //	平均文档长度
  }

  /** 词项的 IDF（BM25+ 平滑，保证非负） */
  private idf(term: string): number {
    //IDF：逆文档频率，衡量这个词的**重要程度**。
    const df = this.df.get(term) ?? 0;
    return Math.log(1 + (this.n - df + 0.5) / (df + 0.5));
  }
  /*
IDF：逆文档频率，衡量这个词的**重要程度**。
公式是 BM25+ 的平滑 IDF，加 0.5 做拉普拉斯平滑：

- 很少文档出现的词（df 很小），IDF 很大，权重高（专有名词，关键词）
- 几乎所有文档都有的词（df 很大），IDF 趋近 0，权重很低（停用词）
- 词不在索引里，df=0，公式依然不会出现负数、除零报错。
*/
  /** 计算 query 对单篇文档的 BM25 分数 */
  private scoreDoc(queryTerms: string[], docIdx: number): number {
    const tf = this.docFreqs[docIdx];
    const lenNorm =
      this.avgDocLen > 0
        ? 1 - BM25_B + BM25_B * (this.docLen[docIdx] / this.avgDocLen)
        : 1;
    let score = 0;
    for (const term of queryTerms) {
      const f = tf.get(term);
      if (!f) continue;
      const idf = this.idf(term);
      score += (idf * (f * (BM25_K1 + 1))) / (f + BM25_K1 * lenNorm);
    }
    return score;
  }

  /**
   * 检索：返回每篇文档的 BM25 原始分（顺序与 documents 一致）。
   * 零分文档也返回（由上层决定是否过滤/参与归一化融合）。
   */
  scoreAll(query: string): number[] {
    //对外暴露的检索入口函数
    if (this.n === 0) return [];
    const queryTerms = tokenize(query); //输入用户查询字符串
    if (queryTerms.length === 0) return new Array(this.n).fill(0);
    // query 内重复词项去重，避免重复计分
    const unique = Array.from(new Set(queryTerms));
    const scores = new Array<number>(this.n);
    for (let i = 0; i < this.n; i++) scores[i] = this.scoreDoc(unique, i);
    return scores;
  }
}
