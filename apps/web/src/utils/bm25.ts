/**
 * BM25 关键词检索（Okapi BM25），纯 JS 实现，无第三方依赖。
 *
 * 分词走 utils/tokenize.ts 的公共实现：中文 bigram + 英文小写单词 + 单字兜底。
 * 与 verifyAnswer 的词法校验共用同一口径（原先两处各写一份，靠注释人肉同步）。
 */

import { tokenize } from "./tokenize";

export { tokenize };

/** BM25 调参：k1 控制词频饱和度，b 控制文档长度归一化强度（行业经验默认值） */
const BM25_K1 = 1.5;
const BM25_B = 0.75;

/** 词项在单篇文档中的词频 */
type TermFreq = Map<string, number>;

export class BM25 {
  /** 文档总数 N（随 add 增长） */
  private n = 0;
  /** 每篇文档的词频表 */
  private readonly docFreqs: TermFreq[] = [];
  /** 每篇文档长度（token 数） */
  private readonly docLen: number[] = [];
  /** 全部文档 token 总数（avgdl 的分子，随 add 累加） */
  private totalLen = 0;
  /** 词项 -> 包含该词项的文档数 n(qi) */
  private readonly df = new Map<string, number>();

  constructor(documents: string[] = []) {
    this.add(documents);
  }

  /** 当前文档数 */
  get size(): number {
    return this.n;
  }

  /**
   * 增量追加文档：只统计新增篇目的 tf/df/len，不重建已有索引。
   * N / avgdl / df 都是查询时实时读取，因此追加后 scoreAll 立即用新统计量。
   * 复杂度 O(新增 token 数)，替代「每批全量重建」的 O(全量 token 数)。
   */
  add(documents: string[]): void {
    for (const doc of documents) {
      const tokens = tokenize(doc);
      const tf: TermFreq = new Map();
      for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
      this.docFreqs.push(tf);
      this.docLen.push(tokens.length);
      this.totalLen += tokens.length;
      // df 只计「包含该词的文档数」，同篇重复词只 +1
      for (const term of tf.keys()) {
        this.df.set(term, (this.df.get(term) ?? 0) + 1);
      }
      this.n += 1;
    }
  }

  /** 清空全部索引（覆盖语义的分批上传首批调用） */
  reset(): void {
    this.n = 0;
    this.docFreqs.length = 0;
    this.docLen.length = 0;
    this.totalLen = 0;
    this.df.clear();
  }

  /** 平均文档长度 avgdl */
  private get avgDocLen(): number {
    return this.n > 0 ? this.totalLen / this.n : 0;
  }

  /** 词项的 IDF（BM25+ 平滑，保证非负） */
  private idf(term: string): number {
    //IDF：逆文档频率，衡量这个词的**重要程度**。
    const df = this.df.get(term) ?? 0;
    return Math.log(1 + (this.n - df + 0.5) / (df + 0.5));
  }

  /** 词项是否出现在语料中（df > 0） */
  hasTerm(term: string): boolean {
    return (this.df.get(term) ?? 0) > 0;
  }

  /**
   * IDF 加权查询覆盖率：查询词的**信息量**里有多大比例能在语料词表中找到。
   *
   * 拒答的第一路信号，而且不需要向量模型 —— 纯词表统计。
   * 语料里根本不存在的词（「股票代码」「银行账号」「碳排放」）df = 0，
   * IDF 取最大值，全额进分母、完全不进分子 → 覆盖率被拉低。
   * 于是「问一个库里没有的主题」在词法层就能看出来，不必等向量余弦。
   *
   * @returns 0~1；查询无有效词项时返回 1（不参与拒答，避免空查询被误拒）
   */
  idfWeightedCoverage(query: string): number {
    const terms = new Set(tokenize(query));
    if (terms.size === 0) return 1;
    let total = 0;
    let covered = 0;
    for (const term of terms) {
      const idf = this.idf(term);
      total += idf;
      if (this.hasTerm(term)) covered += idf;
    }
    return total === 0 ? 1 : covered / total;
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
