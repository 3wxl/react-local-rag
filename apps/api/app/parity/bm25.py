r"""BM25 关键词检索（apps/web/src/utils/bm25.ts 的移植）。

口径要点（改一处排名就变，双端就不可比）：
- 分词走 `tokenize`（中文 bigram + 英文小写单词），与词法校验共用
- k1 = 1.5（词频饱和），b = 0.75（长度归一化）
- IDF 用 BM25+ 平滑：`log(1 + (n - df + 0.5) / (df + 0.5))`，保证非负
- `score_all` 的查询词项去重但**保留首次出现顺序**（浮点累加顺序影响末位精度）
"""

from __future__ import annotations

import math

from .tokenize import tokenize

#: BM25 调参（行业经验默认值，TS 侧同名常量）
BM25_K1 = 1.5
BM25_B = 0.75


class BM25:
    """Okapi BM25，增量可加（add 后 N / avgdl / df 在打分时实时读取）。"""

    def __init__(self, documents: list[str] | None = None) -> None:
        self._n = 0
        self._doc_freqs: list[dict[str, int]] = []
        self._doc_len: list[int] = []
        self._total_len = 0
        self._df: dict[str, int] = {}
        if documents:
            self.add(documents)

    @property
    def size(self) -> int:
        return self._n

    def add(self, documents: list[str]) -> None:
        """增量追加文档（复杂度 O(新增 token 数)，不重建已有统计量）。"""
        for doc in documents:
            tokens = tokenize(doc)
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._doc_freqs.append(tf)
            self._doc_len.append(len(tokens))
            self._total_len += len(tokens)
            for term in tf:
                self._df[term] = self._df.get(term, 0) + 1
            self._n += 1

    def reset(self) -> None:
        self._n = 0
        self._doc_freqs.clear()
        self._doc_len.clear()
        self._total_len = 0
        self._df.clear()

    @property
    def _avg_doc_len(self) -> float:
        return self._total_len / self._n if self._n > 0 else 0.0

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def has_term(self, term: str) -> bool:
        return self._df.get(term, 0) > 0

    def idf_weighted_coverage(self, query: str) -> float:
        """IDF 加权查询覆盖率（拒答信号 1，纯词表统计、不需要向量）。

        语料里根本不存在的词 df = 0 → IDF 取最大 → 全额进分母、不进分子 → 覆盖率被拉低。
        查询无有效词项时返回 1（不参与拒答，避免空查询被误拒）。
        """
        terms = set(tokenize(query))
        if not terms:
            return 1.0
        total = 0.0
        covered = 0.0
        for term in terms:
            idf = self._idf(term)
            total += idf
            if self.has_term(term):
                covered += idf
        return 1.0 if total == 0 else covered / total

    def _score_doc(self, query_terms: list[str], doc_idx: int) -> float:
        tf = self._doc_freqs[doc_idx]
        avg = self._avg_doc_len
        len_norm = (
            1 - BM25_B + BM25_B * (self._doc_len[doc_idx] / avg) if avg > 0 else 1.0
        )
        score = 0.0
        for term in query_terms:
            f = tf.get(term)
            if not f:
                continue
            score += (self._idf(term) * (f * (BM25_K1 + 1))) / (
                f + BM25_K1 * len_norm
            )
        return score

    def score_all(self, query: str) -> list[float]:
        """返回每篇文档的 BM25 原始分（顺序与 add 时一致；零分文档也返回）。"""
        if self._n == 0:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return [0.0] * self._n
        # 查询内重复词项去重（保留首次出现顺序，浮点累加顺序与 TS 一致）
        unique = list(dict.fromkeys(query_terms))
        return [self._score_doc(unique, i) for i in range(self._n)]
