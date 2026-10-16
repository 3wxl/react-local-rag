"""检索指标（apps/web/scripts/eval.ts 的逐条移植，字段名与数值口径一致）。

同名同算：指标名与 `packages/eval/baseline/ts-metrics.json` 完全对齐，
Python Harness 与 `npm run eval` 的结果可以直接并排比较 —— 这就是"双端可对比"的落点。

几个容易抄错的细节（都在 TS 里有注释说明，这里保持一致）：
- recall 的分母是**该条查询的 relevant 条数**（多要点查询天然分散在多个窗口）
- MRR@10 用同一条 hybrid 管线的**宽候选**（topK=10）跑，避免 topK=5 截断把第 6~10 名算成 0
- 要点命中用**原始子串**匹配（不像答案级评测那样去空白）
- 冗余有两套口径：结构性（ord 相邻/重叠）与内容性（bigram Jaccard ≥ 0.8）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..parity.bm25 import BM25
from ..parity.retrieval import (
    DEFAULT_RETRIEVAL_OPTIONS,
    QueryVariantScores,
    RetrievalDoc,
    RetrievalOptions,
    post_process_retrieval,
)

#: 近似重复判定阈（bigram Jaccard）
NEAR_DUP_JACCARD = 0.8


@dataclass
class Metrics:
    """与 ts-metrics.json 的 metrics 字段同名同义。"""

    query_count: int = 0
    chunk_count: int = 0
    hybrid: bool = False
    bm25_only_recall_at_3: float | None = None
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr_at_10: float = 0.0
    key_point_hit_rate: float = 0.0
    top3_duplicate_rate: float = 0.0
    top3_near_duplicate_rate: float = 0.0
    no_hit_query_count: int = 0
    false_refusal_count: int = 0
    #: 受保护人群（q+m，不含口语化 s）的误拒条数 —— 门槛只卡这一批（2026-10-15 起）
    protected_false_refusal_count: int = 0
    #: 受保护人群（q+m）的 recall@3（相对门槛的分子）
    protected_recall_at_3: float = 0.0
    #: 受保护人群的 BM25-only recall@3（相对门槛的分母）；无向量路时为 None
    protected_bm25_only_recall_at_3: float | None = None
    negative_count: int = 0
    negative_refused_count: int = 0
    negative_refusal_rate: float = 0.0
    by_group: dict[str, dict] = field(default_factory=dict)
    per_query: list[dict] = field(default_factory=list)
    negative_detail: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        """API/存档用的 camelCase（与 TS 的 metrics 对象逐字段同名）。"""
        return {
            "queryCount": self.query_count,
            "chunkCount": self.chunk_count,
            "hybrid": self.hybrid,
            "bm25OnlyRecallAt3": self.bm25_only_recall_at_3,
            "recallAt3": self.recall_at_3,
            "recallAt5": self.recall_at_5,
            "mrrAt10": self.mrr_at_10,
            "keyPointHitRate": self.key_point_hit_rate,
            "top3DuplicateRate": self.top3_duplicate_rate,
            "top3NearDuplicateRate": self.top3_near_duplicate_rate,
            "noHitQueryCount": self.no_hit_query_count,
            "falseRefusalCount": self.false_refusal_count,
            "protectedFalseRefusalCount": self.protected_false_refusal_count,
            "protectedRecallAt3": self.protected_recall_at_3,
            "protectedBm25OnlyRecallAt3": self.protected_bm25_only_recall_at_3,
            "negativeCount": self.negative_count,
            "negativeRefusedCount": self.negative_refused_count,
            "negativeRefusalRate": self.negative_refusal_rate,
            "byGroup": {
                g: {
                    "count": v["count"],
                    "recallAt3": v["recallAt3"],
                    "recallAt5": v["recallAt5"],
                    "noHit": v["noHit"],
                }
                for g, v in self.by_group.items()
            },
        }


def bigrams(text: str) -> set[str]:
    """字符 bigram 集合（去掉空白，避免排版差异干扰）。"""
    clean = "".join(text.split())
    return {clean[i : i + 2] for i in range(len(clean) - 1)}


def bigram_jaccard(a: str, b: str) -> float:
    sa, sb = bigrams(a), bigrams(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa) + len(sb) - inter
    return 0.0 if union == 0 else inter / union


def _fallback_bm25_only(
    docs: list[RetrievalDoc],
    bm25: BM25,
    question: str,
    options: RetrievalOptions,
):
    """同一条管线强制关掉向量路（相对门槛的分母，语料无关）。"""
    scores = bm25.score_all(question)
    return post_process_retrieval(
        docs, scores, None, options, idf_coverage=bm25.idf_weighted_coverage(question)
    )


#: 查询扩展变体的 RRF 权重（与 TS 的 EXPANSION_WEIGHT 默认值一致）
EXPANSION_WEIGHT = 0.7


def evaluate(
    docs: list[RetrievalDoc],
    queries,
    *,
    top_k: int = 5,
    options: RetrievalOptions = DEFAULT_RETRIEVAL_OPTIONS,
    query_vector_of: Callable[[object], np.ndarray | None] | None = None,
    expansions: dict[str, list[str]] | None = None,
    expansion_vectors: dict[str, list[np.ndarray]] | None = None,
) -> Metrics:
    """跑一遍检索评测（与 TS evaluate 同序）。

    `query_vector_of`：查询 id → 查询向量。固化向量（CI/可复现）与本地 ONNX（真实模型）
    两种来源都从调用方注入，指标算法本身不关心向量从哪来。
    `expansions` / `expansion_vectors`：固化的查询改写与它们的向量（缺省 = 不开扩展，
    与 TS 侧"没有 expansions.json 时按单查询跑"一致）。
    """
    bm25 = BM25([d.content for d in docs])
    hybrid = any(d.vector is not None for d in docs)

    def vec_scores_for(query) -> list[float] | None:
        if not hybrid:
            return None
        qvec = query_vector_of(query) if query_vector_of is not None else None
        if qvec is None:
            return None
        return [
            float(np.dot(d.vector, qvec)) if d.vector is not None else 0.0 for d in docs
        ]

    def variants_for(query) -> list[QueryVariantScores]:
        """查询扩展变体：与 TS eval.ts 的 expansionRunsFor 同口径（同一份固化产物）。"""
        texts = (expansions or {}).get(query.id) or []
        vectors = (expansion_vectors or {}).get(query.id) or []
        if not texts or not vectors:
            return []
        out: list[QueryVariantScores] = []
        for i, text in enumerate(texts):
            if i >= len(vectors):
                break
            qvec = vectors[i]
            vec_scores = (
                [
                    float(np.dot(d.vector, qvec)) if d.vector is not None else 0.0
                    for d in docs
                ]
                if hybrid
                else None
            )
            out.append(
                QueryVariantScores(
                    bm25_scores=bm25.score_all(text),
                    vec_scores=vec_scores,
                    idf_coverage=bm25.idf_weighted_coverage(text),
                    weight=EXPANSION_WEIGHT,
                )
            )
        return out
    positives = [q for q in queries if not q.negative]
    negatives = [q for q in queries if q.negative]

    metrics = Metrics(
        query_count=len(queries), chunk_count=len(docs), hybrid=hybrid
    )
    recall3 = recall5 = mrr = bm25_only_r3 = 0.0
    point_hits = point_total = 0
    no_hit = 0
    dup_queries = near_dup_queries = 0
    # 受保护人群（q+m）累计：误拒门槛与相对门槛只在这批上判定（s 已按调参口径降为观察项）
    protected_count = 0
    protected_r3 = 0.0
    protected_bm25_only_r3 = 0.0
    protected_no_hit = 0
    by_group: dict[str, dict] = {}

    for q in positives:
        bm25_scores = bm25.score_all(q.question)
        vec_scores = vec_scores_for(q)
        idf = bm25.idf_weighted_coverage(q.question)
        variants = variants_for(q)
        result = post_process_retrieval(
            docs, bm25_scores, vec_scores, options, idf_coverage=idf, variants=variants
        )
        # MRR 与正式指标必须走同一条管线，只把候选放宽到 top10
        wide = post_process_retrieval(
            docs,
            bm25_scores,
            vec_scores,
            RetrievalOptions(**{**options.__dict__, "top_k": 10}),
            idf_coverage=idf,
            variants=variants,
        )

        if result.no_relevant_hit:
            no_hit += 1

        covered = _coverage(result.hits, q.relevant, 3)
        recall3 += covered
        recall5 += _coverage(result.hits, q.relevant, 5)
        # 分组：q=技术措辞，s=口语化改写（观察项），m=单文档多主题
        group = q.id[0] if q.id[0] in ("s", "m") else "q"
        #: 受保护人群：q+m（不含口语化 s）
        is_protected = group != "s"
        # 相对门槛（hybrid ≥ BM25-only × 0.95）同样只算受保护人群：
        # s 已被检索侧刻意粗拒，让它当分母等于用「已知要放弃的人群」去卡向量质量，必然误报
        if hybrid:
            only = _fallback_bm25_only(docs, bm25, q.question, options)
            base_r3 = _coverage(only.hits, q.relevant, 3)
            bm25_only_r3 += base_r3
            if is_protected:
                protected_bm25_only_r3 += base_r3

        if is_protected:
            protected_count += 1
            protected_r3 += covered
            if result.no_relevant_hit:
                protected_no_hit += 1
        bucket = by_group.setdefault(group, {"count": 0, "r3": 0.0, "r5": 0.0, "noHit": 0})
        bucket["count"] += 1
        bucket["r3"] += covered
        bucket["r5"] += _coverage(result.hits, q.relevant, 5)
        if result.no_relevant_hit:
            bucket["noHit"] += 1

        first_rank = next(
            (i for i, h in enumerate(wide.hits) if _is_relevant(h.content, q.relevant)),
            -1,
        )
        mrr += 1 / (first_rank + 1) if first_rank >= 0 else 0.0

        top_text = "".join(h.content for h in result.hits[:top_k])
        point_hits += sum(1 for p in q.key_points if p in top_text)
        point_total += len(q.key_points)

        top3 = result.hits[:3]
        adjacent = near_dup = False
        for i in range(len(top3)):
            for j in range(i + 1, len(top3)):
                jac = bigram_jaccard(top3[i].content, top3[j].content)
                if jac >= NEAR_DUP_JACCARD:
                    near_dup = True
                structurally_adjacent = (
                    top3[i].doc_id == top3[j].doc_id
                    and top3[i].ord_end + 1 >= top3[j].ord_start
                    and top3[j].ord_end + 1 >= top3[i].ord_start
                )
                if structurally_adjacent:
                    adjacent = True
        if adjacent:
            dup_queries += 1
        if near_dup:
            near_dup_queries += 1

        metrics.per_query.append(
            {
                "id": q.id,
                "question": q.question,
                "hitCount": len(result.hits),
                "noRelevantHit": result.no_relevant_hit,
                "relevantCoveredAt3": covered,
                "firstRelevantRank": first_rank + 1 if first_rank >= 0 else 0,
                "pointsHit": sum(1 for p in q.key_points if p in top_text),
                "pointsTotal": len(q.key_points),
            }
        )

    negative_refused = 0
    for q in negatives:
        bm25_scores = bm25.score_all(q.question)
        vec_scores = vec_scores_for(q)
        idf = bm25.idf_weighted_coverage(q.question)
        result = post_process_retrieval(
            docs,
            bm25_scores,
            vec_scores,
            options,
            idf_coverage=idf,
            variants=variants_for(q),
        )
        refused = result.no_relevant_hit
        negative_refused += int(refused)
        metrics.negative_detail.append(
            {
                "id": q.id,
                "question": q.question,
                "refused": refused,
                "hitCount": len(result.hits),
                "topHit": result.hits[0].matched_content[:40] if result.hits else "",
            }
        )

    n = len(positives) or 1
    metrics.recall_at_3 = _round4(recall3 / n)
    metrics.recall_at_5 = _round4(recall5 / n)
    metrics.mrr_at_10 = _round4(mrr / n)
    metrics.bm25_only_recall_at_3 = _round4(bm25_only_r3 / n) if hybrid else None
    metrics.key_point_hit_rate = _round4(point_hits / point_total) if point_total else 0.0
    metrics.top3_duplicate_rate = _round4(dup_queries / n)
    metrics.top3_near_duplicate_rate = _round4(near_dup_queries / n)
    metrics.no_hit_query_count = no_hit
    metrics.false_refusal_count = no_hit
    # 受保护人群（q+m）：误拒门槛与相对门槛的取值来源，与 eval.ts 逐位同口径
    metrics.protected_false_refusal_count = protected_no_hit
    metrics.protected_recall_at_3 = (
        _round4(protected_r3 / protected_count) if protected_count else 0.0
    )
    metrics.protected_bm25_only_recall_at_3 = (
        _round4(protected_bm25_only_r3 / protected_count)
        if hybrid and protected_count
        else None
    )
    metrics.negative_count = len(negatives)
    metrics.negative_refused_count = negative_refused
    metrics.negative_refusal_rate = (
        _round4(negative_refused / len(negatives)) if negatives else 0.0
    )
    metrics.by_group = {
        g: {
            "count": v["count"],
            "recallAt3": _round4(v["r3"] / v["count"]),
            "recallAt5": _round4(v["r5"] / v["count"]),
            "noHit": v["noHit"],
        }
        for g, v in by_group.items()
    }
    return metrics


def _round4(value: float) -> float:
    """与 TS 的 `+x.toFixed(4)` 同口径（两端指标必须逐位可比）。"""
    return round(value + 0.0, 4)


def _is_relevant(content: str, relevant: list[str]) -> bool:
    return any(key in content for key in relevant)


def _coverage(hits, relevant: list[str], k: int) -> float:
    """top-k 窗口文本覆盖了多少条 relevant 原句（无标注时算 0）。"""
    if not relevant:
        return 0.0
    covered = sum(1 for key in relevant if any(key in h.content for h in hits[:k]))
    return covered / len(relevant)
