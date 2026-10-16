"""检索服务：把存储里的块 + 向量喂给与 TS 同源的检索管线。

分工（这是「跨语言同口径」的落点）：
- 存储层：取块、取向量、受 filters 约束（ACL 在召回层生效）
- 这里：调用 `app/parity/retrieval.py` 的 `post_process_retrieval`，**与 apps/web 逐行对应**
- 向量分：本地 ONNX 编码查询后与块向量点积（float32 元素、float64 累加，与 TS 一致）

向量不可用（模型缺失/编码失败）时降级为纯 BM25（mode="bm25"），并启用 BM25 豁免门限 ——
这条降级路径与 Lite 模式的行为一致，不是这里新发明的。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..embedding.bge import BgeOnnxEmbedder
from ..parity.bm25 import BM25
from ..parity.query_expansion import build_query_variants
from ..parity.retrieval import (
    DEFAULT_RETRIEVAL_OPTIONS,
    QueryVariantScores,
    RefusalSignals,
    RetrievalDoc,
    RetrievalOptions,
    SearchHit,
    compute_refusal_signals,
    post_process_retrieval,
)
from ..store import Filters, Store


@dataclass
class SearchOutcome:
    """一次检索的结果与可观测信号（对应 /v1/retrieve 的响应体）。"""

    hits: list[SearchHit]
    no_relevant_hit: bool
    dropped_count: int
    idf_coverage: float
    mode: str  # hybrid | bm25
    latency_ms: int
    #: 命中候选总数（未过门限前），用于可观测性排查
    candidate_count: int = 0
    #: 查询级拒答信号（多信号融合拒答用的取值，进 /v1/retrieve 响应便于归因）
    signals: RefusalSignals = field(default_factory=RefusalSignals)
    #: 融合结果（评测与调试用；不进 API 响应）——见 retrieval.post_process_retrieval
    fused: list = field(default_factory=list)

    @property
    def top_relevance(self) -> float | None:
        return self.hits[0].relevance if self.hits else None


class RetrievalService:
    """检索服务（每进程一份；BM25 按索引版本缓存）。"""

    def __init__(self, store: Store, embedder: BgeOnnxEmbedder) -> None:
        self.store = store
        self.embedder = embedder
        self._bm25_cache: dict[str, BM25] = {}

    # ---------------- 索引缓存 ----------------

    def invalidate(self, version: str | None = None) -> None:
        """摄取完成后调用：BM25 缓存必须失效（否则检索到的是旧语料）。"""
        if version is None:
            self._bm25_cache.clear()
        else:
            self._bm25_cache.pop(version, None)

    async def _load_docs(
        self, version: str, filters: Filters | None
    ) -> list[RetrievalDoc]:
        chunks = await self.store.load_chunks(version, filters=filters)
        vectors = await self.store.load_vectors(version, filters=filters)
        docs: list[RetrievalDoc] = []
        for c in chunks:
            docs.append(
                RetrievalDoc(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    content=c.content,
                    doc_name=c.doc_name or c.doc_id,
                    ord=c.ord,
                    heading_path=c.heading_path,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    char_start=c.char_start,
                    char_end=c.char_end,
                    token_count=c.token_count,
                    content_hash=c.content_hash,
                    vector=vectors.get(c.chunk_id),
                )
            )
        return docs

    def _bm25_for(self, version: str, docs: list[RetrievalDoc]) -> BM25:
        cached = self._bm25_cache.get(version)
        if cached is None or cached.size != len(docs):
            cached = BM25([d.content for d in docs])
            self._bm25_cache[version] = cached
        return cached

    # ---------------- 检索 ----------------

    async def search(
        self,
        question: str,
        *,
        version: str,
        top_k: int = 5,
        options: RetrievalOptions = DEFAULT_RETRIEVAL_OPTIONS,
        filters: Filters | None = None,
        allow_vector: bool = True,
        expansions: list[str] | None = None,
    ) -> SearchOutcome:
        """检索并做后处理（与 worker 的 search 分支同序）。

        `expansions`：术语化改写（可选）。它们作为额外路参与 RRF 与拒答信号，
        但不改变命中上的分数字段 —— 与 TS 侧 postProcessRetrieval 的 variants 同口径。
        """
        started = time.perf_counter()
        docs = await self._load_docs(version, filters)
        if not docs:
            return SearchOutcome(
                hits=[],
                no_relevant_hit=True,
                dropped_count=0,
                idf_coverage=1.0,
                mode="bm25",
                latency_ms=0,
            )

        bm25 = self._bm25_for(version, docs)
        bm25_scores = bm25.score_all(question)
        idf_coverage = bm25.idf_weighted_coverage(question)

        vec_scores: list[float] | None = None
        if allow_vector and self.embedder.available:
            try:
                qvec = self.embedder.embed_query(question)
                vec_scores = [
                    float(np.dot(d.vector, qvec)) if d.vector is not None else 0.0
                    for d in docs
                ]
            except Exception:  # noqa: BLE001 —— 与 Lite 模式一致：向量失败退 BM25
                vec_scores = None

        effective = RetrievalOptions(
            **{**options.__dict__, "top_k": max(1, min(top_k, len(docs)))}
        )
        variants: list[QueryVariantScores] = []
        for text, weight in build_query_variants(question, expansions or [])[1:]:
            v_bm25 = bm25.score_all(text)
            v_vec: list[float] | None = None
            if vec_scores is not None and allow_vector and self.embedder.available:
                try:
                    v_qvec = self.embedder.embed_query(text)
                    v_vec = [
                        float(np.dot(d.vector, v_qvec)) if d.vector is not None else 0.0
                        for d in docs
                    ]
                except Exception:  # noqa: BLE001 —— 变体向量失败时该路降级为纯 BM25
                    v_vec = None
            variants.append(
                QueryVariantScores(
                    bm25_scores=v_bm25,
                    vec_scores=v_vec,
                    idf_coverage=bm25.idf_weighted_coverage(text),
                    weight=weight,
                )
            )
        result = post_process_retrieval(
            docs,
            bm25_scores,
            vec_scores,
            effective,
            idf_coverage=idf_coverage,
            variants=variants or None,
        )
        return SearchOutcome(
            hits=result.hits,
            no_relevant_hit=result.no_relevant_hit,
            dropped_count=result.dropped_count,
            idf_coverage=idf_coverage,
            mode="hybrid" if vec_scores is not None else "bm25",
            latency_ms=int((time.perf_counter() - started) * 1000),
            candidate_count=len(result.fused),
            signals=compute_refusal_signals(bm25_scores, vec_scores),
            fused=result.fused,
        )
