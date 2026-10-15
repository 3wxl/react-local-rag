"""POST /v1/retrieve —— 只检索不生成（契约 §1）。

用途：前端"搜索模式"、Harness 单独量检索指标、调试。
判据与 Lite 模式 `applyThreshold` 完全一致（向量可用时只看余弦门限，不给 BM25 豁免）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...parity.citation import build_evidence
from ...parity.retrieval import DEFAULT_RETRIEVAL_OPTIONS
from ...store import Filters
from ..deps import AppContext, get_ctx
from ..errors import ApiError
from ..schemas import RetrieveRequest
from ..serializers import context_to_dict

router = APIRouter()


@router.post("/retrieve")
async def retrieve(
    body: RetrieveRequest, ctx: AppContext = Depends(get_ctx)
) -> dict:
    active = await ctx.store.get_active_version()
    if active is None:
        raise ApiError("index_not_ready", "知识库索引尚未就绪（还没有摄取任何文档）")

    top_k = body.top_k or DEFAULT_RETRIEVAL_OPTIONS.top_k
    filters = (
        Filters(doc_ids=body.filters.doc_ids, tags=body.filters.tags)
        if body.filters is not None
        else None
    )
    search = await ctx.retrieval.search(
        body.question, version=active.version, top_k=top_k, filters=filters
    )
    return {
        "query_id": _query_id(),
        # M1 未实现查询改写：显式回落到原文（而不是悄悄装作改写过了）
        "rewritten_query": body.question if body.rewrite else None,
        "hits": _hits_to_contexts(search.hits),
        "no_relevant_hit": search.no_relevant_hit,
        "refusal_signals": {
            "idf_coverage": search.idf_coverage,
            # 空候选时是"无信号"，用 null 而不是 0：0 会被误读成"相关度为零"
            "top_relevance": search.top_relevance,
        },
        "dropped_count": search.dropped_count,
        "mode": search.mode,
        "cost": {
            "latency_ms": search.latency_ms,
            "latency_breakdown_ms": {"retrieve": search.latency_ms},
        },
    }


def _hits_to_contexts(hits) -> list[dict]:
    """命中 → 引用卡片（编号即 prompt 里的 [n]，与 /v1/answer 同一套装配）。"""
    _, contexts = build_evidence(hits)
    return [context_to_dict(c) for c in contexts]


def _query_id() -> str:
    import uuid

    return "qry_" + uuid.uuid4().hex[:16]
