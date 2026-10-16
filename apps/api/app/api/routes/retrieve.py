"""POST /v1/retrieve —— 只检索不生成（契约 §1）。

用途：前端"搜索模式"、Harness 单独量检索指标、调试。
判据与 Lite 模式 `applyThreshold` 完全一致（向量可用时只看余弦门限，不给 BM25 豁免）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...generation.base import ModelUnavailableError
from ...parity.citation import build_evidence
from ...parity.expand_prompt import ExpandPromptInput, build_expand_prompt
from ...parity.query_expansion import parse_expanded_queries
from ...parity.retrieval import DEFAULT_RETRIEVAL_OPTIONS
from ...services.reindex_service import select_index_version
from ...store import Filters
from ..deps import AppContext, get_ctx
from ..identity import Identity, get_identity
from ..errors import ApiError
from ..schemas import RetrieveRequest
from ..serializers import context_to_dict

router = APIRouter()


@router.post("/retrieve")
async def retrieve(
    body: RetrieveRequest,
    ctx: AppContext = Depends(get_ctx),
    identity: Identity = Depends(get_identity),
) -> dict:
    # 灰度：同一 query_id 必须永远落到同一版本（否则引用回跳会指向另一份块）
    query_id = _query_id()
    active = select_index_version(await ctx.store.list_ready_versions(), query_id)
    if active is None:
        raise ApiError("index_not_ready", "知识库索引尚未就绪（还没有摄取任何文档）")

    top_k = body.top_k or DEFAULT_RETRIEVAL_OPTIONS.top_k
    # ACL：身份派生的约束在**召回层**生效；客户端传的 doc_ids/tags 只能收紧
    # （它们与身份条件取交集，见 store 层的 _acl_clauses + load_chunks）
    filters = Filters(
        doc_ids=body.filters.doc_ids if body.filters is not None else (),
        tags=body.filters.tags if body.filters is not None else (),
        tenant_id=identity.tenant_id,
        departments=(identity.department,) if identity.department else (),
        max_classification=identity.clearance,
    )
    # 查询改写（M2）：口语化提问 → 术语化检索词。失败一律静默回落原查询
    # （扩展是增强项，不该让一次模型不可用把检索整个挡掉）
    expansions: list[str] = []
    if body.rewrite:
        try:
            completion = ctx.provider.complete(
                build_expand_prompt(ExpandPromptInput(question=body.question, max=3)),
                max_new_tokens=128,
            )
            expansions = parse_expanded_queries(completion.text or "", body.question)
        except ModelUnavailableError:
            expansions = []

    search = await ctx.retrieval.search(
        body.question,
        version=active.version,
        top_k=top_k,
        filters=filters,
        expansions=expansions,
    )
    return {
        "query_id": query_id,
        # 本次实际使用的索引版本（灰度期间必须能看出走的是哪一份）
        "index_version": active.version,
        # 改写结果（契约 §1 的增量字段）：rewrite=true 时给出实际用于检索的改写列表，
        # 空列表 = 模型不可用或没产出可用改写（此时检索只用原查询，如实反映）
        "rewritten_query": expansions[0] if expansions else (body.question if body.rewrite else None),
        "rewritten_queries": expansions if body.rewrite else None,
        "hits": _hits_to_contexts(search.hits),
        "no_relevant_hit": search.no_relevant_hit,
        "refusal_signals": {
            "idf_coverage": search.idf_coverage,
            # 空候选时是"无信号"，用 null 而不是 0：0 会被误读成"相关度为零"
            "top_relevance": search.top_relevance,
            # 多信号融合拒答的四路取值（2026-10-15 起参与判定，见 parity/retrieval.py）。
            # 全部列出是为了「拒答归因」：能看出这次是哪个信号触发的，而不是只看到结论。
            "max_cosine": search.signals.max_cosine,
            "cos_gap": search.signals.cos_gap,
            "max_bm25": search.signals.max_bm25,
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
