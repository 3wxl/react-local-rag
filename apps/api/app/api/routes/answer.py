"""POST /v1/answer 与 GET /v1/answer/{answer_id}/trace（契约 §2、§3）。

不变量（tests/test_api_contract.py 逐条断言）：
1. 答案里的角标 [n] 必在 citations[].index 中
2. verdict == "refused" ⟺ citations 为空且 refusal 非空
3. evidence_snapshot.passages[].index 与 citations[].index 同源同序
4. 拒答是 200 正常响应
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...generation.base import ModelUnavailableError
from ...parity.retrieval import DEFAULT_RETRIEVAL_OPTIONS
from ...services.reindex_service import select_index_version
from ...store import Filters
from ..deps import AppContext, get_ctx
from ..identity import Identity, get_identity
from ..errors import ApiError
from ..schemas import AnswerRequest
from ..serializers import context_to_dict, verification_to_dict

router = APIRouter()


@router.post("/answer")
async def answer(
    body: AnswerRequest,
    ctx: AppContext = Depends(get_ctx),
    identity: Identity = Depends(get_identity),
) -> dict:
    if body.stream:
        # 契约里 stream 是 M3 的能力：现在就明确报不支持，避免前端以为拿到了流
        raise ApiError(
            "invalid_request", "stream=true 尚未实现（M3 提供 SSE 流式）"
        )

    # 灰度版本选择：按 question 哈希确定性分桶（同一问题永远走同一版本）
    active = select_index_version(await ctx.store.list_ready_versions(), body.question)
    if active is None:
        raise ApiError("index_not_ready", "知识库索引尚未就绪（还没有摄取任何文档）")

    # ACL 同样在召回层生效（与 /v1/retrieve 完全一致的口径）
    filters = Filters(
        doc_ids=body.filters.doc_ids if body.filters is not None else (),
        tags=body.filters.tags if body.filters is not None else (),
        tenant_id=identity.tenant_id,
        departments=(identity.department,) if identity.department else (),
        max_classification=identity.clearance,
    )
    try:
        outcome = await ctx.answer.answer(
            body.question,
            version=active.version,
            top_k=body.top_k or DEFAULT_RETRIEVAL_OPTIONS.top_k,
            filters=filters,
        )
    except ModelUnavailableError as err:
        raise ApiError("model_unavailable", str(err)) from err

    return {
        "answer_id": outcome.answer_id,
        "trace_id": outcome.trace_id,
        "question": outcome.question,
        "answer": outcome.answer,
        "verdict": outcome.verdict,
        "citations": [context_to_dict(c) for c in outcome.citations],
        "refusal": outcome.refusal,
        "verification": verification_to_dict(outcome.verification),
        "evidence_snapshot": outcome.evidence_snapshot,
        "cost": outcome.cost,
    }


@router.get("/answer/{answer_id}/trace")
async def trace(answer_id: str, ctx: AppContext = Depends(get_ctx)) -> dict:
    """回放一次问答的完整决策路径（审计 / 轨迹回放 CLI 的数据源）。"""
    record = await ctx.store.get_answer(answer_id)
    if record is None:
        raise ApiError("not_found", f"answer_id 不存在或已过期清理：{answer_id}")
    stored_trace = await ctx.store.get_trace(record.trace_id)
    if stored_trace is None:
        raise ApiError("not_found", f"trace 不存在：{record.trace_id}")
    return {
        "answer_id": record.answer_id,
        "trace_id": record.trace_id,
        "question": record.question,
        "nodes": stored_trace.nodes,
        "final_verdict": stored_trace.final_verdict,
        "cost": stored_trace.cost,
        # 冗余存放：trace 必须自包含，不依赖 answers 行仍存在
        "evidence_snapshot": stored_trace.evidence_snapshot,
    }
