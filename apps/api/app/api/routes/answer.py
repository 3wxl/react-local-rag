"""POST /v1/answer 与 GET /v1/answer/{answer_id}/trace（契约 §2、§3）。

`stream=true` 走 SSE（M3）：事件表与错误语义见 docs/api-contract.md §2。
要点是三条：
1. **预检类错误在流开始前**（版本未就绪、预算超标）→ 仍是真 HTTP 状态码；
   流开始之后的错误一律带内 `error` 事件（HTTP 头已经发出去了）；
2. `final` 事件的数据与 `stream=false` 的响应体**逐字段相同**（同一个对象）；
3. 帧格式与前端既有解析器同构（见 app/api/sse.py）。

不变量（tests/test_api_contract.py 逐条断言）：
1. 答案里的角标 [n] 必在 citations[].index 中
2. verdict == "refused" ⟺ citations 为空且 refusal 非空
3. evidence_snapshot.passages[].index 与 citations[].index 同源同序
4. 拒答是 200 正常响应
"""

from __future__ import annotations

from typing import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ...generation.base import ModelUnavailableError
from ...graphs.budget import BudgetExceeded
from ...parity.retrieval import DEFAULT_RETRIEVAL_OPTIONS
from ...services.reindex_service import select_index_version
from ...store import Filters
from ..deps import AppContext, get_ctx
from ..identity import Identity, get_identity
from ..errors import ApiError
from ..schemas import AnswerRequest
from ..serializers import context_to_dict
from ..sse import DONE_FRAME, sse_frame

router = APIRouter()


@router.post("/answer")
async def answer(
    body: AnswerRequest,
    ctx: AppContext = Depends(get_ctx),
    identity: Identity = Depends(get_identity),
):
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
    top_k = body.top_k or DEFAULT_RETRIEVAL_OPTIONS.top_k

    if body.stream:
        # 预检放在建流之前：超预算时返回真 422（而不是"先 200 再在流里报错"）
        try:
            ctx.answer.preflight(body.question, top_k=top_k)
        except BudgetExceeded as err:
            raise ApiError("budget_exceeded", str(err), err.detail) from err
        return StreamingResponse(
            _sse(ctx, body, version=active.version, top_k=top_k, filters=filters),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                # 反向代理别缓冲：否则流式会退化成"攒完一起发"
                "X-Accel-Buffering": "no",
            },
        )

    try:
        outcome = await ctx.answer.answer(
            body.question,
            version=active.version,
            top_k=top_k,
            filters=filters,
            rewrite=body.rewrite,
        )
    except ModelUnavailableError as err:
        raise ApiError("model_unavailable", str(err)) from err
    except BudgetExceeded as err:
        # 超预算是一等拒绝（契约 §0.5：422 budget_exceeded），detail 说明超的是哪道上限
        raise ApiError("budget_exceeded", str(err), err.detail) from err

    return {
        "answer_id": outcome.answer_id,
        "trace_id": outcome.trace_id,
        "question": outcome.question,
        "answer": outcome.answer,
        "verdict": outcome.verdict,
        "citations": [context_to_dict(c) for c in outcome.citations],
        "refusal": outcome.refusal,
        # M3 起由 QA 图的 verify 节点序列化（同一 serializer 的产物，线上字节不变）
        "verification": outcome.verification,
        "evidence_snapshot": outcome.evidence_snapshot,
        "cost": outcome.cost,
    }


async def _sse(
    ctx: AppContext,
    body: AnswerRequest,
    *,
    version: str,
    top_k: int,
    filters: Filters,
) -> AsyncIterator[str]:
    """图事件 → SSE 帧（逻辑全在服务层，路由只做格式转换）。"""
    async for payload in ctx.answer.answer_stream(
        body.question,
        version=version,
        top_k=top_k,
        filters=filters,
        rewrite=body.rewrite,
    ):
        yield sse_frame(payload["event"], payload["data"])
    yield DONE_FRAME


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
