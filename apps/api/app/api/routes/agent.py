"""Agent 接口（M3 增补，契约 §7）：一次 run 的启动 / 查询 / 审批续跑 / 轨迹。

```jsonc
POST /v1/agent/run              → 200 终态响应，或 {status:"interrupted", pending_approval:{…}}
   （stream=true 时为 SSE：node/token/interrupt/final/error 事件，与非流式同形）
GET  /v1/agent/{run_id}         → 状态 + 待审批动作（前端轮询）
POST /v1/agent/{run_id}/resume  → 202 + 后台续跑（人工审批：批准 / 驳回）
GET  /v1/agent/{run_id}/trace   → 与 answer trace 同形（自包含回放）
```

状态复用 `tasks` 表（kind="agent"）：`queued → running → interrupted → running → succeeded|failed`，
`checkpoint` 里放待审批动作 —— 前端已有的任务轮询范式不用新增一套。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ...generation.base import ModelUnavailableError
from ...graphs.budget import BudgetExceeded
from ...parity.retrieval import DEFAULT_RETRIEVAL_OPTIONS
from ...services.reindex_service import select_index_version
from ...store import Filters, TaskRecord
from ..deps import AppContext, get_ctx
from ..errors import ApiError
from ..identity import Identity, get_identity
from ..schemas import AgentResumeRequest, AgentRunRequest
from ..sse import DONE_FRAME, sse_frame

router = APIRouter()


@router.post("/agent/run")
async def agent_run(
    body: AgentRunRequest,
    ctx: AppContext = Depends(get_ctx),
    identity: Identity = Depends(get_identity),
):
    active = select_index_version(await ctx.store.list_ready_versions(), body.question)
    if active is None:
        raise ApiError("index_not_ready", "知识库索引尚未就绪（还没有摄取任何文档）")

    filters = Filters(
        doc_ids=body.filters.doc_ids if body.filters is not None else (),
        tags=body.filters.tags if body.filters is not None else (),
        tenant_id=identity.tenant_id,
        departments=(identity.department,) if identity.department else (),
        max_classification=identity.clearance,
    )
    top_k = body.top_k or DEFAULT_RETRIEVAL_OPTIONS.top_k
    run_id = "run_" + uuid.uuid4().hex[:16]
    await ctx.store.save_task(
        TaskRecord(task_id=run_id, kind="agent", status="running", checkpoint={"thread_id": run_id})
    )

    if body.stream:
        # 预检在流开始前：超预算仍是真 422
        try:
            ctx.agent.preflight(body.question, top_k=top_k)
        except BudgetExceeded as err:
            raise ApiError("budget_exceeded", str(err), err.detail) from err
        return StreamingResponse(
            _sse_run(ctx, body, run_id=run_id, version=active.version, top_k=top_k, filters=filters),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = await ctx.agent.run(
            body.question,
            run_id=run_id,
            version=active.version,
            top_k=top_k,
            filters=filters,
        )
    except ModelUnavailableError as err:
        await ctx.store.update_task(run_id, status="failed", error=str(err))
        raise ApiError("model_unavailable", str(err)) from err
    except BudgetExceeded as err:
        await ctx.store.update_task(run_id, status="failed", error=str(err))
        raise ApiError("budget_exceeded", str(err), err.detail) from err

    return await _finalize_task(ctx, run_id, result, question=body.question)


@router.get("/agent/{run_id}")
async def agent_status(run_id: str, ctx: AppContext = Depends(get_ctx)) -> dict:
    task = await ctx.store.get_task(run_id)
    if task is None or task.kind != "agent":
        raise ApiError("not_found", f"run 不存在：{run_id}")
    return {
        "run_id": task.task_id,
        "status": task.status,
        "progress": task.progress,
        "pending_approval": (task.checkpoint or {}).get("pending_approval"),
        "error": task.error,
        "updated_at": task.updated_at,
    }


@router.post("/agent/{run_id}/resume", status_code=202)
async def agent_resume(
    run_id: str,
    body: AgentResumeRequest,
    ctx: AppContext = Depends(get_ctx),
) -> dict:
    task = await ctx.store.get_task(run_id)
    if task is None or task.kind != "agent":
        raise ApiError("not_found", f"run 不存在：{run_id}")
    if task.status != "interrupted":
        raise ApiError("invalid_request", f"该 run 没有待审批的工具调用（状态：{task.status}）")

    await ctx.store.update_task(run_id, status="running")

    async def run() -> None:
        try:
            result = await ctx.agent.resume(
                run_id, approved=body.approved, note=body.note
            )
            await _finalize_task(ctx, run_id, result, question=None)
        except Exception as err:  # noqa: BLE001 —— 失败要落到任务状态里
            await ctx.store.update_task(run_id, status="failed", error=str(err))

    ctx.spawn(run_id, run())
    return {"run_id": run_id, "status": "running", "approved": body.approved}


@router.get("/agent/{run_id}/trace")
async def agent_trace(run_id: str, ctx: AppContext = Depends(get_ctx)) -> dict:
    """Agent 轨迹（自包含回放：与 answer trace 同一套字段）。"""
    trace = await ctx.store.get_trace(run_id)
    if trace is None:
        raise ApiError("not_found", f"trace 不存在（run 未完成或已清理）：{run_id}")
    return {
        "run_id": run_id,
        "trace_id": trace.trace_id,
        "question": trace.question,
        "nodes": trace.nodes,
        "final_verdict": trace.final_verdict,
        "cost": trace.cost,
        "evidence_snapshot": trace.evidence_snapshot,
    }


async def _finalize_task(ctx: AppContext, run_id: str, result, *, question: str | None) -> dict:
    """把 run 的结果写回任务表，并组装响应体。"""
    if result.status == "interrupted":
        await ctx.store.update_task(
            run_id,
            status="interrupted",
            progress={"steps": result.steps, "tool_calls": result.tool_calls},
            checkpoint={"thread_id": run_id, "pending_approval": result.pending_approval},
        )
        return {
            "run_id": run_id,
            "trace_id": run_id,
            "status": "interrupted",
            "question": question,
            "pending_approval": result.pending_approval,
            "steps": result.steps,
            "tool_calls": result.tool_calls,
            "trace_nodes": result.trace_nodes,
        }

    outcome = result.outcome or {}
    await ctx.store.update_task(
        run_id,
        status="succeeded",
        progress={"steps": result.steps, "tool_calls": result.tool_calls},
        checkpoint={"thread_id": run_id},
    )
    return {
        "run_id": run_id,
        "trace_id": run_id,
        "status": "succeeded",
        "question": outcome.get("question", question),
        "answer": outcome.get("answer", ""),
        "verdict": outcome.get("verdict", "supported"),
        "citations": outcome.get("citations", []),
        "refusal": outcome.get("refusal"),
        "verification": outcome.get("verification", {}),
        "evidence_snapshot": outcome.get("evidence_snapshot", {}),
        "cost": outcome.get("cost", {}),
        "tool_calls": outcome.get("tool_calls", []),
        "no_progress": outcome.get("no_progress", {}),
        "steps": outcome.get("steps", result.steps),
        "trace_nodes": result.trace_nodes,
    }


async def _sse_run(ctx: AppContext, body: AgentRunRequest, *, run_id, version, top_k, filters):
    events = ctx.agent.run_stream(
        body.question, run_id=run_id, version=version, top_k=top_k, filters=filters
    )
    async for payload in events:
        if payload["event"] == "interrupt":
            await ctx.store.update_task(
                run_id,
                status="interrupted",
                checkpoint={"thread_id": run_id, "pending_approval": payload["data"]},
            )
        elif payload["event"] == "final":
            data = payload["data"]
            await ctx.store.update_task(
                run_id,
                status="succeeded",
                progress={"steps": data.get("steps", 0), "tool_calls": data.get("tool_calls", [])},
                checkpoint={"thread_id": run_id},
            )
        elif payload["event"] == "error":
            await ctx.store.update_task(
                run_id, status="failed", error=payload["data"]["error"]["message"]
            )
        yield sse_frame(payload["event"], payload["data"])
    yield DONE_FRAME
