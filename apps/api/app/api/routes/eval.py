"""POST /v1/eval/run 与 GET /v1/eval/run/{run_id}（契约 §4）。

异步执行用**进程内任务 + 状态表**：不引 Celery/Redis/ARQ（路线图明确砍掉队列框架，
保留的是"异步 + 断点续跑"的机制）。响应体是 202 + run_id，前端轮询同一个 GET。

跑的就是 `apps/api/app/harness` —— 指标名与门槛与 `npm run eval` 一致，
所以这条接口本身就是"双端可对比"的证据来源。
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from ...harness.runner import run_eval
from ...store import EvalRunRecord, TaskRecord
from ..deps import AppContext, get_ctx
from ..errors import ApiError
from ..schemas import EvalRunRequest

router = APIRouter()


@router.post("/eval/run", status_code=202)
async def start_eval(
    body: EvalRunRequest, ctx: AppContext = Depends(get_ctx)
) -> dict:
    if body.suite not in ("retrieval", "all"):
        raise ApiError(
            "invalid_request",
            f"暂不支持的 suite：{body.suite}（M1 支持 retrieval / all；answer 套件需要本地模型）",
        )
    run_id = "run_" + uuid.uuid4().hex[:16]
    await ctx.store.save_eval_run(
        EvalRunRecord(
            run_id=run_id,
            status="queued",
            suite=body.suite,
            golden_version=body.golden_version or "",
            config={"requested_overrides": body.config_overrides},
        )
    )
    await ctx.store.save_task(
        TaskRecord(task_id=run_id, kind="eval", status="queued")
    )

    async def run() -> None:
        await ctx.store.update_eval_run(run_id, status="running")
        await ctx.store.update_task(run_id, status="running")
        try:
            outcome = await asyncio.to_thread(
                run_eval,
                ctx.settings.eval_dir,
                suite=body.suite,
                embedder=ctx.embedder,
                prefer_fixed=True,
                config_overrides=body.config_overrides or None,
            )
            await ctx.store.update_eval_run(
                run_id,
                status="succeeded",
                metrics=_metrics_payload(outcome),
                gates_passed=outcome.gates_passed,
                failed_gates=outcome.failed_gates,
                finished_at=_now(),
            )
            await ctx.store.update_task(
                run_id,
                status="succeeded",
                progress={"gates_passed": outcome.gates_passed, "latency_ms": outcome.latency_ms},
            )
        except Exception as err:  # noqa: BLE001 —— 失败写进状态表，前端轮询可见
            await ctx.store.update_eval_run(
                run_id, status="failed", error=str(err), finished_at=_now()
            )
            await ctx.store.update_task(run_id, status="failed", error=str(err))

    ctx.spawn(run_id, run())
    return {"run_id": run_id, "status": "queued"}


@router.get("/eval/run/{run_id}")
async def get_eval_run(run_id: str, ctx: AppContext = Depends(get_ctx)) -> dict:
    record = await ctx.store.get_eval_run(run_id)
    if record is None:
        raise ApiError("not_found", f"run_id 不存在：{run_id}")
    task = await ctx.store.get_task(run_id)
    metrics = record.metrics or {}
    return {
        "run_id": record.run_id,
        "status": record.status,
        "suite": record.suite,
        "golden_version": record.golden_version,
        "config": record.config,
        "metrics": metrics.get("metrics"),
        "gates_passed": record.gates_passed,
        "failed_gates": record.failed_gates,
        "error": record.error,
        "progress": task.progress if task is not None else {},
        "created_at": record.created_at,
        "finished_at": record.finished_at,
    }


def _metrics_payload(outcome) -> dict:
    """契约 §4 的 metrics 结构（与 packages/eval/baseline/ts-metrics.json 对齐）。"""
    return {
        "metrics": outcome.metrics,
        "config": outcome.config,
        "suite": outcome.suite,
        "latency_ms": outcome.latency_ms,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
