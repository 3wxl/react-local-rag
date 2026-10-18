"""文档摄取接口（契约未覆盖，M1 增补并记录在案）。

- `POST /v1/documents`：multipart 上传 → 202 + task_id（进程内任务，见 deps.AppContext.spawn）
- `GET /v1/tasks/{task_id}`：任务进度（进度写 tasks 表，支持断点续跑的挂点）
- `GET /v1/documents`：已入库文档
- `DELETE /v1/documents/{doc_id}`：软删除（向量同步从检索面移除）

为什么是 202 + 轮询而不是同步等待：摄取要跑解析 + 向量化，大文件耗时几十秒；
契约里 `/v1/eval/run` 已经是这个模式，摄取保持一致，前端只用实现一套轮询。
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel, Field

from ...graphs.ingest_graph import invoke_ingest, upload_seed
from ...services.ingest_service import MAX_FILE_MB
from ...store import TaskRecord
from ..deps import AppContext, get_ctx
from ..errors import ApiError
from ..identity import Identity, get_identity

router = APIRouter()


class RebuildRequest(BaseModel):
    """重建参数。ratio<1 = 灰度上线（先接小比例流量）。"""

    model_config = {"extra": "forbid"}

    ratio: float = Field(default=1.0, ge=0.0, le=1.0)


class ActivateRequest(BaseModel):
    """灰度放量/回滚：目标版本的流量比。"""

    model_config = {"extra": "forbid"}

    ratio: float = Field(ge=0.0, le=1.0)


@router.post("/documents", status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    ctx: AppContext = Depends(get_ctx),
    identity: Identity = Depends(get_identity),
) -> dict:
    if not file.filename:
        raise ApiError("invalid_request", "缺少文件名")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".md", ".markdown", ".mdown", ".mkd", ".txt", ".docx", ".pdf"}:
        raise ApiError(
            "invalid_request",
            f"不支持的文件类型：{suffix}（支持 .md / .txt / .docx / .pdf）",
        )

    payload = await file.read()
    if len(payload) > MAX_FILE_MB * 1024 * 1024:
        raise ApiError("invalid_request", f"文件超过 {MAX_FILE_MB}MB 上限")

    # 原始文件落盘（路线图：文件用本地磁盘，不塞进数据库大字段）
    upload_dir = Path(ctx.settings.data_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f"{uuid.uuid4().hex[:8]}-{file.filename}"
    target.write_bytes(payload)

    task_id = "tsk_" + uuid.uuid4().hex[:16]
    await ctx.store.save_task(
        TaskRecord(task_id=task_id, kind="ingest", status="queued", owner=None)
    )

    async def run() -> None:
        await ctx.store.update_task(task_id, status="running")
        try:
            # 归属由**身份**决定，不由请求方声明（否则等于让调用方自己发权限）
            final = await invoke_ingest(
                ctx.ingest_graph,
                upload_seed(
                    task_id=task_id,
                    path=str(target),
                    doc_name=file.filename or "",
                    acl={
                        "tenant_id": identity.tenant_id,
                        "department": identity.department,
                        "classification": identity.clearance if identity.department else 0,
                    },
                ),
                thread_id=task_id,
                doc_count=1,
                settings=ctx.settings,
            )
            failed = final.get("failed") or []
            if failed:
                # 单文档失败（图把失败隔离进了 failed[]）：任务状态如实置 failed
                raise RuntimeError(str(failed[0].get("error", "摄取失败")))
            result = (final.get("results") or [{}])[0]
            await ctx.store.update_task(
                task_id,
                status="succeeded",
                progress={
                    "doc_id": result.get("doc_id"),
                    "doc_name": result.get("doc_name"),
                    "chunks": result.get("chunk_count"),
                    "deduped": result.get("deduped"),
                    "index_version": result.get("index_version"),
                    "latency_ms": result.get("latency_ms"),
                },
                checkpoint={"done": [result.get("doc_id")]},
            )
        except Exception as err:  # noqa: BLE001 —— 失败要落到任务状态里，前端才看得到原因
            await ctx.store.update_task(task_id, status="failed", error=str(err))

    ctx.spawn(task_id, run())
    return {"task_id": task_id, "status": "queued", "doc_name": file.filename}


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, ctx: AppContext = Depends(get_ctx)) -> dict:
    task = await ctx.store.get_task(task_id)
    if task is None:
        raise ApiError("not_found", f"task_id 不存在：{task_id}")
    return {
        "task_id": task.task_id,
        "kind": task.kind,
        "status": task.status,
        "progress": task.progress,
        "error": task.error,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }


@router.get("/documents")
async def list_documents(ctx: AppContext = Depends(get_ctx)) -> dict:
    documents = await ctx.store.list_documents()
    return {
        "documents": [
            {
                "doc_id": d.doc_id,
                "doc_name": d.doc_name,
                "content_hash": d.content_hash,
                "index_version": d.index_version,
                "size_bytes": d.size_bytes,
                "created_at": d.created_at,
            }
            for d in documents
        ]
    }


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, ctx: AppContext = Depends(get_ctx)) -> dict:
    documents = {d.doc_id for d in await ctx.store.list_documents()}
    if doc_id not in documents:
        raise ApiError("not_found", f"doc_id 不存在：{doc_id}")
    await ctx.ingest.delete_document(doc_id)
    return {"doc_id": doc_id, "deleted": True}


# ---------------- 索引重建与灰度（M2） ----------------


@router.post("/index/rebuild", status_code=202)
async def rebuild_index(body: RebuildRequest | None = None, ctx: AppContext = Depends(get_ctx)) -> dict:
    """重建全量索引（换分块参数/换模型/修复用）。

    - `ratio < 1`：新版本以灰度方式上线（只接该比例流量，旧版本继续服务剩余流量）
    - `ratio = 1`：直接全量切换
    断点续跑：进度与已完成文档写进 tasks 表，进程重启后按 checkpoint 跳过已完成的文档。
    """
    ratio = 1.0 if body is None else body.ratio
    task_id = "tsk_" + uuid.uuid4().hex[:16]
    await ctx.store.save_task(
        TaskRecord(task_id=task_id, kind="reindex", status="queued", owner=None)
    )

    async def run() -> None:
        try:
            await ctx.reindex.rebuild(task_id, ratio=ratio)
        except Exception as err:  # noqa: BLE001 —— 失败要落到任务状态里
            await ctx.store.update_task(task_id, status="failed", error=str(err))

    ctx.spawn(task_id, run())
    return {"task_id": task_id, "status": "queued", "ratio": ratio}


@router.post("/index/{version}/activate")
async def activate_version(
    version: str, body: ActivateRequest, ctx: AppContext = Depends(get_ctx)
) -> dict:
    """灰度放量 / 回滚：设置某版本的流量比（0 = 回滚到旧版本，1 = 全量）。"""
    versions = {v.version for v in await ctx.store.list_ready_versions()}
    if version not in versions:
        raise ApiError("not_found", f"版本不存在或未就绪：{version}")
    if not 0.0 <= body.ratio <= 1.0:
        raise ApiError("invalid_request", "ratio 必须在 0~1 之间")
    if body.ratio >= 1.0:
        await ctx.store.activate_version(version)
    else:
        await ctx.store.set_traffic_ratio(version, body.ratio)
    ctx.retrieval.invalidate()
    return {"version": version, "traffic_ratio": body.ratio}


@router.get("/index/versions")
async def list_versions(ctx: AppContext = Depends(get_ctx)) -> dict:
    """索引版本表（是否 ready、流量比、激活时间）——灰度与回滚的可观测面。"""
    versions = await ctx.store.list_ready_versions()
    current = await ctx.store.get_active_version()
    return {
        "versions": [
            {
                "version": v.version,
                "state": v.state,
                "traffic_ratio": v.traffic_ratio,
                "activated_at": v.activated_at,
                "created_at": v.created_at,
            }
            for v in versions
        ],
        "active": current.version if current else None,
    }
