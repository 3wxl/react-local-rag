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

from ...services.ingest_service import MAX_FILE_MB
from ...store import TaskRecord
from ..deps import AppContext, get_ctx
from ..errors import ApiError

router = APIRouter()


@router.post("/documents", status_code=202)
async def upload_document(
    file: UploadFile = File(...), ctx: AppContext = Depends(get_ctx)
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
            result = await ctx.ingest.ingest_path(target, doc_name=file.filename)
            await ctx.store.update_task(
                task_id,
                status="succeeded",
                progress={
                    "doc_id": result.doc_id,
                    "doc_name": result.doc_name,
                    "chunks": result.chunk_count,
                    "deduped": result.deduped,
                    "index_version": result.index_version,
                    "latency_ms": result.latency_ms,
                },
                checkpoint={"done": [result.doc_id]},
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
