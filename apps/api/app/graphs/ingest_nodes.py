"""Ingestion Graph 的节点实现（M3）：解析 → 分块 → 去重/版本比对 → 向量化 → 索引 → 自检。

**粒度是文档级**：一个节点处理一篇文档，行内的阶段链（解析/分块/去重/向量化/索引）
全部复用 `IngestService` 与 `reindex_service.rebuild_doc` 的既有实现，一行摘要
（`chunks=…, reused=…, embedded=…, deduped=…`）如实反映这条链跑的什么。
为什么不为每个阶段各建一个节点：Traces 表体积与"文档数 × 阶段数"成正比，
而阶段级信息用摘要字段就能表达 —— 轨迹回放要看的是"哪篇文档在哪一步失败"。

失败隔离是这里的**第一等语义**：单文档异常只进 `failed[]`，整批继续（与 M2 的
`ingest_directory` / `rebuild` 同一条纪律），任务最终仍能 succeeded。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, TYPE_CHECKING

from ..core.config import Settings
from .state import IngestState
from .trace import trace_row

if TYPE_CHECKING:  # 避免与 services 层互相 import
    from ..services.ingest_service import IngestService

NodeFn = Callable[[IngestState], Awaitable[dict]]

#: 摄取单个文件的后缀白名单（与 IngestService.ingest_directory 同一口径）
_SUFFIXES = {".md", ".markdown", ".txt", ".docx", ".pdf"}


@dataclass
class IngestDeps:
    """摄取图的外部依赖（只依赖"零件"，不依赖服务类，避免图与服务互相持有）。"""

    store: object  # Store（Protocol，运行时不 import）
    ingest: "IngestService"
    settings: Settings


class IngestError(RuntimeError):
    """摄取流程级错误（不是单文档失败：那种进 failed[]）。"""


def build_ingest_nodes(deps: IngestDeps) -> dict[str, NodeFn]:
    """构造摄取图节点（闭包持有依赖）。"""

    # ---------------- 1. 扫描：确定这次要处理哪些文档 ----------------

    async def scan(state: IngestState) -> dict:
        if state.get("mode") == "upload":
            docs = [
                {
                    "kind": "upload",
                    "doc_id": _upload_doc_id(state),
                    "doc_name": state["upload_name"],
                    "raw_path": state["upload_path"],
                }
            ]
        else:
            # 重建：处理当前在库的全部文档（顺序确定 —— sorted by created_at，与 M2 一致）
            existing = await deps.store.list_documents()
            docs = [
                {
                    "kind": "existing",
                    "doc_id": d.doc_id,
                    "doc_name": d.doc_name,
                    "raw_path": d.raw_path or "",
                }
                for d in existing
            ]

        done = set(state.get("done") or [])
        # 断点续跑：已完成的文档直接跳过（checkpoint 是 tasks 表的 done 列表，兼容 M2）
        pending = [d["doc_id"] for d in docs if d["doc_id"] not in done]
        return {"docs": docs, "pending": pending}

    # ---------------- 2. 逐文档处理（失败隔离） ----------------

    async def process_doc(state: IngestState) -> dict:
        pending = list(state.get("pending") or [])
        if not pending:
            return {}
        doc_id = pending[0]
        doc = next(d for d in state["docs"] if d["doc_id"] == doc_id)
        started = time.perf_counter()
        reused = embedded = chunks = 0
        deduped = False
        failure: str | None = None

        try:
            if doc["kind"] == "upload":
                result = await deps.ingest.ingest_path(
                    Path(doc["raw_path"]),
                    doc_name=doc["doc_name"],
                    **state.get("acl", {}),
                )
                chunks = result.chunk_count
                reused = result.reused_chunks
                embedded = result.embedded_chunks
                deduped = result.deduped
            else:
                from ..services.reindex_service import rebuild_doc
                from ..store import DocumentRecord

                reused, embedded = await rebuild_doc(
                    deps.store,
                    deps.ingest,
                    DocumentRecord(
                        doc_id=doc_id,
                        doc_name=doc["doc_name"],
                        content_hash="",
                        index_version=state["version"],
                        raw_path=doc["raw_path"],
                    ),
                    state["version"],
                )
                chunks = reused + embedded
        except Exception as err:  # noqa: BLE001 —— 失败隔离到单文档
            failure = str(err)

        duration_ms = int((time.perf_counter() - started) * 1000)
        summary = (
            f"parse→chunk→dedup→embed→index ok, chunks={chunks}, "
            f"reused={reused}, embedded={embedded}, deduped={deduped}"
            if failure is None
            else f"失败（已隔离，不影响整批）：{failure}"
        )
        row = trace_row(
            state,
            "ingest_doc",
            status="ok" if failure is None else "failed",
            duration_ms=duration_ms,
            input_summary=f"{doc['doc_name']}（{doc['kind']}）",
            output_summary=summary,
            error=failure,
        )

        done = [*(state.get("done") or []), doc_id]
        failed = list(state.get("failed") or [])
        if failure is not None:
            failed.append({"doc_id": doc_id, "error": failure})
        results = list(state.get("results") or [])
        if failure is None and doc["kind"] == "upload":
            results.append(
                {
                    "doc_id": result.doc_id,
                    "doc_name": result.doc_name,
                    "chunk_count": result.chunk_count,
                    "deduped": result.deduped,
                    "reused_chunks": result.reused_chunks,
                    "embedded_chunks": result.embedded_chunks,
                    "index_version": result.index_version,
                    "latency_ms": duration_ms,
                }
            )
        stats = dict(state.get("stats") or {})
        stats["reused_vectors"] = int(stats.get("reused_vectors", 0)) + reused
        stats["embedded_chunks"] = int(stats.get("embedded_chunks", 0)) + embedded

        # 逐文档提交断点：进程被杀也能从这里续跑（幂等：重建同一文档是覆盖写）
        await deps.store.update_task(
            state["task_id"],
            progress=_progress(state, done=done, failed=failed, stats=stats),
            checkpoint={"done": sorted(done)},
        )
        return {
            "pending": pending[1:],
            "done": done,
            "failed": failed,
            "results": results,
            "stats": stats,
            "trace_nodes": [row],
        }

    def route_more(state: IngestState) -> str:
        return "process" if state.get("pending") else "activate"

    # ---------------- 3. 版本切换（重建路径：全量或灰度） ----------------

    async def activate(state: IngestState) -> dict:
        if state.get("mode") == "upload":
            # 单文档摄取走 IngestService.ingest_path：版本在里面已经激活并通知了检索侧
            return {}
        from ..services.reindex_service import activate_index_version

        await activate_index_version(
            deps.store, deps.ingest, state["version"], ratio=float(state.get("ratio", 1.0))
        )
        return {}

    # ---------------- 4. 自检 ----------------

    async def selfcheck(state: IngestState) -> dict:
        done = list(state.get("done") or [])
        failed = list(state.get("failed") or [])
        stats = dict(state.get("stats") or {})
        row = trace_row(
            state,
            "selfcheck",
            status="ok" if not failed else "ok",
            input_summary=f"docs={len(done)}, failed={len(failed)}",
            output_summary=(
                f"reused_vectors={stats.get('reused_vectors', 0)}, "
                f"embedded_chunks={stats.get('embedded_chunks', 0)}"
            ),
        )
        await deps.store.update_task(
            state["task_id"],
            progress=_progress(state, done=done, failed=failed, stats=stats, finished=True),
            checkpoint={"done": sorted(done)},
        )
        return {"trace_nodes": [row]}

    return {
        "scan": scan,
        "process_doc": process_doc,
        "route_more": route_more,  # type: ignore[dict-item]
        "activate": activate,
        "selfcheck": selfcheck,
    }


def _progress(
    state: IngestState,
    *,
    done: list[str],
    failed: list[dict],
    stats: dict,
    finished: bool = False,
) -> dict:
    """任务进度（字段名与 M2 保持一致：前端轮询与既有测试都读这些键）。"""
    progress = {
        "phase": "done" if finished else "reindex",
        "total_docs": len(state.get("docs") or []),
        "done_docs": len(done),
        "reused_vectors": stats.get("reused_vectors", 0),
        "embedded_chunks": stats.get("embedded_chunks", 0),
        "failed": failed,
    }
    return progress


def _upload_doc_id(state: IngestState) -> str:
    """上传路径的队列键：与摄取层同一份 doc_id 派生（改文件名口径时这里跟着走）。"""
    from ..services.reindex_service import doc_id_for

    return doc_id_for(state["upload_name"])
