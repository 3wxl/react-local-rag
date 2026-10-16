"""索引灰度重建：把语料重算到新索引版本，再按流量比放量切换。

保留的是**机制**，不是框架（路线图：砍队列框架、留异步/断点/增量）：
- **异步 + 断点续跑**：进度写 `tasks.progress`，断点写 `tasks.checkpoint`（逐文档提交），
  进程重启后按 checkpoint 跳过已完成文档继续 —— 不引 Celery/Redis。
- **失败隔离到单文档**：一个文件解析失败只记进 progress.failed，不中断整批。
- **增量**：向量按 contentHash 复用（`chunk_vector_cache`），换分块参数时未变文本不重算。
  这就是"改文档后索引自动增量重建"的落点：只有变化的块会走模型。
- **灰度**：新版本先以 `traffic_ratio` 接小部分流量（按 query_id 确定性分桶），
  指标不退化再放量到 1.0；ratio=0 等于回滚。
"""

from __future__ import annotations

import time
from pathlib import Path

from ..parity.chunk_params import (
    CURRENT_INDEX_VERSION,
    DEFAULT_CHUNK_OPTIONS,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PREFIX_SCHEME,
)
from ..parity.chunker import TextChunkStream
from ..parity.fnv import fnv1a32
from ..parity.parsers import blocks_for_file
from ..store import (
    ChunkRecord,
    DocumentRecord,
    IndexVersionRecord,
    Store,
    TaskRecord,
)
from .ingest_service import IngestService, _doc_id_for


def _chunk_params() -> dict:
    return {
        "maxChunkTokens": DEFAULT_CHUNK_OPTIONS.max_chunk_tokens,
        "minChunkChars": DEFAULT_CHUNK_OPTIONS.min_chunk_chars,
        "maxChunkChars": DEFAULT_CHUNK_OPTIONS.max_chunk_chars,
        "sentenceOverlap": DEFAULT_CHUNK_OPTIONS.sentence_overlap,
    }


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def select_index_version(
    versions: list[IndexVersionRecord], query_id: str
) -> IndexVersionRecord | None:
    """按流量比选版本：**确定性**分桶（同一 query_id 永远选同一个版本）。

    为什么必须确定性：检索、引用回跳、答案快照都带着 index_version，
    同一问题两次请求若落到不同版本，引用校验会指向另一份块 —— 用户看到"引用的原文对不上"。
    哈希取 fnv1a32（与索引版本号同一算法，见 parity/fnv.py），桶宽 1000。
    """
    if not versions:
        return None
    if len(versions) == 1:
        return versions[0]
    newest = versions[0]
    ratio = max(0.0, min(1.0, float(newest.traffic_ratio)))
    if ratio <= 0:
        return versions[1]
    if ratio >= 1:
        return newest
    # fnv1a32 返回十六进制字符串（与索引版本号同表示），分桶前转成整数
    bucket = int(fnv1a32(query_id), 16) % 1000
    return newest if bucket < int(ratio * 1000) else versions[1]


class ReindexService:
    """索引重建（进程内任务；由路由 spawn，状态写 tasks 表）。"""

    def __init__(self, store: Store, ingest: IngestService) -> None:
        self.store = store
        self.ingest = ingest

    async def rebuild(self, task_id: str, *, ratio: float = 1.0) -> None:
        """重建全量语料到新版本；ratio<1 时新版本以灰度方式上线。"""
        started = time.perf_counter()
        version = CURRENT_INDEX_VERSION
        task = await self.store.get_task(task_id)
        done: set[str] = set((task.checkpoint or {}).get("done", [])) if task else set()

        await self.store.upsert_version(
            IndexVersionRecord(
                version=version,
                state="building",
                chunk_params=_chunk_params(),
                embedding_model=EMBEDDING_MODEL_NAME,
                prefix_scheme=EMBEDDING_PREFIX_SCHEME,
                traffic_ratio=ratio,
            )
        )

        docs = await self.store.list_documents()
        progress = {
            "phase": "reindex",
            "total_docs": len(docs),
            "done_docs": len(done),
            "reused_vectors": 0,
            "embedded_chunks": 0,
            "failed": [],
        }
        await self._update(task_id, status="running", progress=progress, checkpoint={"done": sorted(done)})

        for doc in docs:
            if doc.doc_id in done:
                continue
            try:
                reused, embedded = await self._rebuild_doc(doc, version)
                progress["reused_vectors"] += reused
                progress["embedded_chunks"] += embedded
            except Exception as err:  # noqa: BLE001 —— 失败隔离到单文档
                progress["failed"].append({"doc_id": doc.doc_id, "error": str(err)})
            done.add(doc.doc_id)
            progress["done_docs"] = len(done)
            # 逐文档提交断点：进程被杀也能从这里续跑（幂等：重建同一文档是覆盖写）
            await self._update(
                task_id,
                progress=progress,
                checkpoint={"done": sorted(done)},
            )

        if ratio >= 1.0:
            # 全量切换：旧版本转 stale（保留数据，便于回滚与对比）
            await self.store.activate_version(version)
        else:
            # 灰度：新版本直接 ready 但只接 ratio 比例的流量；旧版本保持 ready
            current = await self.store.get_active_version()
            await self.store.upsert_version(
                IndexVersionRecord(
                    version=version,
                    state="ready",
                    chunk_params=_chunk_params(),
                    embedding_model=EMBEDDING_MODEL_NAME,
                    prefix_scheme=EMBEDDING_PREFIX_SCHEME,
                    activated_at=_now(),
                    traffic_ratio=ratio,
                )
            )
            if current is not None and current.version != version:
                # 旧版本继续吃剩余流量（选择逻辑见 select_index_version）
                await self.store.set_traffic_ratio(current.version, 1.0)

        progress["phase"] = "done"
        progress["latency_ms"] = int((time.perf_counter() - started) * 1000)
        await self._update(task_id, status="succeeded", progress=progress)

        # 索引变了 → 让检索侧的 BM25 缓存失效（同一进程内的回调）
        self.ingest.notify_index_changed(version)

    async def _rebuild_doc(self, doc: DocumentRecord, version: str) -> tuple[int, int]:
        """重算单文档（解析 → 分块 → 复用/编码向量 → 写新版本）。返回 (复用块数, 新编码块数)。"""
        raw_path = Path(doc.raw_path) if doc.raw_path else None
        if raw_path is None or not raw_path.exists():
            raise FileNotFoundError(f"原文不存在，无法重建：{doc.raw_path or '(未记录)'}")

        blocks = blocks_for_file(raw_path, filename=doc.doc_name)
        stream = TextChunkStream(doc.doc_id, doc.doc_name, DEFAULT_CHUNK_OPTIONS)
        for block in blocks:
            stream.push(block)
        chunks = stream.finish()
        if not chunks:
            raise ValueError(f"未切出任何分块：{doc.doc_name}")

        records = [
            ChunkRecord(
                chunk_id=c.chunk_id,
                doc_id=doc.doc_id,
                index_version=version,
                ord=c.ord,
                content=c.content,
                content_hash=c.content_hash,
                heading_path=c.heading_path,
                page_start=c.page_start,
                page_end=c.page_end,
                char_start=c.char_start,
                char_end=c.char_end,
                token_count=c.token_count,
                parent_id=c.parent_id,
                prev_id=c.prev_id,
                next_id=c.next_id,
                doc_name=doc.doc_name,
            )
            for c in chunks
        ]
        await self.store.replace_doc_chunks(doc.doc_id, version, records)

        reused = embedded = 0
        if self.ingest.embedder.available:
            vectors, reused, embedded = await self.ingest.vectors_for_chunks(chunks)
            await self.store.put_vectors(
                version, [(c.chunk_id, vectors[c.content_hash]) for c in chunks]
            )
        return reused, embedded

    async def _update(self, task_id: str, **fields) -> None:
        await self.store.update_task(task_id, **fields)


def doc_id_for(name: str) -> str:
    """与摄取同一份 doc_id 口径（迁移脚本与测试共用，避免两处派生规则漂移）。"""
    return _doc_id_for(name)


def new_task_record(task_id: str, *, kind: str = "reindex") -> TaskRecord:
    now = _now()
    return TaskRecord(task_id=task_id, kind=kind, status="queued", created_at=now, updated_at=now)
