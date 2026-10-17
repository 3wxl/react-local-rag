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
    """索引重建（M3 起由 Ingestion Graph 编排；进程内任务，状态写 tasks 表）。

    本类的职责收敛为三件：版本行的建立/切换、任务状态的终态、把图跑起来。
    逐文档的处理与断点提交在图内（见 app/graphs/ingest_graph.py），
    但断点格式与 M2 完全一致（`tasks.checkpoint["done"]`），前端轮询与既有测试不受影响。
    """

    def __init__(self, store: Store, ingest: IngestService, *, graph=None) -> None:
        self.store = store
        self.ingest = ingest
        self._graph = graph

    @property
    def graph(self):
        """摄取图（进程内编译一次；未注入时按需自建 —— 测试与脚本可只给 store/ingest）。"""
        if self._graph is None:
            from ..core.config import settings as app_settings
            from ..graphs.ingest_graph import build_ingest_graph
            from ..graphs.ingest_nodes import IngestDeps

            self._graph = build_ingest_graph(
                IngestDeps(store=self.store, ingest=self.ingest, settings=app_settings)
            )
        return self._graph

    async def activate_version(self, version: str, *, ratio: float = 1.0) -> None:
        """把版本切上线（灰度/全量逻辑见模块级同名函数的说明）。"""
        await activate_index_version(self.store, self.ingest, version, ratio=ratio)

    async def rebuild(self, task_id: str, *, ratio: float = 1.0) -> None:
        """重建全量语料到新版本（图编排：断点续跑 + 单文档失败隔离）。"""
        from ..core.config import settings as app_settings
        from ..graphs.ingest_graph import invoke_ingest, reindex_seed

        started = time.perf_counter()
        version = CURRENT_INDEX_VERSION
        task = await self.store.get_task(task_id)
        done: list[str] = sorted((task.checkpoint or {}).get("done", [])) if task else []

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
        await self._update(
            task_id,
            status="running",
            progress={
                "phase": "reindex",
                "total_docs": len(docs),
                "done_docs": len(done),
                "reused_vectors": 0,
                "embedded_chunks": 0,
                "failed": [],
            },
            checkpoint={"done": done},
        )

        await invoke_ingest(
            self.graph,
            reindex_seed(task_id=task_id, version=version, ratio=ratio, done=done),
            thread_id=task_id,
            doc_count=len(docs),
            settings=app_settings,
        )

        # 终态与耗时由服务层写（图只维护 progress/checkpoint，见 ingest_graph 的说明）
        after = await self.store.get_task(task_id)
        progress = dict(after.progress or {}) if after else {}
        progress["phase"] = "done"
        progress["latency_ms"] = int((time.perf_counter() - started) * 1000)
        await self._update(task_id, status="succeeded", progress=progress)

    async def _rebuild_doc(self, doc: DocumentRecord, version: str) -> tuple[int, int]:
        """重算单文档（解析 → 分块 → 复用/编码向量 → 写新版本）。返回 (复用块数, 新编码块数)。"""
        return await rebuild_doc(self.store, self.ingest, doc, version)

    async def _update(self, task_id: str, **fields) -> None:
        await self.store.update_task(task_id, **fields)


async def activate_index_version(
    store: Store, ingest: IngestService, version: str, *, ratio: float = 1.0
) -> None:
    """把版本切上线：全量 activate；ratio<1 走灰度（旧版本继续吃剩余流量）。

    灰度选择逻辑见 `select_index_version`；ratio=0 事实上等于回滚。
    模块级函数：Ingestion Graph 的 activate 节点直接调用（不经过服务类，避免图与服务互相持有）。
    """
    if ratio >= 1.0:
        # 全量切换：旧版本转 stale（保留数据，便于回滚与对比）
        await store.activate_version(version)
    else:
        # 灰度：新版本直接 ready 但只接 ratio 比例的流量；旧版本保持 ready
        current = await store.get_active_version()
        await store.upsert_version(
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
        # 显式设置流量比：upsert 在版本行已存在时**不更新 traffic_ratio**
        # （ON CONFLICT 的 SET 里没有这一列）—— 靠 upsert 传值会在"同版本重现灰度"时静默失效
        await store.set_traffic_ratio(version, ratio)
        if current is not None and current.version != version:
            await store.set_traffic_ratio(current.version, 1.0)

    # 索引变了 → 让检索侧的 BM25 缓存失效（同一进程内的回调）
    ingest.notify_index_changed(version)


async def rebuild_doc(
    store: Store, ingest: IngestService, doc: DocumentRecord, version: str
) -> tuple[int, int]:
    """重算单文档（解析 → 分块 → 复用/编码向量 → 写新版本）。返回 (复用块数, 新编码块数)。

    抽成模块级函数是为了让 **Ingestion Graph 的节点**直接复用它 —— 摄取图不能重写这段
    逻辑（分块参数、向量缓存键、replace_doc_chunks 的版本语义都在这里，重写必然漂移）。
    """
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
    await store.replace_doc_chunks(doc.doc_id, version, records)

    reused = embedded = 0
    if ingest.embedder.available:
        vectors, reused, embedded = await ingest.vectors_for_chunks(chunks)
        await store.put_vectors(
            version, [(c.chunk_id, vectors[c.content_hash]) for c in chunks]
        )
    return reused, embedded


def doc_id_for(name: str) -> str:
    """与摄取同一份 doc_id 口径（迁移脚本与测试共用，避免两处派生规则漂移）。"""
    return _doc_id_for(name)


def new_task_record(task_id: str, *, kind: str = "reindex") -> TaskRecord:
    now = _now()
    return TaskRecord(task_id=task_id, kind=kind, status="queued", created_at=now, updated_at=now)
