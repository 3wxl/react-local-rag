"""摄取服务：上传 → 解析 → 结构化分块 → embedding → 入索引。

与浏览器端 App.tsx 的 processFile 同序（parseFile → createTextChunkStream.push → finish →
embedPassages → saveVectorIndex），只是目标从 IndexedDB 换成服务端存储。

几个刻意的设计：
- **索引版本**：`CURRENT_INDEX_VERSION` 与 TS 同值；首次摄取建版本行，成功后才 activate。
  换分块参数/换模型 → 版本变化 → 旧向量不可比 → 提示重建（灰度重建的挂点，M2 细化）。
- **失败隔离到单文档**：一个文件解析失败不影响同批次其它文件（对应路线图"失败隔离"机制）。
- **去重**：同一 doc_id 重新上传按整篇替换（旧块与旧向量一并删除），不做"按块增量"，
  因为块边界随分块参数变化，按块增量在换参数时必然错乱。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..embedding.bge import BgeOnnxEmbedder
from ..parity.chunk_params import (
    CURRENT_INDEX_VERSION,
    DEFAULT_CHUNK_OPTIONS,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PREFIX_SCHEME,
)
from ..parity.chunker import TextChunkStream
from ..parity.fnv import fnv1a32
from ..parity.parsers import blocks_for_file
from ..store import ChunkRecord, DocumentRecord, IndexVersionRecord, Store

#: 单个文件上限（MB）：与浏览器端 MAX_FILE_MB 一致，超限直接拒绝而不是悄悄截断
MAX_FILE_MB = 50


@dataclass
class IngestResult:
    doc_id: str
    doc_name: str
    chunk_count: int
    index_version: str
    latency_ms: int
    #: 已存在同内容文档（按整篇哈希判定）：跳过重复向量化
    deduped: bool = False


class IngestService:
    """摄取服务（进程内同步执行；HTTP 层用后台任务包装成 202 + 轮询）。"""

    def __init__(
        self,
        store: Store,
        embedder: BgeOnnxEmbedder,
        *,
        data_dir: Path,
        on_index_changed=None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.data_dir = Path(data_dir)
        self._on_index_changed = on_index_changed

    # ---------------- 索引版本 ----------------

    async def ensure_index_version(self) -> str:
        """确保当前索引版本存在（building → ready 由 activate 完成）。"""
        version = CURRENT_INDEX_VERSION
        existing = await self.store.get_active_version()
        if existing is not None and existing.version == version:
            return version
        await self.store.upsert_version(
            IndexVersionRecord(
                version=version,
                state="building",
                chunk_params={
                    "maxChunkTokens": DEFAULT_CHUNK_OPTIONS.max_chunk_tokens,
                    "minChunkChars": DEFAULT_CHUNK_OPTIONS.min_chunk_chars,
                    "maxChunkChars": DEFAULT_CHUNK_OPTIONS.max_chunk_chars,
                    "sentenceOverlap": DEFAULT_CHUNK_OPTIONS.sentence_overlap,
                },
                embedding_model=EMBEDDING_MODEL_NAME,
                prefix_scheme=EMBEDDING_PREFIX_SCHEME,
            )
        )
        return version

    # ---------------- 摄取 ----------------

    async def ingest_path(self, path: Path, *, doc_name: str | None = None) -> IngestResult:
        """摄取一个文件（解析 → 分块 → 向量化 → 入库）。"""
        started = time.perf_counter()
        name = doc_name or path.name
        size_bytes = path.stat().st_size
        if size_bytes > MAX_FILE_MB * 1024 * 1024:
            raise ValueError(f"文件超过 {MAX_FILE_MB}MB 上限：{name}")

        blocks = blocks_for_file(path, filename=name)
        content_hash = fnv1a32("\n".join(b.text for b in blocks))
        doc_id = _doc_id_for(name, content_hash)
        version = await self.ensure_index_version()

        # 同一文档同名同内容重复上传：直接复用已有块（省一次向量化）
        existing = {d.doc_id: d for d in await self.store.list_documents()}
        previous = existing.get(doc_id)
        if previous is not None and previous.content_hash == content_hash:
            chunks = await self.store.load_chunks(version, filters=None)
            own = [c for c in chunks if c.doc_id == doc_id]
            if own:
                return IngestResult(
                    doc_id=doc_id,
                    doc_name=name,
                    chunk_count=len(own),
                    index_version=version,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    deduped=True,
                )

        stream = TextChunkStream(doc_id, name, DEFAULT_CHUNK_OPTIONS)
        for block in blocks:
            stream.push(block)
        chunks = stream.finish()
        if not chunks:
            raise ValueError(f"未切出任何分块（可能是空文档或纯标题）：{name}")

        records = [
            ChunkRecord(
                chunk_id=c.chunk_id,
                doc_id=doc_id,
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
                doc_name=name,
            )
            for c in chunks
        ]

        await self.store.upsert_document(
            DocumentRecord(
                doc_id=doc_id,
                doc_name=name,
                content_hash=content_hash,
                index_version=version,
                size_bytes=size_bytes,
                raw_path=str(path),
            )
        )
        await self.store.replace_doc_chunks(doc_id, records)

        if self.embedder.available:
            vectors = self.embedder.embed_passages([c.content for c in chunks])
            await self.store.put_vectors(
                version, [(c.chunk_id, vectors[i]) for i, c in enumerate(chunks)]
            )

        await self.store.activate_version(version)
        if self._on_index_changed is not None:
            self._on_index_changed(version)

        return IngestResult(
            doc_id=doc_id,
            doc_name=name,
            chunk_count=len(chunks),
            index_version=version,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def ingest_directory(self, directory: Path) -> list[IngestResult | Exception]:
        """批量摄取（目录下 .md/.txt/.docx/.pdf）。

        单个文件失败**不中断整批**：返回列表里那一项是异常对象，
        调用方（任务运行器）把它写进 progress.failed，对应"失败隔离到单文档"。
        """
        results: list[IngestResult | Exception] = []
        suffixes = {".md", ".markdown", ".txt", ".docx", ".pdf"}
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in suffixes:
                continue
            try:
                results.append(await self.ingest_path(path))
            except Exception as err:  # noqa: BLE001 —— 失败隔离：记录并继续
                results.append(err)
        return results

    async def delete_document(self, doc_id: str) -> None:
        """软删除（向量同步从检索面移除，见 store.soft_delete_doc）。"""
        await self.store.soft_delete_doc(doc_id)
        if self._on_index_changed is not None:
            self._on_index_changed()


def _doc_id_for(name: str, content_hash: str) -> str:
    """doc_id 由「文件名 + 内容哈希」派生：同名同内容 → 同一个 doc（天然幂等）。

    不随机生成 id：重复上传要能覆盖而不是堆出第二份文档。
    """
    return "doc_" + uuid.uuid5(uuid.NAMESPACE_URL, f"{name}:{content_hash}").hex[:16]
