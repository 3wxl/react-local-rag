"""摄取服务：上传 → 解析 → 结构化分块 → embedding → 入索引。

与浏览器端 App.tsx 的 processFile 同序（parseFile → createTextChunkStream.push → finish →
embedPassages → saveVectorIndex），只是目标从 IndexedDB 换成服务端存储。

几个刻意的设计：
- **索引版本**：`CURRENT_INDEX_VERSION` 与 TS 同值；首次摄取建版本行，成功后才 activate。
  换分块参数/换模型 → 版本变化 → 旧向量不可比 → 提示重建（灰度重建见 reindex_service）。
- **失败隔离到单文档**：一个文件解析失败不影响同批次其它文件（对应路线图"失败隔离"机制）。
- **文档身份稳定**：doc_id 只由文件名派生（**不含内容哈希**）—— 见 `_doc_id_for` 的注释，
  含哈希会让"改一次内容"派生出第二个文档，替换路径永远走不到。
- **按块增量**：块级别不做"块边界对齐"，而是按 **contentHash 复用向量**
  （`chunk_vector_cache`）：相同文本 + 同模型同前缀 ⇒ 向量逐位相同。
  这样改一段文字只重算那几块，换分块参数也不会重算未变文本 ——
  既拿到了增量的收益，又避开了"块边界随参数变化导致错位"的坑。
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
    #: 增量指标：命中向量缓存（复用）与真正重算的块数 —— 改一段只该重算一小撮
    reused_chunks: int = 0
    embedded_chunks: int = 0


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

    async def ingest_path(
        self,
        path: Path,
        *,
        doc_name: str | None = None,
        tenant_id: str = "default",
        department: str = "",
        classification: int = 0,
    ) -> IngestResult:
        """摄取一个文件（解析 → 分块 → 向量化 → 入库）。

        ACL 三元组随文档一起落库：**召回层**过滤要靠它（不是取回后再筛）。
        默认值等价于"公开的公司级文档"，与本机单租户场景一致。
        """
        started = time.perf_counter()
        name = doc_name or path.name
        size_bytes = path.stat().st_size
        if size_bytes > MAX_FILE_MB * 1024 * 1024:
            raise ValueError(f"文件超过 {MAX_FILE_MB}MB 上限：{name}")

        blocks = blocks_for_file(path, filename=name)
        content_hash = fnv1a32("\n".join(b.text for b in blocks))
        doc_id = _doc_id_for(name)
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
                tenant_id=tenant_id,
                department=department,
                classification=int(classification),
            )
        )
        await self.store.replace_doc_chunks(doc_id, version, records)

        reused = 0
        embedded = 0
        if self.embedder.available:
            vectors_by_hash, reused, embedded = await self.vectors_for_chunks(chunks)
            await self.store.put_vectors(
                version,
                [(c.chunk_id, vectors_by_hash[c.content_hash]) for c in chunks],
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
            reused_chunks=reused,
            embedded_chunks=embedded,
        )

    async def vectors_for_chunks(self, chunks) -> tuple[dict, int, int]:
        """按 contentHash 复用向量：只对缓存里没有的块做向量化。

        复用的正确性来自确定性：相同文本 + 同 embedding_model + 同 prefix_scheme
        ⇒ 同一个模型对同一输入必然给出同一个向量。三者的组合是缓存键，
        换模型/换前缀会让缓存**自动失效**（查不到），不需要手工清理。
        同一文档里重复的块（页眉页脚之类的重复段落）也只编码一次。
        """
        hashes: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            if chunk.content_hash not in seen:
                seen.add(chunk.content_hash)
                hashes.append(chunk.content_hash)

        cached = await self.store.load_cached_vectors(
            hashes,
            embedding_model=EMBEDDING_MODEL_NAME,
            prefix_scheme=EMBEDDING_PREFIX_SCHEME,
        )
        missing = [h for h in hashes if h not in cached]
        vectors_by_hash = dict(cached)
        if missing:
            # 文本按哈希取回：同一哈希的块文本必然相同，用第一处出现的文本即可
            text_by_hash: dict[str, str] = {}
            for chunk in chunks:
                text_by_hash.setdefault(chunk.content_hash, chunk.content)
            fresh = self.embedder.embed_passages([text_by_hash[h] for h in missing])
            rows = [(h, fresh[i]) for i, h in enumerate(missing)]
            await self.store.put_cached_vectors(
                rows,
                embedding_model=EMBEDDING_MODEL_NAME,
                prefix_scheme=EMBEDDING_PREFIX_SCHEME,
            )
            vectors_by_hash.update(dict(rows))
        return vectors_by_hash, len(cached), len(missing)

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

    def notify_index_changed(self, version: str | None = None) -> None:
        """索引内容变化后通知订阅方（检索侧据此让 BM25 缓存失效）。"""
        if self._on_index_changed is not None:
            self._on_index_changed(version)

    async def delete_document(self, doc_id: str) -> None:
        """软删除（向量同步从检索面移除，见 store.soft_delete_doc）。"""
        await self.store.soft_delete_doc(doc_id)
        if self._on_index_changed is not None:
            self._on_index_changed()


def _doc_id_for(name: str) -> str:
    """doc_id 由**文件名**派生（不含内容哈希）：同一个文件的不同版本 = 同一份文档。

    这里踩过坑：旧实现是 `uuid5(f"{name}:{content_hash}")`，把内容哈希并进了身份。
    后果是文档一改内容就派生出**新的 doc_id**，"同名替换"分支永远走不到 ——
    知识库里会同时存在同一文档的多个版本，旧版本继续参与检索。
    内容哈希的正确位置是**文档版本属性**（documents.content_hash），不是主键。
    """
    return "doc_" + uuid.uuid5(uuid.NAMESPACE_URL, f"rag-doc:{name}").hex[:16]
