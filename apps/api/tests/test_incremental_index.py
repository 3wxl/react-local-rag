"""增量索引与灰度重建（M2 验收：「改文档后索引自动增量重建」）。

用**计数假 embedder** 证明"增量"不是口号：直接数模型被调用了多少次。
真实的 bge 编码器在这里没有意义 —— 要验的是"哪些块被送去编码"，不是编码质量。

覆盖三类场景：
1. 同内容重传：一次模型都不调（整篇哈希命中）
2. 改一段：同一 doc_id 下只重算变化的那几块（其余命中 contentHash 缓存）
3. 全量重建（换分块参数/换模型的路径）：未变文本全部复用，只补编码新块
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from app.services.ingest_service import IngestService
from app.services.reindex_service import ReindexService, select_index_version
from app.store import IndexVersionRecord, SqliteStore, TaskRecord

DIM = 512


class CountingEmbedder:
    """确定性假编码器：同一文本必得同一向量（这是"可复用"的前提）。"""

    available = True

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def embed_passages(self, texts: list[str]) -> list[np.ndarray]:
        out = []
        for text in texts:
            self.encoded.append(text)
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(DIM).astype(np.float32)
            out.append(vec / np.linalg.norm(vec))
        return out

    @property
    def encoded_count(self) -> int:
        return len(self.encoded)


@pytest.fixture
async def env(tmp_path: Path):
    store = SqliteStore(tmp_path / "rag.sqlite3")
    await store.init()
    embedder = CountingEmbedder()
    ingest = IngestService(store, embedder, data_dir=tmp_path)
    # 文档要够长才会切成多块：切块器 minChunkChars=100 / maxChunkChars=600，
    # 太短的文档整篇就是一块，验不出"只重算变化的那块"
    for i in range(1, 7):
        paragraphs = "\n\n".join(
            f"第 {j} 段：这是文档 {i} 的第 {j} 段内容，"
            + "用来把文档撑到多块的长度，确保切块器不会把整篇合成一块。" * 2
            for j in range(1, 31)
        )
        (tmp_path / f"doc{i}.md").write_text(f"# 文档 {i}\n\n{paragraphs}\n", encoding="utf-8")
    yield store, ingest, embedder, tmp_path
    await store.close()


async def _ingest_all(ingest: IngestService, tmp_path: Path) -> list:
    return [await ingest.ingest_path(tmp_path / f"doc{i}.md") for i in range(1, 7)]


@pytest.mark.asyncio
async def test_same_content_reupload_costs_zero_embeddings(env) -> None:
    store, ingest, embedder, tmp_path = env
    first = await _ingest_all(ingest, tmp_path)
    after_first = embedder.encoded_count
    assert after_first > 0

    again = await _ingest_all(ingest, tmp_path)
    assert all(r.deduped for r in again), "同名同内容必须走整篇去重"
    assert embedder.encoded_count == after_first, "重传不应该再调模型"
    assert {r.doc_id for r in first} == {r.doc_id for r in again}


@pytest.mark.asyncio
async def test_edit_one_paragraph_only_reembeds_changed_chunks(env) -> None:
    store, ingest, embedder, tmp_path = env
    results = await _ingest_all(ingest, tmp_path)
    target = tmp_path / "doc1.md"
    doc_id = results[0].doc_id
    before = embedder.encoded_count

    # 只改一段，且**保持长度不变** —— 这很重要：
    # 改动段落长度会移动后续块边界，导致块文本整片变化（缓存是按文本启用用了，但文本确实变了）。
    # 这条测试要验的是"增量机制生效"，不是"任意编辑都只重算一块"。
    text = target.read_text(encoding="utf-8")
    target.write_text(
        text.replace(
            "第 2 段：这是文档 1 的第 2 段内容", "第 2 段：这是文档 1 的第 2 段改动"
        ),
        encoding="utf-8",
    )
    result = await ingest.ingest_path(target)

    assert result.doc_id == doc_id, "同名文档必须保持同一个 doc_id（改内容不是新增文档）"
    assert not result.deduped
    # 关键断言：只重算了变化的那一块，其余走了 contentHash 缓存
    assert embedder.encoded_count - before < result.chunk_count, (
        "改一段却把所有块重算了：增量没生效"
    )
    assert result.embedded_chunks <= 1
    assert result.reused_chunks >= result.chunk_count - 1

    # 旧版本被替换：同一文档在库里只有一份（不会堆出第二个版本）
    version = (await store.get_active_version()).version
    chunks = await store.load_chunks(version)
    assert len([c for c in chunks if c.doc_id == doc_id]) == result.chunk_count


@pytest.mark.asyncio
async def test_reindex_reuses_all_unchanged_vectors(env) -> None:
    store, ingest, embedder, tmp_path = env
    await _ingest_all(ingest, tmp_path)

    task = TaskRecord(task_id="tsk_reindex", kind="reindex", status="queued")
    await store.save_task(task)
    before = embedder.encoded_count
    reindex = ReindexService(store, ingest)
    await reindex.rebuild("tsk_reindex")

    assert embedder.encoded_count == before, "语料没变：重建应该 100% 命中向量缓存"
    done = await store.get_task("tsk_reindex")
    assert done is not None and done.status == "succeeded"
    assert done.progress["embedded_chunks"] == 0
    assert done.progress["done_docs"] == 6


@pytest.mark.asyncio
async def test_reindex_is_resumable_from_checkpoint(env) -> None:
    store, ingest, embedder, tmp_path = env
    await _ingest_all(ingest, tmp_path)
    # 伪造一个"上次跑到一半"的任务：已完成 3 篇
    done_ids = sorted(d.doc_id for d in await store.list_documents())[:3]
    await store.save_task(
        TaskRecord(
            task_id="tsk_resume",
            kind="reindex",
            status="interrupted",
            checkpoint={"done": done_ids},
        )
    )
    reindex = ReindexService(store, ingest)
    await reindex.rebuild("tsk_resume")
    task = await store.get_task("tsk_resume")
    assert task is not None
    assert task.progress["done_docs"] == 6, "断点续跑：已完成的文档被跳过，剩下的补上"


def test_canary_version_selection_is_deterministic() -> None:
    """灰度分桶必须确定性：同一 query_id 永远落到同一版本。"""
    stable = IndexVersionRecord("v1", "ready", {}, "m", "p", activated_at="2026-10-01")
    canary = IndexVersionRecord("v2", "ready", {}, "m", "p", activated_at="2026-10-02", traffic_ratio=0.2)
    versions = [canary, stable]

    picks = [select_index_version(versions, f"qry_{i}") for i in range(200)]
    again = [select_index_version(versions, f"qry_{i}") for i in range(200)]
    assert [p.version for p in picks] == [p.version for p in again], "同一 id 两次选择必须一致"

    share = sum(1 for p in picks if p.version == "v2") / len(picks)
    assert 0.1 < share < 0.3, f"20% 灰度的实际占比 {share} 偏离太多"

    # ratio=0 等价回滚；ratio=1 全量
    canary.traffic_ratio = 0.0
    assert all(select_index_version(versions, f"qry_{i}").version == "v1" for i in range(50))
    canary.traffic_ratio = 1.0
    assert all(select_index_version(versions, f"qry_{i}").version == "v2" for i in range(50))


@pytest.mark.asyncio
async def test_doc_id_is_stable_across_content_changes(env) -> None:
    """doc_id 只由文件名派生：改内容不是"新文档"，而是同一份文档的新版本。

    旧口径 `uuid5(name:content_hash)` 会让每次改内容都派生出新 doc_id，
    知识库里堆出同一文档的多个版本，旧版本继续被检索到。
    """
    from app.services.ingest_service import _doc_id_for
    from app.services.reindex_service import doc_id_for

    store, ingest, embedder, tmp_path = env
    target = tmp_path / "doc1.md"
    first = await ingest.ingest_path(target)
    target.write_text(target.read_text(encoding="utf-8") + "\n\n新增一段。\n", encoding="utf-8")
    second = await ingest.ingest_path(target)

    assert first.doc_id == second.doc_id
    assert first.doc_id == _doc_id_for("doc1.md") == doc_id_for("doc1.md")
    docs = await store.list_documents()
    assert len([d for d in docs if d.doc_name == "doc1.md"]) == 1, "同名文档只能有一份"
