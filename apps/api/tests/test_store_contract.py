"""存储契约：SQLite 与 Postgres/pgvector 的行为必须一致。

同一份用例跑两种后端（Postgres 只在配了 DATABASE_URL 时跑，否则 skip）：
向量 float32 往返必须**逐位相等**（pgvector 若走文本 I/O 会丢精度，这条能抓出来），
索引版本生命周期、软删除的召回层效果、过滤条件、任务/评测状态都要一致。
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from app.store import (
    AnswerRecord,
    ChunkRecord,
    DocumentRecord,
    EvalRunRecord,
    Filters,
    IndexVersionRecord,
    SqliteStore,
    TaskRecord,
    TraceRecord,
)
from app.store.factory import make_store

VERSION = "0635b636"


@pytest.fixture(params=["sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "sqlite":
        sqlite = SqliteStore(tmp_path / "rag.sqlite3")
        yield sqlite
        return
    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn.startswith("postgres"):
        pytest.skip("未配置 DATABASE_URL，跳过 Postgres 契约用例")
    pg = make_store(dsn, tmp_path)
    yield pg


@pytest.fixture
async def ready_store(store):
    await store.init()
    await store.upsert_version(
        IndexVersionRecord(
            version=VERSION,
            state="building",
            chunk_params={"maxChunkTokens": 256},
            embedding_model="Xenova/bge-small-zh-v1.5",
            prefix_scheme="passage:none|query:x|verify:none",
        )
    )
    yield store
    await store.close()


def _chunk(
    ordinal: int, doc_id: str, seed: int, *, version: str = VERSION, tag: str = ""
) -> ChunkRecord:
    """构造一条块记录。

    chunk_id 全局唯一（真实切块每次都是新 UUID），所以换版本重切时要换 id ——
    复用同一个 id 会撞 chunk_id 主键，那是**测试夹具**的问题，不是生产路径的问题。
    """
    return ChunkRecord(
        chunk_id=f"{doc_id}-c{ordinal}{tag}",
        doc_id=doc_id,
        index_version=version,
        ord=ordinal,
        content=f"第 {ordinal} 块内容 {seed}",
        content_hash=f"{seed:08x}",
        heading_path=["第一章", None, f"第{ordinal}节"],
        page_start=1,
        page_end=2,
        char_start=ordinal * 10,
        char_end=ordinal * 10 + 8,
        token_count=5,
        doc_name=f"{doc_id}.md",
    )


async def _seed_document(store, doc_id: str, count: int = 3) -> None:
    await store.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            doc_name=f"{doc_id}.md",
            content_hash="deadbeef",
            index_version=VERSION,
            size_bytes=123,
        )
    )
    await store.replace_doc_chunks(
        doc_id, VERSION, [_chunk(i, doc_id, seed=ord(doc_id[0]) + i) for i in range(count)]
    )


@pytest.mark.asyncio
async def test_chunk_roundtrip_and_ordering(ready_store) -> None:
    await _seed_document(ready_store, "alpha")
    await _seed_document(ready_store, "beta")
    chunks = await ready_store.load_chunks(VERSION)
    assert [c.doc_id for c in chunks] == ["alpha"] * 3 + ["beta"] * 3
    assert [c.ord for c in chunks] == [0, 1, 2, 0, 1, 2]
    first = chunks[0]
    assert first.heading_path == ["第一章", None, "第0节"], "标题路径含 None（层级跳跃）必须原样往返"
    assert first.doc_name == "alpha.md", "doc_name 由 documents 表 join 而来"


@pytest.mark.asyncio
async def test_vector_roundtrip_is_bit_exact(ready_store) -> None:
    await _seed_document(ready_store, "alpha")
    # 单位向量才是"余弦 = 点积"的前提；这里用归一化后的向量做往返与检索断言
    raw = np.linspace(-0.9876543, 0.9876543, 512, dtype=np.float32)
    vec = (raw / np.linalg.norm(raw)).astype(np.float32)
    await ready_store.put_vectors(VERSION, [("alpha-c0", vec)])
    loaded = await ready_store.get_vector("alpha-c0")
    assert loaded is not None
    assert np.array_equal(loaded, vec), "向量往返必须逐位相等（pgvector 文本 I/O 会丢精度）"

    scores = await ready_store.cosine_all(VERSION, vec)
    assert len(scores) == 1
    assert scores[0][0] == "alpha-c0"
    assert scores[0][1] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.asyncio
async def test_filters_apply_in_query(ready_store) -> None:
    await _seed_document(ready_store, "alpha")
    await _seed_document(ready_store, "beta")
    chunk = (await ready_store.load_chunks(VERSION))[0]
    vec = np.ones(512, dtype=np.float32)
    await ready_store.put_vectors(
        VERSION, [(c.chunk_id, vec) for c in await ready_store.load_chunks(VERSION)]
    )
    only_alpha = await ready_store.load_chunks(VERSION, filters=Filters(doc_ids=["alpha"]))
    assert {c.doc_id for c in only_alpha} == {"alpha"}
    scored = await ready_store.cosine_all(VERSION, vec, filters=Filters(doc_ids=["alpha"]))
    assert {chunk_id.split("-")[0] for chunk_id, _ in scored} == {"alpha"}
    assert chunk.chunk_id in {chunk_id for chunk_id, _ in scored}


@pytest.mark.asyncio
async def test_soft_delete_removes_from_recall(ready_store) -> None:
    await _seed_document(ready_store, "alpha")
    vec = np.ones(512, dtype=np.float32)
    await ready_store.put_vectors(VERSION, [("alpha-c0", vec)])

    await ready_store.soft_delete_doc("alpha")
    assert await ready_store.load_chunks(VERSION) == []
    assert await ready_store.cosine_all(VERSION, vec) == []
    # 软删除：文档记录仍在（可恢复/可审计），但默认列表不显示
    assert await ready_store.list_documents() == []
    assert len(await ready_store.list_documents(include_deleted=True)) == 1


@pytest.mark.asyncio
async def test_index_version_lifecycle(ready_store) -> None:
    assert await ready_store.get_active_version() is None
    await ready_store.activate_version(VERSION)
    active = await ready_store.get_active_version()
    assert active is not None and active.version == VERSION and active.state == "ready"

    # 新版本激活后，旧版本转 stale（数据保留，便于回滚与灰度）
    await ready_store.upsert_version(
        IndexVersionRecord(
            version="newversion",
            state="building",
            chunk_params={},
            embedding_model="m",
            prefix_scheme="p",
        )
    )
    await ready_store.activate_version("newversion")
    active = await ready_store.get_active_version()
    assert active.version == "newversion"
    stale = await ready_store.load_chunks(VERSION)
    assert stale == [], "版本切换后旧版本的块不参与检索"


@pytest.mark.asyncio
async def test_answer_trace_and_eval_roundtrip(ready_store) -> None:
    await ready_store.save_answer(
        AnswerRecord(
            answer_id="ans_1",
            trace_id="trc_1",
            question="q",
            answer="a[1]",
            verdict="supported",
            citations=[{"index": 1}],
            refusal=None,
            verification={"source": "lexical", "sentences": []},
            evidence_snapshot={"passages": []},
            cost={"latency_ms": 1},
            config={},
        )
    )
    answer = await ready_store.get_answer("ans_1")
    assert answer is not None and answer.citations == [{"index": 1}]
    assert await ready_store.get_answer("ans_missing") is None

    await ready_store.save_trace(
        TraceRecord(
            trace_id="trc_1",
            answer_id="ans_1",
            question="q",
            nodes=[{"seq": 1, "name": "retrieve"}],
            final_verdict="supported",
            evidence_snapshot={"passages": []},
            cost={"latency_ms": 1},
        )
    )
    trace = await ready_store.get_trace("trc_1")
    assert trace is not None and trace.nodes[0]["name"] == "retrieve"

    await ready_store.save_eval_run(
        EvalRunRecord(
            run_id="run_1",
            status="queued",
            suite="retrieval",
            golden_version="fnv1a32:abc",
            config={"top_k": 5},
        )
    )
    await ready_store.update_eval_run(
        "run_1", status="succeeded", gates_passed=True, metrics={"recallAt3": 0.7255}
    )
    run = await ready_store.get_eval_run("run_1")
    assert run is not None and run.status == "succeeded" and run.gates_passed is True
    assert run.metrics["recallAt3"] == 0.7255

    with pytest.raises(ValueError):
        await ready_store.update_eval_run("run_1", unknown_field=1)


@pytest.mark.asyncio
async def test_task_state_table(ready_store) -> None:
    await ready_store.save_task(
        TaskRecord(task_id="tsk_1", kind="ingest", status="queued")
    )
    await ready_store.update_task(
        "tsk_1", status="running", progress={"done": 1, "total": 3}
    )
    task = await ready_store.get_task("tsk_1")
    assert task is not None and task.status == "running" and task.progress["total"] == 3
    with pytest.raises(ValueError):
        await ready_store.update_task("tsk_1", nope=1)


# ---------------- 增量索引与灰度（M2） ----------------


@pytest.mark.asyncio
async def test_chunk_replacement_is_version_scoped(ready_store) -> None:
    """灰度重建期间，写新版本**不能**动到在服务的旧版本。"""
    await _seed_document(ready_store, "alpha")
    await ready_store.upsert_version(
        IndexVersionRecord(
            version="v2",
            state="building",
            chunk_params={},
            embedding_model="m",
            prefix_scheme="p",
        )
    )
    await ready_store.replace_doc_chunks(
        "alpha", "v2", [_chunk(0, "alpha", seed=999, version="v2", tag="-v2")]
    )
    assert len(await ready_store.load_chunks(VERSION)) == 3, "旧版本的块必须原样保留"
    assert len(await ready_store.load_chunks("v2")) == 1


@pytest.mark.asyncio
async def test_vector_cache_reuse_keyed_by_hash_and_model(ready_store) -> None:
    vec = np.linspace(-1, 1, 512, dtype=np.float32)
    await ready_store.put_cached_vectors(
        [("hash-a", vec)], embedding_model="m1", prefix_scheme="p1"
    )
    hit = await ready_store.load_cached_vectors(
        ["hash-a", "hash-missing"], embedding_model="m1", prefix_scheme="p1"
    )
    assert list(hit) == ["hash-a"]
    assert np.array_equal(hit["hash-a"], vec), "缓存往返必须逐位相等"

    # 换模型或换前缀 → 缓存自动失效（不靠手工清理）
    assert (
        await ready_store.load_cached_vectors(
            ["hash-a"], embedding_model="m2", prefix_scheme="p1"
        )
        == {}
    )
    assert (
        await ready_store.load_cached_vectors(
            ["hash-a"], embedding_model="m1", prefix_scheme="p2"
        )
        == {}
    )


@pytest.mark.asyncio
async def test_gc_only_removes_non_ready_versions(ready_store) -> None:
    await _seed_document(ready_store, "alpha")
    await ready_store.activate_version(VERSION)
    with pytest.raises(ValueError):
        await ready_store.gc_version(VERSION)  # ready 版本还在服务，拒绝 GC

    await ready_store.upsert_version(
        IndexVersionRecord(
            version="old",
            state="building",
            chunk_params={},
            embedding_model="m",
            prefix_scheme="p",
        )
    )
    await ready_store.replace_doc_chunks(
        "alpha", "old", [_chunk(0, "alpha", seed=7, version="old", tag="-old")]
    )
    await ready_store.activate_version("old")  # VERSION 转 stale
    removed = await ready_store.gc_version(VERSION)
    assert removed == 3
    assert await ready_store.load_chunks(VERSION) == []


# ---------------- ACL：召回层过滤（两种后端必须一致） ----------------


@pytest.mark.asyncio
async def test_acl_filter_applies_in_recall(ready_store) -> None:
    """租户/密级/部门在**召回层**生效：过滤与打分同一条 SQL，不是取回后再筛。"""
    await ready_store.upsert_document(
        DocumentRecord(
            doc_id="v",
            doc_name="v.md",
            content_hash="hv",
            index_version=VERSION,
            tenant_id="t1",
            department="",
            classification=0,
        )
    )
    await ready_store.upsert_document(
        DocumentRecord(
            doc_id="s",
            doc_name="s.md",
            content_hash="hs",
            index_version=VERSION,
            tenant_id="t1",
            department="eng",
            classification=2,
        )
    )
    await ready_store.replace_doc_chunks(
        "v", VERSION, [_chunk(0, "v", seed=1)]
    )
    await ready_store.replace_doc_chunks(
        "s", VERSION, [_chunk(0, "s", seed=2)]
    )
    await ready_store.upsert_document(
        DocumentRecord(
            doc_id="t2",
            doc_name="t2.md",
            content_hash="h2",
            index_version=VERSION,
            tenant_id="t2",
        )
    )
    await ready_store.replace_doc_chunks("t2", VERSION, [_chunk(0, "t2", seed=3)])

    vec = np.ones(512, dtype=np.float32)
    await ready_store.put_vectors(
        VERSION, [(c.chunk_id, vec) for c in await ready_store.load_chunks(VERSION)]
    )

    # 低密级 + 无部门限制：看不到机密文档
    low = await ready_store.load_chunks(
        VERSION, filters=Filters(tenant_id="t1", max_classification=1)
    )
    assert {c.doc_id for c in low} == {"v"}

    # 跨租户：命中数与向量检索都为空
    other = await ready_store.load_chunks(VERSION, filters=Filters(tenant_id="t9"))
    assert other == []
    assert await ready_store.cosine_all(VERSION, vec, filters=Filters(tenant_id="t9")) == []

    # 部门过滤：本部门 + 全员可见
    eng = await ready_store.load_chunks(
        VERSION,
        filters=Filters(tenant_id="t1", max_classification=2, departments=("eng",)),
    )
    assert {c.doc_id for c in eng} == {"v", "s"}
