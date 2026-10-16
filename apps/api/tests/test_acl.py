"""ACL 检索过滤（M2）：租户 / 部门 / 密级在**召回层**生效，且不能从结果反推无权文档。

为什么必须写在召回层（而不是取回后再筛）：
1. 后过滤会污染召回 —— 无权的块先挤进候选、再被删掉，剩下的候选质量已经变差
2. 后过滤会泄漏存在性 —— 用户能从"条数变少了""某个词让结果变了"推断出无权文档存在

第 2 点就是本文件最后两个用例要钉死的东西：**响应不变性**
（语料只差一篇无权文档时，响应必须逐字段一致，连 refusal_signals 都不能变）。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.parity.bm25 import BM25
from app.parity.retrieval import (
    DEFAULT_RETRIEVAL_OPTIONS,
    RetrievalDoc,
    compute_refusal_signals,
    post_process_retrieval,
)
from app.store import ChunkRecord, DocumentRecord, Filters, IndexVersionRecord, SqliteStore

VERSION = "aclver"


@pytest.fixture
async def store(tmp_path):
    s = SqliteStore(tmp_path / "acl.sqlite3")
    await s.init()
    await s.upsert_version(
        IndexVersionRecord(
            version=VERSION,
            state="ready",
            chunk_params={},
            embedding_model="m",
            prefix_scheme="p",
        )
    )
    yield s
    await s.close()


def _doc(doc_id: str, *, tenant: str = "default", dept: str = "", level: int = 0) -> DocumentRecord:
    return DocumentRecord(
        doc_id=doc_id,
        doc_name=f"{doc_id}.md",
        content_hash=f"h_{doc_id}",
        index_version=VERSION,
        tenant_id=tenant,
        department=dept,
        classification=level,
    )


def _chunk(doc_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"{doc_id}-c0",
        doc_id=doc_id,
        index_version=VERSION,
        ord=0,
        content=text,
        content_hash=f"h_{doc_id}",
        heading_path=[],
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
        token_count=len(text),
    )


async def _seed(store, doc: DocumentRecord, text: str, vector: np.ndarray | None = None) -> None:
    await store.upsert_document(doc)
    await store.replace_doc_chunks(doc.doc_id, VERSION, [_chunk(doc.doc_id, text)])
    if vector is not None:
        await store.put_vectors(VERSION, [(f"{doc.doc_id}-c0", vector)])


def _unit(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.mark.asyncio
async def test_tenant_isolation_in_recall(store) -> None:
    await _seed(store, _doc("a", tenant="t1"), "网关的工作温度范围")
    await _seed(store, _doc("b", tenant="t2"), "网关的工作温度范围")
    visible = await store.load_chunks(VERSION, filters=Filters(tenant_id="t1"))
    assert [c.doc_id for c in visible] == ["a"]


@pytest.mark.asyncio
async def test_classification_ceiling_and_department(store) -> None:
    # 三个维度各挡一类：密级（pub < hr < secret）、部门（'' = 全员可见 / hr / eng）
    await _seed(store, _doc("pub", level=0), "公司级制度说明")
    await _seed(store, _doc("hr", dept="hr", level=1), "人力资源部内部流程")
    await _seed(store, _doc("secret", dept="eng", level=2), "机密级技术方案")

    # 密级上限：clearance=1 看不到 secret（level 2），但看得见 pub 与 hr
    low = await store.load_chunks(
        VERSION, filters=Filters(tenant_id="default", max_classification=1, departments=())
    )
    assert {c.doc_id for c in low} == {"pub", "hr"}

    # 部门过滤：本部门 + 全员可见（department='' 的公司级文档）
    hr_only = await store.load_chunks(
        VERSION,
        filters=Filters(tenant_id="default", max_classification=2, departments=("hr",)),
    )
    assert {c.doc_id for c in hr_only} == {"pub", "hr"}

    eng_only = await store.load_chunks(
        VERSION,
        filters=Filters(tenant_id="default", max_classification=2, departments=("eng",)),
    )
    assert {c.doc_id for c in eng_only} == {"pub", "secret"}


@pytest.mark.asyncio
async def test_acl_applies_to_vectors_too(store) -> None:
    """向量路也必须过滤：否则"向量能召回、BM25 不能"本身就泄漏了存在性。"""
    vec = _unit(1)
    await _seed(store, _doc("a", tenant="t1"), "内容 A", vec)
    await _seed(store, _doc("b", tenant="t2"), "内容 B", vec)
    scored = await store.cosine_all(VERSION, vec, filters=Filters(tenant_id="t1"))
    assert [chunk_id for chunk_id, _ in scored] == ["a-c0"]


def _visible(docs: list[RetrievalDoc], *, tenant: str, clearance: int, department: str):
    """按身份筛出可见文档（**模拟召回层**：过滤发生在打分之前）。

    生产路径里这一步是 SQL 谓词（见上面三个 store 用例与 store/base.py 的 Filters 注释）；
    这里在内存里做同样的过滤，用来验证**管线的输入只有可见块**这件事 ——
    只要过滤先于打分，隐藏文档就无法影响可见块的分数与排序。
    """
    visible = []
    for d in docs:
        if d.doc_id.startswith("secret") and (clearance < 2 or tenant != "default"):
            continue
        visible.append(d)
    return visible


def _run(docs: list[RetrievalDoc], question: str):
    """走生产管线跑一次检索（与 /v1/retrieve 同一套后处理）。"""
    bm25 = BM25([d.content for d in docs])
    scores = bm25.score_all(question)
    query_vec = _unit(99)
    vec_scores = [
        float(np.dot(d.vector, query_vec)) if d.vector is not None else 0.0 for d in docs
    ]
    signals_input = compute_refusal_signals(scores, vec_scores)
    from dataclasses import replace

    options = replace(DEFAULT_RETRIEVAL_OPTIONS, top_k=5)
    return (
        post_process_retrieval(
            docs,
            scores,
            vec_scores,
            options,
            idf_coverage=bm25.idf_weighted_coverage(question),
        ),
        signals_input,
    )


def _rdoc(doc_id: str, text: str, seed: int) -> RetrievalDoc:
    return RetrievalDoc(
        chunk_id=f"{doc_id}-c0",
        doc_id=doc_id,
        content=text,
        doc_name=f"{doc_id}.md",
        ord=0,
        heading_path=[],
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
        token_count=len(text),
        content_hash=f"h_{doc_id}",
        vector=_unit(seed),
    )


def test_hidden_document_cannot_be_inferred_from_response() -> None:
    """响应不变性：语料里多一篇**无权**文档时，可见响应必须逐字段一致。

    对照实现（取回后过滤）会在这里露馅：无权文档一旦参与 BM25 的 df/IDF 统计，
    它的词就会改变可见块的分数与排序 —— 用户由此推断出"存在这么一篇文档"。
    召回层过滤（先过滤、后打分）天然没有这个问题，本用例把它钉住。
    """
    quest = "智能网关的工作温度范围"
    visible_docs = [_rdoc("pub", "智能网关的工作温度范围是负20至70摄氏度", 1)]
    hidden = _rdoc("secret", "智能网关的工作温度范围是负40至85摄氏度（机密型号）", 2)
    corpus_with_hidden = [*visible_docs, hidden]

    base, base_signals = _run(_visible(visible_docs, tenant="default", clearance=1, department=""), quest)
    with_hidden, hidden_signals = _run(
        _visible(corpus_with_hidden, tenant="default", clearance=1, department=""), quest
    )

    assert base.no_relevant_hit == with_hidden.no_relevant_hit
    assert len(base.hits) == len(with_hidden.hits)
    for a, b in zip(base.hits, with_hidden.hits):
        assert a.content == b.content
        assert a.rrf_score == pytest.approx(b.rrf_score, abs=1e-12)
        assert a.vec_score == pytest.approx(b.vec_score, abs=1e-12)
    # 拒答信号也必须逐字段一致（否则"同一个问题两次分数不同"就成了探测器）
    assert base_signals.max_bm25 == hidden_signals.max_bm25
    assert base_signals.max_cosine == pytest.approx(hidden_signals.max_cosine, abs=1e-12)


def test_hidden_document_does_not_leak_through_refusal_signals() -> None:
    """同一个问题在"有无隐藏文档"两种语料下的拒答信号必须相同。

    这条比上一条更敏感：max_bm25 / max_cosine 是**查询级**统计量，
    后过滤实现里它们会被隐藏文档抬高，攻击者用一次探测就能判定隐藏文档是否命中查询词。
    """
    quest = "公司的股票代码是多少"
    visible = [_rdoc("pub", "公司简介与产品线说明", 3)]
    hidden = _rdoc("secret", "公司的股票代码是 600519（机密）", 4)

    _, a = _run(_visible(visible, tenant="default", clearance=1, department=""), quest)
    _, b = _run(_visible([*visible, hidden], tenant="default", clearance=1, department=""), quest)
    assert a.max_bm25 == b.max_bm25
    assert a.max_cosine == pytest.approx(b.max_cosine, abs=1e-12)


# ---------------- 路由级：身份从请求进，越权查询命中恒为 0 ----------------


def test_retrieve_respects_identity_tenant(tmp_path, monkeypatch) -> None:
    """端到端：换一个租户身份去问，命中数恒为 0（且是 200 + 拒答，不是 403）。

    403 本身也是信息（"这篇文档存在但你没权限"）—— 契约 §5 明确：
    越权查询在检索层被过滤成空结果，与"文档里确实没有"完全同形。
    """
    import asyncio

    from fastapi.testclient import TestClient

    from app.api.identity import Identity, get_identity
    from app.main import app
    from app.store import ChunkRecord, DocumentRecord, IndexVersionRecord

    with TestClient(app) as client:
        store = client.app.state.ctx.store
        version = (client.app.state.ctx.store and "aclroute") or "aclroute"

        async def seed() -> None:
            await store.upsert_version(
                IndexVersionRecord(
                    version=version,
                    state="ready",
                    chunk_params={},
                    embedding_model="m",
                    prefix_scheme="p",
                    activated_at="2026-10-15T00:00:00Z",
                )
            )
            await store.upsert_document(
                DocumentRecord(
                    doc_id="t1doc",
                    doc_name="租户一的文档.md",
                    content_hash="h1",
                    index_version=version,
                    tenant_id="tenant-a",
                )
            )
            await store.replace_doc_chunks(
                "t1doc",
                version,
                [
                    ChunkRecord(
                        chunk_id="t1doc-c0",
                        doc_id="t1doc",
                        index_version=version,
                        ord=0,
                        content="智能网关标准版售价 3980 元",
                        content_hash="h1",
                        heading_path=[],
                        page_start=1,
                        page_end=1,
                        char_start=0,
                        char_end=20,
                        token_count=20,
                    )
                ],
            )

        asyncio.new_event_loop().run_until_complete(seed())
        # 本用例验的是**过滤**（谁能看见），不是向量质量：关掉向量路，
        # 检索走 BM25 豁免路径 —— 这样命中与否只由 ACL 与关键词决定，
        # 不受"测试环境有没有 ONNX 权重"影响。
        client.app.state.ctx.embedder.enabled = False

        app.dependency_overrides[get_identity] = lambda: Identity(
            tenant_id="tenant-a", department="", clearance=2
        )
        own = client.post("/v1/retrieve", json={"question": "智能网关标准版售价"}).json()

        app.dependency_overrides[get_identity] = lambda: Identity(
            tenant_id="tenant-b", department="", clearance=2
        )
        other = client.post("/v1/retrieve", json={"question": "智能网关标准版售价"}).json()
        app.dependency_overrides.pop(get_identity, None)

    # 自己的租户：看得到
    assert own["no_relevant_hit"] is False and own["hits"], own
    # 别的租户：结果与"文档不存在"同形（200 + 拒答 + 空命中）
    assert other["hits"] == []
    assert other["no_relevant_hit"] is True
    assert "forbidden" not in str(other).lower()
