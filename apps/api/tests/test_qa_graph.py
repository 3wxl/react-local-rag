"""QA 图（M3）：把直线流程换成图之后，轨迹与契约不变量必须原样成立。

断言的重点不是"图能跑"，而是：
1. **默认路径的轨迹与 M2 逐行一致**（retrieve → gate → generate → verify）；
2. 拒答是两条一等分支（检索侧粗拒 / 答案侧细拒），不是异常路径；
3. 四条契约不变量的图路径验证；
4. State 保持 JSON 原生（检查点前提）且同题两次确定性一致。

**为什么要自建 store 环境**：整条测试链共享一个 DATA_DIR，别的用例会软删文档、
灌入合成语料，检索统计（IDF / 拒答门限）随之漂移 —— 依赖共享语料断言"这题不该拒答"
会随文件顺序红绿闪烁（本文件第一版就踩了这个坑）。所以图行为用独立 tmp 目录 + 真实摄取，
HTTP 只留"路由确实接到了图上"这类与语料状态无关的断言。
"""

from __future__ import annotations

import json
import re
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from .parity_utils import EVAL_CORPUS_DIR

CITATION_RE = re.compile(r"\[(\d+(?:\s*[,，]\s*\d+)*)\]")

POSITIVE_Q = "智能网关标准版的售价是多少？"
NEGATIVE_Q = "公司的股票代码是多少？"
CORPUS_DOCS = ["产品手册.md", "员工手册.md", "会议纪要.md"]


# ---------------- 自建环境（独立数据目录 + 真实摄取） ----------------


@pytest.fixture
async def qa_env(tmp_path):
    from app.core.config import Settings
    from app.embedding.bge import BgeOnnxEmbedder
    from app.generation.fake import FakeProvider
    from app.graphs.qa_graph import build_qa_graph
    from app.graphs.qa_nodes import QaDeps
    from app.services.answer_service import AnswerService
    from app.services.ingest_service import IngestService
    from app.services.retrieval_service import RetrievalService
    from app.store import make_store

    settings = Settings(data_dir=tmp_path)
    store = make_store("", tmp_path)
    await store.init()
    # 自建实例（不用 get_embedder 的进程内单例）：单例会被别的用例改 enabled，
    # 本环境必须与测试顺序无关。权重缺失时 available=False，检索退化为 BM25-only。
    embedder = BgeOnnxEmbedder(settings.bge_model_dir)
    provider = FakeProvider()
    retrieval = RetrievalService(store, embedder)
    ingest = IngestService(
        store,
        embedder,
        data_dir=tmp_path,
        on_index_changed=lambda version=None: retrieval.invalidate(version),
    )
    for name in CORPUS_DOCS:
        await ingest.ingest_path(EVAL_CORPUS_DIR / name, doc_name=name)

    deps = QaDeps(
        store=store,
        retrieval=retrieval,
        embedder=embedder,
        provider=provider,
        settings=settings,
    )
    graph = build_qa_graph(deps, None)
    version = (await store.list_ready_versions())[0].version
    try:
        yield SimpleNamespace(
            settings=settings, store=store, deps=deps, graph=graph, version=version
        )
    finally:
        await store.close()


async def _answer(env, question: str, *, provider=None):
    import dataclasses

    from app.graphs.qa_graph import build_qa_graph
    from app.services.answer_service import AnswerService
    from app.store import Filters

    deps = env.deps
    if provider is not None:
        deps = dataclasses.replace(deps, provider=provider)
    graph = env.graph if provider is None else build_qa_graph(deps, None)
    service = AnswerService(graph, settings=env.settings)
    return await service.answer(
        question, version=env.version, top_k=3, filters=Filters(tenant_id="default")
    )


async def test_success_path_trace_is_the_legacy_flow(qa_env) -> None:
    """成功路径：轨迹仍是 retrieve → gate → generate → verify（无 rewrite 行）。"""
    outcome = await _answer(qa_env, POSITIVE_Q)

    assert outcome.verdict != "refused", outcome.refusal
    assert [node["name"] for node in outcome.trace_nodes] == [
        "retrieve",
        "gate",
        "generate",
        "verify",
    ]
    assert [node["seq"] for node in outcome.trace_nodes] == [1, 2, 3, 4]
    for node in outcome.trace_nodes:
        assert set(node) == {
            "seq",
            "name",
            "status",
            "duration_ms",
            "input_summary",
            "output_summary",
        }
        assert node["status"] == "ok"

    # 阶段耗时表来自图节点的自测耗时（键与 M2 相同）
    assert set(outcome.cost["latency_breakdown_ms"]) == {"retrieve", "generate", "verify"}

    # 不变量 1 / 3 在图路径下同样成立
    indexes = {c.index for c in outcome.citations}
    for match in CITATION_RE.finditer(outcome.answer):
        for part in re.split(r"[,，]", match.group(1)):
            assert int(part) in indexes
    assert [p["index"] for p in outcome.evidence_snapshot["passages"]] == [
        c.index for c in outcome.citations
    ]


async def test_retrieval_refusal_branch_skips_generation(qa_env) -> None:
    """检索侧粗拒：gate 之后直接终止（没有 generate/verify 行）。"""
    outcome = await _answer(qa_env, NEGATIVE_Q)

    assert outcome.verdict == "refused"
    assert outcome.citations == []
    assert outcome.refusal is not None
    assert outcome.refusal["reason"] == "no_relevant_hit"
    assert [node["name"] for node in outcome.trace_nodes] == ["retrieve", "gate"]
    assert "拒答" in outcome.trace_nodes[-1]["output_summary"]
    assert outcome.cost["tokens"] == {"prompt": 0, "completion": 0, "total": 0}


async def test_answer_side_refusal_is_first_class_branch(qa_env) -> None:
    """答案侧细拒：检索有命中，但模型自己拒答 → adjudicate 行 + 可复盘 detail。"""
    from app.generation.base import Completion

    class _RefusingProvider:
        name = "stub"
        model = "stub-refusal"

        def available(self) -> bool:
            return True

        def complete(self, prompt: str, *, max_new_tokens: int = 512) -> Completion:
            return Completion(
                text="文档中没有找到相关内容。",
                prompt_tokens=12,
                completion_tokens=9,
                model=self.model,
                provider=self.name,
                sampling={"deterministic": True},
            )

    outcome = await _answer(qa_env, POSITIVE_Q, provider=_RefusingProvider())

    assert outcome.verdict == "refused"
    assert outcome.citations == []
    assert outcome.refusal is not None
    assert outcome.refusal["reason"] == "answer_unsupported"
    # 拒答要能复盘到逐句证据（verdict 来自 verify 节点，不是"因为所以"）
    assert outcome.refusal["verify_source"] in ("semantic", "lexical")
    assert "uncited_claims" in outcome.refusal
    assert [node["name"] for node in outcome.trace_nodes] == [
        "retrieve",
        "gate",
        "generate",
        "verify",
        "adjudicate",
    ]
    assert [node["seq"] for node in outcome.trace_nodes] == [1, 2, 3, 4, 5]
    # 拒答返回的校验结果是"拒答形态"（逐句明细在 refusal.detail）
    assert outcome.verification["source"] == "lexical"


async def test_state_is_json_native_and_deterministic(qa_env) -> None:
    """State 必须能被 JSON 序列化（检查点前提），且同题两次的裁决一致。"""
    first = await _answer(qa_env, POSITIVE_Q)
    second = await _answer(qa_env, POSITIVE_Q)

    assert [n["name"] for n in first.trace_nodes] == [n["name"] for n in second.trace_nodes]
    assert first.verdict == second.verdict
    assert [s["level"] for s in first.verification["sentences"]] == [
        s["level"] for s in second.verification["sentences"]
    ]
    assert first.answer_id != second.answer_id

    # 用带检查点的图再跑一次：检查点里的 State 必须能原样 JSON dump（能 dump 就能落库/读回）
    from app.graphs.qa_graph import build_qa_graph
    from app.services.answer_service import AnswerService
    from app.store import Filters
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = InMemorySaver()
    service = AnswerService(
        build_qa_graph(qa_env.deps, checkpointer), settings=qa_env.settings
    )
    outcome = await service.answer(
        POSITIVE_Q, version=qa_env.version, top_k=3, filters=Filters(tenant_id="default")
    )
    snapshot = await service.graph.aget_state(
        {"configurable": {"thread_id": outcome.trace_id}}
    )
    json.dumps(snapshot.values)  # 不抛异常即为通过
    assert snapshot.values["verdict"] == outcome.verdict
    assert snapshot.values["outcome"]["answer_id"] == outcome.answer_id


# ---------------- HTTP 层：路由确实接到了图上 ----------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as test_client:
        for name in CORPUS_DOCS:
            with (EVAL_CORPUS_DIR / name).open("rb") as fh:
                response = test_client.post(
                    "/v1/documents", files={"file": (name, fh, "text/markdown")}
                )
            assert response.status_code == 202, response.text
            task_id = response.json()["task_id"]
            for _ in range(200):
                task = test_client.get(f"/v1/tasks/{task_id}").json()
                if task["status"] in ("succeeded", "failed"):
                    break
                time.sleep(0.2)
            assert task["status"] == "succeeded", task
        yield test_client


def test_http_answer_serves_graph_trace(client: TestClient) -> None:
    """HTTP 层拿到的 nodes[] 来自图：前两步固定为 retrieve/gate，seq 连续。

    这里刻意不断言"这题该不该拒答"：共享 DATA_DIR 下语料会被其它用例增删，
    拒答与否随语料漂移（图行为本身由上面自建环境的用例钉住）。
    """
    body = client.post("/v1/answer", json={"question": POSITIVE_Q}).json()
    trace = client.get(f"/v1/answer/{body['answer_id']}/trace").json()

    names = [node["name"] for node in trace["nodes"]]
    assert names[:2] == ["retrieve", "gate"]
    assert set(names) <= {"retrieve", "gate", "generate", "verify", "adjudicate"}
    assert [node["seq"] for node in trace["nodes"]] == list(range(1, len(names) + 1))
    assert trace["final_verdict"] == body["verdict"]
    # 拒答 ⟺ 空引用 + refusal 非空（不变量 2 的 HTTP 断言）
    if body["verdict"] == "refused":
        assert body["citations"] == [] and body["refusal"] is not None
    else:
        assert body["citations"] and body["refusal"] is None
