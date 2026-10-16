"""API 契约测试（docs/api-contract.md 的字段与不变量）。

用 FakeProvider 跑完整问答链路：把「HTTP 契约」与「模型是否可用」解耦，
CI 里没有模型也能断言引用/拒答/轨迹这些关键行为。

不变量（契约 §2 明写、实现必须满足）：
1. 答案里出现的每个角标 [n] 必在 citations[].index 中
2. verdict == "refused" ⟺ citations 为空且 refusal != null
3. evidence_snapshot.passages[].index 与 citations[].index 同源同序
4. 拒答是 200 正常响应，不是错误
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from .parity_utils import EVAL_CORPUS_DIR, EVAL_DIR

CITATION_RE = re.compile(r"\[(\d+(?:\s*[,，]\s*\d+)*)\]")


@pytest.fixture(scope="module")
def client() -> TestClient:
    """独立数据目录（conftest 在任何 app 导入前设好）+ 假生成模型。

    用假模型的原因：契约行为（引用装配、拒答、轨迹、错误体）与模型无关，
    把它们和"模型是否下载"绑在一起，CI 就永远跑不了。
    """
    from app.main import app

    with TestClient(app) as test_client:
        _ingest(test_client, ["产品手册.md", "员工手册.md", "会议纪要.md"])
        yield test_client


def _ingest(client: TestClient, names: list[str]) -> None:
    for name in names:
        with (EVAL_CORPUS_DIR / name).open("rb") as fh:
            response = client.post(
                "/v1/documents", files={"file": (name, fh, "text/markdown")}
            )
        assert response.status_code == 202, response.text
        task_id = response.json()["task_id"]
        for _ in range(200):
            task = client.get(f"/v1/tasks/{task_id}").json()
            if task["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.2)
        assert task["status"] == "succeeded", task


def test_health_reports_dependencies(client: TestClient) -> None:
    body = client.get("/v1/health").json()
    assert body["status"] == "ok"
    assert body["storage"]["backend"] == "sqlite"
    assert body["storage"]["chunks"] > 0
    assert body["index"]["version"].startswith("fnv1a32:")
    assert body["models"]["embed"] in ("ready", "missing")


def test_retrieve_shape_and_invariants(client: TestClient) -> None:
    body = client.post(
        "/v1/retrieve", json={"question": "智能网关标准版的售价是多少？", "top_k": 3}
    ).json()
    assert body["query_id"].startswith("qry_")
    assert body["rewritten_query"] is None  # 未开启改写
    assert "no_relevant_hit" in body
    # 多信号融合拒答的四路取值都必须回传：拒答要能归因到具体信号（2026-10-15 起）
    assert set(body["refusal_signals"]) == {
        "idf_coverage",
        "top_relevance",
        "max_cosine",
        "cos_gap",
        "max_bm25",
    }
    if not body["no_relevant_hit"]:
        hit = body["hits"][0]
        # 契约 §0.1 的字段一个都不能少（前端按这些字段渲染引用卡片）
        for field in (
            "index",
            "doc_id",
            "doc_name",
            "heading_path",
            "page_start",
            "page_end",
            "score",
            "relevance",
            "chunk_ids",
            "window_chunk_ids",
            "matched_content",
            "content",
        ):
            assert field in hit, f"Citation 缺字段 {field}"


def test_answer_citations_and_evidence_invariants(client: TestClient) -> None:
    body = client.post(
        "/v1/answer", json={"question": "智能网关标准版的售价是多少？"}
    ).json()
    assert body["answer_id"].startswith("ans_")
    assert body["trace_id"].startswith("trc_")

    indexes = {c["index"] for c in body["citations"]}
    # 不变量 1：答案里的每个角标都要能在 citations 里找到
    for match in CITATION_RE.finditer(body["answer"]):
        for part in re.split(r"[,，]", match.group(1)):
            assert int(part) in indexes, f"角标 [{part}] 不在 citations 中"

    # 不变量 3：evidence_snapshot.passages 与 citations 同源同序
    passages = body["evidence_snapshot"]["passages"]
    assert [p["index"] for p in passages] == [c["index"] for c in body["citations"]]
    assert body["evidence_snapshot"]["index_version"].startswith("fnv1a32:")

    # cost 契约字段
    assert set(body["cost"]) == {
        "latency_ms",
        "latency_breakdown_ms",
        "tokens",
        "estimated_cost_usd",
    }
    # verification 契约字段
    assert set(body["verification"]) >= {"source", "scoped_chunk_count", "sentences"}


def test_refusal_is_200_with_empty_citations(client: TestClient) -> None:
    body = client.post("/v1/answer", json={"question": "公司的股票代码是多少？"}).json()
    # 不变量 2 与 4：拒答是正常响应
    assert body["verdict"] == "refused"
    assert body["citations"] == []
    assert body["refusal"] is not None
    assert body["refusal"]["reason"]
    assert body["answer"]


def test_trace_is_self_contained(client: TestClient) -> None:
    answer = client.post("/v1/answer", json={"question": "年假制度是怎么规定的？"}).json()
    trace = client.get(f"/v1/answer/{answer['answer_id']}/trace").json()
    assert trace["answer_id"] == answer["answer_id"]
    assert [node["name"] for node in trace["nodes"]][:2] == ["retrieve", "gate"]
    for node in trace["nodes"]:
        for field in ("seq", "name", "status", "duration_ms", "input_summary", "output_summary"):
            assert field in node, f"trace node 缺字段 {field}"
    # trace 自带证据快照（不依赖 answers 行仍存在）
    assert "passages" in trace["evidence_snapshot"]
    assert trace["final_verdict"] == answer["verdict"]


def test_error_body_shape(client: TestClient) -> None:
    response = client.get("/v1/answer/ans_not_exists/trace")
    assert response.status_code == 404
    body = response.json()
    assert set(body["error"]) == {"code", "message", "detail"}
    assert body["error"]["code"] == "not_found"


def test_invalid_request_is_400(client: TestClient) -> None:
    # 拼错字段名必须报错，而不是被静默忽略（静默忽略是口径漂移的温床）
    response = client.post("/v1/retrieve", json={"question": "x", "topK": 3})
    assert response.status_code in (400, 422)


def test_stream_not_silently_ignored(client: TestClient) -> None:
    response = client.post(
        "/v1/answer", json={"question": "年假制度是怎么规定的？", "stream": True}
    )
    assert response.status_code == 400, "stream=true 必须明确报不支持"


def test_eval_run_lifecycle(client: TestClient) -> None:
    started = client.post("/v1/eval/run", json={"suite": "retrieval"})
    assert started.status_code == 202
    run_id = started.json()["run_id"]
    for _ in range(600):
        body = client.get(f"/v1/eval/run/{run_id}").json()
        if body["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.5)
    assert body["status"] == "succeeded", body.get("error")
    assert body["gates_passed"] is True, body["failed_gates"]
    metrics = body["metrics"]
    # 指标名必须与 ts-metrics.json 对齐（前端与 CI 都按这些名字取值）
    for field in ("recallAt3", "recallAt5", "mrrAt10", "negativeRefusedCount", "byGroup"):
        assert field in metrics


def test_documents_listing_and_delete(client: TestClient) -> None:
    docs = client.get("/v1/documents").json()["documents"]
    assert docs, "至少应有一条已入库文档"
    doc_id = docs[-1]["doc_id"]
    assert client.delete(f"/v1/documents/{doc_id}").json()["deleted"] is True
    remaining = client.get("/v1/documents").json()["documents"]
    assert doc_id not in {d["doc_id"] for d in remaining}
    # 软删除后立刻从检索面消失（召回层过滤，不是取回后再筛）
    listing = client.post("/v1/retrieve", json={"question": "年假", "top_k": 5}).json()
    assert all(hit["doc_id"] != doc_id for hit in listing["hits"])
