"""SSE 流式（M3）：事件顺序、`final` 与非流式响应体逐字段一致、错误在带内。

分层（理由同 test_qa_graph.py）：**图行为**在自建环境里断言（共享 DATA_DIR 的语料会漂移，
"这题会不会被拒答"不再是稳定前提）；**HTTP 层**只断言与语料无关的传输契约
（content-type / 帧格式 / final 键集合 / seq 连续）。

帧解析走 `apps/web/src/agent/cloudPlanner.ts` 的同款口径（event: 行 + data: 行 + [DONE]），
所以这里也顺带钉住"前后端帧格式没有分叉"。
"""

from __future__ import annotations

import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

from .test_qa_graph import CORPUS_DOCS, NEGATIVE_Q, POSITIVE_Q, _ingest


@pytest.fixture(scope="module")
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as test_client:
        _ingest(test_client, CORPUS_DOCS)
        yield test_client


def _read_sse(client: TestClient, payload: dict) -> list[tuple[str, dict]]:
    """读完整条流并解析成 [(event, data)]（不含终止帧）。"""
    events: list[tuple[str, dict]] = []
    with client.stream("POST", "/v1/answer", json=payload) as response:
        assert response.status_code == 200, response.read()
        assert response.headers["content-type"].startswith("text/event-stream")
        event = ""
        saw_done = False
        for line in response.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                body = line.split(":", 1)[1].strip()
                if body == "[DONE]":
                    saw_done = True
                    break
                events.append((event, json.loads(body)))
        assert saw_done, "流必须以 [DONE] 收尾（前端解析器据此停止）"
    return events


async def _collect(service, env, question: str) -> list[dict]:
    from app.store import Filters

    return [
        payload
        async for payload in service.answer_stream(
            question,
            version=env.version,
            top_k=3,
            filters=Filters(tenant_id="default"),
        )
    ]


# ---------------- HTTP 层：传输契约（与语料状态无关） ----------------


def test_http_stream_transport_contract(client: TestClient) -> None:
    streamed = _read_sse(client, {"question": POSITIVE_Q, "stream": True})
    plain = client.post("/v1/answer", json={"question": POSITIVE_Q}).json()

    node_events = [data for event, data in streamed if event == "node"]
    assert node_events, "至少要发节点事件"
    assert [row["seq"] for row in node_events] == list(range(1, len(node_events) + 1))
    # 节点名来自图的节点集合（具体走到哪一步取决于语料状态）
    assert set(row["name"] for row in node_events) <= {
        "rewrite",
        "retrieve",
        "gate",
        "generate",
        "verify",
        "adjudicate",
    }

    finals = [data for event, data in streamed if event == "final"]
    assert len(finals) == 1
    # final 事件 = 非流式响应体（键集合逐字段一致）
    assert set(finals[0]) == set(plain)
    assert finals[0]["question"] == plain["question"]
    # final 必须是最后一个事件
    assert streamed[-1][0] == "final"


def test_stream_preflight_budget_is_real_422(client: TestClient, monkeypatch) -> None:
    """预检超预算：发生在流开始**之前** → 真 422（而不是 200 + 带内 error）。"""
    ctx = client.app.state.ctx
    monkeypatch.setattr(ctx.settings, "request_token_budget", 8)

    response = client.post("/v1/answer", json={"question": POSITIVE_Q, "stream": True})

    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "budget_exceeded"


# ---------------- 图行为（自建环境，语料可控） ----------------


async def test_stream_events_order_and_final_equivalence(qa_env) -> None:
    from app.services.answer_service import AnswerService
    from app.store import Filters

    service = AnswerService(qa_env.graph, settings=qa_env.settings)
    streamed = await _collect(service, qa_env, POSITIVE_Q)
    plain = await service.answer(
        POSITIVE_Q,
        version=qa_env.version,
        top_k=3,
        filters=Filters(tenant_id="default"),
    )

    node_events = [data for e, data in [(p["event"], p["data"]) for p in streamed] if e == "node"]
    assert [row["name"] for row in node_events] == ["retrieve", "gate", "generate", "verify"]
    assert [row["seq"] for row in node_events] == [1, 2, 3, 4]

    finals = [p["data"] for p in streamed if p["event"] == "final"]
    assert len(finals) == 1
    final = finals[0]
    assert set(final) == {
        "answer_id",
        "trace_id",
        "question",
        "answer",
        "verdict",
        "citations",
        "refusal",
        "verification",
        "evidence_snapshot",
        "cost",
    }
    # 同一问题、同一语料：流式与非流式的产物逐字段可比（id/耗时除外）
    assert final["answer"] == plain.answer
    assert final["verdict"] == plain.verdict
    assert [c["index"] for c in final["citations"]] == [c.index for c in plain.citations]
    assert final["cost"]["tokens"] == plain.cost["tokens"]
    assert final["evidence_snapshot"]["passages"] == plain.evidence_snapshot["passages"]
    assert streamed[-1]["event"] == "final"


async def test_stream_emits_tokens_that_final_overrides(qa_env) -> None:
    """FakeProvider 支持 stream()：token 事件拼接结果与 final.answer 一致。

    实际客户端应以 `final.answer` 为准（token 是未去 <think> 的原始文本）——
    fake 不带思考标签，所以两者恰好相等，正好用来钉住"拼接无损"。
    """
    from app.services.answer_service import AnswerService

    service = AnswerService(qa_env.graph, settings=qa_env.settings)
    streamed = await _collect(service, qa_env, POSITIVE_Q)

    deltas = "".join(p["data"]["delta"] for p in streamed if p["event"] == "token")
    final = [p["data"] for p in streamed if p["event"] == "final"][0]

    assert deltas, "支持流式的 provider 必须产出 token 事件"
    assert deltas == final["answer"]


async def test_stream_refusal_still_emits_final(qa_env) -> None:
    """拒答也要走 final（拒答是正常响应，不是错误）——通过流交付时同样如此。"""
    from app.services.answer_service import AnswerService

    service = AnswerService(qa_env.graph, settings=qa_env.settings)
    streamed = await _collect(service, qa_env, NEGATIVE_Q)

    final = [p["data"] for p in streamed if p["event"] == "final"][0]
    assert final["verdict"] == "refused"
    assert final["citations"] == []
    assert final["refusal"] is not None
    assert [p["data"]["name"] for p in streamed if p["event"] == "node"] == ["retrieve", "gate"]


async def test_stream_errors_are_in_band(qa_env) -> None:
    """流开始之后的错误一律带内：模型不可用 / 运行中超预算，都是 error 事件。"""
    from app.generation.base import Completion, ModelUnavailableError
    from app.graphs.qa_graph import build_qa_graph
    from app.services.answer_service import AnswerService

    class _UnavailableOnGenerate:
        name = "stub"
        model = "stub-unavailable"

        def available(self) -> bool:
            return False

        def complete(self, prompt: str, *, max_new_tokens: int = 512) -> Completion:
            raise ModelUnavailableError("模型未就绪（测试桩）")

    deps = dataclasses.replace(qa_env.deps, provider=_UnavailableOnGenerate())
    service = AnswerService(build_qa_graph(deps, None), settings=qa_env.settings)
    streamed = await _collect(service, qa_env, POSITIVE_Q)

    errors = [p for p in streamed if p["event"] == "error"]
    assert len(errors) == 1, streamed
    assert errors[0]["data"]["error"]["code"] == "model_unavailable"
    # 流里已经发过节点事件（检索/判级跑完了），错误在带内交付而不是断连
    assert [p["data"]["name"] for p in streamed if p["event"] == "node"] == ["retrieve", "gate"]

    # 运行中超预算：同样是带内 error（预检时 token 数还看不出来）
    class _HugeUsage:
        name = "stub"
        model = "stub-huge"

        def available(self) -> bool:
            return True

        def complete(self, prompt: str, *, max_new_tokens: int = 512) -> Completion:
            return Completion(
                text="答案[1]。",
                prompt_tokens=50_000,
                completion_tokens=50_000,
                model=self.model,
                provider=self.name,
                sampling={},
            )

    monkey = pytest.MonkeyPatch()
    monkey.setattr(qa_env.settings, "request_token_budget", 4096)
    try:
        deps = dataclasses.replace(qa_env.deps, provider=_HugeUsage())
        service = AnswerService(build_qa_graph(deps, None), settings=qa_env.settings)
        streamed = await _collect(service, qa_env, POSITIVE_Q)
    finally:
        monkey.undo()

    errors = [p for p in streamed if p["event"] == "error"]
    assert len(errors) == 1, streamed
    assert errors[0]["data"]["error"]["code"] == "budget_exceeded"
    assert errors[0]["data"]["error"]["detail"]["budget"]["exceeded"] == "tokens"
    assert not [p for p in streamed if p["event"] == "final"], "超预算的请求不该有 final"
