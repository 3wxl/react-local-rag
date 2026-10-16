"""Agent 接口（M3）：/v1/agent/* 的 HTTP 契约与状态机。

用默认的假生成模型（`LLM_PROVIDER=fake`）跑：planner 提示词里没有【文档内容】段，
假模型给出的是标准拒答文本 → 解析成"结束检索" → 无证据 → 拒答。
这条路径**恰好覆盖了"planner 输出不可用/无证据时也必须优雅收尾"**，
而且完全确定性（不受语料状态影响），适合做 HTTP 层冒烟。
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from .test_qa_graph import CORPUS_DOCS, _ingest


@pytest.fixture(scope="module")
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as test_client:
        _ingest(test_client, CORPUS_DOCS)
        yield test_client


def _read_sse(response) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    event = ""
    for line in response.iter_lines():
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            body = line.split(":", 1)[1].strip()
            if body == "[DONE]":
                break
            events.append((event, json.loads(body)))
    return events


def test_agent_run_lifecycle_and_trace(client: TestClient) -> None:
    response = client.post("/v1/agent/run", json={"question": "智能网关标准版的售价是多少？"})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["status"] == "succeeded"
    assert body["run_id"].startswith("run_")
    assert body["answer"], body
    # 拒答形态：planner 没调工具 → 没有证据 → 拒答（而不是凭空作答）
    assert body["verdict"] == "refused"
    assert body["citations"] == [] and body["refusal"] is not None
    assert body["refusal"]["reason"] == "no_evidence"
    assert [row["name"] for row in body["trace_nodes"]][:3] == [
        "plan",
        "synthesize",
        "verify",
    ]

    # 状态可轮询（tasks 表，kind=agent）
    status = client.get(f"/v1/agent/{body['run_id']}").json()
    assert status["status"] == "succeeded"
    assert status["pending_approval"] is None

    # 轨迹自包含可回放
    trace = client.get(f"/v1/agent/{body['run_id']}/trace").json()
    assert [row["name"] for row in trace["nodes"]][:3] == ["plan", "synthesize", "verify"]
    assert trace["final_verdict"] == "refused"
    assert trace["evidence_snapshot"]["tool_calls"] == []


def test_agent_stream_emits_final(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/agent/run",
        json={"question": "智能网关标准版的售价是多少？", "stream": True},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _read_sse(response)

    names = [data["name"] for event, data in events if event == "node"]
    assert names[:3] == ["plan", "synthesize", "verify"]
    finals = [data for event, data in events if event == "final"]
    assert len(finals) == 1
    assert set(finals[0]) >= {"run_id", "answer", "verdict", "citations", "cost", "tool_calls"}
    assert events[-1][0] == "final"


def test_agent_resume_requires_interrupted_run(client: TestClient) -> None:
    started = client.post("/v1/agent/run", json={"question": "年假制度怎么规定的？"}).json()
    run_id = started["run_id"]

    response = client.post(f"/v1/agent/{run_id}/resume", json={"approved": True})
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "invalid_request"


def test_agent_unknown_run_is_404(client: TestClient) -> None:
    response = client.get("/v1/agent/run_not_exists")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_agent_budget_rejection_is_422(client: TestClient, monkeypatch) -> None:
    ctx = client.app.state.ctx
    monkeypatch.setattr(ctx.settings, "request_token_budget", 8)

    response = client.post("/v1/agent/run", json={"question": "智能网关标准版的售价是多少？"})

    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "budget_exceeded"


def test_agent_trace_not_found_before_completion(client: TestClient) -> None:
    """未完成的 run 没有轨迹（自包含回放的前提是"跑完了"）。"""
    response = client.get("/v1/agent/run_pending/trace")
    assert response.status_code == 404


def test_hitl_run_and_streaming_resume(client: TestClient, monkeypatch) -> None:
    """HITL 全链路（HTTP）：需审批工具 → interrupted → 批准 → 流式续跑到 final。

    注入脚本化 planner 与一个"需要审批的工具"：默认配置里没有 MCP server，
    真模型又不会稳定地规划出工具调用 —— 而这里要验的是**状态机与事件流**。
    """
    from app.graphs.tools.registry import ToolOutcome, ToolSpec

    from .test_agent_graph import ScriptedProvider, _ANSWER, _tool_call

    async def handler(args: dict) -> ToolOutcome:
        return ToolOutcome(text=f"已写入：{args.get('text', '')}")

    approval_tool = ToolSpec(
        name="mcp:write_note",
        description="写外部系统（需要审批）",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        handler=handler,
        source="mcp",
        requires_approval=True,
    )

    ctx = client.app.state.ctx
    monkeypatch.setattr(
        ctx.agent,
        "provider",
        ScriptedProvider([_tool_call("mcp:write_note", {"text": "记一下"}), _ANSWER]),
    )
    monkeypatch.setattr(ctx.agent, "mounted_tools", [approval_tool])

    started = client.post("/v1/agent/run", json={"question": "把结论记到外部系统"})
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["status"] == "interrupted"
    assert body["pending_approval"]["tool"] == "mcp:write_note"
    assert "args_preview" in body["pending_approval"]

    run_id = body["run_id"]
    status = client.get(f"/v1/agent/{run_id}").json()
    assert status["status"] == "interrupted"
    assert status["pending_approval"]["tool"] == "mcp:write_note"

    # 批准 → 流式续跑：事件协议与首次运行相同
    with client.stream(
        "POST",
        f"/v1/agent/{run_id}/resume",
        json={"approved": True, "stream": True},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _read_sse(response)

    finals = [data for event, data in events if event == "final"]
    assert len(finals) == 1
    assert finals[0]["tool_calls"][0]["tool"] == "mcp:write_note"
    assert finals[0]["tool_calls"][0]["ok"] is True

    # 终态与轨迹都落地（自包含回放）
    assert client.get(f"/v1/agent/{run_id}").json()["status"] == "succeeded"
    trace = client.get(f"/v1/agent/{run_id}/trace").json()
    names = [row["name"] for row in trace["nodes"]]
    assert names.count("act") == 1, names
    assert trace["evidence_snapshot"]["tool_calls"][0]["ok"] is True
