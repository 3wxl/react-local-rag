"""Agentic Graph（M3）：工具循环、无进展检测、HITL 审批与续跑、轨迹可回放。

planner 用**测试专用的脚本化 provider**（按调用顺序给出预设决策）：
真模型的规划是概率性的，而这里要验的是编排本身 —— 循环边界、无进展判据、
中断/续跑的语义。`generation/fake.py` 刻意不动（它的输出被契约测试与跨语言口径依赖）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.generation.base import Completion
from app.graphs.agent_graph import build_agent_graph
from app.graphs.agent_nodes import AgentDeps, parse_decision
from app.graphs.tools.registry import ToolOutcome, ToolRegistry, ToolSpec
from app.services.agent_service import AgentService
from app.services.ingest_service import IngestService
from app.services.retrieval_service import RetrievalService
from app.store import Filters, SqliteStore

from .parity_utils import EVAL_CORPUS_DIR
from .test_incremental_index import CountingEmbedder


class ScriptedProvider:
    """按顺序吐出预设文本的 provider（脚本用尽后重复最后一条）。"""

    name = "scripted"
    model = "scripted-planner"

    def __init__(self, scripts: list[str]) -> None:
        self.scripts = list(scripts)
        self.prompts: list[str] = []

    def available(self) -> bool:
        return True

    def complete(self, prompt: str, *, max_new_tokens: int = 512) -> Completion:
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self.scripts) - 1)
        text = self.scripts[index]
        return Completion(
            text=text,
            prompt_tokens=len(prompt),
            completion_tokens=len(text),
            model=self.model,
            provider=self.name,
            sampling={"deterministic": True},
        )


def _tool_call(name: str, args: dict, reason: str = "查一下") -> str:
    return json.dumps({"action": "tool", "tool": name, "args": args, "reason": reason})


_ANSWER = json.dumps({"action": "answer", "reason": "观察够了"})


@pytest.fixture
async def env(tmp_path: Path):
    store = SqliteStore(tmp_path / "rag.sqlite3")
    await store.init()
    embedder = CountingEmbedder()
    ingest = IngestService(store, embedder, data_dir=tmp_path)
    for name in ("产品手册.md", "员工手册.md"):
        await ingest.ingest_path(EVAL_CORPUS_DIR / name, doc_name=name)
    retrieval = RetrievalService(store, embedder)
    version = (await store.list_ready_versions())[0].version
    try:
        yield store, retrieval, embedder, version, tmp_path
    finally:
        await store.close()


def _service(*, store, retrieval, embedder, settings, provider, checkpointer, tools=()) -> AgentService:
    return AgentService(
        store=store,
        retrieval=retrieval,
        embedder=embedder,
        provider=provider,
        settings=settings,
        checkpointer=checkpointer,
        mounted_tools=list(tools),
    )


def test_parse_decision_is_tolerant() -> None:
    known = {"search_docs"}
    # 包着解释文字也能解析（模型爱在 JSON 外面写一句话）
    decision = parse_decision(
        '好的，我来查：{"action":"tool","tool":"search_docs","args":{"query":"售价"}}',
        known_tools=known,
    )
    assert decision["action"] == "tool" and decision["tool"] == "search_docs"
    # 未知工具 → 退化为直接回答（不让一次幻觉出的工具名把循环卡死）
    assert parse_decision('{"action":"tool","tool":"rm_rf"}', known_tools=known)["action"] == "answer"
    # 非 JSON → 退化为直接回答
    assert parse_decision("我觉得可以直接回答", known_tools=known)["action"] == "answer"


async def test_tool_loop_then_answer(env) -> None:
    """正常链路：planner 调一次检索 → 汇总 → 校验 → 落轨迹。"""
    store, retrieval, embedder, version, _ = env
    provider = ScriptedProvider([_tool_call("search_docs", {"query": "售价"}), _ANSWER])
    service = _service(
        store=store,
        retrieval=retrieval,
        embedder=embedder,
        settings=_settings(),
        provider=provider,
        checkpointer=InMemorySaver(),
    )

    result = await service.run(
        "智能网关标准版的售价是多少？", run_id="run_loop", version=version, top_k=3
    )

    assert result.status == "succeeded", result
    outcome = result.outcome or {}
    assert [row["name"] for row in result.trace_nodes] == [
        "plan",
        "act",
        "reflect",
        "plan",
        "synthesize",
        "verify",
    ]
    assert [call["tool"] for call in outcome["tool_calls"]] == ["search_docs"]
    assert outcome["tool_calls"][0]["ok"] is True
    # 契约不变量（Agent 路径同样成立）：refused ⟺ 空引用 + refusal 非空；
    # 非拒答时证据快照与引用同源同序。这里不假设"一定判 supported"——
    # 测试用的假编码器给的是随机向量，语义分无意义，裁决可能是拒答（真实环境由真模型决定）。
    snapshot = outcome["evidence_snapshot"]
    if outcome["verdict"] == "refused":
        assert outcome["citations"] == [] and outcome["refusal"] is not None
    else:
        assert outcome["citations"], outcome
        assert [p["index"] for p in snapshot["passages"]] == [
            c["index"] for c in outcome["citations"]
        ]
    assert snapshot["passages"], "工具检索到的证据必须进快照"

    # 轨迹落库（自包含回放）
    trace = await store.get_trace("run_loop")
    assert trace is not None
    assert [row["name"] for row in trace.nodes][:3] == ["plan", "act", "reflect"]


async def test_no_progress_detection_stops_the_loop(env) -> None:
    """同一个调用重复出现 → 判定无进展并提前收敛（不烧到步数上限）。"""
    store, retrieval, embedder, version, _ = env
    same_call = _tool_call("search_docs", {"query": "售价"})
    provider = ScriptedProvider([same_call, same_call, same_call, same_call])
    service = _service(
        store=store,
        retrieval=retrieval,
        embedder=embedder,
        settings=_settings(),
        provider=provider,
        checkpointer=InMemorySaver(),
    )

    result = await service.run(
        "智能网关标准版的售价是多少？", run_id="run_noprogress", version=version, top_k=3
    )

    outcome = result.outcome or {}
    assert outcome["no_progress"]["reason"], outcome
    assert "重复调用" in outcome["no_progress"]["reason"]
    reflect_rows = [r for r in result.trace_nodes if r["name"] == "reflect"]
    assert any("无进展" in r["output_summary"] for r in reflect_rows)


async def test_hitl_interrupt_and_resume_across_engines(env) -> None:
    """HITL：需审批的工具挂起 → 换一个引擎实例（模拟重启）审批 → 继续跑完。"""
    store, retrieval, embedder, version, tmp_path = env
    provider = ScriptedProvider(
        [_tool_call("mcp:write_note", {"text": "记一下"}, "写外部系统"), _ANSWER]
    )
    checkpoint_db = str(tmp_path / "checkpoints.sqlite3")

    # ── 引擎 1：跑到审批门 ──
    async with AsyncSqliteSaver.from_conn_string(checkpoint_db) as saver:
        service = _service(
            store=store,
            retrieval=retrieval,
            embedder=embedder,
            settings=_settings(),
            provider=provider,
            checkpointer=saver,
            tools=[_write_note_tool()],
        )
        result = await service.run(
            "把结论记到外部系统", run_id="run_hitl", version=version, top_k=3
        )
        assert result.status == "interrupted", result
        assert result.pending_approval["tool"] == "mcp:write_note"
        pending = await service.pending_approval("run_hitl")
        assert pending is not None and pending["kind"] == "tool_approval"

    # ── 引擎 2（新实例）：审批通过 → 工具执行 → 收尾 ──
    async with AsyncSqliteSaver.from_conn_string(checkpoint_db) as saver:
        provider2 = ScriptedProvider([_ANSWER, _ANSWER])
        service2 = _service(
            store=store,
            retrieval=retrieval,
            embedder=embedder,
            settings=_settings(),
            provider=provider2,
            checkpointer=saver,
            tools=[_write_note_tool()],
        )
        resumed = await service2.resume("run_hitl", approved=True)
        assert resumed.status == "succeeded", resumed
        calls = (resumed.outcome or {})["tool_calls"]
        assert [c["tool"] for c in calls] == ["mcp:write_note"]
        assert calls[0]["ok"] is True


async def test_hitl_rejection_feeds_back_to_planner(env) -> None:
    """驳回：工具不执行，planner 收到"被驳回"的观察后改为直接回答。"""
    store, retrieval, embedder, version, tmp_path = env
    provider = ScriptedProvider([_tool_call("mcp:write_note", {"text": "x"}), _ANSWER])
    checkpoint_db = str(tmp_path / "checkpoints-reject.sqlite3")

    async with AsyncSqliteSaver.from_conn_string(checkpoint_db) as saver:
        service = _service(
            store=store,
            retrieval=retrieval,
            embedder=embedder,
            settings=_settings(),
            provider=provider,
            checkpointer=saver,
            tools=[_write_note_tool()],
        )
        first = await service.run("写点什么", run_id="run_reject", version=version, top_k=3)
        assert first.status == "interrupted"

    async with AsyncSqliteSaver.from_conn_string(checkpoint_db) as saver:
        service2 = _service(
            store=store,
            retrieval=retrieval,
            embedder=embedder,
            settings=_settings(),
            provider=ScriptedProvider([_ANSWER]),
            checkpointer=saver,
            tools=[_write_note_tool()],
        )
        resumed = await service2.resume("run_reject", approved=False)

    assert resumed.status == "succeeded", resumed
    calls = (resumed.outcome or {})["tool_calls"]
    assert calls and calls[0]["denied"] is True
    # 驳回后没有证据 → 拒答（而不是凭空作答）
    assert (resumed.outcome or {})["verdict"] == "refused"
    assert (resumed.outcome or {})["refusal"]["reason"] == "no_evidence"


async def test_budget_step_cap_stops_agent(env, monkeypatch) -> None:
    """步数上限：planner 每轮都要求继续查 → 预算守卫拒绝并说明原因。"""
    from app.graphs.budget import BudgetExceeded

    store, retrieval, embedder, version, _ = env
    settings = _settings()
    monkeypatch.setattr(settings, "graph_max_steps", 2)
    provider = ScriptedProvider(
        [
            _tool_call("search_docs", {"query": "a"}),
            _tool_call("search_docs", {"query": "b"}),
            _tool_call("search_docs", {"query": "c"}),
        ]
    )
    service = _service(
        store=store,
        retrieval=retrieval,
        embedder=embedder,
        settings=settings,
        provider=provider,
        checkpointer=InMemorySaver(),
    )

    with pytest.raises(BudgetExceeded) as excinfo:
        await service.run("问题", run_id="run_steps", version=version, top_k=3)

    detail = excinfo.value.detail["budget"]
    assert detail["exceeded"] == "steps"
    assert detail["limit"]["steps"] == 2


# ---------------- 辅助 ----------------


def _settings():
    from app.core.config import settings as app_settings

    return app_settings


def _write_note_tool() -> ToolSpec:
    """模拟一个 MCP 写操作工具（需要人工审批）。"""
    from app.graphs.tools.registry import ToolOutcome as Outcome

    async def handler(args: dict) -> Outcome:
        return Outcome(text=f"已写入：{args.get('text', '')}")

    return ToolSpec(
        name="mcp:write_note",
        description="写外部系统（需要审批）",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        handler=handler,
        source="mcp",
        requires_approval=True,
    )
