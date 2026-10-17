"""检查点持久化（M3 验收：中断后能恢复）。

用一张最小图（审批门 → 完成）证明两件事：
1. `sqlite` 后端下，中断状态落在 DATA_DIR 的检查点库里 —— **换一个引擎实例**（模拟进程重启）
   仍能读到挂起的 `interrupt` 并续跑到完成；
2. `memory` 后端只在进程内有效（测试默认用它换速度，这里钉住它的边界）。

图本身是测试自带的，不依赖 app/graphs 下的业务图：检查点行为是基础设施属性，
与具体节点实现无关。
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.core.config import Settings
from app.graphs import open_checkpointer


class _State(TypedDict, total=False):
    question: str
    steps: Annotated[list, operator.add]
    approved: bool
    answer: str


def _gate(state: _State) -> dict:
    decision = interrupt({"kind": "tool_approval", "tool": "search_docs"})
    return {"approved": bool(decision.get("approved")), "steps": [{"name": "gate"}]}


def _finish(state: _State) -> dict:
    return {"answer": "approved" if state.get("approved") else "rejected"}


def _build(checkpointer):
    graph = StateGraph(_State)
    graph.add_node("gate", _gate)
    graph.add_node("finish", _finish)
    graph.add_edge(START, "gate")
    graph.add_edge("gate", "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)


def _thread(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


async def test_sqlite_checkpoint_survives_new_engine(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, checkpoint_backend="sqlite")
    config = _thread("run-restart")

    # ── 引擎 1：跑到审批门被 interrupt 挡住 ──
    async with open_checkpointer(settings) as checkpointer:
        graph = _build(checkpointer)
        result = await graph.ainvoke({"question": "停机保养前要做什么"}, config)
        assert "__interrupt__" in result
        pending = await graph.aget_state(config)
        assert pending.next == ("gate",)
        assert pending.tasks[0].interrupts[0].value["tool"] == "search_docs"

    # 检查点确实落了盘（文件在 DATA_DIR 下，不是内存对象）
    assert settings.checkpoint_db_path.exists()

    # ── 引擎 2（新实例 = 模拟进程重启）：读到挂起的中断并续跑 ──
    async with open_checkpointer(settings) as checkpointer:
        graph = _build(checkpointer)
        resumed_state = await graph.aget_state(config)
        assert resumed_state.next == ("gate",), "重启后必须仍能看到挂起的审批"

        final = await graph.ainvoke(Command(resume={"approved": True}), config)
        assert final["answer"] == "approved"
        assert [s["name"] for s in final["steps"]] == ["gate"]


async def test_memory_backend_is_process_local(tmp_path: Path) -> None:
    """memory 后端不落盘：换实例就丢状态（所以生产默认是 sqlite）。"""
    settings = Settings(data_dir=tmp_path, checkpoint_backend="memory")
    config = _thread("run-memory")

    async with open_checkpointer(settings) as checkpointer:
        graph = _build(checkpointer)
        result = await graph.ainvoke({"question": "q"}, config)
        assert "__interrupt__" in result
    assert not settings.checkpoint_db_path.exists()

    async with open_checkpointer(settings) as checkpointer:
        graph = _build(checkpointer)
        fresh = await graph.aget_state(config)
        assert fresh.next == (), "新实例里没有历史检查点"
