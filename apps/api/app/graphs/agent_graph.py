"""Agentic Graph（M3）：planner → 工具（含 MCP）→ 无进展检测 → HITL 审批 → 汇总。

形状：`plan` ⇄（`act` → `reflect`）循环，收敛到 `synthesize` → `verify` → `finalize`。
终止条件有三个，都是显式的：planner 说够了、无进展检测触发、预算/步数用尽。

检查点按 `thread_id = run_id` 落盘，所以 `interrupt`（人工审批）挂起后可以跨进程续跑 ——
这是 HITL 值得用图编排而不是"回调 + 全局状态"的核心原因。
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from .agent_nodes import AgentDeps, build_agent_nodes
from .state import AgentState

#: 相对步数预算的余量（planner 一轮 = plan + act + reflect 三个超级步 + 汇总/校验）
RECURSION_HEADROOM = 3


def build_agent_graph(deps: AgentDeps, checkpointer: BaseCheckpointSaver | None = None):
    """编译 Agent 图（进程内一次）。"""
    nodes = build_agent_nodes(deps)

    graph = StateGraph(AgentState)
    graph.add_node("plan", nodes["plan"])
    graph.add_node("act", nodes["act"])
    graph.add_node("reflect", nodes["reflect"])
    graph.add_node("synthesize", nodes["synthesize"])
    graph.add_node("verify", nodes["verify"])
    graph.add_node("finalize", nodes["finalize"])

    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan",
        nodes["route_plan"],
        {"act": "act", "synthesize": "synthesize"},
    )
    graph.add_edge("act", "reflect")
    graph.add_conditional_edges(
        "reflect",
        nodes["route_reflect"],
        {"plan": "plan", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", "verify")
    graph.add_edge("verify", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


def agent_recursion_limit(settings) -> int:
    """`recursion_limit` 兜底：步数预算 × 3 + 余量（预算守卫先给出友好原因）。"""
    return max(12, settings.graph_max_steps * 3 + RECURSION_HEADROOM)
