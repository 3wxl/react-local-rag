"""QA 图组装（M3）：改写 → 检索 → 判级 → 生成 → 引用校验 → 拒答。

形状刻意做成"两条拒答分支 + 一条正常分支"的显式图，而不是直线流程里的 if：
判级（`gate`）与答案侧裁决（`verify` → `adjudicate`）都是**一等分支**，
拒答不是异常路径 —— 这与契约"拒答是 200 正常响应"是同一条设计。

分支语义：
- `gate` 拒答 = 检索侧粗拒（多信号融合，reason=no_relevant_hit）
- `verify` 后的拒答 = 答案侧细拒（reason=answer_unsupported，逐句证据不足）
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from .qa_nodes import QaDeps, build_qa_nodes
from .state import QaState


def build_qa_graph(deps: QaDeps, checkpointer: BaseCheckpointSaver | None = None):
    """编译 QA 图（进程内一次；per-request 参数走 State）。"""
    nodes = build_qa_nodes(deps)

    graph = StateGraph(QaState)
    # 改写是条件入口：默认（rewrite=false）不进图，默认路径轨迹与 M2 逐行一致
    graph.add_node("rewrite", nodes["rewrite"])
    graph.add_node("retrieve", nodes["retrieve"])
    graph.add_node("gate", nodes["gate"])
    graph.add_node("generate", nodes["generate"])
    graph.add_node("verify", nodes["verify"])
    graph.add_node("refuse_no_relevant", nodes["refuse_no_relevant"])
    graph.add_node("refuse_unsupported", nodes["refuse_unsupported"])
    graph.add_node("finalize", nodes["finalize"])

    graph.add_conditional_edges(
        START,
        nodes["route_rewrite"],
        {"rewrite": "rewrite", "retrieve": "retrieve"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "gate")
    graph.add_conditional_edges(
        "gate",
        nodes["route_gate"],
        {"refuse": "refuse_no_relevant", "generate": "generate"},
    )
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify",
        nodes["route_verify"],
        {"refuse": "refuse_unsupported", "finalize": "finalize"},
    )
    graph.add_edge("refuse_no_relevant", END)
    graph.add_edge("refuse_unsupported", END)
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
