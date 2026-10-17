"""图内事件 → SSE 通道（M3）。

节点只调 `emit(...)`，不关心自己跑在流式还是非流式路径上：
- 流式（`graph.astream(..., stream_mode="custom")`）：事件被路由转成 SSE 帧；
- 非流式（`graph.ainvoke`）：LangGraph 的 writer 存在但事件被丢弃，是安全 no-op。

同一份节点函数必须在两种模式下行为一致 —— 这是"流式只是多了一条输出通道"的落点，
而不是两套编排逻辑。
"""

from __future__ import annotations

from langgraph.config import get_stream_writer

#: SSE 事件名（契约 §2）：node / token / final / error / interrupt
EVENT_NODE = "node"
EVENT_TOKEN = "token"
EVENT_FINAL = "final"
EVENT_ERROR = "error"
EVENT_INTERRUPT = "interrupt"


def emit(event: str, data: dict) -> None:
    """发一条事件（非流式下静默丢弃）。"""
    try:
        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 —— 版本差异兜底：拿不到 writer 就当非流式
        return
    if writer is None:
        return
    writer({"event": event, "data": data})
