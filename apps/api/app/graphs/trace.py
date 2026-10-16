"""轨迹行构造 + 流式事件（M3）。

行字段与 /v1/answer/{id}/trace 的 nodes[] 同一条口径（tests/test_api_contract.py 逐字段断言）：
`seq / name / status / duration_ms / input_summary / output_summary`
—— M3 起这些行由 LangGraph 节点产出，但**摘要文案保持与 M2 的直线流程逐字符一致**，
否则已有的 trace 断言与人工对照都会失效。

`seq` 由"当前已累积的行数 + 1"决定：节点按拓扑顺序执行，进入节点时 State 里
已是此前所有节点追加后的结果，所以这里读到的长度就是正确序号。

构造行时**同时发一条 SSE `node` 事件**（非流式下 emit 是 no-op）：前端看到的步骤链
与事后回放的轨迹是同一份对象，不存在"流式和落库各写一套"的漂移可能。
"""

from __future__ import annotations

from .events import EVENT_NODE, emit


def trace_row(
    state: dict,
    name: str,
    *,
    status: str = "ok",
    duration_ms: int = 0,
    input_summary: str = "",
    output_summary: str = "",
    error: str | None = None,
) -> dict:
    """构造一条轨迹行（调用方把它放进返回值的 `trace_nodes` 通道）。"""
    row = {
        "seq": len(state.get("trace_nodes") or []) + 1,
        "name": name,
        "status": status,
        "duration_ms": duration_ms,
        "input_summary": input_summary,
        "output_summary": output_summary,
    }
    if error is not None:
        # 失败行才带 error（成功行不加，避免把契约里没有的字段塞进历史数据）
        row["error"] = error
    emit(EVENT_NODE, row)
    return row
