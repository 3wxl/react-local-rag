"""LangGraph 图定义（M3）：QA 图 / 摄取图 / Agentic 图。

分工（与 docs/ROADMAP-FULLSTACK.md §5 对齐）：
- `qa_graph.py`   问答链：改写 → 检索 → 判级 → 生成 → 引用校验 → 拒答
- `ingest_graph.py` 摄取链：解析 → 分块 → 去重/版本比对 → 向量化 → 索引 → 自检
- `agent_graph.py`  Agent 链：planner → 工具调用（含 MCP）→ 无进展检测 → HITL 审批 → 汇总

三条共同纪律：
1. 图只**调用** app/parity 与 app/services 的既有实现，不重写算法（跨语言口径靠这个保持）；
2. State 只放 JSON 原生类型（检查点要序列化；dataclass/numpy 不进 state）；
3. 轨迹节点沿用 traces 表的字段口径（seq/name/status/duration_ms/input_summary/output_summary）。
"""

from .checkpointer import open_checkpointer
from .events import (
    EVENT_ERROR,
    EVENT_FINAL,
    EVENT_INTERRUPT,
    EVENT_NODE,
    EVENT_TOKEN,
    emit,
)

__all__ = [
    "open_checkpointer",
    "emit",
    "EVENT_NODE",
    "EVENT_TOKEN",
    "EVENT_FINAL",
    "EVENT_ERROR",
    "EVENT_INTERRUPT",
]
