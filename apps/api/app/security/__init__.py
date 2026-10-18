"""安全边界（M5）：间接提示注入防护。

工具返回的内容是**数据**，不是用户意图。这里的函数负责识别其中的指令性文本、
在送进 planner / 生成模型前隔离它，并把命中情况留在轨迹与证据快照里。
"""

from .injection import (
    InjectionFinding,
    InjectionScan,
    fence_for_planner,
    redact_untrusted,
    scan_untrusted,
    sanitize_tool_output,
)

__all__ = [
    "InjectionFinding",
    "InjectionScan",
    "fence_for_planner",
    "redact_untrusted",
    "scan_untrusted",
    "sanitize_tool_output",
]
