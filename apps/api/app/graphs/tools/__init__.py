"""工具面（M3）：内置工具 + MCP 挂载，统一由 ToolRegistry 分发。

统一抽象的理由：Agentic Graph 的 planner 只该看到"有哪些工具、参数是什么"，
至于工具是内置实现还是外部 MCP 服务提供的，属于注册表的事 ——
安全属性（超时、allowlist、是否需人工审批）也都在注册表这一层声明。
"""

from .registry import ToolRegistry, ToolResult, ToolSpec

__all__ = ["ToolRegistry", "ToolSpec", "ToolResult"]
