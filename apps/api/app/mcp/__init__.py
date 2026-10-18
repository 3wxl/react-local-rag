"""MCP 接入（M3：stdio 客户端；M5：allowlist + 每次调用审计）。

为什么手写 JSON-RPC 而不引官方 SDK：MCP 的 stdio 传输就是"换行分隔的 JSON-RPC 2.0"，
本仓库需要的只是 initialize / tools/list / tools/call 三条报文；为此增加一个依赖
（以及它的版本面）不划算。客户端只做三件事，其余（权限、超时、审计）都在这一层收口。

MCP **Server** 不在这个包里：它是独立的 TS 进程（apps/web/scripts/mcp/server.ts，
stdio + Streamable HTTP），避免把 Node 运行时塞进 Python 服务。
"""

from .audit import McpAuditLogger, audit_context
from .client import McpClientError, McpStdioClient, McpToolDenied, McpToolResult

__all__ = [
    "McpAuditLogger",
    "McpClientError",
    "McpStdioClient",
    "McpToolDenied",
    "McpToolResult",
    "audit_context",
]
