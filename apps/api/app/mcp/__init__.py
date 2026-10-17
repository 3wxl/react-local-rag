"""MCP 接入（M3：stdio 客户端；M5 再补 server 面）。

为什么手写 JSON-RPC 而不引官方 SDK：MCP 的 stdio 传输就是"换行分隔的 JSON-RPC 2.0"，
本仓库需要的只是 initialize / tools/list / tools/call 三条报文；为此增加一个依赖
（以及它的版本面）不划算。客户端只做三件事，其余（权限、超时、审计）都在这一层收口。
"""

from .client import McpClientError, McpStdioClient, McpToolDenied, McpToolResult

__all__ = ["McpStdioClient", "McpToolResult", "McpClientError", "McpToolDenied"]
