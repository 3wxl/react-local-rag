"""测试用假 MCP server（stdio，换行分隔 JSON-RPC 2.0）。

为什么用 Python 假 server 而不是直接挂仓库里的 TS 真 server：
- CI 的 api job 里 Node 只是为了跨语言导出器，不该让 MCP 测试也依赖它编译/启动；
- 这里要验的是**客户端行为**（握手、allowlist、超时、错误映射），协议对端越简单越能定位问题。

真 server 的挂载由 `test_mcp_client.py::test_mounts_real_ts_mcp_server` 单独验（需要 node，缺失即跳过）。
"""

from __future__ import annotations

import json
import sys
import time

TOOLS = [
    {
        "name": "echo",
        "description": "回显给定文本（测试用）",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "slow",
        "description": "永远比客户端超时慢（测试超时路径）",
        "inputSchema": {"type": "object", "properties": {"seconds": {"type": "number"}}},
    },
    {
        "name": "restricted",
        "description": "用来验证 allowlist 的工具",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def main() -> None:
    # stdio 传输按规范是 UTF-8；Windows 上子进程默认可能是 GBK（cp936），
    # 不显式重设会把中文参数解错（曾把"你好，世界"解成乱码再回显）
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        method = message.get("method")
        request_id = message.get("id")

        if method == "initialize":
            _reply(request_id, {"protocolVersion": "2025-06-18", "serverInfo": {"name": "fake"}})
        elif method == "tools/list":
            _reply(request_id, {"tools": TOOLS})
        elif method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "echo":
                _reply(request_id, {"content": [{"type": "text", "text": str(args.get("text", ""))}]})
            elif name == "slow":
                time.sleep(float(args.get("seconds", 5)))
                _reply(request_id, {"content": [{"type": "text", "text": "slow done"}]})
            elif name == "restricted":
                _reply(
                    request_id,
                    {
                        "content": [{"type": "text", "text": "should not be reachable"}],
                        "isError": True,
                    },
                )
            else:
                _reply(
                    request_id,
                    {"content": [{"type": "text", "text": f"unknown tool: {name}"}], "isError": True},
                )
        # 通知（notifications/initialized 等）没有 id：不回。


def _reply(request_id: object, result: dict) -> None:
    if request_id is None:
        return
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
