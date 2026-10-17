"""MCP stdio 客户端（M3）：握手、allowlist、超时、错误映射。

对端是 `tests/fixtures/fake_mcp_server.py`（极简换行分隔 JSON-RPC），
因为这里要验的是**客户端行为**；挂真 TS server 另有一条需要 node 的集成用例。
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

import pytest

from app.mcp import McpStdioClient, McpToolDenied

FIXTURE = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"


def _command() -> str:
    return f'"{sys.executable}" "{FIXTURE}"'


@pytest.fixture
async def client():
    mcp = McpStdioClient(_command(), allowlist=("echo", "slow"), timeout_s=1.0)
    await mcp.start()
    try:
        yield mcp
    finally:
        await mcp.close()


async def test_handshake_and_list_tools(client: McpStdioClient) -> None:
    tools = await client.list_tools()
    # allowlist 生效：restricted 连"列出来"都不该出现（默认拒绝，不是"隐藏但可调"）
    assert [t["name"] for t in tools] == ["echo", "slow"]
    assert client.server_info.get("name") == "fake"


async def test_call_tool_returns_text_and_latency(client: McpStdioClient) -> None:
    result = await client.call_tool("echo", {"text": "你好，世界"})
    assert result.ok is True
    assert result.text == "你好，世界"
    assert result.latency_ms >= 0
    assert result.error is None


async def test_allowlist_denies_calls(client: McpStdioClient) -> None:
    with pytest.raises(McpToolDenied):
        await client.call_tool("restricted", {})


async def test_timeout_is_isolated_per_call(client: McpStdioClient) -> None:
    """工具卡住不能把请求拖死：超时返回 ok=False（而不是抛异常），且连接不被污染。

    注意对端是**单线程** server：它对超时的那次调用仍在跑，之后的请求要排队 ——
    所以这里先等它跑完再发下一条，验的是"客户端没有因为一次超时坏掉"，
    而不是"服务端能并发"。
    """
    result = await client.call_tool("slow", {"seconds": 2})
    assert result.ok is False
    assert "超时" in (result.error or "")

    # 等服务端把那次调用跑完（超时只解除客户端的等待，不会杀掉服务端的工作）
    await asyncio.sleep(1.5)
    again = await client.call_tool("echo", {"text": "still alive"})
    assert again.ok is True and again.text == "still alive"


async def test_unknown_tool_maps_to_error(client: McpStdioClient) -> None:
    """服务端报错的工具：映射成 ok=False（而不是抛异常打断 Agent 循环）。"""
    mcp = McpStdioClient(_command(), allowlist=(), timeout_s=2.0)
    await mcp.start()
    try:
        result = await mcp.call_tool("nope", {})
    finally:
        await mcp.close()
    assert result.ok is False
    assert "unknown tool" in (result.error or "")


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 才能起 TS MCP server")
async def test_mounts_real_ts_mcp_server(tmp_path: Path) -> None:
    """集成：Python 客户端能挂上仓库里既有的 TS MCP server（M5 的接口面验证）。

    备份文件为空知识库即可 —— 这条用例验的是**协议互操作**（握手 + tools/list），
    不是检索质量（那由 TS 侧自己的测试覆盖）。
    """
    backup = tmp_path / "backup.json"
    backup.write_text(
        '{"formatVersion":"2.0","exportedAt":"2026-10-16T00:00:00Z",'
        '"conversations":[{"id":"c1","title":"t","docName":"d.md","messages":[],'
        '"vectorChunks":[]}]}',
        encoding="utf-8",
    )
    web_dir = Path(__file__).resolve().parents[2] / "web"
    if not (web_dir / "node_modules").exists():
        pytest.skip("apps/web 未安装依赖（npm ci）")

    mcp = McpStdioClient(
        f'node --import tsx scripts/mcp/server.ts --backup "{backup}"',
        allowlist=(),
        timeout_s=30.0,
        cwd=str(web_dir),
    )
    await mcp.start()
    try:
        tools = await mcp.list_tools()
        names = {t["name"] for t in tools}
        assert {"search_docs", "doc_outline", "read_chunks"} <= names, names
    finally:
        await mcp.close()
