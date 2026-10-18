"""MCP stdio 客户端（M3）：握手、allowlist、超时、错误映射。

对端是 `tests/fixtures/fake_mcp_server.py`（极简换行分隔 JSON-RPC），
因为这里要验的是**客户端行为**；挂真 TS server 另有一条需要 node 的集成用例。
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest

from app.mcp import McpAuditLogger, McpStdioClient, McpToolDenied, audit_context

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
        assert {"kb.list_docs", "kb.search", "kb.get_chunk", "kb.answer"} <= names, names
        # 兼容别名仍在（M3 客户端不受影响）
        assert {"search_docs", "doc_outline", "read_chunks"} <= names, names
    finally:
        await mcp.close()


# ---------------- M5：审计日志 ----------------


def _read_audit(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


async def test_audit_records_call_and_deny(tmp_path: Path) -> None:
    """每次调用进审计：成功、拒绝都要有记录；参数预览要脱敏。"""
    audit_path = tmp_path / "audit" / "mcp.jsonl"
    audit = McpAuditLogger(audit_path)
    mcp = McpStdioClient(_command(), allowlist=("echo",), timeout_s=2.0, audit=audit)
    token = audit_context.set({"run_id": "run_audit"})
    try:
        await mcp.start()
        await mcp.list_tools()
        result = await mcp.call_tool(
            "echo", {"text": "你好", "api_key": "sk-should-not-appear"}
        )
        assert result.ok is True
        with pytest.raises(McpToolDenied):
            await mcp.call_tool("restricted", {})
    finally:
        audit_context.reset(token)
        await mcp.close()

    entries = _read_audit(audit_path)
    events = [entry["event"] for entry in entries]
    assert "mcp.start" in events
    assert "mcp.tools/list" in events
    calls = [entry for entry in entries if entry["event"] == "mcp.tools/call"]
    assert len(calls) == 2
    ok_call = next(entry for entry in calls if entry["tool"] == "echo")
    assert ok_call["ok"] is True and ok_call["decision"] == "allow"
    assert ok_call["ctx_run_id"] == "run_audit"
    assert "sk-should-not-appear" not in json.dumps(ok_call, ensure_ascii=False)
    assert "***" in ok_call["args_preview"]
    # 审计不记工具返回正文，只记长度/指纹（参数预览里出现调用方自己传的文本是允许的）
    assert "result_text" not in ok_call
    assert ok_call["result_chars"] > 0 and ok_call["result_digest"]

    denied = next(entry for entry in calls if entry["tool"] == "restricted")
    assert denied["decision"] == "deny" and denied["ok"] is False
    assert denied["ctx_run_id"] == "run_audit"


async def test_audit_is_optional_and_never_breaks_calls(tmp_path: Path) -> None:
    """审计关闭时调用照常；审计路径不可写也不会让工具调用失败。"""
    mcp = McpStdioClient(_command(), allowlist=("echo",), timeout_s=2.0)
    await mcp.start()
    try:
        result = await mcp.call_tool("echo", {"text": "ok"})
    finally:
        await mcp.close()
    assert result.ok is True

    # 路径是一个目录 → 追加写失败；客户端只告警，不抛异常
    broken = McpAuditLogger(tmp_path, enabled=True)
    mcp2 = McpStdioClient(_command(), allowlist=("echo",), timeout_s=2.0, audit=broken)
    await mcp2.start()
    try:
        result2 = await mcp2.call_tool("echo", {"text": "still ok"})
    finally:
        await mcp2.close()
    assert result2.ok is True and result2.text == "still ok"
