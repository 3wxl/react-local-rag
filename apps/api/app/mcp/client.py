"""MCP stdio 客户端（M3）：换行分隔 JSON-RPC 2.0。

报文只有三条：`initialize` → `notifications/initialized` → `tools/list` / `tools/call`。
安全与可运维的收口点全在这里：
- **allowlist**：不在名单里的工具连列都不列出、更不允许调用（默认拒绝）；
- **超时**：每次调用独立 `asyncio.wait_for`，外部工具卡住不能把请求一起拖死；
- **审计素材**：调用结果里带耗时与是否成功，由调用方（工具注册表）写进轨迹。

挂载的就是仓库里既有的 TS MCP server（apps/web/scripts/mcp/server.ts）——
它能被 Python 端挂上，本身就是"MCP 作为接口而不是框架"的验证。
"""

from __future__ import annotations

import asyncio
import json
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

#: 我们实现的协议版本（与 MCP 规范一致；服务端回什么以它为准）
PROTOCOL_VERSION = "2025-06-18"


class McpClientError(RuntimeError):
    """客户端级错误（启动失败 / 协议错误 / 服务端返回 error）。"""


class McpToolDenied(McpClientError):
    """工具不在 allowlist 里（默认拒绝，不是"未找到"）。"""


@dataclass
class McpToolResult:
    """一次 tools/call 的结果。"""

    ok: bool
    text: str
    latency_ms: int
    error: str | None = None
    raw: dict = field(default_factory=dict)


class McpStdioClient:
    """stdio 传输的 MCP 客户端（一个实例 = 一个子进程）。"""

    def __init__(
        self,
        command: str,
        *,
        allowlist: Sequence[str] = (),
        timeout_s: float = 20.0,
        cwd: str | None = None,
    ) -> None:
        self.command = command
        self.allowlist = tuple(allowlist)
        self.timeout_s = timeout_s
        self.cwd = cwd
        self._proc: asyncio.subprocess.Process | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._next_id = 0
        self.server_info: dict = {}

    # ---------------- 生命周期 ----------------

    async def start(self) -> None:
        if self._proc is not None:
            return
        args = shlex.split(self.command)
        if not args:
            raise McpClientError("MCP 启动命令为空")
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=self.cwd,
            )
        except OSError as err:
            raise McpClientError(f"MCP 服务启动失败：{err}") from err
        self._reader_task = asyncio.create_task(self._read_loop())

        result = await self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "react-local-rag-api", "version": "0.1.0"},
            },
        )
        self.server_info = result.get("serverInfo", {}) or {}
        # 按规范回一条 initialized 通知（通知没有 id，不期待应答）
        self._notify("notifications/initialized", {})

    async def close(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
            self._reader_task = None
        proc = self._proc
        if proc is not None and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:  # pragma: no cover —— 子进程不听话就强杀
                proc.kill()
                await proc.wait()
        # 显式关掉 stdin 管道并让事件循环跑一轮：Windows 上留给 GC 的 transport
        # 会在循环关闭后抛 "I/O operation on closed pipe"（pytest 会当作 unraisable 警告刷屏）
        if proc is not None and proc.stdin is not None:
            try:
                proc.stdin.close()
            except Exception:  # noqa: BLE001 —— 已经关掉的管道无需再管
                pass
        await asyncio.sleep(0)
        self._proc = None
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()

    # ---------------- 工具面 ----------------

    async def list_tools(self) -> list[dict]:
        """列出工具（按 allowlist 过滤；空 allowlist = 全部允许）。"""
        result = await self._request("tools/list", {})
        tools = list(result.get("tools", []) or [])
        if not self.allowlist:
            return tools
        return [t for t in tools if t.get("name") in self.allowlist]

    async def call_tool(self, name: str, arguments: dict | None = None) -> McpToolResult:
        if self.allowlist and name not in self.allowlist:
            raise McpToolDenied(f"工具不在 allowlist 中，已拒绝：{name}")
        started = time.perf_counter()
        try:
            result = await self._request(
                "tools/call",
                {"name": name, "arguments": arguments or {}},
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            return McpToolResult(
                ok=False,
                text="",
                latency_ms=int((time.perf_counter() - started) * 1000),
                error=f"MCP 工具调用超时（>{self.timeout_s}s）：{name}",
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return McpToolResult(
            ok=not bool(result.get("isError")),
            text=_content_text(result),
            latency_ms=latency_ms,
            error=_content_text(result) if result.get("isError") else None,
            raw=result,
        )

    # ---------------- JSON-RPC 细节 ----------------

    async def _request(self, method: str, params: dict, *, timeout: float | None = None) -> dict:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise McpClientError("MCP 服务未启动")
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        payload = json.dumps(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
            ensure_ascii=False,
        )
        proc.stdin.write((payload + "\n").encode("utf-8"))
        await proc.stdin.drain()
        try:
            return await asyncio.wait_for(future, timeout=timeout or self.timeout_s)
        finally:
            self._pending.pop(request_id, None)

    def _notify(self, method: str, params: dict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        payload = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
        proc.stdin.write((payload + "\n").encode("utf-8"))

    async def _read_loop(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break  # 服务端退出：未完成的调用由 close()/超时兜底
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                continue  # 非 JSON 行（有些实现会往 stdout 打日志）：跳过而不是崩掉
            future = self._pending.get(message.get("id"))
            if future is None or future.done():
                continue
            if "error" in message:
                future.set_exception(
                    McpClientError(
                        f"MCP 返回错误：{message['error'].get('message', message['error'])}"
                    )
                )
            else:
                future.set_result(message.get("result", {}) or {})


def _content_text(result: dict) -> str:
    """把 `content: [{type: "text", text}]` 拼成纯文本（非 text 类型如实标注）。"""
    parts: list[str] = []
    for item in result.get("content", []) or []:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
        else:
            parts.append(f"[{item.get('type', 'unknown') if isinstance(item, dict) else 'unknown'}]")
    return "\n".join(parts).strip()


def tool_schema_to_parameters(tool: dict) -> dict:
    """MCP 的 inputSchema 直接当工具参数声明（不重写、不裁剪）。"""
    schema: Any = tool.get("inputSchema") or {}
    return schema if isinstance(schema, dict) else {}
