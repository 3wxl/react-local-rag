"""MCP 调用审计（M5）：每次 list / call / deny / 超时都落一行 JSONL。

审计的边界与红线：
- **只记元数据**：工具名、参数指纹、参数预览（脱敏 + 截断）、耗时、成败、错误类型；
  不记工具返回正文、不记 token/密钥。工具返回可能含文档原文与注入载荷，
  把它们写进审计日志等于又复制了一份敏感数据。
- **写失败不打断调用**：审计是旁路；磁盘满/权限错时打 stderr 告警后继续。
- `run_id` 等编排上下文通过 `audit_context`（ContextVar）传入，异步任务天然继承。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from ..parity.fnv import fnv1a32

#: 参数预览上限（审计要能定位"调了什么"，但不该成为数据副本）
ARGS_PREVIEW_CHARS = 300

#: Agent 运行上下文：call_tool 在深层工具 handler 里，显式传参会污染每个签名
audit_context: ContextVar[dict] = ContextVar("mcp_audit_context", default={})

_SECRET_KEY_RE = ("token", "secret", "password", "passwd", "api_key", "apikey", "authorization")


def _redact(value: Any, key: str = "") -> Any:
    """递归脱敏：敏感键一律替换；其余值原样保留（预览会再截断）。"""
    lowered = key.lower()
    if any(marker in lowered for marker in _SECRET_KEY_RE):
        return "***"
    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def args_preview(args: dict | None) -> str:
    try:
        text = json.dumps(_redact(args or {}), ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        text = repr(args)[:ARGS_PREVIEW_CHARS]
    return text if len(text) <= ARGS_PREVIEW_CHARS else text[:ARGS_PREVIEW_CHARS] + "…"


def args_digest(name: str, args: dict | None) -> str:
    canonical = json.dumps(args or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return fnv1a32(f"{name}:{canonical}")


class McpAuditLogger:
    """JSONL 审计日志（每行一个事件；追加写，进程内串行化）。"""

    def __init__(
        self,
        path: str | Path | None,
        *,
        server_name: str = "mcp",
        enabled: bool = True,
    ) -> None:
        self.path = Path(path) if path else None
        self.server_name = server_name
        self.enabled = bool(enabled and self.path)
        self._lock = asyncio.Lock()

    async def record(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
            "event": event,
            "server": self.server_name,
            **{k: v for k, v in fields.items() if v is not None},
            **{f"ctx_{k}": v for k, v in audit_context.get().items() if v is not None},
        }
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        async with self._lock:
            try:
                await asyncio.to_thread(self._append, line)
            except OSError as err:  # 审计是旁路：写不进去也不能让工具调用失败
                print(f"[mcp-audit] 写入失败：{err}", file=sys.stderr)

    def _append(self, line: str) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)
