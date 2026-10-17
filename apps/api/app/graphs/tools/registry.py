"""工具注册表（M3）：声明式描述 + 统一调用出口。

三个属性都在这里声明，因为它们是**安全属性**，不能散落在各工具实现里：
- `timeout_s`：外部工具（MCP）卡住不能把请求拖死；
- `requires_approval`：写操作/外部副作用工具默认要人工审批（HITL）；
- `source`：builtin / mcp —— 审计与轨迹要能区分"这个结果是谁给的"。

调用永不抛异常（未知工具、超时、工具内部错误都映射成 `ok=False` 的结果）：
Agent 循环靠"观察里的错误文本"自我修正，把异常抛出去只会让整轮请求失败。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from ...parity.fnv import fnv1a32

#: 工具结果的默认超时（与前端 toolAgent 的 DEFAULT_TOOL_STEP_TIMEOUT 同量级）
DEFAULT_TOOL_TIMEOUT_S = 45.0


@dataclass
class ToolSpec:
    """一个工具的声明。"""

    name: str
    description: str
    parameters: dict  # JSON Schema（直接喂给 planner 的提示词）
    handler: Callable[[dict], Awaitable["ToolOutcome"]]
    source: str = "builtin"  # builtin | mcp
    timeout_s: float = DEFAULT_TOOL_TIMEOUT_S
    #: 需要人工审批才执行（HITL interrupt）：写操作、外部副作用工具默认开启
    requires_approval: bool = False


@dataclass
class ToolOutcome:
    """工具处理器返回的东西（文本给人/模型看，data 给编排用）。"""

    text: str
    data: dict = field(default_factory=dict)


@dataclass
class ToolResult:
    """一次工具调用的结果（进轨迹与观测）。"""

    name: str
    ok: bool
    text: str
    latency_ms: int
    source: str = "builtin"
    error: str | None = None
    #: 调用指纹（工具名 + 规范化参数）：无进展检测按它判"重复调了同一个东西"
    call_digest: str = ""
    data: dict = field(default_factory=dict)


def args_digest(name: str, args: dict) -> str:
    """调用指纹：工具名 + 键排序后的参数 JSON（同参数 ⇒ 同指纹）。"""
    canonical = json.dumps(args or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return fnv1a32(f"{name}:{canonical}")


class ToolRegistry:
    """工具注册与调用。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict]:
        """给 planner 的工具清单（JSON Schema 原文，不改写）。"""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
                "source": spec.source,
                "requires_approval": spec.requires_approval,
            }
            for spec in self._tools.values()
        ]

    def with_spec(self, spec: ToolSpec, *, requires_approval: bool | None = None) -> None:
        """注册/覆盖一个工具（MCP 挂载用：只改安全属性，不改实现）。"""
        if requires_approval is not None:
            spec.requires_approval = requires_approval
        self.register(spec)

    async def call(self, name: str, args: dict | None = None) -> ToolResult:
        """调用工具。永不抛异常：未知工具/超时/内部错误都是 ok=False 的结果。"""
        args = args or {}
        spec = self._tools.get(name)
        digest = args_digest(name, args)
        started = time.perf_counter()

        def _elapsed() -> int:
            return int((time.perf_counter() - started) * 1000)

        if spec is None:
            return ToolResult(
                name=name,
                ok=False,
                text="",
                latency_ms=_elapsed(),
                error=f"未知工具：{name}（可用：{', '.join(self.names())}）",
                call_digest=digest,
            )
        try:
            outcome = await asyncio.wait_for(spec.handler(args), timeout=spec.timeout_s)
        except asyncio.TimeoutError:
            return ToolResult(
                name=name,
                ok=False,
                text="",
                latency_ms=_elapsed(),
                source=spec.source,
                error=f"工具超时（>{spec.timeout_s}s）：{name}",
                call_digest=digest,
            )
        except Exception as err:  # noqa: BLE001 —— 工具内部错误也要变成"观察"喂回模型
            return ToolResult(
                name=name,
                ok=False,
                text="",
                latency_ms=_elapsed(),
                source=spec.source,
                error=f"工具执行失败：{err}",
                call_digest=digest,
            )
        return ToolResult(
            name=name,
            ok=True,
            text=outcome.text,
            latency_ms=_elapsed(),
            source=spec.source,
            call_digest=digest,
            data=outcome.data,
        )
