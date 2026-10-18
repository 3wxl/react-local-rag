"""工具注册表（M3）：声明式描述 + 统一调用出口。

三个属性都在这里声明，因为它们是**安全属性**，不能散落在各工具实现里：
- `timeout_s`：外部工具（MCP）卡住不能把请求拖死；
- `requires_approval`：写操作/外部副作用工具默认要人工审批（HITL）；
- `source`：builtin / mcp / skill —— 审计与轨迹要能区分"这个结果是谁给的"；
- `permissions` / `cost_estimate_usd` / `sandboxed`：Skills manifest 声明的能力与成本，
  进 planner 工具清单与轨迹；内置/MCP 工具保持默认空值。

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
from ...security import redact_untrusted, sanitize_tool_output, scan_untrusted

#: 工具结果的默认超时（与前端 toolAgent 的 DEFAULT_TOOL_STEP_TIMEOUT 同量级）
DEFAULT_TOOL_TIMEOUT_S = 45.0


@dataclass
class ToolSpec:
    """一个工具的声明。"""

    name: str
    description: str
    parameters: dict  # JSON Schema（直接喂给 planner 的提示词）
    handler: Callable[[dict], Awaitable["ToolOutcome"]]
    source: str = "builtin"  # builtin | mcp | skill
    timeout_s: float = DEFAULT_TOOL_TIMEOUT_S
    #: 需要人工审批才执行（HITL interrupt）：写操作、外部副作用工具默认开启
    requires_approval: bool = False
    # ---------------- Skills manifest 字段（内置/MCP 工具留空） ----------------
    #: 声明式权限（如 net / fs / process）；沙箱按它放行系统能力
    permissions: list[str] = field(default_factory=list)
    #: 成本预估（美元/次）：进 planner 工具清单与轨迹，预算守卫可据此取舍
    cost_estimate_usd: float = 0.0
    #: 是否运行在最小沙箱里（Skills 为 True；内置/MCP 为 False）
    sandboxed: bool = False


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
    # ---------------- 间接提示注入防护（M5） ----------------
    #: 已脱敏的观察文本（进 planner）；`text` 保留原文供审计/调试
    safe_text: str = ""
    #: evidence 文本已脱敏的数据（进生成模型）；`data` 保留原文供引用卡片
    data_for_model: dict = field(default_factory=dict)
    #: 注入扫描结果 {risk,count,categories,findings}
    injection: dict = field(default_factory=dict)
    #: 工具声明的单次成本预估（Skills manifest；内置/MCP 默认 0）
    cost_estimate_usd: float = 0.0


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
                "permissions": list(spec.permissions),
                "cost_estimate_usd": spec.cost_estimate_usd,
                "sandboxed": spec.sandboxed,
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
            error = f"未知工具：{name}（可用：{', '.join(self.names())}）"
            scan = scan_untrusted(error)
            return ToolResult(
                name=name,
                ok=False,
                text="",
                latency_ms=_elapsed(),
                error=error,
                safe_text=redact_untrusted(error, scan),
                call_digest=digest,
                injection=scan.to_dict(),
            )
        try:
            outcome = await asyncio.wait_for(spec.handler(args), timeout=spec.timeout_s)
        except asyncio.TimeoutError:
            error = f"工具超时（>{spec.timeout_s}s）：{name}"
            scan = scan_untrusted(error)
            return ToolResult(
                name=name,
                ok=False,
                text="",
                latency_ms=_elapsed(),
                source=spec.source,
                error=error,
                safe_text=redact_untrusted(error, scan),
                call_digest=digest,
                injection=scan.to_dict(),
            )
        except Exception as err:  # noqa: BLE001 —— 工具内部错误也要变成"观察"喂回模型
            # 工具错误文本同样不可信（MCP server / 技能可能把注入内容塞进异常消息）
            error = f"工具执行失败：{err}"
            scan = scan_untrusted(error)
            return ToolResult(
                name=name,
                ok=False,
                text="",
                latency_ms=_elapsed(),
                source=spec.source,
                error=error,
                safe_text=redact_untrusted(error, scan),
                call_digest=digest,
                injection=scan.to_dict(),
            )
        safe_text, safe_data, scan = sanitize_tool_output(
            outcome.text, outcome.data, source=spec.source
        )
        return ToolResult(
            name=name,
            ok=True,
            text=outcome.text,
            latency_ms=_elapsed(),
            source=spec.source,
            call_digest=digest,
            data=outcome.data,
            safe_text=safe_text,
            data_for_model=safe_data,
            injection=scan.to_dict(),
            cost_estimate_usd=spec.cost_estimate_usd,
        )
