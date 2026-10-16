"""单请求预算（M3）：步数 / token / 成本三道上限 —— 超限拒绝并说明原因。

M3 验收的一半就在这个文件：**超预算请求被拒绝且说明原因**（契约 §0.5 的
422 `budget_exceeded`）。两层执法：

1. **预检**（进图前）：用 `parity/token_count`（与 TS 同口径的估算器）估上界
   —— 问题 + top_k×窗口块上限 + 生成长度；超限连一次模型都不调就拒绝。
   流式请求也在这一步被拦：错误发生在流开始**之前**，所以仍是真正的 HTTP 422
   （流开始之后的错误才走带内 error 事件，见 docs/api-contract.md §2）。
2. **逐步记账**（真实用量）：每次模型调用后累加步数/token/成本，超限抛
   `BudgetExceeded`；非流式由路由转 422，流式转带内 error 事件。

为什么三样都要：步数是"Agent 跑飞了"的兜底，token 是"这次请求本身太贵"的判据，
成本是接云端通道后的账单口径 —— 归因不同，提示必须指到具体哪一个（detail.exceeded）。

成本口径：本地模型按 0 计（价目默认 0），所以 `estimated_cost_usd` 默认仍是 0；
接云端通道时配 `COST_PER_1K_PROMPT_USD / COST_PER_1K_COMPLETION_USD` 即生效。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.config import Settings
from ..parity.chunk_params import DEFAULT_CHUNK_OPTIONS
from ..parity.token_count import count_tokens


@dataclass(frozen=True)
class BudgetLimits:
    """单请求三道上限（来自 Settings，进程内固定）。"""

    max_steps: int
    max_tokens: int
    max_cost_usd: float

    @classmethod
    def from_settings(cls, settings: Settings) -> "BudgetLimits":
        return cls(
            max_steps=settings.graph_max_steps,
            max_tokens=settings.request_token_budget,
            max_cost_usd=settings.request_cost_limit_usd,
        )


class BudgetExceeded(RuntimeError):
    """超预算（路由把它转成 422 budget_exceeded 或流内 error 事件）。"""

    def __init__(self, message: str, *, detail: dict) -> None:
        super().__init__(message)
        #: 原样进错误体（detail.budget.exceeded 给出 exceeded 的具体那道上限）
        self.detail = detail


def new_ledger() -> dict:
    """空账本（进 State；纯 JSON 原生，能进检查点）。"""
    return {
        "steps": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tokens": 0,
        "cost_usd": 0.0,
    }


def price_usd(settings: Settings, prompt_tokens: int, completion_tokens: int) -> float:
    """按价目折算这一次调用的成本（默认价目为 0 → 本地通道成本恒为 0）。"""
    return round(
        prompt_tokens / 1000 * settings.cost_per_1k_prompt_usd
        + completion_tokens / 1000 * settings.cost_per_1k_completion_usd,
        8,
    )


def cost_dict(
    state: dict,
    *,
    latency_ms: int,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict:
    """契约 §0.4 的 cost 对象（QA 图与 Agent 图共用同一处口径）。

    `estimated_cost_usd` 取账本里的真实累计成本：默认价目为 0（本地通道）时与 M2 的
    固定 0.0 逐字段一致；接云端通道配了价目后才是真金白银的数字。
    """
    ledger = state.get("budget") or {}
    return {
        "latency_ms": latency_ms,
        "latency_breakdown_ms": dict(state.get("breakdown") or {}),
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        },
        "estimated_cost_usd": float(ledger.get("cost_usd", 0.0) or 0.0),
    }


def estimate_tokens(question: str, top_k: int, max_new_tokens: int) -> int:
    """预检的 token 上界估算。

    每段窗口全文按 `max_chunk_chars` 计（中文里 1 字符≈1 token，count_tokens 也是
    这个口径），加上问题本身与生成长度 —— 是上界而非精确值，预检只用来"提前拒绝明显超标的请求"。
    """
    return (
        count_tokens(question)
        + top_k * DEFAULT_CHUNK_OPTIONS.max_chunk_chars
        + max_new_tokens
    )


def _limits_payload(limits: BudgetLimits) -> dict:
    return {"tokens": limits.max_tokens, "steps": limits.max_steps, "cost_usd": limits.max_cost_usd}


def _exceeded(
    kind: str,
    *,
    limits: BudgetLimits,
    used: dict,
    stage: str,
    estimated: bool,
    message: str,
) -> BudgetExceeded:
    return BudgetExceeded(
        message,
        detail={
            "budget": {
                "limit": _limits_payload(limits),
                "used": used,
                "exceeded": kind,
                "estimated": estimated,
                "stage": stage,
            }
        },
    )


def preflight(
    settings: Settings,
    *,
    question: str,
    top_k: int,
    max_new_tokens: int,
) -> None:
    """进图前的预检：明显超标的请求不进图（也就不会产生任何模型调用）。"""
    limits = BudgetLimits.from_settings(settings)
    est_tokens = estimate_tokens(question, top_k, max_new_tokens)
    est_cost = price_usd(settings, est_tokens, max_new_tokens)
    used = {"tokens": est_tokens, "steps": 0, "cost_usd": est_cost}

    if est_tokens > limits.max_tokens:
        raise _exceeded(
            "tokens",
            limits=limits,
            used=used,
            stage="preflight",
            estimated=True,
            message=(
                f"超出单请求 token 预算：预计 tokens={est_tokens} > 上限 {limits.max_tokens}"
                "（可调低 top_k，或调大 REQUEST_TOKEN_BUDGET）"
            ),
        )
    if est_cost > limits.max_cost_usd:
        raise _exceeded(
            "cost_usd",
            limits=limits,
            used=used,
            stage="preflight",
            estimated=True,
            message=(
                f"超出单请求成本上限：预计 ${est_cost:.6f} > 上限 ${limits.max_cost_usd}"
                "（价目见 COST_PER_1K_* 配置）"
            ),
        )


def charge(
    ledger: dict | None,
    *,
    limits: BudgetLimits,
    settings: Settings,
    prompt_tokens: int,
    completion_tokens: int,
    stage: str,
) -> dict:
    """记一次模型调用（步数 + token + 成本）并检查上限；超限抛 BudgetExceeded。

    先记账再判：拒绝时 detail.used 里是**真实已用量**（含这次调用），
    而不是"还没跑"的估计值 —— 事后归因要看真实账。
    """
    led = dict(ledger or new_ledger())
    led["steps"] = int(led.get("steps", 0)) + 1
    led["prompt_tokens"] = int(led.get("prompt_tokens", 0)) + int(prompt_tokens or 0)
    led["completion_tokens"] = int(led.get("completion_tokens", 0)) + int(completion_tokens or 0)
    led["tokens"] = led["prompt_tokens"] + led["completion_tokens"]
    led["cost_usd"] = round(
        float(led.get("cost_usd", 0.0))
        + price_usd(settings, int(prompt_tokens or 0), int(completion_tokens or 0)),
        8,
    )

    used = {"tokens": led["tokens"], "steps": led["steps"], "cost_usd": led["cost_usd"]}
    if led["tokens"] > limits.max_tokens:
        raise _exceeded(
            "tokens",
            limits=limits,
            used=used,
            stage=stage,
            estimated=False,
            message=(
                f"超出单请求 token 预算：已用 tokens={led['tokens']} > 上限 {limits.max_tokens}"
                f"（发生在 {stage} 阶段）"
            ),
        )
    if led["steps"] > limits.max_steps:
        raise _exceeded(
            "steps",
            limits=limits,
            used=used,
            stage=stage,
            estimated=False,
            message=(
                f"超出单请求步数上限：已用 {led['steps']} 步 > 上限 {limits.max_steps}"
                f"（发生在 {stage} 阶段，GRAPH_MAX_STEPS 可调）"
            ),
        )
    if led["cost_usd"] > limits.max_cost_usd:
        raise _exceeded(
            "cost_usd",
            limits=limits,
            used=used,
            stage=stage,
            estimated=False,
            message=(
                f"超出单请求成本上限：已用 ${led['cost_usd']:.6f} > 上限 ${limits.max_cost_usd}"
                f"（发生在 {stage} 阶段）"
            ),
        )
    return led
