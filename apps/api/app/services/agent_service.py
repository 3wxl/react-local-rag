"""Agent 运行服务（M3）：把 Agentic Graph 包成"一次 run 的完整生命周期"。

与 AnswerService 的分工不同：问答是"一问一答"，Agent 是"可能挂起、可能续跑"的长事务，
所以这里要额外负责三件事：
1. **中断与续跑**：非流式被 `interrupt` 挡住时返回 `interrupted` + 待审批动作；
   续跑从检查点读回 State（版本与 ACL 过滤也在 State 里，续跑不会串到别的请求上）；
2. **事件流**：`run_stream` 把图的自定义事件（node/token/interrupt/error）原样吐出；
3. **每 run 一个工具注册表**：内置工具的检索必须绑定本次 run 的索引版本与 ACL 过滤，
   共享一个注册表会让并发请求互相串权限 —— 编译一张图的成本远低于一次模型调用，
   所以选择"每 run 编译"换取"隔离上不可能出错"。
"""

from __future__ import annotations

import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Iterator, Sequence

from ..core.config import Settings
from ..embedding.bge import BgeOnnxEmbedder
from ..generation.base import LlmProvider, ModelUnavailableError
from ..graphs.agent_graph import agent_recursion_limit, build_agent_graph
from ..graphs.agent_nodes import AgentDeps
from ..graphs.budget import BudgetExceeded, new_ledger, preflight
from ..graphs.state import filters_from_dict, filters_to_dict
from ..graphs.tools.builtin import build_builtin_tools
from ..graphs.tools.registry import ToolOutcome, ToolRegistry, ToolSpec
from ..mcp.audit import audit_context
from ..mcp.client import McpStdioClient, tool_schema_to_parameters
from ..skills import SkillRegistry
from ..store import Filters, Store
from .retrieval_service import RetrievalService


@dataclass
class RunOutcome:
    """一次 run 的结果（HTTP 层直接映射）。"""

    status: str  # succeeded | interrupted
    outcome: dict | None = None
    pending_approval: dict | None = None
    steps: int = 0
    tool_calls: list[dict] | None = None
    trace_nodes: list[dict] | None = None


class AgentService:
    """Agent 编排（进程内；检查点由 lifespan 注入的 saver 负责落盘）。"""

    def __init__(
        self,
        *,
        store: Store,
        retrieval: RetrievalService,
        embedder: BgeOnnxEmbedder,
        provider: LlmProvider,
        settings: Settings,
        checkpointer=None,
        mounted_tools: Sequence[ToolSpec] = (),
    ) -> None:
        self.store = store
        self.retrieval = retrieval
        self.embedder = embedder
        self.provider = provider
        self.settings = settings
        self.checkpointer = checkpointer
        #: 启动时挂载的外部工具（MCP）；每次 run 复制进新注册表
        self.mounted_tools = list(mounted_tools)
        #: Skills 注册表（manifest 加载期完成；调用时才起沙箱子进程）
        self.skills = (
            SkillRegistry.discover(settings.skill_dirs)
            if settings.skills_enabled
            else SkillRegistry([])
        )
        #: 外部技能目录里加载失败的条目（`GET /v1/skills` 原样返回，便于排查）
        self.skill_errors = list(self.skills.errors)

    # ---------------- 工具注册表 ----------------

    def build_registry(self, *, version: str, filters: Filters | None) -> ToolRegistry:
        registry = ToolRegistry()
        for spec in build_builtin_tools(
            store=self.store,
            retrieval=self.retrieval,
            embedder=self.embedder,
            settings=self.settings,
            version_getter=lambda: version,
            filters_getter=lambda: filters,
        ):
            registry.register(spec)
        for spec in self.mounted_tools:
            registry.register(spec)
        for spec in self.skills.tool_specs():
            registry.register(spec)
        return registry

    def skill_manifests(self) -> list[dict]:
        return self.skills.public_manifests()

    def _build_graph(self, registry: ToolRegistry):
        return build_agent_graph(
            AgentDeps(
                store=self.store,
                retrieval=self.retrieval,
                embedder=self.embedder,
                provider=self.provider,
                registry=registry,
                settings=self.settings,
            ),
            self.checkpointer,
        )

    def _config(self, run_id: str) -> dict:
        return {
            "configurable": {"thread_id": run_id},
            "recursion_limit": agent_recursion_limit(self.settings),
        }

    # ---------------- 启动一次 run ----------------

    def preflight(self, question: str, *, top_k: int) -> None:
        preflight(
            self.settings,
            question=question,
            top_k=top_k,
            max_new_tokens=self.settings.max_new_tokens,
        )

    def _seed(
        self,
        *,
        run_id: str,
        question: str,
        version: str,
        top_k: int,
        filters: Filters | None,
    ) -> dict:
        return {
            "run_id": run_id,
            "trace_id": run_id,
            "question": question,
            "version": version,
            "top_k": top_k,
            "filters_dict": filters_to_dict(filters),
            "started_iso": _now(),
            "steps": 0,
            "observations": [],
            "signatures": [],
            "tool_calls": [],
            "evidence": [],
            "citations": [],
            "approved": [],
            # 工具输出的不可信数据标签用随机 nonce：数据里无法预知标签，无法提前闭合
            "untrusted_nonce": secrets.token_hex(8),
            "injection_findings": [],
            "budget": new_ledger(),
            "trace_nodes": [],
            "breakdown": {},
        }

    async def run(
        self,
        question: str,
        *,
        run_id: str,
        version: str,
        top_k: int,
        filters: Filters | None = None,
    ) -> RunOutcome:
        self.preflight(question, top_k=top_k)
        graph = self._build_graph(self.build_registry(version=version, filters=filters))
        with _audit_run(run_id):
            result = await graph.ainvoke(
                self._seed(
                    run_id=run_id,
                    question=question,
                    version=version,
                    top_k=top_k,
                    filters=filters,
                ),
                config=self._config(run_id),
            )
        return _outcome_of(result)

    async def run_stream(
        self,
        question: str,
        *,
        run_id: str,
        version: str,
        top_k: int,
        filters: Filters | None = None,
    ) -> AsyncIterator[dict]:
        self.preflight(question, top_k=top_k)
        graph = self._build_graph(self.build_registry(version=version, filters=filters))
        seed = self._seed(
            run_id=run_id, question=question, version=version, top_k=top_k, filters=filters
        )
        try:
            with _audit_run(run_id):
                async for payload in graph.astream(
                    seed, config=self._config(run_id), stream_mode="custom"
                ):
                    yield payload
            # 中断不会产生 custom 事件：流结束后回读检查点，把"待审批"补发出去
            pending = await self.pending_approval(run_id)
            if pending is not None:
                from ..graphs.events import EVENT_INTERRUPT

                yield {"event": EVENT_INTERRUPT, "data": {"run_id": run_id, **pending}}
        except ModelUnavailableError as err:
            yield _error_event("model_unavailable", str(err))
        except BudgetExceeded as err:
            yield _error_event("budget_exceeded", str(err), err.detail)
        except Exception as err:  # noqa: BLE001 —— 流内错误必须带内交付
            yield _error_event("internal_error", f"Agent 运行失败：{err}")

    # ---------------- 续跑（HITL） ----------------

    async def pending_approval(self, run_id: str) -> dict | None:
        state_text = await self.aget_state(run_id)
        if state_text is None:
            return None
        pending = state_text.get("pending_approval")
        return pending if pending else None

    async def aget_state(self, run_id: str) -> dict | None:
        """读回一次 run 的检查点状态（含待审批动作）。"""
        if self.checkpointer is None:
            return None
        probe = self._build_graph(ToolRegistry())
        config = self._config(run_id)
        snapshot = await probe.aget_state(config)
        if not snapshot.values:
            return None
        pending = None
        for task in snapshot.tasks or ():
            for item in task.interrupts or ():
                pending = dict(item.value or {})
        return {
            "values": snapshot.values,
            "next": list(snapshot.next or ()),
            "pending_approval": pending,
        }

    async def resume(
        self,
        run_id: str,
        *,
        approved: bool,
        note: str | None = None,
    ) -> RunOutcome:
        from langgraph.types import Command

        state = await self.aget_state(run_id)
        if state is None:
            raise KeyError(run_id)
        if not state.get("pending_approval"):
            raise ValueError("该 run 没有待审批的工具调用")
        values = state["values"]
        registry = self.build_registry(
            version=str(values.get("version", "")),
            filters=filters_from_dict(values.get("filters_dict") or {}),
        )
        graph = self._build_graph(registry)
        with _audit_run(run_id):
            result = await graph.ainvoke(
                Command(resume={"approved": approved, "note": note or ""}),
                config=self._config(run_id),
            )
        return _outcome_of(result)

    async def resume_stream(
        self, run_id: str, *, approved: bool, note: str | None = None
    ) -> AsyncIterator[dict]:
        """续跑的事件流版本（供前端"批准后继续看流式输出"）。"""
        from langgraph.types import Command

        registry_probe = await self.aget_state(run_id)
        if registry_probe is None:
            raise KeyError(run_id)
        registry = self.build_registry(
            version=str(registry_probe["values"].get("version", "")),
            filters=filters_from_dict(registry_probe["values"].get("filters_dict") or {}),
        )
        graph = self._build_graph(registry)
        try:
            with _audit_run(run_id):
                async for payload in graph.astream(
                    Command(resume={"approved": approved, "note": note or ""}),
                    config=self._config(run_id),
                    stream_mode="custom",
                ):
                    yield payload
            pending = await self.pending_approval(run_id)
            if pending is not None:
                from ..graphs.events import EVENT_INTERRUPT

                yield {"event": EVENT_INTERRUPT, "data": {"run_id": run_id, **pending}}
        except ModelUnavailableError as err:
            yield _error_event("model_unavailable", str(err))
        except BudgetExceeded as err:
            yield _error_event("budget_exceeded", str(err), err.detail)
        except Exception as err:  # noqa: BLE001
            yield _error_event("internal_error", f"Agent 续跑失败：{err}")


async def mount_mcp_tools(client: McpStdioClient) -> list[ToolSpec]:
    """把 MCP server 的工具挂成内置工具同构的 ToolSpec（默认需要人工审批）。

    外部工具是"别人写的代码 + 可能写数据"：默认 `requires_approval=True`，
    宁可让用户多点一次批准，也不要让一次提示词注入顺手调用了外部写操作。
    """
    specs: list[ToolSpec] = []
    for tool in await client.list_tools():
        name = f"mcp:{tool.get('name', '')}"

        def _make(tool_name: str):
            async def handler(args: dict):
                result = await client.call_tool(tool_name, args)
                if not result.ok:
                    raise RuntimeError(result.error or "MCP 工具执行失败")
                return _mcp_outcome(result.text)

            return handler

        specs.append(
            ToolSpec(
                name=name,
                description=str(tool.get("description", "")),
                parameters=tool_schema_to_parameters(tool),
                handler=_make(str(tool.get("name", ""))),
                source="mcp",
                timeout_s=client.timeout_s,
                requires_approval=True,
            )
        )
    return specs


def _mcp_outcome(text: str) -> ToolOutcome:
    return ToolOutcome(text=text, data={"source": "mcp"})


def _outcome_of(result: dict) -> RunOutcome:
    interrupts = result.get("__interrupt__")
    if interrupts:
        pending = dict(getattr(interrupts[0], "value", {}) or {})
        return RunOutcome(
            status="interrupted",
            pending_approval=pending,
            steps=int(result.get("steps", 0)),
            tool_calls=list(result.get("tool_calls") or []),
            trace_nodes=list(result.get("trace_nodes") or []),
        )
    return RunOutcome(
        status="succeeded",
        outcome=result.get("outcome") or {},
        steps=int(result.get("steps", 0)),
        tool_calls=list(result.get("tool_calls") or []),
        trace_nodes=list(result.get("trace_nodes") or []),
    )


def _error_event(code: str, message: str, detail: dict | None = None) -> dict:
    return {
        "event": "error",
        "data": {"error": {"code": code, "message": message, "detail": detail or {}}},
    }


@contextmanager
def _audit_run(run_id: str) -> Iterator[None]:
    """把 run_id 注入审计上下文：MCP 客户端在深层工具 handler 里仍能记到归属。"""
    token = audit_context.set({"run_id": run_id})
    try:
        yield
    finally:
        audit_context.reset(token)


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
