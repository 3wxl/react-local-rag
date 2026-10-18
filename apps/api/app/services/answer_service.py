"""问答服务（M3 起是 QA 图的薄适配层）。

M2 之前这里是一条直线流程（检索 → 生成 → 校验 → 拒答），M3 把它搬进
`app/graphs/qa_nodes.py`（图编排：判级与拒答是一等分支，见 `app/graphs/qa_graph.py`），
本类只负责三件事：
1. 生成 `answer_id` / `trace_id`（LangGraph 的 `thread_id` 就用 trace_id，便于按轨迹回放）；
2. 播种 State 并调用图（`graph.ainvoke`，`recursion_limit` 兜底防跑飞）；
3. 把终态映射回 `AnswerOutcome`（HTTP 契约层直接消费，字段与 M2 完全一致）。

几个容易被忽略但契约要求的不变量（tests/test_api_contract.py 逐条断言，
改动图时逐条对照）：
1. 答案里出现的每个角标 [n] 必在 citations[].index 里；citations 可以包含没被用到的候选
2. verdict == "refused" ⟺ citations 为空且 refusal 非空
3. evidence_snapshot.passages[].index 与 citations[].index 同源同序
   （靠 retrieve 节点里 build_evidence 的单点装配保证）
4. 拒答是 200 正常响应，不是错误
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

from ..api.serializers import context_from_dict
from ..core.config import Settings
from ..generation.base import ModelUnavailableError
from ..graphs.budget import BudgetExceeded
from ..graphs.state import filters_to_dict
from ..parity.citation import ContextRef
from ..store import Filters

#: `recursion_limit` 相对步数预算的余量（见 answer() 里的注释）
RECURSION_HEADROOM = 4


@dataclass
class AnswerOutcome:
    """一次问答的全部产物（HTTP 层直接映射成契约响应）。"""

    answer_id: str
    trace_id: str
    question: str
    answer: str
    verdict: str
    citations: list[ContextRef]
    refusal: dict | None
    #: 已序列化的逐句校验（图节点产出，值等同 verification_to_dict）
    verification: dict
    evidence_snapshot: dict
    cost: dict
    trace_nodes: list[dict] = field(default_factory=list)


class AnswerService:
    """问答编排（M3：LangGraph QA 图，进程内，无队列依赖）。"""

    def __init__(self, graph, *, settings: Settings) -> None:
        self.graph = graph
        self.settings = settings

    def preflight(self, question: str, *, top_k: int) -> None:
        """进图前的预算预检（流式请求也必须在**流开始前**跑，才会返回真 422）。

        超限抛 `BudgetExceeded`；调用方（路由）把它转成 422 budget_exceeded。
        """
        from ..graphs.budget import preflight

        preflight(
            self.settings,
            question=question,
            top_k=top_k,
            max_new_tokens=self.settings.max_new_tokens,
        )

    async def answer(
        self,
        question: str,
        *,
        version: str,
        top_k: int = 5,
        filters: Filters | None = None,
        prompt_template_id: str = "qa-v1",
        rewrite: bool = False,
    ) -> AnswerOutcome:
        answer_id = "ans_" + uuid.uuid4().hex[:16]
        trace_id = "trc_" + uuid.uuid4().hex[:16]
        # 预检（超限直接拒绝，一次模型都不调）；路由在流式路径也会先调一遍
        self.preflight(question, top_k=top_k)

        seed = {
            "answer_id": answer_id,
            "trace_id": trace_id,
            "question": question,
            "version": version,
            "top_k": top_k,
            "filters_dict": filters_to_dict(filters),
            "prompt_template_id": prompt_template_id,
            "rewrite": rewrite,
            "started_perf": time.perf_counter(),
            "trace_nodes": [],
            "breakdown": {},
        }
        final = await self.graph.ainvoke(seed, config=self._config(trace_id))
        outcome = final.get("outcome") or {}
        return AnswerOutcome(
            answer_id=outcome["answer_id"],
            trace_id=outcome["trace_id"],
            question=outcome["question"],
            answer=outcome["answer"],
            verdict=outcome["verdict"],
            citations=[context_from_dict(c) for c in outcome["citations"]],
            refusal=outcome["refusal"],
            verification=outcome["verification"],
            evidence_snapshot=outcome["evidence_snapshot"],
            cost=outcome["cost"],
            trace_nodes=outcome["trace_nodes"],
        )

    async def answer_stream(
        self,
        question: str,
        *,
        version: str,
        top_k: int = 5,
        filters: Filters | None = None,
        prompt_template_id: str = "qa-v1",
        rewrite: bool = False,
    ) -> AsyncIterator[dict]:
        """流式问答：逐条产出 `{"event", "data"}`（路由负责转 SSE 帧）。

        与非流式**同一个图、同一条节点路径**（`stream_mode="custom"` 只是多开了一条
        输出通道）；`final` 事件的数据就是非流式响应体本身。

        错误语义（契约 §2）：预检类错误在流开始前由路由抛真 HTTP 状态码；
        流开始之后的错误一律**带内**（`error` 事件），因为 HTTP 头已经发出去了。
        """
        answer_id = "ans_" + uuid.uuid4().hex[:16]
        trace_id = "trc_" + uuid.uuid4().hex[:16]
        self.preflight(question, top_k=top_k)

        seed = {
            "answer_id": answer_id,
            "trace_id": trace_id,
            "question": question,
            "version": version,
            "top_k": top_k,
            "filters_dict": filters_to_dict(filters),
            "prompt_template_id": prompt_template_id,
            "rewrite": rewrite,
            "stream": True,
            "started_perf": time.perf_counter(),
            "trace_nodes": [],
            "breakdown": {},
        }
        try:
            async for payload in self.graph.astream(
                seed, config=self._config(trace_id), stream_mode="custom"
            ):
                yield payload
        except ModelUnavailableError as err:
            yield _error_event("model_unavailable", str(err))
        except BudgetExceeded as err:
            yield _error_event("budget_exceeded", str(err), err.detail)
        except Exception as err:  # noqa: BLE001 —— 流内错误必须带内交付，不能只留 500
            yield _error_event("internal_error", f"流式问答失败：{err}")

    def _config(self, trace_id: str) -> dict:
        return {
            # thread_id = trace_id：检查点按轨迹可查（M4 的轨迹回放依赖这一点）
            "configurable": {"thread_id": trace_id},
            # 兜底防跑飞：**预算守卫先触发**（给出"超了哪道上限"的友好原因），
            # 这里只是最后的保险，所以留出余量，避免 LangGraph 的 GraphRecursionError
            # 抢在 budget_exceeded 之前把请求打断（错误语义会变成"内部错误"）
            "recursion_limit": self.settings.graph_max_steps + RECURSION_HEADROOM,
        }


def _error_event(code: str, message: str, detail: dict | None = None) -> dict:
    """带内错误事件（形状与错误体 §0.5 一致）。"""
    return {
        "event": "error",
        "data": {"error": {"code": code, "message": message, "detail": detail or {}}},
    }
