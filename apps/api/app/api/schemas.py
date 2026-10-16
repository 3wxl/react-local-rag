"""请求体模型（与 docs/api-contract.md 逐字段对齐）。

- 请求一律 `extra="forbid"`：拼错字段名要报 400，而不是被静默忽略
  （静默忽略是"评测按 A 跑、线上按 B 跑"这类漂移的温床）
- 响应体直接用 dict 组装（字段多且嵌套深，用 Pydantic 再声明一遍反而容易与契约不一致；
  不变量由 tests/test_api_contract.py 断言）
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FiltersIn(StrictModel):
    """检索过滤（M2 起 ACL 在检索层生效；客户端传入的 filters 只能收紧不能放宽）。"""

    doc_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class RetrieveRequest(StrictModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    filters: FiltersIn | None = None
    #: 是否先做查询改写（M1 未实现改写，置 true 会在响应里显式回落到原文）
    rewrite: bool = False


class AnswerRequest(StrictModel):
    question: str = Field(min_length=1)
    conversation_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    filters: FiltersIn | None = None
    #: 流式（M3 支持 SSE）；拒绝而不是静默忽略，避免前端以为拿到了流
    stream: bool = False
    #: 查询改写（M3 起 QA 图有 rewrite 节点）。**默认关**：线上默认配置 = 评测默认配置，
    #: 且要与浏览器 Lite 模式的普通 RAG 分支同口径（那边不改写）。
    #: 开启与否都会冻结进 evidence_snapshot.rewrite，事后可复核"这次到底改没改"。
    rewrite: bool = False


class AgentRunRequest(StrictModel):
    """POST /v1/agent/run（M3）：planner + 工具循环，可能因人工审批挂起。"""

    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    filters: FiltersIn | None = None
    stream: bool = False


class AgentResumeRequest(StrictModel):
    """POST /v1/agent/{run_id}/resume：人工审批（HITL）的决定。"""

    approved: bool
    note: str | None = None


class EvalRunRequest(StrictModel):
    suite: str = "retrieval"
    config_overrides: dict = Field(default_factory=dict)
    golden_version: str | None = None
