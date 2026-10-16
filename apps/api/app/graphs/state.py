"""图的 State 定义（M3）。

**State 只放 JSON 原生类型**（str/int/float/list/dict），不放 dataclass / numpy：
- 检查点要把 State 序列化进 SQLite（官方 saver 的 serde 对任意对象不保证稳定），
- 轨迹回放要求 State 能原样落库、原样读回。

领域对象（`SearchOutcome` / `Completion` / `VerificationResult` / `ContextRef`）在节点内
构造与消费，跨节点只传它们的 dict 形式（走 `app/api/serializers.py` 的成对映射）。

State 字段名与 docs/ROADMAP-FULLSTACK.md §5 的强制清单对齐：
`question / rewritten_query / evidence[] / citations[] / verdict / tool_calls[] / cost / trace_id`
—— Harness 的指标与轨迹回放都依赖这个 schema。
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from ..store import Filters


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """dict 通道的合并 reducer（阶段耗时表这类"逐节点累加"的字段用）。"""
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class QaState(TypedDict, total=False):
    """QA 图 State（改写 → 检索 → 判级 → 生成 → 校验 → 拒答）。"""

    # ---- 输入（服务层播种） ----
    answer_id: str
    trace_id: str
    question: str
    version: str
    top_k: int
    filters_dict: dict
    prompt_template_id: str
    rewrite: bool
    #: 请求开始时刻（perf_counter）。QA 图不跨进程续跑，所以单调时钟可用；
    #: 需要跨进程的 Agent 图改用 ISO 时间戳（见 agent 图）。
    started_perf: float

    # ---- 强制字段（roadmap §5） ----
    rewritten_query: str | None
    evidence: list[dict]  # AnswerPassage: {index,label,text}
    citations: list[dict]  # 契约 §0.1 Citation（context_to_dict 的产物）
    verdict: str
    tool_calls: list[dict]
    cost: dict

    # ---- 中间产物 ----
    expansions: list[str]
    search_summary: dict
    gate: dict
    answer: str
    completion_meta: dict
    verification: dict
    verify_detail: dict
    must_refuse: bool
    refusal: dict | None
    outcome: dict

    # ---- 轨迹与耗时（reducer 通道） ----
    #: 节点轨迹行（seq/name/status/duration_ms/input_summary/output_summary），逐节点追加
    trace_nodes: Annotated[list[dict], operator.add]
    #: 阶段耗时表 {retrieve: ms, generate: ms, verify: ms}
    breakdown: Annotated[dict, merge_dicts]


def filters_to_dict(filters: Filters | None) -> dict:
    """Filters → State 里的普通 dict（dataclass 不进 State；检索节点再重建）。"""
    if filters is None:
        return {}
    return {
        "doc_ids": list(filters.doc_ids),
        "tags": list(filters.tags),
        "tenant_id": filters.tenant_id,
        "departments": list(filters.departments),
        "max_classification": filters.max_classification,
    }


def filters_from_dict(data: dict) -> Filters | None:
    """State dict → Filters（与 filters_to_dict 成对）。空 dict = 不过滤。"""
    if not data:
        return None
    return Filters(
        doc_ids=tuple(data.get("doc_ids") or ()),
        tags=tuple(data.get("tags") or ()),
        tenant_id=data.get("tenant_id"),
        departments=tuple(data.get("departments") or ()),
        max_classification=data.get("max_classification"),
    )
