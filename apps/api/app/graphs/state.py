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
    #: 是否流式请求（generate 节点据此决定发不发 token 事件）。
    #: 注意：**必须是声明过的字段** —— LangGraph 只保留 schema 里声明过的 State 键，
    #: 没声明的键在进图时会被丢掉（曾因此让 token 事件静默消失）。
    stream: bool
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
    #: 预算账本（app/graphs/budget.py 的 new_ledger()；每次模型调用后整份写回）
    budget: dict


class IngestState(TypedDict, total=False):
    """摄取图 State（解析 → 分块 → 去重/版本比对 → 向量化 → 索引 → 自检，逐文档）。

    `done` 是**断点**（与 tasks.checkpoint["done"] 同源同序）：重跑时按它跳过已完成文档。
    """

    task_id: str
    #: upload（单文档上传）| reindex（全量重建，含灰度 ratio）
    mode: str
    ratio: float
    version: str
    #: upload 模式的输入
    upload_path: str
    upload_name: str
    #: 文档归属（由身份决定，不由请求方声明）
    acl: dict
    #: 本次要处理的文档清单 / 未完成队列（doc_id）
    docs: list[dict]
    pending: list[str]
    done: list[str]
    failed: list[dict]
    #: 单文档模式的摄取结果（HTTP 层用它写任务进度）
    results: list[dict]
    #: 增量统计：reused_vectors / embedded_chunks
    stats: dict
    error: str | None
    trace_nodes: Annotated[list[dict], operator.add]


class AgentState(TypedDict, total=False):
    """Agentic Graph State（planner → 工具 → 无进展检测 → HITL → 汇总）。

    `observations` / `tool_calls` / `signatures` 等是**整体替换**的通道：每个节点返回
    追加后的完整列表。这种写法在检查点恢复时是幂等的（节点重跑不会重复追加）。
    """

    run_id: str
    trace_id: str
    question: str
    version: str
    top_k: int
    filters_dict: dict
    #: run 开始时刻（ISO）：Agent 可能跨进程续跑，不能用单调时钟
    started_iso: str

    steps: int
    plan: dict
    observations: list[dict]
    signatures: list[dict]
    tool_calls: list[dict]
    evidence: list[dict]
    citations: list[dict]
    approved: list[str]
    #: 每次 run 的随机 nonce：planner 观察的不可信数据标签用它，工具内容无法预知/伪造
    untrusted_nonce: str
    #: 间接提示注入命中记录（工具 / 类别 / 风险 / 处置），进证据快照与轨迹
    injection_findings: list[dict]

    answer: str
    completion_meta: dict
    verification: dict
    verify_detail: dict
    must_refuse: bool
    verdict: str
    refusal: dict | None
    no_progress: dict
    outcome: dict
    cost: dict

    breakdown: Annotated[dict, merge_dicts]
    budget: dict
    trace_nodes: Annotated[list[dict], operator.add]


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
