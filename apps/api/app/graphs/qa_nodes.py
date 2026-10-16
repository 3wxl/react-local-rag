"""QA 图的节点实现（M3）：改写 → 检索 → 判级 → 生成 → 引用校验 → 拒答。

这些函数是 `services/answer_service.py` 原直线流程的**搬迁**，不是重写：
- 每个节点的轨迹摘要与 M2 逐字符一致（否则既有 trace 断言与人工对照都失效）；
- 检索、证据装配、Prompt、校验、拒答判定全部调用既有实现（parity/* 与 services/*），
  图只是把"什么时候调哪一步"变成显式编排（判级与拒答是一等分支）。

依赖用闭包注入（`build_qa_nodes(deps)`）：单进程应用里图在 lifespan 编译一次，
per-request 参数走 State，运行时对象（store/embedder/provider）不进 State —— State 要能序列化进检查点。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Awaitable, Callable

import numpy as np

from ..api.serializers import context_from_dict, context_to_dict, verification_to_dict
from ..core.config import Settings
from ..embedding.bge import BgeOnnxEmbedder
from ..generation.base import LlmProvider, ModelUnavailableError
from ..parity.answer_prompt import AnswerPromptInput, PromptPassage, build_answer_prompt
from ..parity.citation import build_evidence, to_verify_passages
from ..parity.expand_prompt import ExpandPromptInput, build_expand_prompt
from ..parity.query_expansion import parse_expanded_queries
from ..parity.retrieval import DEFAULT_RETRIEVAL_OPTIONS
from ..parity.verify import (
    DEFAULT_VERIFY_THRESHOLDS,
    NO_RELEVANT_ANSWER,
    build_refusal_result,
    should_refuse,
    strip_think_tags,
    verify_answer,
)
from ..store import AnswerRecord, Store, TraceRecord
from .budget import BudgetLimits, charge
from .events import EVENT_FINAL, EVENT_TOKEN, emit
from .state import QaState, filters_from_dict
from .trace import trace_row

if TYPE_CHECKING:  # 只为类型标注：避免与 services 层互相 import
    from ..services.retrieval_service import RetrievalService

#: 本地模型成本为 0（账单口径：只统计 token；接云端通道时按 provider 价目折算，见 budget.py）
LOCAL_COST_USD = 0.0

NodeFn = Callable[[QaState], Awaitable[dict]]


@dataclass
class QaDeps:
    """QA 图的外部依赖（生命周期与进程一致）。"""

    store: Store
    retrieval: "RetrievalService"
    embedder: BgeOnnxEmbedder
    provider: LlmProvider
    settings: Settings


def fired_signals(summary: dict) -> list[str]:
    """列出本次触发的拒答信号（多信号融合的归因）。

    向量降级（mode="bm25"）时融合拒答**不参与**判定，此时拒答只可能来自
    逐命中相关度门限/相对截断 —— 如实说明，避免把降级路径误读成"信号触发了"。
    """
    if summary.get("mode") != "hybrid":
        return ["向量路降级：融合拒答不参与（仅逐命中相关度门限）"]
    opts = DEFAULT_RETRIEVAL_OPTIONS
    fired: list[str] = []
    idf_coverage = summary.get("idf_coverage")
    if idf_coverage is not None and idf_coverage < opts.min_idf_coverage:
        fired.append(f"idf_coverage={idf_coverage:.4f}<{opts.min_idf_coverage}")
    max_cosine = summary.get("max_cosine")
    if max_cosine is not None and max_cosine < opts.min_max_cosine:
        fired.append(f"max_cosine={max_cosine:.4f}<{opts.min_max_cosine}")
    cos_gap = summary.get("cos_gap")
    if cos_gap is not None and cos_gap < opts.min_cos_gap:
        fired.append(f"cos_gap={cos_gap:.4f}<{opts.min_cos_gap}")
    max_bm25 = summary.get("max_bm25")
    if max_bm25 is not None and max_bm25 < opts.min_bm25:
        fired.append(f"max_bm25={max_bm25:.4f}<{opts.min_bm25}")
    if not fired:
        fired.append("融合信号均未触发（逐命中相关度门限/相对截断）")
    return fired


def build_qa_nodes(deps: QaDeps) -> dict[str, NodeFn]:
    """构造 QA 图节点（闭包持有依赖）。"""

    # ---------------- 0. 改写（按请求开启；默认关，见 AnswerRequest.rewrite） ----------------

    async def rewrite(state: QaState) -> dict:
        """口语化提问 → 术语化检索关键词（与 /v1/retrieve 的 rewrite 同一份实现）。

        为什么默认关：线上默认配置 = 评测默认配置，且浏览器 Lite 模式的普通 RAG 分支
        不改写 —— 默认开启会让双端口径分叉（M2 的改写收益在 /v1/retrieve 已验证，
        随时可以用 `rewrite: true` 升级默认值，但那要两端一起改）。
        """
        t0 = time.perf_counter()
        expansions: list[str] = []
        ledger = state.get("budget")
        status = "ok"
        summary = "无可用改写 → 仅原查询"
        try:
            completion = deps.provider.complete(
                build_expand_prompt(ExpandPromptInput(question=state["question"], max=3)),
                max_new_tokens=128,
            )
            # 改写也是一次模型调用：同样计入预算（超限抛 BudgetExceeded）
            ledger = charge(
                ledger,
                limits=BudgetLimits.from_settings(deps.settings),
                settings=deps.settings,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                stage="rewrite",
            )
            expansions = parse_expanded_queries(completion.text or "", state["question"])
            if expansions:
                summary = f"expansions={len(expansions)}：" + "；".join(expansions)
            else:
                # 模型没产出可用改写（输出是废话/复述原句）也如实记录，不假装改写过
                status = "skipped"
        except ModelUnavailableError:
            # 扩展是增强项，模型不可用不该把问答整个挡掉（与 /v1/retrieve 同一条降级纪律）
            status = "skipped"
            summary = "模型不可用 → 仅原查询"
        duration_ms = int((time.perf_counter() - t0) * 1000)
        row = trace_row(
            state,
            "rewrite",
            status=status,
            duration_ms=duration_ms,
            input_summary=state["question"],
            output_summary=summary,
        )
        return {
            "expansions": expansions,
            "rewritten_query": expansions[0] if expansions else None,
            "budget": ledger,
            "breakdown": {"rewrite": duration_ms},
            "trace_nodes": [row],
        }

    def route_rewrite(state: QaState) -> str:
        """条件入口：只有真的请求了改写才让 rewrite 节点进图。

        默认路径的轨迹因此与 M2 逐行一致（rewrite 行只在执行过时才出现），
        trace 差异即"这次有没有改写"，不留模糊。
        """
        return "rewrite" if state.get("rewrite") else "retrieve"

    # ---------------- 1. 检索 ----------------

    async def retrieve(state: QaState) -> dict:
        t0 = time.perf_counter()
        search = await deps.retrieval.search(
            state["question"],
            version=state["version"],
            top_k=state["top_k"],
            filters=filters_from_dict(state.get("filters_dict") or {}),
            expansions=state.get("expansions") or None,
        )
        duration_ms = int((time.perf_counter() - t0) * 1000)
        # 段落编号与引用卡片**同一次调用**产出：契约不变量 3（passages 与 citations 同源同序）
        # 靠的就是这个"单点装配"，两条路径分别构造迟早会错位。
        passages, contexts = build_evidence(search.hits)
        row = trace_row(
            state,
            "retrieve",
            duration_ms=duration_ms,
            input_summary=state["question"],
            output_summary=(
                f"hits={len(search.hits)}, no_relevant_hit={search.no_relevant_hit}, "
                f"dropped={search.dropped_count}, mode={search.mode}"
            ),
        )
        return {
            "evidence": [
                {"index": p.index, "label": p.label, "text": p.text} for p in passages
            ],
            "citations": [context_to_dict(c) for c in contexts],
            "search_summary": {
                "hits": len(search.hits),
                "no_relevant_hit": search.no_relevant_hit,
                "dropped_count": search.dropped_count,
                "mode": search.mode,
                "idf_coverage": search.idf_coverage,
                "top_relevance": search.top_relevance,
                "max_cosine": search.signals.max_cosine,
                "cos_gap": search.signals.cos_gap,
                "max_bm25": search.signals.max_bm25,
            },
            "breakdown": {"retrieve": duration_ms},
            "trace_nodes": [row],
        }

    # ---------------- 2. 判级（拒答的一等分支） ----------------

    async def gate(state: QaState) -> dict:
        summary = state.get("search_summary") or {}
        if summary.get("no_relevant_hit"):
            row = trace_row(
                state,
                "gate",
                input_summary=f"idf_coverage={summary['idf_coverage']:.4f}",
                # 拒答要能归因到具体信号：只说"拒了"无法区分是阈值太紧还是查询真的无关
                output_summary="多信号融合拒答命中 → 拒答；触发：" + "，".join(fired_signals(summary)),
            )
            return {
                "gate": {"decision": "refuse", "reason": "no_relevant_hit"},
                "trace_nodes": [row],
            }
        row = trace_row(
            state,
            "gate",
            input_summary=f"idf_coverage={summary['idf_coverage']:.4f}",
            output_summary=(
                f"top_relevance={(summary.get('top_relevance') or 0):.4f} ≥ "
                f"{DEFAULT_RETRIEVAL_OPTIONS.min_vec_score} → pass"
            ),
        )
        return {"gate": {"decision": "pass"}, "trace_nodes": [row]}

    def route_gate(state: QaState) -> str:
        return "refuse" if (state.get("gate") or {}).get("decision") == "refuse" else "generate"

    # ---------------- 3. 生成 ----------------

    async def generate(state: QaState) -> dict:
        prompt = build_answer_prompt(
            AnswerPromptInput(
                question=state["question"],
                passages=[
                    PromptPassage(index=p["index"], label=p["label"], text=p["text"])
                    for p in state["evidence"]
                ],
            )
        )
        t1 = time.perf_counter()
        # 流式：provider 支持 stream() 才发 token 事件（best-effort，见 generation/base.py）。
        # 发的是**原始**文本（未去思考标签），客户端以 final 事件里的 answer 为准。
        stream_fn = getattr(deps.provider, "stream", None)
        if state.get("stream") and callable(stream_fn):
            completion = stream_fn(
                prompt,
                max_new_tokens=deps.settings.max_new_tokens,
                on_delta=lambda delta: emit(EVENT_TOKEN, {"delta": delta}),
            )
        else:
            completion = deps.provider.complete(
                prompt, max_new_tokens=deps.settings.max_new_tokens
            )
        ledger = charge(
            state.get("budget"),
            limits=BudgetLimits.from_settings(deps.settings),
            settings=deps.settings,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            stage="generate",
        )
        duration_ms = int((time.perf_counter() - t1) * 1000)
        final_text = strip_think_tags(completion.text or "")
        row = trace_row(
            state,
            "generate",
            duration_ms=duration_ms,
            input_summary=f"prompt_tokens={completion.prompt_tokens}",
            output_summary=f"completion_tokens={completion.completion_tokens}",
        )
        return {
            "answer": final_text,
            "budget": ledger,
            "completion_meta": {
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "model": completion.model,
                "provider": completion.provider,
                "sampling": completion.sampling,
            },
            "breakdown": {"generate": duration_ms},
            "trace_nodes": [row],
        }

    # ---------------- 4. 逐句校验（语义范围收敛到被引用的窗口块） ----------------

    async def verify(state: QaState) -> dict:
        contexts = [context_from_dict(c) for c in state["citations"]]
        t2 = time.perf_counter()
        verification = await _verify(deps, state["answer"], contexts, version=state["version"])
        duration_ms = int((time.perf_counter() - t2) * 1000)
        row = trace_row(
            state,
            "verify",
            duration_ms=duration_ms,
            input_summary=f"sentences={len(verification.sentences)}",
            output_summary=(
                f"verdict={verification.verdict}, source={verification.source}, "
                f"scoped_chunks={verification.scoped_chunk_count}, "
                f"uncited={verification.convergence.uncited_claims}, "
                f"out_of_range={verification.convergence.out_of_range_claims}"
            ),
        )
        return {
            "verification": verification_to_dict(verification),
            "verdict": verification.verdict,
            "must_refuse": should_refuse(verification),
            # 拒答要能复盘：把逐句裁决与引用闭环的摘要一并落库
            "verify_detail": {
                "verdict": verification.verdict,
                "verify_source": verification.source,
                "uncited_claims": verification.convergence.uncited_claims,
                "out_of_range_claims": verification.convergence.out_of_range_claims,
            },
            "breakdown": {"verify": duration_ms},
            "trace_nodes": [row],
        }

    def route_verify(state: QaState) -> str:
        return "refuse" if state.get("must_refuse") else "finalize"

    # ---------------- 5. 拒答（两条分支共用同一套产物口径） ----------------

    async def refuse_no_relevant(state: QaState) -> dict:
        outcome = _refusal_outcome(state, reason="no_relevant_hit")
        await _persist(deps.store, outcome, version=state["version"])
        emit(EVENT_FINAL, public_outcome(outcome))
        return _refusal_updates(outcome)

    async def refuse_unsupported(state: QaState) -> dict:
        detail = state.get("verify_detail") or {}
        row = trace_row(
            state,
            "adjudicate",
            input_summary=f"verdict={detail.get('verdict')}, source={detail.get('verify_source')}",
            output_summary=(
                "逐句证据不足以支撑回答 → 拒答"
                "（模型自己拒答 / 全部主张句无依据 / 语义证据不可用且非全绿）"
            ),
        )
        outcome = _refusal_outcome(state, reason="answer_unsupported", detail=detail, extra_rows=[row])
        await _persist(deps.store, outcome, version=state["version"])
        emit(EVENT_FINAL, public_outcome(outcome))
        return {**_refusal_updates(outcome), "trace_nodes": [row]}

    # ---------------- 6. 成功收尾 ----------------

    async def finalize(state: QaState) -> dict:
        meta = state.get("completion_meta") or {}
        latency_ms = int((time.perf_counter() - state["started_perf"]) * 1000)
        cost = _cost_dict(
            state,
            latency_ms=latency_ms,
            prompt_tokens=int(meta.get("prompt_tokens", 0) or 0),
            completion_tokens=int(meta.get("completion_tokens", 0) or 0),
        )
        snapshot = evidence_snapshot(
            passages=state["evidence"],
            version=state["version"],
            prompt_template_id=state["prompt_template_id"],
            model={"name": meta.get("model", ""), "provider": meta.get("provider", "")},
            sampling=meta.get("sampling") or {},
            rewrite_info=_rewrite_info(state),
        )
        outcome = {
            "answer_id": state["answer_id"],
            "trace_id": state["trace_id"],
            "question": state["question"],
            "answer": state["answer"],
            "verdict": state["verdict"],
            "citations": state["citations"],
            "refusal": None,
            "verification": state["verification"],
            "evidence_snapshot": snapshot,
            "cost": cost,
            "trace_nodes": state["trace_nodes"],
        }
        await _persist(deps.store, outcome, version=state["version"])
        # final 事件与非流式响应体是同一个对象（落库与发事件同点，见 docs/api-contract.md §2）
        emit(EVENT_FINAL, public_outcome(outcome))
        return {"outcome": outcome, "cost": cost}

    return {
        "rewrite": rewrite,
        "route_rewrite": route_rewrite,  # type: ignore[dict-item]
        "retrieve": retrieve,
        "gate": gate,
        "route_gate": route_gate,  # type: ignore[dict-item]
        "generate": generate,
        "verify": verify,
        "route_verify": route_verify,  # type: ignore[dict-item]
        "refuse_no_relevant": refuse_no_relevant,
        "refuse_unsupported": refuse_unsupported,
        "finalize": finalize,
    }


# ---------------- 校验实现（从 AnswerService._verify 搬迁） ----------------


async def _verify(
    deps: QaDeps,
    answer_text: str,
    contexts: list,
    *,
    version: str,
):
    """语义比对范围收敛到引用段落的窗口块（与 Lite 模式 toVerifyPassages 一致）。"""
    passages = to_verify_passages(contexts)
    scorer = None
    if deps.embedder.available and passages:
        wanted = {cid for p in passages for cid in p.chunk_ids}
        all_vectors = await deps.store.load_vectors(version)
        scope_vectors = [all_vectors[cid] for cid in wanted if cid in all_vectors]

        def scorer(texts: list[str], scope: list[str] | None = None) -> list[float]:
            if not scope_vectors:
                return [0.0] * len(texts)
            sent = deps.embedder.embed_verify(texts)
            # 每句取「被引用块」内的最大余弦（与 worker 的 maxSimilarity 同义）
            return [
                float(np.max(sent[i] @ np.asarray(scope_vectors).T)) for i in range(len(texts))
            ]

    return verify_answer(
        answer_text,
        # 词法覆盖率的分母 = 喂进 prompt 的段落文本
        #（AnswerPassage.text 与 ContextRef.content 同源，都是窗口全文）
        "\n\n".join(c.content for c in contexts),
        passages=passages,
        thresholds=DEFAULT_VERIFY_THRESHOLDS,
        semantic_scorer=scorer,
    )


# ---------------- 拒答 / 成本 / 快照 / 落库 ----------------


def _cost_dict(
    state: QaState,
    *,
    latency_ms: int,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict:
    return {
        "latency_ms": latency_ms,
        "latency_breakdown_ms": dict(state.get("breakdown") or {}),
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        },
        # 成本口径：按价目折算（默认价目 0 → 本地通道仍是 0.0，与 M2 逐字段一致）
        "estimated_cost_usd": float((state.get("budget") or {}).get("cost_usd", LOCAL_COST_USD) or LOCAL_COST_USD),
    }


def _refusal_outcome(
    state: QaState,
    *,
    reason: str,
    detail: dict | None = None,
    extra_rows: list[dict] | None = None,
) -> dict:
    """拒答产物（与 M2 的 `AnswerService._refuse` 逐字段一致）。"""
    summary = state.get("search_summary") or {}
    latency_ms = int((time.perf_counter() - state["started_perf"]) * 1000)
    nodes = list(state.get("trace_nodes") or [])
    if extra_rows:
        nodes = [*nodes, *extra_rows]
    return {
        "answer_id": state["answer_id"],
        "trace_id": state["trace_id"],
        "question": state["question"],
        "answer": NO_RELEVANT_ANSWER,
        "verdict": "refused",
        "citations": [],
        "refusal": {
            "reason": reason,
            "message": NO_RELEVANT_ANSWER,
            "signals": {
                "idf_coverage": summary.get("idf_coverage"),
                "top_relevance": None,
                # 融合拒答的四路取值（与 /v1/retrieve 同一组字段，便于归因）
                "max_cosine": summary.get("max_cosine"),
                "cos_gap": summary.get("cos_gap"),
                "max_bm25": summary.get("max_bm25"),
            },
            # 答案侧细拒的判据（reason="answer_unsupported" 时给出逐句证据摘要）
            **(detail or {}),
        },
        # 拒答一律返回「拒答形态」的校验结果（逐句明细在 refusal.detail 里）
        "verification": verification_to_dict(build_refusal_result(NO_RELEVANT_ANSWER)),
        "evidence_snapshot": evidence_snapshot(
            passages=[],
            version=state["version"],
            prompt_template_id=state["prompt_template_id"],
            model={"name": "n/a", "provider": "none"},
            sampling={},
            rewrite_info=_rewrite_info(state),
        ),
        "cost": _cost_dict(state, latency_ms=latency_ms, prompt_tokens=0, completion_tokens=0),
        "trace_nodes": nodes,
    }


def public_outcome(outcome: dict) -> dict:
    """契约响应体（`final` 事件的载荷）：与 `/v1/answer` 非流式响应**逐字段相同**。

    `outcome` 里额外带着 `trace_nodes`（落库用），但它不属于响应体 —— 直接把它当
    final 事件发出去会让"流式 vs 非流式"多出一个字段（`test_sse_stream.py` 会红）。
    """
    return {
        "answer_id": outcome["answer_id"],
        "trace_id": outcome["trace_id"],
        "question": outcome["question"],
        "answer": outcome["answer"],
        "verdict": outcome["verdict"],
        "citations": outcome["citations"],
        "refusal": outcome["refusal"],
        "verification": outcome["verification"],
        "evidence_snapshot": outcome["evidence_snapshot"],
        "cost": outcome["cost"],
    }


def _refusal_updates(outcome: dict) -> dict:
    """拒答节点对 State 的更新（把终态字段也写回去，便于流式路径读 State）。"""
    return {
        "outcome": outcome,
        "verdict": outcome["verdict"],
        "answer": outcome["answer"],
        "citations": [],
        "refusal": outcome["refusal"],
        "verification": outcome["verification"],
        "cost": outcome["cost"],
        "trace_nodes": [],
    }


def _rewrite_info(state: QaState) -> dict | None:
    """改写节点的实际执行情况（只在请求了改写时进快照）。

    默认路径（rewrite=false）**不加这个字段**：契约「只增不改」允许新增字段，
    但默认路径连"多一个字段"都不要 —— 双端快照逐字节可比是评测纪律的一部分。
    """
    if not state.get("rewrite"):
        return None
    expansions = list(state.get("expansions") or [])
    return {"enabled": True, "applied": bool(expansions), "expansions": expansions}


def evidence_snapshot(
    *,
    passages: list[dict],
    version: str,
    prompt_template_id: str,
    model: dict,
    sampling: dict,
    rewrite_info: dict | None = None,
) -> dict:
    """把「模型当时看到了什么 + 用什么配置」冻结下来（事后不依赖在线状态即可复核）。"""
    opts = DEFAULT_RETRIEVAL_OPTIONS
    snapshot = {
        "passages": [
            {"index": p["index"], "label": p["label"], "text": p["text"]} for p in passages
        ],
        "prompt_template_id": prompt_template_id,
        "model": {**model, "sampling": sampling},
        "retrieval_config": {
            "top_k": opts.top_k,
            "rrf_k": opts.rrf_k,
            "vec_weight": opts.vec_weight,
            "bm25_weight": opts.bm25_weight,
            "min_vec_score": opts.min_vec_score,
            # 多信号融合拒答的四路门限（2026-10-15 起生效）：快照必须冻结它们，
            # 否则「当时为什么拒答」事后无法复核（口径纪律：线上配置 = 评测配置）
            "min_idf_coverage": opts.min_idf_coverage,
            "min_max_cosine": opts.min_max_cosine,
            "min_cos_gap": opts.min_cos_gap,
            "min_bm25": opts.min_bm25,
            "window_radius": opts.window_radius,
            "mmr_lambda": opts.mmr_lambda,
        },
        "verify_thresholds": {
            "supported": DEFAULT_VERIFY_THRESHOLDS.supported,
            "weak": DEFAULT_VERIFY_THRESHOLDS.weak,
            "sem_weight": DEFAULT_VERIFY_THRESHOLDS.sem_weight,
            "lex_weight": DEFAULT_VERIFY_THRESHOLDS.lex_weight,
        },
        # 契约样例写的是 sha256:…，但 TS 侧索引版本是 fnv1a32（8 位十六进制）。
        # 这里保留真实算法名 + 同值，避免"两端索引版本无法对照"——
        # 换哈希算法会让旧索引全部失效却换不来任何东西（见 docs/api-contract.md 的 PR 说明）。
        "index_version": f"fnv1a32:{version}",
        "created_at": _now(),
    }
    if rewrite_info is not None:
        # 只增字段：请求了改写才出现，默认路径快照逐字节不变
        snapshot["rewrite"] = rewrite_info
    return snapshot


async def _persist(store: Store, outcome: dict, *, version: str) -> None:
    """落库（answers + 自包含的 traces）。成功与拒答走同一处，流式路径才不用分叉。"""
    await store.save_answer(
        AnswerRecord(
            answer_id=outcome["answer_id"],
            trace_id=outcome["trace_id"],
            question=outcome["question"],
            answer=outcome["answer"],
            verdict=outcome["verdict"],
            citations=outcome["citations"],
            refusal=outcome["refusal"],
            verification=outcome["verification"],
            evidence_snapshot=outcome["evidence_snapshot"],
            cost=outcome["cost"],
            config={"index_version": version},
        )
    )
    # trace 自包含（冗余存 evidence_snapshot）：answers 行被清理也能回放
    await store.save_trace(
        TraceRecord(
            trace_id=outcome["trace_id"],
            answer_id=outcome["answer_id"],
            question=outcome["question"],
            nodes=outcome["trace_nodes"],
            final_verdict=outcome["verdict"],
            evidence_snapshot=outcome["evidence_snapshot"],
            cost=outcome["cost"],
        )
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
