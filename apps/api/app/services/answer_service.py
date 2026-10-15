"""问答服务：检索 → 生成 → 逐句校验 → 引用/拒答 → 证据快照 → 轨迹。

与浏览器端 Lite 模式的普通 RAG 分支**同序同口径**：
`searchTopK → buildEvidence → generateAnswer → verifyAnswer → patchMessage`
（见 apps/web/src/App.tsx 的普通 RAG 分支）。

几个容易被忽略但契约要求的不变量（tests/test_api_contract.py 逐条断言）：
1. 答案里出现的每个角标 [n] 必在 citations[].index 里；citations 可以包含没被用到的候选
2. verdict == "refused" ⟺ citations 为空且 refusal 非空
3. evidence_snapshot.passages[].index 与 citations[].index 同源同序
4. 拒答是 200 正常响应，不是错误
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from ..embedding.bge import BgeOnnxEmbedder
from ..generation.base import LlmProvider, ModelUnavailableError
from ..parity.answer_prompt import AnswerPromptInput, PromptPassage, build_answer_prompt
from ..parity.chunk_params import CURRENT_INDEX_VERSION
from ..parity.citation import ContextRef, build_evidence, to_verify_passages
from ..parity.retrieval import DEFAULT_RETRIEVAL_OPTIONS
from ..parity.verify import (
    NO_RELEVANT_ANSWER,
    DEFAULT_VERIFY_THRESHOLDS,
    VerificationResult,
    build_refusal_result,
    strip_think_tags,
    verify_answer,
)
from ..store import AnswerRecord, Filters, Store, TraceRecord
from .retrieval_service import RetrievalService, SearchOutcome

#: 本地模型成本为 0（账单口径：只统计 token，成本为 0；接云端通道时按 provider 价目折算）
LOCAL_COST_USD = 0.0


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
    verification: VerificationResult
    evidence_snapshot: dict
    cost: dict
    trace_nodes: list[dict] = field(default_factory=list)
    #: 检索阶段快照（`/v1/retrieve` 复用同一套信号）
    search: SearchOutcome | None = None


class AnswerService:
    """问答编排（进程内，无队列依赖）。"""

    def __init__(
        self,
        store: Store,
        retrieval: RetrievalService,
        embedder: BgeOnnxEmbedder,
        provider: LlmProvider,
    ) -> None:
        self.store = store
        self.retrieval = retrieval
        self.embedder = embedder
        self.provider = provider

    # ---------------- 主流程 ----------------

    async def answer(
        self,
        question: str,
        *,
        version: str,
        top_k: int = 5,
        filters: Filters | None = None,
        prompt_template_id: str = "qa-v1",
    ) -> AnswerOutcome:
        answer_id = "ans_" + uuid.uuid4().hex[:16]
        trace_id = "trc_" + uuid.uuid4().hex[:16]
        started = time.perf_counter()
        nodes: list[dict] = []
        breakdown: dict[str, int] = {}

        # ── 1. 检索 ──
        t0 = time.perf_counter()
        search = await self.retrieval.search(
            question, version=version, top_k=top_k, filters=filters
        )
        breakdown["retrieve"] = int((time.perf_counter() - t0) * 1000)
        nodes.append(
            {
                "seq": 1,
                "name": "retrieve",
                "status": "ok",
                "duration_ms": breakdown["retrieve"],
                "input_summary": question,
                "output_summary": (
                    f"hits={len(search.hits)}, no_relevant_hit={search.no_relevant_hit}, "
                    f"dropped={search.dropped_count}, mode={search.mode}"
                ),
            }
        )

        # ── 2. 门限判定（拒答的一等分支）──
        if search.no_relevant_hit:
            nodes.append(
                {
                    "seq": 2,
                    "name": "gate",
                    "status": "ok",
                    "duration_ms": 0,
                    "input_summary": f"idf_coverage={search.idf_coverage:.4f}",
                    "output_summary": (
                        f"idf_coverage={search.idf_coverage:.4f} ≥ "
                        f"{DEFAULT_RETRIEVAL_OPTIONS.min_idf_coverage}; "
                        f"top_relevance=∅ → 拒答"
                    ),
                }
            )
            return await self._refuse(
                answer_id=answer_id,
                trace_id=trace_id,
                question=question,
                reason="no_relevant_hit",
                nodes=nodes,
                breakdown=breakdown,
                search=search,
                version=version,
                prompt_template_id=prompt_template_id,
                started=started,
            )

        nodes.append(
            {
                "seq": 2,
                "name": "gate",
                "status": "ok",
                "duration_ms": 0,
                "input_summary": f"idf_coverage={search.idf_coverage:.4f}",
                "output_summary": (
                    f"top_relevance={(search.top_relevance or 0):.4f} ≥ "
                    f"{DEFAULT_RETRIEVAL_OPTIONS.min_vec_score} → pass"
                ),
            }
        )

        # ── 3. 编号段落 + Prompt ──
        passages, contexts = build_evidence(search.hits)
        prompt = build_answer_prompt(
            AnswerPromptInput(
                question=question,
                passages=[
                    PromptPassage(index=p.index, label=p.label, text=p.text)
                    for p in passages
                ],
            )
        )

        # ── 4. 生成 ──
        t1 = time.perf_counter()
        try:
            completion = self.provider.complete(prompt)
        except ModelUnavailableError:
            raise
        breakdown["generate"] = int((time.perf_counter() - t1) * 1000)
        final_text = strip_think_tags(completion.text or "")
        nodes.append(
            {
                "seq": 3,
                "name": "generate",
                "status": "ok",
                "duration_ms": breakdown["generate"],
                "input_summary": f"prompt_tokens={completion.prompt_tokens}",
                "output_summary": f"completion_tokens={completion.completion_tokens}",
            }
        )

        # ── 5. 逐句校验（语义范围收敛到被引用的窗口块）──
        t2 = time.perf_counter()
        verification = await self._verify(
            final_text,
            contexts,
            version=version,
        )
        breakdown["verify"] = int((time.perf_counter() - t2) * 1000)
        nodes.append(
            {
                "seq": 4,
                "name": "verify",
                "status": "ok",
                "duration_ms": breakdown["verify"],
                "input_summary": f"sentences={len(verification.sentences)}",
                "output_summary": (
                    f"verdict={verification.verdict}, source={verification.source}, "
                    f"scoped_chunks={verification.scoped_chunk_count}"
                ),
            }
        )

        latency_ms = int((time.perf_counter() - started) * 1000)
        cost = {
            "latency_ms": latency_ms,
            "latency_breakdown_ms": breakdown,
            "tokens": {
                "prompt": completion.prompt_tokens,
                "completion": completion.completion_tokens,
                "total": completion.prompt_tokens + completion.completion_tokens,
            },
            "estimated_cost_usd": LOCAL_COST_USD,
        }
        snapshot = self._evidence_snapshot(
            passages=passages,
            version=version,
            prompt_template_id=prompt_template_id,
            model={"name": completion.model, "provider": completion.provider},
            sampling=completion.sampling,
        )
        outcome = AnswerOutcome(
            answer_id=answer_id,
            trace_id=trace_id,
            question=question,
            answer=final_text,
            verdict=verification.verdict,
            citations=contexts,
            refusal=None,
            verification=verification,
            evidence_snapshot=snapshot,
            cost=cost,
            trace_nodes=nodes,
            search=search,
        )
        await self._persist(outcome, version=version)
        return outcome

    # ---------------- 分支：拒答 ----------------

    async def _refuse(
        self,
        *,
        answer_id: str,
        trace_id: str,
        question: str,
        reason: str,
        nodes: list[dict],
        breakdown: dict[str, int],
        search: SearchOutcome,
        version: str,
        prompt_template_id: str,
        started: float,
    ) -> AnswerOutcome:
        latency_ms = int((time.perf_counter() - started) * 1000)
        cost = {
            "latency_ms": latency_ms,
            "latency_breakdown_ms": breakdown,
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "estimated_cost_usd": LOCAL_COST_USD,
        }
        outcome = AnswerOutcome(
            answer_id=answer_id,
            trace_id=trace_id,
            question=question,
            answer=NO_RELEVANT_ANSWER,
            verdict="refused",
            citations=[],
            refusal={
                "reason": reason,
                "message": NO_RELEVANT_ANSWER,
                "signals": {
                    "idf_coverage": search.idf_coverage,
                    "top_relevance": None,
                },
            },
            verification=build_refusal_result(NO_RELEVANT_ANSWER),
            evidence_snapshot=self._evidence_snapshot(
                passages=[],
                version=version,
                prompt_template_id=prompt_template_id,
                model={"name": "n/a", "provider": "none"},
                sampling={},
            ),
            cost=cost,
            trace_nodes=nodes,
            search=search,
        )
        await self._persist(outcome, version=version)
        return outcome

    # ---------------- 校验 ----------------

    async def _verify(
        self,
        answer_text: str,
        contexts: list[ContextRef],
        *,
        version: str,
    ) -> VerificationResult:
        """语义比对范围收敛到引用段落的窗口块（与 Lite 模式 toVerifyPassages 一致）。"""
        passages = to_verify_passages(contexts)
        scorer = None
        if self.embedder.available and passages:
            wanted = {cid for p in passages for cid in p.chunk_ids}
            all_vectors = await self.store.load_vectors(version)
            scope_vectors = [all_vectors[cid] for cid in wanted if cid in all_vectors]

            def scorer(
                texts: list[str], scope: list[str] | None = None
            ) -> list[float]:
                if not scope_vectors:
                    return [0.0] * len(texts)
                sent = self.embedder.embed_verify(texts)
                # 每句取「被引用块」内的最大余弦（与 worker 的 maxSimilarity 同义）
                return [float(np.max(sent[i] @ np.asarray(scope_vectors).T)) for i in range(len(texts))]

        return verify_answer(
            answer_text,
            # 词法覆盖率的分母 = 喂进 prompt 的段落文本
            #（AnswerPassage.text 与 ContextRef.content 同源，都是窗口全文）
            "\n\n".join(c.content for c in contexts),
            passages=passages,
            thresholds=DEFAULT_VERIFY_THRESHOLDS,
            semantic_scorer=scorer,
        )

    # ---------------- 快照与落库 ----------------

    def _evidence_snapshot(
        self,
        *,
        passages,
        version: str,
        prompt_template_id: str,
        model: dict,
        sampling: dict,
    ) -> dict:
        """把「模型当时看到了什么 + 用什么配置」冻结下来（事后不依赖在线状态即可复核）。"""
        opts = DEFAULT_RETRIEVAL_OPTIONS
        return {
            "passages": [
                {"index": p.index, "label": p.label, "text": p.text} for p in passages
            ],
            "prompt_template_id": prompt_template_id,
            "model": {**model, "sampling": sampling},
            "retrieval_config": {
                "top_k": opts.top_k,
                "rrf_k": opts.rrf_k,
                "vec_weight": opts.vec_weight,
                "bm25_weight": opts.bm25_weight,
                "min_vec_score": opts.min_vec_score,
                "min_idf_coverage": opts.min_idf_coverage,
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

    async def _persist(self, outcome: AnswerOutcome, *, version: str) -> None:
        # 序列化统一走 app/api/serializers.py（同一字段不允许两处写法）
        from ..api.serializers import context_to_dict, verification_to_dict

        record = AnswerRecord(
            answer_id=outcome.answer_id,
            trace_id=outcome.trace_id,
            question=outcome.question,
            answer=outcome.answer,
            verdict=outcome.verdict,
            citations=[context_to_dict(c) for c in outcome.citations],
            refusal=outcome.refusal,
            verification=verification_to_dict(outcome.verification),
            evidence_snapshot=outcome.evidence_snapshot,
            cost=outcome.cost,
            config={"index_version": version},
        )
        await self.store.save_answer(record)
        # trace 自包含（冗余存 evidence_snapshot）：answers 行被清理也能回放
        await self.store.save_trace(
            TraceRecord(
                trace_id=outcome.trace_id,
                answer_id=outcome.answer_id,
                question=outcome.question,
                nodes=outcome.trace_nodes,
                final_verdict=outcome.verdict,
                evidence_snapshot=outcome.evidence_snapshot,
                cost=outcome.cost,
            )
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


