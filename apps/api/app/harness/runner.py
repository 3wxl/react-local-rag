"""Harness v1：跑检索套件 → 指标 + 门槛判定。

两条向量来源（`mode`）：
- `fixed`：读 packages/eval/embeddings.json（**CI 与双端对比用**：确定、可复现、无需模型）
- `live`：用本地 ONNX 现场编码查询（验证"真实服务端"这条路径，数值与 fixed 有 ~1e-3 差异）

配置纪律（H8）：`config` 字段完整记录本次生效的全部参数，
杜绝"评测开精排、线上默认关"这类漂移 —— 所以这里是**读**默认值而不是另写一份。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from ..embedding.bge import BgeOnnxEmbedder
from ..embedding.fixed import assert_embeddings_consistent, load_fixed_embeddings
from ..parity.chunk_params import DEFAULT_CHUNK_OPTIONS
from ..parity.retrieval import DEFAULT_RETRIEVAL_OPTIONS
from .gates import check_gates
from .golden import Golden, golden_version, load_corpus_docs, load_golden
from .metrics import Metrics, evaluate

SUPPORTED_SUITES = ("retrieval", "all")


@dataclass
class EvalOutcome:
    suite: str
    golden_version: str
    config: dict
    metrics: dict
    gates_passed: bool
    failed_gates: list[str] = field(default_factory=list)
    metrics_raw: Metrics | None = None
    latency_ms: int = 0


def run_eval(
    eval_dir: Path,
    *,
    suite: str = "retrieval",
    embedder: BgeOnnxEmbedder | None = None,
    prefer_fixed: bool = True,
    top_k: int | None = None,
    config_overrides: dict | None = None,
) -> EvalOutcome:
    """跑一次评测（同步；HTTP 层用后台任务包成 202 + 轮询）。"""
    if suite not in SUPPORTED_SUITES:
        raise ValueError(f"暂不支持的 suite：{suite}（M1 支持 {SUPPORTED_SUITES}）")
    started = time.perf_counter()
    eval_dir = Path(eval_dir)
    golden: Golden = load_golden(eval_dir)
    top_k = top_k or golden.top_k

    options = DEFAULT_RETRIEVAL_OPTIONS
    if config_overrides:
        options = _apply_overrides(options, config_overrides)

    fixed = load_fixed_embeddings(eval_dir / "embeddings.json") if prefer_fixed else None
    if fixed is not None:
        # 固化向量与当前切块必须严格一致：不一致直接抛错，禁止"缺向量按 0 分"悄悄降级
        docs = load_corpus_docs(eval_dir)
        assert_embeddings_consistent(
            fixed,
            [d.content_hash for d in docs],
            [q.id for q in golden.queries],
        )
        docs = load_corpus_docs(eval_dir, chunk_vectors=fixed.chunks)
        query_vectors = fixed.queries
        # 查询扩展：与 TS 用同一份固化产物（expansions.json 的文本 + embeddings.json 的向量）
        expansions = load_expansions(eval_dir)

        def query_vector_of(query) -> np.ndarray | None:
            return query_vectors.get(query.id)

        mode = "fixed"
    else:
        docs = load_corpus_docs(eval_dir)
        if embedder is not None and embedder.available:
            vectors = embedder.embed_passages([d.content for d in docs])
            for doc, vec in zip(docs, vectors):
                doc.vector = vec

        expansions = {} if fixed is None else load_expansions(eval_dir)

        def query_vector_of(query) -> np.ndarray | None:
            if embedder is None or not embedder.available:
                return None
            return embedder.embed_query(query.question)

        mode = "live"

    metrics = evaluate(
        docs,
        golden.queries,
        top_k=top_k,
        options=options,
        query_vector_of=query_vector_of,
        expansions=(expansions or {}).get("texts"),
        expansion_vectors=fixed.expansions if fixed is not None else None,
    )
    passed, failed = check_gates(metrics)
    config = {
        "mode": mode,
        "top_k": top_k,
        "chunk_params": {
            "maxChunkTokens": DEFAULT_CHUNK_OPTIONS.max_chunk_tokens,
            "minChunkChars": DEFAULT_CHUNK_OPTIONS.min_chunk_chars,
            "maxChunkChars": DEFAULT_CHUNK_OPTIONS.max_chunk_chars,
            "sentenceOverlap": DEFAULT_CHUNK_OPTIONS.sentence_overlap,
        },
        "retrieval_config": {
            "rrf_k": options.rrf_k,
            "vec_weight": options.vec_weight,
            "bm25_weight": options.bm25_weight,
            "min_vec_score": options.min_vec_score,
            "bm25_exempt_rank": options.bm25_exempt_rank,
            # 多信号融合拒答的四路门限（2026-10-15 起生效）：评测配置必须记录生效值
            "min_idf_coverage": options.min_idf_coverage,
            "min_max_cosine": options.min_max_cosine,
            "min_cos_gap": options.min_cos_gap,
            "min_bm25": options.min_bm25,
            "relative_cut_ratio": options.relative_cut_ratio,
            "max_window_chars": options.max_window_chars,
            "window_radius": options.window_radius,
            "mmr_lambda": options.mmr_lambda,
            "mmr_pool_size": options.mmr_pool_size,
        },
    }
    return EvalOutcome(
        suite=suite,
        golden_version=golden_version(eval_dir),
        config=config,
        metrics=metrics.as_dict(),
        gates_passed=passed,
        failed_gates=failed,
        metrics_raw=metrics,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


def load_expansions(eval_dir: Path) -> dict | None:
    """读取查询扩展的固化产物（packages/eval/expansions.json）；不存在返回 None。

    与 TS 侧 loadFixedExpansions 同一份文件 —— 双端评测必须吃同一批改写，
    否则"双端同口径"就只剩检索算法本身，输入却已经不同了。
    """
    path = Path(eval_dir) / "expansions.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {"texts": raw.get("expansions", {})}


def _apply_overrides(options, overrides: dict):
    """影子评测（M4-H6）用的配置覆盖：只允许覆盖已存在的字段，避免拼错字段被忽略。"""
    from ..parity.retrieval import RetrievalOptions

    unknown = set(overrides) - set(options.__dict__)
    if unknown:
        raise ValueError(f"未知的检索参数：{sorted(unknown)}")
    return RetrievalOptions(**{**options.__dict__, **overrides})
