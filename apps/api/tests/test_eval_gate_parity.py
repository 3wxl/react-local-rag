"""Harness 与 `npm run eval` 同口径：指标逐字段相等、门槛判定一致。

基准是 `packages/eval/baseline/ts-metrics.json`（TS 端实测指标的冻结副本，
因为 docs/eval/results 是 gitignore 的，CI 拿不到）。

这条测试是"双端可对比"的地基：指标算法只要有一处不同（分母、口径、四舍五入），
双端对比表就会得出错误结论。
"""

from __future__ import annotations

from app.harness.runner import run_eval

from .parity_utils import EVAL_DIR

#: 逐字段比对（含分组；这些字段名就是 CI 门槛取值的来源）
COMPARED_FIELDS = (
    "queryCount",
    "chunkCount",
    "hybrid",
    "bm25OnlyRecallAt3",
    "recallAt3",
    "recallAt5",
    "mrrAt10",
    "keyPointHitRate",
    "top3DuplicateRate",
    "top3NearDuplicateRate",
    "noHitQueryCount",
    "falseRefusalCount",
    "negativeCount",
    "negativeRefusedCount",
    "negativeRefusalRate",
)


def test_python_harness_matches_ts_baseline(ts_baseline: dict) -> None:
    # 不传 embedder：强制走固化向量路径（与 TS 的 `npm run eval` 完全同源）
    outcome = run_eval(EVAL_DIR, suite="retrieval", embedder=None, prefer_fixed=True)
    expected = ts_baseline["metrics"]
    assert ts_baseline["chunkParams"] == outcome.config["chunk_params"], (
        "分块参数与 TS 基线不一致（改分块=改索引口径）"
    )
    for field in COMPARED_FIELDS:
        assert outcome.metrics[field] == expected[field], (
            f"{field} 与 TS 基线不一致：PY={outcome.metrics[field]} TS={expected[field]}"
        )
    assert set(outcome.metrics["byGroup"]) == set(expected["byGroup"])
    for group, bucket in expected["byGroup"].items():
        py_bucket = outcome.metrics["byGroup"][group]
        for field in ("count", "recallAt3", "recallAt5", "noHit"):
            assert py_bucket[field] == bucket[field], (
                f"分组 {group}.{field} 不一致：PY={py_bucket[field]} TS={bucket[field]}"
            )


def test_gates_pass_on_frozen_baseline(ts_baseline: dict) -> None:
    outcome = run_eval(EVAL_DIR, suite="retrieval", embedder=None, prefer_fixed=True)
    assert outcome.gates_passed, f"Harness 门槛未通过：{outcome.failed_gates}"


def test_config_records_effective_values(ts_baseline: dict) -> None:
    """口径纪律：config 必须完整记录本次生效值（杜绝"评测开 A、线上默认 B"）。"""
    outcome = run_eval(EVAL_DIR, suite="retrieval", embedder=None, prefer_fixed=True)
    config = outcome.config
    assert config["mode"] == "fixed"
    assert config["top_k"] == ts_baseline["topK"]
    rc = config["retrieval_config"]
    for field in (
        "rrf_k",
        "vec_weight",
        "bm25_weight",
        "min_vec_score",
        "min_idf_coverage",
        "window_radius",
        "mmr_lambda",
    ):
        assert field in rc, f"config.retrieval_config 缺字段 {field}"
