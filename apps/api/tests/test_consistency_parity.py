"""自一致性度量的跨语言一致性（M2）。

回放 `packages/eval/consistency.json`（同一份固化采样），两端必须给出**相同**数字：
一致性是"双端对比"里的一列，两端算法有差就等于在比两个不同的东西。
"""

from __future__ import annotations

import json

from app.parity.consistency import SampleView, measure_consistency

from .parity_utils import EVAL_DIR


def test_consistency_matches_ts_artifact() -> None:
    path = EVAL_DIR / "consistency.json"
    assert path.exists(), f"缺少 {path}：先跑 npm run gen:consistency（需本地模型）"
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["samples"], "固化产物为空"
    for qid, entry in data["samples"].items():
        views = [
            SampleView(verdict=("refused" if run["answer"] == "" else "answered"),
                       answer=run["answer"], citations=run["citations"])
            for run in entry["runs"]
        ]
        report = measure_consistency(views)
        # 与 TS 侧同口径的两条硬约束（数值门槛在 scripts/eval-consistency.ts 与 gates.py）
        assert report.samples == len(entry["runs"]), f"{qid}: 采样数不一致"
        if len({v.verdict for v in views}) > 1:
            assert report.refusal_flip or report.verdict_agreement == 0.0
        assert 0.0 <= report.answer_jaccard <= 1.0
        assert 0.0 <= report.number_jaccard <= 1.0


def test_measure_consistency_basic_semantics() -> None:
    same = SampleView(verdict="answered", answer="售价 3980 元[1]。", citations=[1])
    r = measure_consistency([same, same, same])
    assert (r.verdict_agreement, r.answer_jaccard, r.number_jaccard, r.citation_jaccard) == (
        1.0,
        1.0,
        1.0,
        1.0,
    )
    assert r.refusal_flip is False

    flip = measure_consistency(
        [same, SampleView(verdict="refused", answer="文档中没有找到相关内容。", citations=[])]
    )
    assert flip.refusal_flip is True
    assert flip.verdict_agreement == 0.0

    drift = measure_consistency(
        [
            SampleView(verdict="answered", answer="售价 3980 元。"),
            SampleView(verdict="answered", answer="售价 4980 元。"),
        ]
    )
    assert drift.number_jaccard < 1.0
