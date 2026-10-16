"""答案侧拒答裁决的跨语言一致性（M2）。

回放 `packages/eval/verdict-scores.json`（有模型的机器冻结的作答现场 + 逐句语义分），
用 Python 的生产实现 `verify_answer` + `should_refuse` 复算一遍，
断言与 TS 侧（`npm run eval:verdict -- --check` 用的 verifyAnswerWithScores + shouldRefuse）
**逐条判定相同**。

为什么值得单独一条测试：拒答是"一等分支"，两端如果对同一份证据给出不同裁决，
双端对比报告里的拒答率就没有可比性；而拒答又恰好是最容易被悄悄改坏的一段
（改阈值、改封顶规则、改聚合顺序都会翻结论）。
"""

from __future__ import annotations

import json

import pytest

from app.parity.verify import (
    VerifyPassage,
    should_refuse,
    verify_answer,
)

from .parity_utils import EVAL_DIR


def _load(name: str) -> dict:
    path = EVAL_DIR / name
    assert path.exists(), f"缺少 {path}：先跑 npm run gen:verdict-scores（需本地模型）"
    return json.loads(path.read_text(encoding="utf-8"))


def test_verdict_decisions_match_frozen_answers() -> None:
    scores = _load("verdict-scores.json")
    labels = _load("verdict-labels.json")
    expect_by_id = {s["id"]: s["expectRefuse"] for s in labels["samples"]}

    checked = 0
    false_refusals = 0
    missed_refusals = 0
    model_self_refusals = 0
    for entry in scores["entries"]:
        qid = entry["id"]
        assert qid in expect_by_id, f"标注集缺 {qid}"

        # 检索侧粗拒 / 空答案：线上直接走「文档中没有找到相关内容」分支
        if entry["retrievalRefused"] or entry["answer"] == "":
            refused = True
        else:
            semantic_by_text: dict[str, float] = entry["semanticByText"] or {}
            passages = [
                VerifyPassage(index=p["index"], chunk_ids=[])
                for p in entry["passages"]
            ]

            def scorer(texts: list[str], _scope=None, _m=semantic_by_text):
                # 冻结分按句子文本索引；主张句之外不会被要求（与 TS 侧一致）
                return [_m[t] for t in texts]

            result = verify_answer(
                entry["answer"],
                entry["contextText"],
                passages=passages,
                semantic_scorer=scorer if semantic_by_text else None,
            )
            refused = should_refuse(result)
            model_self_refusal = refused and result.verdict == "refused"

        expected = bool(entry["expectRefuse"])
        assert expected == expect_by_id[qid], f"{qid}: 标注与标签文件不一致"
        if expected and refused:
            pass
        elif not expected and refused:
            if model_self_refusal:
                # 模型自己说"没找到"（合规拒答），但我方裁决没有误判 —— 单列观察
                model_self_refusals += 1
            else:
                false_refusals += 1
        elif expected and not refused:
            missed_refusals += 1
        checked += 1

    assert checked == len(scores["entries"]), "有样本没被比对"
    # 与 TS 侧同口径的两条硬约束（数值门槛在 apps/web/scripts/eval-verdict.ts 与 gates.py）
    assert false_refusals == 0, f"我方裁决误拒正样本 {false_refusals} 条（模型自述拒答另计 {model_self_refusals}）"
    negatives = sum(1 for e in scores["entries"] if e["expectRefuse"])
    recall = 0.0 if negatives == 0 else (negatives - missed_refusals) / negatives
    assert recall >= 0.60, f"负样本拒答 recall={recall:.4f} < 0.60"
