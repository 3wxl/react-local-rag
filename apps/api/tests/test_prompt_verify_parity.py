"""Prompt 与校验算法的跨语言一致性。

- Prompt：**逐字节**相同（模板里一个空格/换行不同，模型行为就可能不同，双端对比失真）
- 校验：切句、词法覆盖率、数字核查、角标解析、加权打分与等级判定全部一致

用例的输入与 TS 侧期望值都来自 `scripts/parity-dump.ts --section prompt|verify`，
Python 只用**生产实现**重算，不复制 TS 的逻辑。
"""

from __future__ import annotations

import pytest

from app.parity.answer_prompt import (
    AnswerPromptInput,
    HistoryTurn,
    PromptPassage,
    build_answer_prompt,
)
from app.parity.verify import (
    DEFAULT_VERIFY_THRESHOLDS,
    VerifyPassage,
    VerifyThresholds,
    compute_sentence_score,
    level_of,
    lexical_coverage,
    parse_citations,
    verify_answer,
)
from app.parity.tokenize import tokenize_set

from .parity_utils import requires_node, run_parity_dump


def _thresholds_of(spec: dict) -> VerifyThresholds:
    """TS 侧 camelCase 阈值 → Python dataclass（默认值兜底，避免字段缺省时静默用错口径）。"""
    t = spec["thresholds"]
    return VerifyThresholds(
        supported=t["supported"],
        weak=t["weak"],
        sem_weight=t["semWeight"],
        lex_weight=t["lexWeight"],
    )


@requires_node
def test_prompt_bytes_match_ts() -> None:
    dump = run_parity_dump("prompt")
    assert dump["prompt"]["cases"], "prompt 导出为空"
    for case in dump["prompt"]["cases"]:
        py_prompt = build_answer_prompt(
            AnswerPromptInput(
                question=case["question"],
                passages=[PromptPassage(**p) for p in case["passages"]],
                history=tuple(HistoryTurn(**h) for h in case.get("history", [])),
                history_summary=case.get("historySummary"),
            )
        )
        assert py_prompt == case["prompt"], (
            f"{case['name']}: Prompt 与 TS 不一致\n"
            f"TS={case['prompt']!r}\nPY={py_prompt!r}"
        )


@requires_node
def test_expand_prompt_bytes_and_parser_match_ts() -> None:
    """查询扩展的 Prompt 逐字节一致；解析器对弱模型的乱格式处理也一致。"""
    from app.parity.expand_prompt import ExpandPromptInput, build_expand_prompt
    from app.parity.query_expansion import parse_expanded_queries

    dump = run_parity_dump("expand")
    spec = dump["expand"]
    for case in spec["cases"]:
        py_prompt = build_expand_prompt(
            ExpandPromptInput(question=case["question"], max=case["max"])
        )
        assert py_prompt == case["prompt"], (
            f"{case['name']}: 扩展 Prompt 与 TS 不一致\n"
            f"TS={case['prompt']!r}\nPY={py_prompt!r}"
        )
    for case in spec["parseCases"]:
        assert parse_expanded_queries(case["raw"], case["original"]) == case["parsed"], (
            f"{case['name']}: 改写解析与 TS 不一致"
        )


@requires_node
def test_verify_sentences_match_ts() -> None:
    dump = run_parity_dump("verify")
    spec = dump["verify"]
    context_text = spec["contextText"]
    thresholds = _thresholds_of(spec)

    # TS 侧语义分按句给出；这里按文本建索引回放（verify_answer 的分组调用会逐句取用）
    for case in spec["cases"]:
        expected = case["sentences"]
        # 用例可以带自己的编号段落表（如「编号不连续」边界），缺省用共享的那张
        passages = [
            VerifyPassage(index=p["index"], chunk_ids=p["chunkIds"])
            for p in case.get("passages", spec["passages"])
        ]
        semantic_by_text = {
            row["text"]: row["semantic"]
            for row in expected
            if row["semantic"] is not None
        }

        def scorer(texts: list[str], _scope=None, _m=semantic_by_text):
            # 未在 TS 侧打过分的句子不会被要求（TS 也只对主张句打分）
            return [_m[t] for t in texts]

        result = verify_answer(
            case["answer"],
            context_text,
            passages=passages,
            thresholds=thresholds,
            semantic_scorer=scorer,
        )
        assert len(result.sentences) == len(expected), (
            f"{case['name']}: 句子数不一致 PY={len(result.sentences)} TS={len(expected)}"
        )
        for py_s, ts_s in zip(result.sentences, expected):
            where = f"{case['name']} 句[{ts_s['index']}]"
            assert py_s.text == ts_s["text"], f"{where}: 句子文本不一致"
            assert py_s.trivial == ts_s["trivial"], f"{where}: trivial 判定不一致"
            assert py_s.lexical == pytest.approx(ts_s["lexical"], abs=1e-12), (
                f"{where}: 词法覆盖率不一致 PY={py_s.lexical} TS={ts_s['lexical']}"
            )
            assert py_s.unknown_numbers == ts_s["unknownNumbers"], (
                f"{where}: 数字核查不一致 PY={py_s.unknown_numbers} TS={ts_s['unknownNumbers']}"
            )
            assert py_s.citations == ts_s["citations"], f"{where}: 角标解析不一致"
            assert py_s.invalid_citations == ts_s["invalidCitations"], (
                f"{where}: 越界角标判定不一致"
            )
            assert py_s.uncited == ts_s["uncited"], f"{where}: uncited 判定不一致"
            assert py_s.score == pytest.approx(ts_s["score"], abs=1e-12), (
                f"{where}: 依据分不一致 PY={py_s.score} TS={ts_s['score']}"
            )
            assert py_s.level == ts_s["level"], (
                f"{where}: 等级不一致 PY={py_s.level} TS={ts_s['level']}"
            )
            if not ts_s["trivial"]:
                assert py_s.semantic == pytest.approx(ts_s["semantic"], abs=1e-12)

        # 引用闭环收敛（逐句角标 → 越界 → 收敛）：汇总规则也必须同口径
        ts_conv = case["convergence"]
        assert result.convergence.uncited_claims == ts_conv["uncitedClaims"], (
            f"{case['name']}: uncitedClaims 不一致"
        )
        assert result.convergence.out_of_range_claims == ts_conv["outOfRangeClaims"], (
            f"{case['name']}: outOfRangeClaims 不一致"
        )
        assert (
            result.convergence.citations_subset_of_passages
            == ts_conv["citationsSubsetOfPassages"]
        ), f"{case['name']}: citationsSubsetOfPassages 不一致"


@requires_node
def test_lexical_layer_matches_ts_dump() -> None:
    """词法层单独再比一遍（不依赖 verify_answer 的胶水逻辑）。"""
    dump = run_parity_dump("verify")
    spec = dump["verify"]
    context_text = spec["contextText"]
    context_tokens = tokenize_set(context_text)
    assert sorted(context_tokens) == spec["contextTokens"], "上下文 token 集合不一致"
    thresholds = _thresholds_of(spec)
    refusal_phrases = ("没有找到", "未找到", "找不到", "无法回答", "未能找到", "不含相关", "没有相关")
    for case in spec["cases"]:
        # 拒答分支的 lexical 是构造值（固定 1），不是覆盖率 —— 那一支由上一个测试覆盖
        is_refusal = len(case["answer"]) <= 80 and any(
            p in case["answer"] for p in refusal_phrases
        )
        if is_refusal:
            continue
        for ts_s in case["sentences"]:
            py_lexical = lexical_coverage(ts_s["text"], context_tokens)
            assert py_lexical == pytest.approx(ts_s["lexical"], abs=1e-12)
            py_citations = parse_citations(ts_s["text"])
            assert py_citations == ts_s["citations"]
            if ts_s["trivial"]:
                continue
            py_score = compute_sentence_score(
                ts_s["semantic"],
                ts_s["lexical"],
                thresholds,
                fabricated_number=bool(ts_s["unknownNumbers"]),
                dangling_citation=bool(ts_s["invalidCitations"]),
                uncited_claim=bool(ts_s["uncited"]),
            )
            assert py_score == pytest.approx(ts_s["score"], abs=1e-12)
            assert level_of(py_score, thresholds) == ts_s["level"]
