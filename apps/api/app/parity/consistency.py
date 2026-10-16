"""自一致性度量（apps/web/src/utils/selfConsistency.ts 的移植）。

定位：弱模型同题多次采样的**稳定性**是可信度下界 —— 三次给出三套数字说明它在编，
这时不该把答案直接交给用户。四个维度分别抓一类漂移（结论 / 措辞 / 数字 / 角标）。

与 TS 侧同口径是硬要求：`packages/eval/consistency.json` 是同一份固化产物，
两端的度量函数必须给出相同数字，否则"双端对比"里的稳定性一列没有可比性。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence


@dataclass
class SampleView:
    """一次采样的可比较视图（只保留参与度量的字段）。"""

    verdict: str
    answer: str
    citations: Sequence[int] = ()


@dataclass
class ConsistencyReport:
    samples: int
    verdict_agreement: float
    answer_jaccard: float
    number_jaccard: float
    citation_jaccard: float
    refusal_flip: bool


def bigrams(text: str) -> set[str]:
    """字符 bigram 集合（去空白，与检索侧近重复判定同一口径）。"""
    clean = "".join(text.split())
    return {clean[i : i + 2] for i in range(len(clean) - 1)}


def numbers_of(answer: str) -> set[str]:
    """从答案里抽数字事实（与校验层同一口径：带单位的数字串，去空白）。"""
    from .verify import extract_numbers

    return set(extract_numbers(answer))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return 1.0 if union == 0 else inter / union


def measure_consistency(views: Sequence[SampleView]) -> ConsistencyReport:
    """与 TS measureConsistency 同口径（两两组合求均值）。"""
    if len(views) <= 1:
        return ConsistencyReport(
            samples=len(views),
            verdict_agreement=1.0,
            answer_jaccard=1.0,
            number_jaccard=1.0,
            citation_jaccard=1.0,
            refusal_flip=False,
        )

    pairs = list(combinations(views, 2))

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    verdicts = {v.verdict for v in views}
    refused_states = {v.verdict == "refused" for v in views}
    return ConsistencyReport(
        samples=len(views),
        verdict_agreement=1.0 if len(verdicts) == 1 else 0.0,
        answer_jaccard=mean([jaccard(bigrams(a.answer), bigrams(b.answer)) for a, b in pairs]),
        number_jaccard=mean(
            [jaccard(numbers_of(a.answer), numbers_of(b.answer)) for a, b in pairs]
        ),
        citation_jaccard=mean(
            [jaccard(set(a.citations), set(b.citations)) for a, b in pairs]
        ),
        refusal_flip=len(refused_states) > 1,
    )
