"""Harness v1：与 apps/web/scripts/eval.ts 同口径的评测（指标名、公式、门槛逐条对齐）。"""

from .golden import Golden, GoldenQuery, golden_version, load_corpus_docs, load_golden
from .metrics import Metrics, bigram_jaccard, evaluate
from .runner import EvalOutcome, run_eval

__all__ = [
    "EvalOutcome",
    "Golden",
    "GoldenQuery",
    "Metrics",
    "bigram_jaccard",
    "evaluate",
    "golden_version",
    "load_corpus_docs",
    "load_golden",
    "run_eval",
]
