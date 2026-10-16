"""查询扩展的解析与清洗（apps/web/src/utils/queryExpansion.ts 的移植）。

模型输出是自由文本：带编号、带引号、带解释、复述原句、空行……
解析层必须全部吃掉，且**任何异常都退化为"没有扩展"**（空列表）——
查询扩展是增强项，不是必经路径，它的失败不该拖垮检索。
"""

from __future__ import annotations

import re

MAX_EXPANSION_CHARS = 40
MAX_EXPANSIONS = 3

_PUNCT_RE = re.compile(r"[。！？!?；;，,、：:\"'（）()【】\[\]\-—…·]")
_LEADING_MARK_RE = re.compile(r"^(?:\d+[.、)]|[-*•·]|[①②③④⑤])\s*")
_QUOTES = "“”\"'‘’《》"


def _normalize(text: str) -> str:
    return _PUNCT_RE.sub("", re.sub(r"\s+", "", text)).lower()


def parse_expanded_queries(raw: str, original: str) -> list[str]:
    """解析模型输出的改写列表（与 TS parseExpandedQueries 同口径）。"""
    if not raw:
        return []
    seen = {_normalize(original)}
    out: list[str] = []
    for line in raw.splitlines():
        text = _LEADING_MARK_RE.sub("", line.strip()).strip().strip(_QUOTES).strip()
        if not text or len(text) > MAX_EXPANSION_CHARS:
            continue
        # 去掉行尾解释（模型爱写「——这是术语化改写」）
        text = re.split(r"——|（注|\(注|：这|,这", text)[0].strip()
        if not text:
            continue
        key = _normalize(text)
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= MAX_EXPANSIONS:
            break
    return out


def build_query_variants(
    question: str, expansions: list[str], expansion_weight: float = 0.7
) -> list[tuple[str, float]]:
    """原查询与改写合成变体列表（原查询权重 1，改写默认 0.7）。"""
    return [(question, 1.0), *[(text, expansion_weight) for text in expansions]]
