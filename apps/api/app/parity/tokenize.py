r"""统一分词（与 apps/web/src/utils/tokenize.ts 同口径）。

BM25 关键词检索与词法幻觉校验共用同一套切分，两端也必须共用同一套 ——
分词差一个 bigram，BM25 排名就会变，双端指标再也不可比。

    WORD_RE = /[a-z0-9]+(?:\.\d+)?%?/gi        // 英文/数字（含小数、百分号），转小写
    CJK_RE  = /[一-鿿]+/g                       // 中文连续段 → 每段产出 bigram

细节：中文段长度 1 时保留该单字（给短句兜底）；tokenize 保留重复（BM25 的 TF 需要重复），
tokenize_set 去重（覆盖率计算用）。
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+(?:\.\d+)?%?", re.IGNORECASE)
_CJK_RE = re.compile(r"[一-鿿]+")


def tokenize(text: str) -> list[str]:
    """分词（保留重复）。顺序与 TS 一致：先英文/数字，后中文 bigram。"""
    tokens = [m.group(0).lower() for m in _WORD_RE.finditer(text)]
    for m in _CJK_RE.finditer(text):
        seg = m.group(0)
        for i in range(len(seg) - 1):
            tokens.append(seg[i : i + 2])
        if len(seg) == 1:
            tokens.append(seg)
    return tokens


def tokenize_set(text: str) -> set[str]:
    """分词去重（覆盖率用）。"""
    return set(tokenize(text))
