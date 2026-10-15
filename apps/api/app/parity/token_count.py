r"""token 估算（与 apps/web/src/utils/tokenCount.ts 同口径）。

    const CJK_RE = /[一-鿿]/g;
    const WORD_RE = /[a-zA-Z0-9]+(?:\.\d+)?/g;
    return words.length + cjk.length;

用途：分块参数以 token 为单位（bge-small-zh 上限 512），
不依赖真实 tokenizer（体积大、加载慢），纯正则估算。
"""

from __future__ import annotations

import re

# [一-鿿]：CJK 统一表意文字基本区
_CJK_RE = re.compile(r"[一-鿿]")
_WORD_RE = re.compile(r"[a-zA-Z0-9]+(?:\.\d+)?")


def count_tokens(text: str) -> int:
    """汉字数 + 英文/数字串数（与 TS countTokens 逐字对齐）。"""
    return len(_WORD_RE.findall(text)) + len(_CJK_RE.findall(text))
