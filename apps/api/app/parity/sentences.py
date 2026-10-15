"""句子切分（与 apps/web/src/utils/chunk.ts、verifyAnswer.ts 同口径）。

两处切分**必须一致**：
- 分块器用 `split_sentences_keep`（保留分隔符，句子是打包的原子单位）
- 校验/UI 用 `split_sentences`（在 keep 的基础上把游离角标归并回前一句）

正则（TS 原文）：`/[^。！？!?；;\n]+[。！？!?；;\n]?|\n+/g`
—— 即「非分隔符字符 + 可选的一个分隔符」，分隔符留在句子里；换行单独成段。
"""

from __future__ import annotations

import re

_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;\n]?|\n+")

# 整段只由引用角标组成（模型把 [1] 写在了句号之后）
_ONLY_CITATION_RE = re.compile(r"^(?:\s*\[\d+(?:\s*[,，]\s*\d+)*\])+\s*$")
# 段首引用角标（模型把 [1] 写在了下一句开头）
_LEADING_CITATION_RE = re.compile(
    r"^((?:\s*\[\d+(?:\s*[,，]\s*\d+)*\])+)\s*([\s\S]*)$"
)


def split_sentences_keep(text: str) -> list[str]:
    """分块器口径：切句并保留分隔符，丢弃 trim 后为空的片段。"""
    return [s for s in _SENTENCE_RE.findall(text) if s.strip()]


def split_sentences(text: str) -> list[str]:
    """校验口径：在 split_sentences_keep 之上把游离角标归并回前一句。

    模型可能把角标写在句号之后（「…元。[1]」）或下一句开头（「[1]周期…」），
    直接切会切出孤立的 "[1]" 段，引用就挂不到它证明的那句话上。
    """
    parts = _SENTENCE_RE.findall(text)
    merged: list[str] = []
    for part in parts:
        if len(part) == 0:
            continue
        if merged and _ONLY_CITATION_RE.match(part):
            merged[-1] += part
            continue
        lead = _LEADING_CITATION_RE.match(part) if merged else None
        if lead:
            merged[-1] += lead.group(1)
            if lead.group(2).strip():
                merged.append(lead.group(2))
            continue
        merged.append(part)
    return merged
