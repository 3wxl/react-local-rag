"""评测语料文本归一化（apps/web/src/utils/evalCorpus.ts 的移植）。

评测路径的预处理：剥掉所有 Markdown 标题行、折叠连续空行，整篇作为**一个块**送进分块器。
`packages/eval/embeddings.json` 里的 119 个 contentHash 就是这条路径的产物 ——
Python 端要能对上它们，靠的就是这个函数 + 同一个分块器。
"""

from __future__ import annotations

import re

from .jsstr import js_trim

#: markdown 标题行（# ~ ###### 起首，含空格）
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s")
#: 3 个及以上连续换行压成 2 个（最多保留一个空行）
_EXCESS_BLANK_RE = re.compile(r"\n{3,}")


def normalize_markdown_for_eval(raw: str) -> str:
    """与 TS 同序：按行过滤 → 以 `\\n` 拼回 → 折叠空行 → trim。"""
    lines = [line for line in re.split(r"\r?\n", raw) if not _HEADING_LINE_RE.match(line)]
    return js_trim(_EXCESS_BLANK_RE.sub("\n\n", "\n".join(lines)))
