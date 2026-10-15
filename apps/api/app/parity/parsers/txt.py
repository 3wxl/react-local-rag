"""纯文本 → 结构化块（apps/web/src/utils/parseBlocks.ts 的对应实现）。

TXT / DOCX / PDF 走的都是"清洗后整篇一个块"或"逐页一个块"的形态，
只有 Markdown 有标题结构（见 markdown.py）。
"""

from __future__ import annotations

import re

from ..chunker import ChunkBlock
from ..jsstr import js_trim

_HTML_TAG_RE = re.compile(r"<[^>]*>")
_HSPACE_RE = re.compile(r"[ \t]+")
_EXCESS_BLANK_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """清洗脏文本：去 HTML 标签、压缩水平空白与连续空行、trim（与 TS cleanText 同序）。"""
    no_html = _HTML_TAG_RE.sub("", text)
    return js_trim(_EXCESS_BLANK_RE.sub("\n\n", _HSPACE_RE.sub(" ", no_html)))


def txt_to_blocks(raw: str) -> list[ChunkBlock]:
    """整篇清洗后作为一个块（页码恒为 1）；清洗后为空则返回空列表。"""
    cleaned = clean_text(raw)
    return [ChunkBlock(text=cleaned, page=1)] if cleaned else []
