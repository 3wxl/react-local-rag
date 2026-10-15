"""DOCX → 结构化块（整篇一个块），用 python-docx。

**不承诺跨端一致**（SPEC 第 8 节）：浏览器用 mammoth.extractRawText，服务端用 python-docx，
段落的空白与表格处理可能不同。这里取与 mammoth 最接近的口径：按段落取文本、以换行连接，
再走与 TXT 相同的 clean_text。
"""

from __future__ import annotations

from pathlib import Path

from ..chunker import ChunkBlock
from .txt import clean_text


def docx_to_blocks(path: Path) -> list[ChunkBlock]:
    """整篇清洗后作为一个块（无分页，页码恒为 1）。"""
    import docx  # python-docx

    document = docx.Document(str(path))
    lines = [p.text for p in document.paragraphs]
    text = clean_text("\n".join(lines))
    if not text:
        raise ValueError("DOCX 未提取到任何文本内容（可能是空文档）")
    return [ChunkBlock(text=text, page=1)]
