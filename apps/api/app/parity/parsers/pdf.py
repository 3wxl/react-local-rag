"""PDF → 结构化块（逐页），用 pypdfium2。

**不承诺跨端一致**（SPEC 第 8 节）：浏览器用 pdfjs、服务端用 pypdfium2，
同一份 PDF 的文本抽取在换行、连字符、阅读顺序上都可能不同。
所以跨语言 contentHash 门槛只覆盖 Markdown / TXT；PDF 走"能答就行"的路径。

选 pypdfium2 而不是 PyMuPDF：许可更宽松（BSD/Apache vs AGPL），私有化交付时少一个法律问题。
"""

from __future__ import annotations

from pathlib import Path

from ..chunker import ChunkBlock
from .txt import clean_text


def pdf_to_blocks(path: Path) -> list[ChunkBlock]:
    """逐页抽取文本，每页一个块（页码从 1 起，页尾补空格与浏览器解析层一致）。"""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(path))
    blocks = []
    try:
        for page_index in range(len(document)):
            page = document[page_index]
            text_page = page.get_textpage()
            text = text_page.get_text_bounded() or ""
            cleaned = clean_text(text)
            if cleaned:
                # 页尾补空格做分隔，避免相邻两页的单词被拼在一起（与 pdfParse.ts 同处理）
                blocks.append(ChunkBlock(text=cleaned + " ", page=page_index + 1))
            text_page.close()
            page.close()
    finally:
        document.close()
    if not blocks:
        raise ValueError("PDF 未提取到任何文本内容（可能是扫描件或空文档）")
    return blocks
