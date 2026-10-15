"""解析层：文件字节 → 结构化文本块（ChunkBlock）。

与浏览器端 apps/web/src/utils/parseBlocks.ts + pdfParse.ts 对应。
跨端一致性**只承诺 Markdown / TXT**：PDF/DOCX 的文本抽取依赖不同库（pdfjs/mammoth vs
pypdfium2/python-docx），换行、连字符、阅读顺序都可能不同，见 packages/chunker-spec/SPEC.md 第 8 节。

`blocks_for_file` 是统一入口（按扩展名分派），供摄取与命令行脚本共用。
"""

from __future__ import annotations

from pathlib import Path

from ..chunker import ChunkBlock
from .docx import docx_to_blocks
from .markdown import markdown_to_blocks, strip_markdown_front_matter
from .pdf import pdf_to_blocks
from .txt import clean_text, txt_to_blocks

#: 支持的上传格式（与浏览器端 parseFile 的 50MB 上限一并由接口层校验）
MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdown", ".mkd"}


def blocks_for_file(path: Path, *, filename: str | None = None) -> list[ChunkBlock]:
    """按文件名分派解析（Markdown 保留标题结构，其余整篇/逐页成块）。"""
    name = filename or path.name
    suffix = Path(name).suffix.lower()
    if suffix in MARKDOWN_SUFFIXES:
        return markdown_to_blocks(path.read_text(encoding="utf-8"))
    if suffix == ".docx":
        return docx_to_blocks(path)
    if suffix == ".pdf":
        return pdf_to_blocks(path)
    return txt_to_blocks(path.read_text(encoding="utf-8"))


__all__ = [
    "MARKDOWN_SUFFIXES",
    "blocks_for_file",
    "clean_text",
    "docx_to_blocks",
    "markdown_to_blocks",
    "pdf_to_blocks",
    "strip_markdown_front_matter",
    "txt_to_blocks",
]
