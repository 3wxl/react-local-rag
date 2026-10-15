"""Markdown → 结构化块（apps/web/src/utils/parseBlocks.ts 的移植）。

规则（每条都影响切块边界，改动前先看 SPEC）：
- 先删 HTML 注释，再剥 YAML front matter（仅当匹配上且中间内容为空或含 `key:` 才剥）
- 行尾空白去掉；标题行判定 `^(#{1,6})\\s+(.*)$`（必须在行首、# 后必须有空白）
- 标题：**先 flush 当前段落** → `heading_path` 截断到 `级数-1`（层级跳跃留 None）→ push 标题 → 产出 heading 块
- 正文：连续非空行以 `\\n` 连接成一个 text 块；空行 flush

`heading_path` 里的 None 对应 TS 空洞（`headingPath.length = level - 1`）经 JSON 往返的 null，
分块器把整条路径序列化进 chunk，Python 端必须一致，否则切块分组不同。
"""

from __future__ import annotations

import re

from ..chunker import ChunkBlock
from ..jsstr import JS_WS_CLASS, js_trim, js_trim_end

_COMMENT_RE = re.compile(r"<!--[\s\S]*?-->")
_FRONT_MATTER_RE = re.compile(r"^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)")
#: 首行 `---` 之后、`---` 之前的内容是否像 YAML（含 key: value）
_YAML_KEY_RE = re.compile(r"^[\w-]+\s*:", re.MULTILINE)
_HEADING_RE = re.compile(f"^(#{{1,6}}){JS_WS_CLASS}+(.*)$")


def strip_markdown_front_matter(text: str) -> str:
    """去掉 YAML front matter（文件开头 --- ... ---），兼容 CRLF。"""
    # 1. 移除 UTF-8 BOM
    normalized = text[1:] if text.startswith("﻿") else text
    m = _FRONT_MATTER_RE.match(normalized)
    if not m:
        return normalized
    body = m.group(1)
    # 校验：中间内容是空 OR 存在 yaml key:value 格式，才判定是真 front matter
    if js_trim(body) == "" or _YAML_KEY_RE.search(body):
        return normalized[m.end() :]
    return normalized


def markdown_to_blocks(raw: str) -> list[ChunkBlock]:
    """按标题层级产出结构化块（标题文本只进 heading_path，不进正文）。"""
    blocks: list[ChunkBlock] = []
    no_comment = _COMMENT_RE.sub("", raw)
    no_front_matter = strip_markdown_front_matter(no_comment)
    lines = [js_trim_end(line) for line in re.split(r"\r?\n", no_front_matter)]

    heading_path: list[str | None] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph:
            return
        text = js_trim("\n".join(paragraph))
        if text:
            blocks.append(
                ChunkBlock(
                    text=text,
                    page=1,
                    heading_path=list(heading_path),
                    type="text",
                )
            )
        paragraph.clear()

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            flush_paragraph()
            level = len(m.group(1))
            title = js_trim(m.group(2))
            # 标题层级堆栈：同层替换，低层出栈，高层压栈；层级跳跃留空洞（None）
            del heading_path[level - 1 :]
            heading_path.extend([None] * ((level - 1) - len(heading_path)))
            heading_path.append(title)
            blocks.append(
                ChunkBlock(
                    text=title,
                    page=1,
                    heading_path=list(heading_path),
                    type="heading",
                )
            )
            continue
        if js_trim(line) == "":
            flush_paragraph()
            continue
        paragraph.append(js_trim(line))
    flush_paragraph()
    return blocks
