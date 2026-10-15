"""引用装配（apps/web/src/utils/citation.ts 的移植）。

编号是回答与原文之间唯一的对接点：
- prompt 里给每段打上 [1][2]… 并附「文件名 · 章节 · 页码」标签
- 模型被要求逐句带角标，无依据必须拒答
- 校验阶段按角标把语义比对范围收敛到「被引用的那几段」
- UI 把角标渲染成可展开的引用卡片
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .retrieval import SearchHit


@dataclass
class AnswerPassage:
    """喂给 LLM 的编号段落。"""

    index: int
    text: str
    label: str


@dataclass
class ContextRef:
    """引用卡片数据（与 TS ContextRef 逐字段对齐，snake_case 供 API 直接序列化）。"""

    index: int
    doc_id: str
    doc_name: str
    heading_path: list[str | None]
    page_start: int
    page_end: int
    score: float
    relevance: float
    chunk_ids: list[str]
    window_chunk_ids: list[str]
    matched_content: str
    content: str


@dataclass
class CitationSegment:
    """句子切片：普通文本 或 引用角标（UI 把角标渲染成可点击上标）。"""

    text: str
    citation: int | None = None


@dataclass
class VerifyPassage:
    """编号段落 -> 校验范围（块 id 集合）。"""

    index: int
    chunk_ids: list[str]


_CITATION_RE = re.compile(r"\[(\d+(?:\s*[,，]\s*\d+)*)\]")


def heading_label(heading_path: Sequence[str | None]) -> str:
    """章节显示名：取标题路径末级；无标题的文档显示「正文」。"""
    parts = [p for p in heading_path if p]
    return parts[-1] if parts else "正文"


def page_label(page_start: int, page_end: int) -> str:
    """页码显示：单页 P12，跨页 P12-13。"""
    return f"P{page_start}" if page_start == page_end else f"P{page_start}-{page_end}"


def build_citation_label(hit: SearchHit) -> str:
    """引用标签：产品手册.md · 定价与购买 · P1"""
    return " · ".join(
        [
            hit.doc_name or "未命名文档",
            heading_label(hit.heading_path),
            page_label(hit.page_start, hit.page_end),
        ]
    )


def build_evidence(
    hits: Sequence[SearchHit],
) -> tuple[list[AnswerPassage], list[ContextRef]]:
    """命中列表 → 编号段落 + 引用卡片（编号从 1 开始，与答案角标一致）。"""
    passages: list[AnswerPassage] = []
    contexts: list[ContextRef] = []
    for i, hit in enumerate(hits):
        index = i + 1
        passages.append(
            AnswerPassage(index=index, text=hit.content, label=build_citation_label(hit))
        )
        contexts.append(
            ContextRef(
                index=index,
                doc_id=hit.doc_id,
                doc_name=hit.doc_name,
                heading_path=hit.heading_path,
                page_start=hit.page_start,
                page_end=hit.page_end,
                score=hit.rrf_score,
                relevance=hit.relevance,
                chunk_ids=hit.chunk_ids,
                window_chunk_ids=hit.window_chunk_ids,
                matched_content=hit.matched_content,
                content=hit.content,
            )
        )
    return passages, contexts


def split_by_citations(text: str) -> list[CitationSegment]:
    """把句子按角标切成「文本 / 角标」交替片段，供 UI 渲染可点击上标。"""
    segments: list[CitationSegment] = []
    last = 0
    for m in _CITATION_RE.finditer(text):
        if m.start() > last:
            segments.append(CitationSegment(text=text[last : m.start()]))
        for part in re.split(r"[,，]", m.group(1)):
            n = int(part.strip())
            if n > 0:
                segments.append(CitationSegment(text=m.group(0), citation=n))
        last = m.end()
    if last < len(text):
        segments.append(CitationSegment(text=text[last:]))
    return segments


def to_verify_passages(contexts: Sequence[ContextRef]) -> list[VerifyPassage]:
    """引用卡片 → 校验范围（用窗口块：窗口就是模型实际看到的原文范围）。"""
    return [
        VerifyPassage(
            index=c.index,
            chunk_ids=list(c.window_chunk_ids) if c.window_chunk_ids else list(c.chunk_ids),
        )
        for c in contexts
    ]
