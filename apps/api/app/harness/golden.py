"""评测数据装载（与 apps/web/scripts/eval.ts 同源）。

一条纪律：**评测口径 = 线上默认配置**。所以这里：
- 语料用与 TS 相同的评测路径切块（normalize_markdown_for_eval → 整篇一个块 → 分块器）
- 向量优先用固化文件（packages/eval/embeddings.json）：确定、可复现、CI 无需模型
- 参数取 DEFAULT_RETRIEVAL_OPTIONS / DEFAULT_CHUNK_OPTIONS，不在这里另设一套
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..parity.chunk_params import CURRENT_INDEX_VERSION
from ..parity.chunker import create_text_chunks
from ..parity.eval_corpus import normalize_markdown_for_eval
from ..parity.fnv import fnv1a32
from ..parity.retrieval import RetrievalDoc


@dataclass
class GoldenQuery:
    id: str
    question: str
    relevant: list[str]
    key_points: list[str]
    negative: bool
    type: str | None = None


@dataclass
class Golden:
    name: str
    queries: list[GoldenQuery]
    top_k: int


def load_golden(eval_dir: Path) -> Golden:
    raw = json.loads((Path(eval_dir) / "golden.json").read_text(encoding="utf-8"))
    queries = [
        GoldenQuery(
            id=q["id"],
            question=q["question"],
            relevant=q.get("relevant", []),
            key_points=q.get("keyPoints", []),
            negative=bool(q.get("negative", False)),
            type=q.get("type"),
        )
        for q in raw["queries"]
    ]
    return Golden(
        name=raw.get("name", "golden"), queries=queries, top_k=raw.get("retrievalTopK", 5)
    )


def golden_version(eval_dir: Path) -> str:
    """H8 配置版本化：golden 集内容哈希（改动即可追溯是哪版数据跑出的指标）。"""
    text = (Path(eval_dir) / "golden.json").read_text(encoding="utf-8")
    return f"fnv1a32:{fnv1a32(text)}"


def load_corpus_docs(
    eval_dir: Path, *, chunk_vectors: dict[str, np.ndarray] | None = None
) -> list[RetrievalDoc]:
    """语料切块（评测路径）+ 挂向量。

    `chunk_vectors` 传固化向量（contentHash → 向量）时用固化，否则留空（纯 BM25 口径）。
    """
    corpus_dir = Path(eval_dir) / "corpus"
    docs: list[RetrievalDoc] = []
    for path in sorted(p for p in corpus_dir.iterdir() if p.suffix == ".md"):
        raw = path.read_text(encoding="utf-8")
        for chunk in create_text_chunks(
            path.name, path.name, normalize_markdown_for_eval(raw)
        ):
            vector = None
            if chunk_vectors is not None:
                vector = chunk_vectors.get(chunk.content_hash)
            docs.append(
                RetrievalDoc(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=chunk.content,
                    doc_name=chunk.doc_name,
                    ord=chunk.ord,
                    heading_path=chunk.heading_path,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    token_count=chunk.token_count,
                    content_hash=chunk.content_hash,
                    vector=vector,
                )
            )
    return docs


def index_version() -> str:
    return CURRENT_INDEX_VERSION
