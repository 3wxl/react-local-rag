"""固化向量加载（packages/eval/embeddings.json）。

用途与浏览器端 scripts/fixedEmbeddings.ts 一致：**没有模型的环境也能跑评测**
（CI 里没有 ONNX 权重），而且结果确定、可复现。

一致性校验是致命的（不静默降级）：语料内容、分块参数或模型变了，contentHash 必然对不上。
这时宁可报错退出，也不能"缺向量按 0 分参与排序" —— 那等于拿旧向量测新切块，
向量路悄悄退化成近似纯 BM25，指标看起来正常却毫无意义。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..parity.chunk_params import DEFAULT_CHUNK_OPTIONS, EMBEDDING_MODEL_NAME


@dataclass
class FixedEmbeddings:
    model: str
    chunk_params: dict
    chunks: dict[str, np.ndarray]
    queries: dict[str, np.ndarray]
    #: 查询扩展的改写向量（id → 各条改写的向量，顺序与 expansions.json 一致）
    expansions: dict[str, list[np.ndarray]] = field(default_factory=dict)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def query_count(self) -> int:
        return len(self.queries)


def load_fixed_embeddings(path: Path) -> FixedEmbeddings | None:
    """读取固化向量；文件不存在返回 None（调用方据此标注 mode=bm25）。"""
    path = Path(path)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return FixedEmbeddings(
        model=raw.get("model", ""),
        chunk_params=raw.get("chunkParams", {}),
        chunks={k: _to_float32(v) for k, v in raw["chunks"].items()},
        queries={k: _to_float32(v) for k, v in raw["queries"].items()},
        expansions={
            k: [_to_float32(vec) for vec in vectors]
            for k, vectors in (raw.get("expansions") or {}).items()
        },
    )


def _to_float32(values: list[float]) -> np.ndarray:
    """float32 存储：与 TS 的 Float32Array 同口径（见 parity/retrieval.py 的说明）。"""
    return np.asarray(values, dtype=np.float32)


def assert_embeddings_consistent(
    fixed: FixedEmbeddings,
    chunk_hashes: list[str],
    query_ids: list[str],
) -> None:
    """严格一致性校验：任何一项不一致都抛错（调用方应 exit 1）。"""
    problems: list[str] = []
    if fixed.model and fixed.model != EMBEDDING_MODEL_NAME:
        problems.append(
            f"模型不一致：固化文件是「{fixed.model}」，当前代码是「{EMBEDDING_MODEL_NAME}」"
        )
    if fixed.chunk_params:
        expected = {
            "maxChunkTokens": DEFAULT_CHUNK_OPTIONS.max_chunk_tokens,
            "minChunkChars": DEFAULT_CHUNK_OPTIONS.min_chunk_chars,
            "maxChunkChars": DEFAULT_CHUNK_OPTIONS.max_chunk_chars,
            "sentenceOverlap": DEFAULT_CHUNK_OPTIONS.sentence_overlap,
        }
        if fixed.chunk_params != expected:
            problems.append(
                f"分块参数不一致：固化文件={fixed.chunk_params}，当前默认={expected}"
            )
    missing_chunks = [h for h in dict.fromkeys(chunk_hashes) if h not in fixed.chunks]
    if missing_chunks:
        problems.append(
            f"{len(missing_chunks)} 个块的向量缺失（示例：{', '.join(missing_chunks[:3])}…）"
        )
    missing_queries = [q for q in dict.fromkeys(query_ids) if q not in fixed.queries]
    if missing_queries:
        problems.append(f"缺失查询向量：{', '.join(missing_queries[:5])}")
    if problems:
        raise ValueError(
            "固化向量与当前语料/代码不一致，禁止用旧向量跑评测：\n  - "
            + "\n  - ".join(problems)
            + "\n请用浏览器端（apps/web）重新生成：npm run gen:embeddings"
        )
