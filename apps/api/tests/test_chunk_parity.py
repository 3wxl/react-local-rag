"""跨语言分块一致性：TS 与 Python 对同一语料必须切出**相同分块**。

这是 M1 的核心证据链之一（见 docs/ROADMAP-FULLSTACK.md 与 packages/chunker-spec/SPEC.md）：
先有相同分块，才有相同 contentHash，才有可对比的检索与指标。

三层断言，一层比一层硬：
1. 冻结期望值（无 Node 也能跑）：fixtures 的块序列
2. 实时 TS 导出：fixtures 两条路径（评测路径 / 摄取路径）
3. 真实语料评测路径 + **所有 contentHash 必须命中 packages/eval/embeddings.json**
   —— 后者意味着 Python 切出的块能直接复用已固化向量，两端索引可互认
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.parity.chunk_params import CURRENT_INDEX_VERSION
from app.parity.chunker import TextChunkStream, create_text_chunks
from app.parity.eval_corpus import normalize_markdown_for_eval
from app.parity.parsers.markdown import markdown_to_blocks

from .parity_utils import (
    EVAL_CORPUS_DIR,
    FIXTURES_DIR,
    corpus_docs,
    requires_node,
    run_parity_dump,
)

#: 比对字段：不含 chunkId（随机值），其余逐字段相等
COMPARED_FIELDS = (
    "ord",
    "content",
    "contentHash",
    "charStart",
    "charEnd",
    "pageStart",
    "pageEnd",
    "tokenCount",
    "headingPath",
)


def py_chunks_eval(doc_name: str, raw: str) -> list[dict]:
    """评测路径：剥标题行 → 整篇一个块 → 分块（与 scripts/eval.ts 同序）。"""
    chunks = create_text_chunks(
        doc_name, doc_name, normalize_markdown_for_eval(raw)
    )
    return [_dump(c) for c in chunks]


def py_chunks_ingest(doc_name: str, raw: str) -> tuple[list[dict], list[dict]]:
    """摄取路径：Markdown 解析产块 → 逐块 push → 分块。"""
    blocks = markdown_to_blocks(raw)
    stream = TextChunkStream(doc_name, doc_name)
    for b in blocks:
        stream.push(b)
    chunks = stream.finish()
    dump_blocks = [
        {
            "text": b.text,
            "page": b.page,
            "headingPath": b.heading_path,
            "type": b.type,
        }
        for b in blocks
    ]
    return [_dump(c) for c in chunks], dump_blocks


def _dump(chunk) -> dict:
    """与 parity-dump.ts 的导出结构对齐（字段名用 TS 侧的驼峰）。"""
    return {
        "ord": chunk.ord,
        "content": chunk.content,
        "contentHash": chunk.content_hash,
        "charStart": chunk.char_start,
        "charEnd": chunk.char_end,
        "pageStart": chunk.page_start,
        "pageEnd": chunk.page_end,
        "tokenCount": chunk.token_count,
        "headingPath": chunk.heading_path,
    }


def assert_chunks_equal(doc_name: str, ts_chunks: list[dict], py: list[dict]) -> None:
    """逐块比对，失败时给出**第一处**差异的细节（否则只剩一个 assert 失败无从下手）。"""
    assert len(py) == len(ts_chunks), (
        f"{doc_name}: 块数不同 TS={len(ts_chunks)} PY={len(py)}；"
        f"TS 首块={ts_chunks[0]['content'][:40] if ts_chunks else None!r} "
        f"PY 首块={py[0]['content'][:40] if py else None!r}"
    )
    for ts_chunk, py_chunk in zip(ts_chunks, py):
        for field in COMPARED_FIELDS:
            assert py_chunk[field] == ts_chunk[field], (
                f"{doc_name} ord={ts_chunk['ord']} 字段 {field} 不一致："
                f"TS={ts_chunk[field]!r} PY={py_chunk[field]!r}"
            )


# ---------------- 1. 冻结期望值（无 Node 也能跑） ----------------


def test_fixture_chunks_match_frozen_expectation(expected_chunks: dict) -> None:
    """fixtures 两条路径都要与冻结期望一致（保证没有 Node 时也有回归保护）。"""
    assert expected_chunks["indexVersion"] == CURRENT_INDEX_VERSION, (
        "索引版本不一致：分块参数 / 前缀方案 / 模型名与冻结期望值不同 → 旧索引不可复用"
    )
    for mode in ("eval", "ingest"):
        for doc in expected_chunks["modes"][mode]:
            name = doc["docName"]
            raw = (FIXTURES_DIR / name).read_text(encoding="utf-8")
            if mode == "eval":
                py = py_chunks_eval(name, raw)
            else:
                py, py_blocks = py_chunks_ingest(name, raw)
                # 解析层也要一致：块序列不同的话，切块迟早会分叉
                assert py_blocks == doc["blocks"], f"{name}: 解析块序列与 TS 不一致"
            assert_chunks_equal(f"{name}({mode}, 冻结)", doc["chunks"], py)


# ---------------- 2. 实时 TS 导出（fixtures，两条路径） ----------------


@requires_node
@pytest.mark.parametrize("mode", ["eval", "ingest"])
def test_fixture_chunks_match_live_ts(mode: str) -> None:
    dump = run_parity_dump("chunks", mode=mode, corpus=FIXTURES_DIR)
    assert dump["indexVersion"] == CURRENT_INDEX_VERSION
    for doc in dump["chunks"]["docs"]:
        name = doc["docName"]
        raw = (FIXTURES_DIR / name).read_text(encoding="utf-8")
        if mode == "eval":
            py = py_chunks_eval(name, raw)
        else:
            py, py_blocks = py_chunks_ingest(name, raw)
            assert py_blocks == doc["blocks"], f"{name}: 解析块序列与 TS 不一致"
        assert_chunks_equal(f"{name}({mode}, 实时)", doc["chunks"], py)


# ---------------- 3. 真实语料（评测路径 = 固化向量的来源） ----------------


@requires_node
def test_corpus_chunks_match_live_ts() -> None:
    """packages/eval/corpus 全量比对（119 块基准）。"""
    dump = run_parity_dump("chunks", mode="eval", corpus=EVAL_CORPUS_DIR)
    for doc in dump["chunks"]["docs"]:
        name = doc["docName"]
        raw = (EVAL_CORPUS_DIR / name).read_text(encoding="utf-8")
        assert_chunks_equal(name, doc["chunks"], py_chunks_eval(name, raw))


def test_corpus_hashes_covered_by_fixed_embeddings(fixed_embeddings: dict) -> None:
    """Python 切出的每个 contentHash 都要能在固化向量里找到。

    这条是"两端索引可互认"的硬证据：命中即说明 Python 的切块与生成
    packages/eval/embeddings.json 时的 TS 切块逐字符相同。
    """
    known = set(fixed_embeddings["chunks"].keys())
    missing: list[str] = []
    total = 0
    for name, raw in corpus_docs():
        for chunk in py_chunks_eval(name, raw):
            total += 1
            if chunk["contentHash"] not in known:
                missing.append(f"{name}#{chunk['ord']}({chunk['contentHash']})")
    assert total == fixed_embeddings["chunkCount"], (
        f"块数不一致：Python={total} 固化文件={fixed_embeddings['chunkCount']}"
    )
    assert not missing, f"{len(missing)} 个块的哈希不在固化向量里：{missing[:5]}"
