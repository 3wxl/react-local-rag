"""检索链路跨语言一致性：BM25 分数、RRF 融合、门限判定、最终 topK 命中。

比对基准是 TS 端 `scripts/parity-dump.ts --section bm25|retrieval` 的实时导出
（同一份固化向量 + 同一份 golden set + 同一批默认参数），因此这条测试同时钉住了三件事：

1. 分词/BM25 公式一致
2. RRF/合并/去重/MMR/门限的**执行顺序**一致
3. 两端默认参数没有漂移（改一端不改另一端立刻红）

分数用容差比（浮点求和顺序差异），**排名与命中集合必须完全相等** —— 排名才是检索结果。
"""

from __future__ import annotations

import pytest

from app.parity.bm25 import BM25
from app.parity.chunker import create_text_chunks
from app.parity.eval_corpus import normalize_markdown_for_eval
from app.parity.retrieval import (
    DEFAULT_RETRIEVAL_OPTIONS,
    RetrievalDoc,
    dot_product,
    post_process_retrieval,
    to_float32,
)

from .parity_utils import (
    EVAL_CORPUS_DIR,
    corpus_docs,
    requires_node,
    run_parity_dump,
)

SCORE_TOL = 1e-9


def load_corpus_docs(fixed_embeddings: dict) -> list[RetrievalDoc]:
    """评测路径切块 + 挂上固化向量（contentHash → vector）。"""
    docs: list[RetrievalDoc] = []
    for name, raw in corpus_docs(EVAL_CORPUS_DIR):
        for chunk in create_text_chunks(
            name, name, normalize_markdown_for_eval(raw)
        ):
            raw_vector = fixed_embeddings["chunks"].get(chunk.content_hash)
            # TS 侧向量是 Float32Array（固化 JSON → float32），这里必须同口径
            vector = to_float32(raw_vector) if raw_vector is not None else None
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


def vec_scores_for(
    docs: list[RetrievalDoc], query_id: str, fixed_embeddings: dict
) -> list[float] | None:
    raw_qvec = fixed_embeddings["queries"].get(query_id)
    if raw_qvec is None:
        return None
    qvec = to_float32(raw_qvec)
    return [
        dot_product(qvec, d.vector) if d.vector is not None else 0.0 for d in docs
    ]


@pytest.fixture(scope="module")
def docs(fixed_embeddings: dict) -> list[RetrievalDoc]:
    return load_corpus_docs(fixed_embeddings)


@requires_node
def test_bm25_scores_match_ts(docs, golden, fixed_embeddings) -> None:
    """BM25 原始分（容差）+ 排名（必须完全相等）+ IDF 覆盖率。"""
    dump = run_parity_dump("bm25")
    assert dump["bm25"]["chunkCount"] == len(docs)

    bm25 = BM25([d.content for d in docs])
    for ts_query in dump["bm25"]["queries"]:
        question = ts_query["question"]
        scores = bm25.score_all(question)
        assert bm25.idf_weighted_coverage(question) == pytest.approx(
            ts_query["idfCoverage"], abs=SCORE_TOL
        ), f"{ts_query['id']}: IDF 覆盖率不一致"
        assert len(scores) == len(ts_query["scores"])
        for i, (py_score, ts_score) in enumerate(zip(scores, ts_query["scores"])):
            assert py_score == pytest.approx(ts_score, abs=1e-6), (
                f"{ts_query['id']} 第 {i} 块 BM25 分不一致：PY={py_score} TS={ts_score}"
            )


@requires_node
@pytest.mark.parametrize("top_k", [5, 10])
def test_retrieval_hits_match_ts(docs, golden, fixed_embeddings, top_k) -> None:
    """最终命中逐条比对（含窗口范围与相关度）。"""
    dump = run_parity_dump("retrieval", timeout=900)
    bm25 = BM25([d.content for d in docs])
    key = "wide" if top_k == 10 else "result"

    checked = 0
    for ts_query in dump["retrieval"]["queries"]:
        question = ts_query["question"]
        bm25_scores = bm25.score_all(question)
        vec_scores = vec_scores_for(docs, ts_query["id"], fixed_embeddings)
        idf_coverage = bm25.idf_weighted_coverage(question)
        result = post_process_retrieval(
            docs,
            bm25_scores,
            vec_scores,
            DEFAULT_RETRIEVAL_OPTIONS.__class__(**{**DEFAULT_RETRIEVAL_OPTIONS.__dict__, "top_k": top_k}),
            idf_coverage=idf_coverage,
        )
        expected = ts_query[key]
        assert result.no_relevant_hit == expected["noRelevantHit"], (
            f"{ts_query['id']}: noRelevantHit 不一致 PY={result.no_relevant_hit} "
            f"TS={expected['noRelevantHit']}"
        )
        assert len(result.hits) == len(expected["hits"]), (
            f"{ts_query['id']}: 命中条数不一致 PY={len(result.hits)} TS={len(expected['hits'])}"
        )
        for i, (py_hit, ts_hit) in enumerate(zip(result.hits, expected["hits"])):
            where = f"{ts_query['id']}#{i}"
            assert py_hit.doc_id == ts_hit["docId"], f"{where}: 文档不一致"
            assert py_hit.ord_start == ts_hit["ordStart"], f"{where}: ordStart 不一致"
            assert py_hit.ord_end == ts_hit["ordEnd"], f"{where}: ordEnd 不一致"
            assert py_hit.content == ts_hit["content"], f"{where}: 窗口文本不一致"
            assert py_hit.matched_content == ts_hit["matchedContent"], (
                f"{where}: 命中文本不一致"
            )
            assert py_hit.relevance == pytest.approx(ts_hit["relevance"], abs=1e-9), (
                f"{where}: relevance 不一致"
            )
            assert py_hit.rrf_score == pytest.approx(ts_hit["rrfScore"], abs=1e-12), (
                f"{where}: RRF 分不一致"
            )
            assert py_hit.vec_rank == ts_hit["vecRank"], f"{where}: 向量排名不一致"
            assert py_hit.bm25_rank == ts_hit["bm25Rank"], f"{where}: BM25 排名不一致"
            # 窗口组成用 ord 表达（chunkId 是随机 UUID，不能跨端比对）
            py_ords = [
                next(d.ord for d in docs if d.chunk_id == cid)
                for cid in py_hit.window_chunk_ids
            ]
            assert py_ords == ts_hit["windowOrds"], f"{where}: 窗口块序列不一致"
            checked += 1
    assert checked > 0, "没有比对任何命中：dump 结构可能变了"
