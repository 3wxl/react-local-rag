"""向量化跨运行时一致性：Python ONNX 与固化向量（TS/transformers.js 产出）比。

两端用**同一份权重**（`model_quantized.onnx`，q8）与同一套前缀规则，
但运行时不同（onnxruntime-web WASM vs onnxruntime CPU）。所以这里不比"逐位相等"，
而是比**语义等价**：余弦 ≥ 0.99，且用它做检索时排序不变。

没有模型权重的环境（CI）自动跳过 —— 跨语言 contentHash 一致性不依赖模型，
这条测试是"真实服务端路径"的补充证据。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import settings
from app.embedding.bge import BgeOnnxEmbedder
from app.embedding.fixed import load_fixed_embeddings
from app.harness.golden import load_corpus_docs
from app.parity.retrieval import dot_product

MODEL_READY = (settings.bge_model_dir / "onnx" / "model_quantized.onnx").exists()

requires_model = pytest.mark.skipif(
    not MODEL_READY, reason="本机没有 bge-small-zh ONNX 权重（apps/web/public/models）"
)

#: 允许的余弦下界：低于它就说明前缀/池化/权重口径不同，而不是"数值抖动"
MIN_COSINE = 0.99


@requires_model
def test_passage_vectors_match_frozen() -> None:
    fixed = load_fixed_embeddings(settings.fixed_embeddings_path)
    assert fixed is not None
    docs = load_corpus_docs(settings.eval_dir, chunk_vectors=fixed.chunks)
    embedder = BgeOnnxEmbedder(settings.bge_model_dir)
    vectors = embedder.embed_passages([d.content for d in docs])
    cosines = [dot_product(vectors[i], docs[i].vector) for i in range(len(docs))]
    assert min(cosines) >= MIN_COSINE, (
        f"文档向量与固化向量余弦过低：min={min(cosines):.4f}（阈值 {MIN_COSINE}）"
    )


@requires_model
def test_query_vectors_match_frozen() -> None:
    fixed = load_fixed_embeddings(settings.fixed_embeddings_path)
    assert fixed is not None
    from app.harness.golden import load_golden

    embedder = BgeOnnxEmbedder(settings.bge_model_dir)
    worst = 1.0
    for query in load_golden(settings.eval_dir).queries[:8]:
        vector = embedder.embed_query(query.question)
        worst = min(worst, dot_product(vector, fixed.queries[query.id]))
    assert worst >= MIN_COSINE, f"查询向量余弦过低：min={worst:.4f}"


@requires_model
def test_retrieval_ranking_stable_under_runtime_difference() -> None:
    """同一批查询：用 Python 现场向量得到的 top-1 文档，与固化向量一致。

    这条比余弦阈值更有意义：端到端的排序不变，才说明运行时差异不影响用户看到的结果。
    """
    fixed = load_fixed_embeddings(settings.fixed_embeddings_path)
    assert fixed is not None
    docs = load_corpus_docs(settings.eval_dir, chunk_vectors=fixed.chunks)
    embedder = BgeOnnxEmbedder(settings.bge_model_dir)

    from app.harness.golden import load_golden

    queries = load_golden(settings.eval_dir).queries
    same = 0
    checked = 0
    for query in queries[:12]:
        live = embedder.embed_query(query.question)
        fixed_scores = [dot_product(d.vector, fixed.queries[query.id]) for d in docs]
        live_scores = [dot_product(d.vector, live) for d in docs]
        fixed_top = int(np.argmax(fixed_scores))
        live_top = int(np.argmax(live_scores))
        checked += 1
        same += int(fixed_top == live_top)
    assert same == checked, f"top-1 块在两种向量来源下不一致：{same}/{checked} 相同"
