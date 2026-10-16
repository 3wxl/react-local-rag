"""pytest 公共 fixture：评测资产（packages/eval）与冻结期望值。

路径与 Node 调用等工具函数在 tests/parity_utils.py，供测试直接 import。

⚠ 测试数据目录必须在**任何 app 模块被导入之前**设好：
`app.core.config.settings` 是模块级单例（首次 import 时构造），
在 fixture 里改环境变量已经来不及 —— 那会让测试进程写到开发用的 .data 库上
（曾真的把开发库里的文档软删掉一条）。
"""

from __future__ import annotations

import os
import tempfile

# 必须在 import app.* 之前：测试只允许写临时目录，且默认用假生成模型（不下载权重）
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="rag-api-test-"))
os.environ.setdefault("LLM_PROVIDER", "fake")
# 检查点默认走内存：大多数测试不关心落盘；持久化本身由 test_checkpoint.py 用临时 SQLite 单独证明
os.environ.setdefault("CHECKPOINT_BACKEND", "memory")

import pytest

from .parity_utils import CHUNKER_SPEC_DIR, EVAL_CORPUS_DIR, EVAL_DIR, load_json


@pytest.fixture
async def qa_env(tmp_path):
    """自建的 QA 图运行环境（独立数据目录 + 真实摄取 + 假生成模型）。

    为什么不用共享的 DATA_DIR：那条链上别的用例会软删文档、灌合成语料，
    检索统计（IDF / 拒答门限）随之漂移 —— 依赖共享语料的裁决断言会随文件顺序红绿闪烁。
    另外 embedder 用**新实例**而不是 get_embedder 单例：单例会被别的用例改 `enabled`。
    """
    from app.core.config import Settings
    from app.embedding.bge import BgeOnnxEmbedder
    from app.generation.fake import FakeProvider
    from app.graphs.qa_graph import build_qa_graph
    from app.graphs.qa_nodes import QaDeps
    from app.services.ingest_service import IngestService
    from app.services.retrieval_service import RetrievalService
    from app.store import make_store

    settings = Settings(data_dir=tmp_path)
    store = make_store("", tmp_path)
    await store.init()
    # 权重缺失时 available=False，检索自动退化为 BM25-only（CI 环境就是这样）
    embedder = BgeOnnxEmbedder(settings.bge_model_dir)
    provider = FakeProvider()
    retrieval = RetrievalService(store, embedder)
    ingest = IngestService(
        store,
        embedder,
        data_dir=tmp_path,
        on_index_changed=lambda version=None: retrieval.invalidate(version),
    )
    for name in ("产品手册.md", "员工手册.md", "会议纪要.md"):
        await ingest.ingest_path(EVAL_CORPUS_DIR / name, doc_name=name)

    deps = QaDeps(
        store=store,
        retrieval=retrieval,
        embedder=embedder,
        provider=provider,
        settings=settings,
    )
    from types import SimpleNamespace

    try:
        yield SimpleNamespace(
            settings=settings,
            store=store,
            deps=deps,
            graph=build_qa_graph(deps, None),
            version=(await store.list_ready_versions())[0].version,
        )
    finally:
        await store.close()


@pytest.fixture(scope="session")
def expected_chunks() -> dict:
    """fixtures 的冻结期望值（无 Node 环境时的兜底比对基准）。"""
    return load_json(CHUNKER_SPEC_DIR / "expected-chunks.json")


@pytest.fixture(scope="session")
def golden() -> dict:
    return load_json(EVAL_DIR / "golden.json")


@pytest.fixture(scope="session")
def fixed_embeddings() -> dict:
    return load_json(EVAL_DIR / "embeddings.json")


@pytest.fixture(scope="session")
def ts_baseline() -> dict:
    """TS 端评测指标的冻结副本（apps/web/scripts/eval.ts 的产出）。"""
    return load_json(EVAL_DIR / "baseline" / "ts-metrics.json")
