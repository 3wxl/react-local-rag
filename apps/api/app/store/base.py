"""存储接口与记录结构（SQLite 与 Postgres/pgvector 两种实现共用）。

设计要点：
- **检索语义留在 Python**：BM25、RRF、邻近合并、去重、MMR、门限全部在 `app/parity/retrieval.py`，
  存储层只负责"把块、向量、状态存取出来"。原因很直接：SQL 里的 `ts_rank`/`pg_trgm` 与
  TS 的 BM25 公式必然不同，一旦把排序下推到 SQL，跨语言一致性就无从谈起。
- 向量按 **float32 小端二进制** 往返（SQLite 用 BLOB，Postgres 用 pgvector 二进制编解码），
  绝不走文本 I/O —— pgvector 的文本输出只有 ~6 位有效数字，会悄悄改掉余弦值。
- `filters` 现在就贯穿到 SQL（M1 是单租户空实现）：M2 的 ACL 必须在**召回层**过滤，
  而不是取回后再筛（后过滤会污染召回，还会让人从结果条数推断无权文档存在）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


@dataclass
class ChunkRecord:
    """块（与 TS TextChunk 字段一一对应，便于两端比对）。"""

    chunk_id: str
    doc_id: str
    index_version: str
    ord: int
    content: str
    content_hash: str
    heading_path: list[str | None]
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    token_count: int
    parent_id: str = ""
    prev_id: str = ""
    next_id: str = ""
    #: 文档名（从 documents 表 join 出来；引用标签「文件名 · 章节 · 页码」要用）
    doc_name: str = ""


@dataclass
class DocumentRecord:
    doc_id: str
    doc_name: str
    content_hash: str
    index_version: str
    tags: list[str] = field(default_factory=list)
    raw_path: str | None = None
    size_bytes: int = 0
    page_count: int = 1
    created_at: str = ""
    deleted_at: str | None = None


@dataclass
class IndexVersionRecord:
    version: str
    state: str  # building | ready | stale | failed
    chunk_params: dict
    embedding_model: str
    prefix_scheme: str
    created_at: str = ""
    activated_at: str | None = None


@dataclass
class AnswerRecord:
    answer_id: str
    trace_id: str
    question: str
    answer: str
    verdict: str
    citations: list[dict]
    refusal: dict | None
    verification: dict
    evidence_snapshot: dict
    cost: dict
    config: dict
    created_at: str = ""


@dataclass
class TraceRecord:
    trace_id: str
    answer_id: str
    question: str
    nodes: list[dict]
    final_verdict: str
    evidence_snapshot: dict
    cost: dict
    created_at: str = ""


@dataclass
class EvalRunRecord:
    run_id: str
    status: str  # queued | running | succeeded | failed
    suite: str
    golden_version: str
    config: dict
    metrics: dict | None = None
    gates_passed: bool | None = None
    failed_gates: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: str = ""
    finished_at: str | None = None


@dataclass
class TaskRecord:
    task_id: str
    kind: str  # ingest | eval
    status: str  # queued | running | succeeded | failed | interrupted
    progress: dict = field(default_factory=dict)
    checkpoint: dict = field(default_factory=dict)
    owner: str | None = None
    error: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Filters:
    """检索过滤条件（M1 单租户为空实现，M2 的 ACL 从这里进 SQL）。"""

    doc_ids: Sequence[str] = ()
    tags: Sequence[str] = ()


class Store(Protocol):
    """存储接口（两种实现必须行为一致，见 tests/test_store_contract.py）。"""

    backend: str

    async def init(self) -> None: ...
    async def close(self) -> None: ...

    # ---------------- 索引版本 ----------------
    async def get_active_version(self) -> IndexVersionRecord | None: ...
    async def upsert_version(self, record: IndexVersionRecord) -> None: ...
    async def activate_version(self, version: str) -> None: ...

    # ---------------- 文档与块 ----------------
    async def upsert_document(self, record: DocumentRecord) -> None: ...
    async def replace_doc_chunks(self, doc_id: str, chunks: list[ChunkRecord]) -> None: ...
    async def soft_delete_doc(self, doc_id: str) -> None: ...
    async def list_documents(self, *, include_deleted: bool = False) -> list[DocumentRecord]: ...
    async def load_chunks(
        self, version: str, *, filters: Filters | None = None
    ) -> list[ChunkRecord]: ...

    # ---------------- 向量 ----------------
    async def put_vectors(
        self, version: str, rows: list[tuple[str, np.ndarray]]
    ) -> None: ...
    async def cosine_all(
        self, version: str, query_vec: np.ndarray, *, filters: Filters | None = None
    ) -> list[tuple[str, float]]: ...
    async def load_vectors(
        self, version: str, *, filters: Filters | None = None
    ) -> dict[str, np.ndarray]: ...
    async def get_vector(self, chunk_id: str) -> np.ndarray | None: ...

    # ---------------- 答案 / 轨迹 / 评测 / 任务 ----------------
    async def save_answer(self, record: AnswerRecord) -> None: ...
    async def get_answer(self, answer_id: str) -> AnswerRecord | None: ...
    async def save_trace(self, record: TraceRecord) -> None: ...
    async def get_trace(self, trace_id: str) -> TraceRecord | None: ...
    async def save_eval_run(self, record: EvalRunRecord) -> None: ...
    async def update_eval_run(self, run_id: str, **fields) -> None: ...
    async def get_eval_run(self, run_id: str) -> EvalRunRecord | None: ...
    async def save_task(self, record: TaskRecord) -> None: ...
    async def update_task(self, task_id: str, **fields) -> None: ...
    async def get_task(self, task_id: str) -> TaskRecord | None: ...
