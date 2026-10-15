"""存储层：SQLite（默认，零依赖）与 Postgres/pgvector（生产路径）。

检索语义不在这里 —— 见 app/parity/retrieval.py。存储只负责存取块、向量与运行状态。
"""

from .base import (
    AnswerRecord,
    ChunkRecord,
    DocumentRecord,
    EvalRunRecord,
    Filters,
    IndexVersionRecord,
    Store,
    TaskRecord,
    TraceRecord,
)
from .factory import make_store
from .sqlite_store import SqliteStore

__all__ = [
    "AnswerRecord",
    "ChunkRecord",
    "DocumentRecord",
    "EvalRunRecord",
    "Filters",
    "IndexVersionRecord",
    "SqliteStore",
    "Store",
    "TaskRecord",
    "TraceRecord",
    "make_store",
]
