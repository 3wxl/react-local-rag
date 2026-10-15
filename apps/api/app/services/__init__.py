"""业务服务层：摄取、检索、问答、评测任务。

服务层只编排顺序与状态，算法一律来自 app/parity（与浏览器端同源），
存储细节来自 app/store（SQLite / pgvector 两种实现）。
"""

from .answer_service import AnswerOutcome, AnswerService
from .ingest_service import IngestResult, IngestService
from .retrieval_service import RetrievalService, SearchOutcome

__all__ = [
    "AnswerOutcome",
    "AnswerService",
    "IngestResult",
    "IngestService",
    "RetrievalService",
    "SearchOutcome",
]
