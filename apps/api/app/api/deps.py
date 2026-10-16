"""应用级依赖：把 store / 模型 / 服务挂到 app.state，路由从这里取。

进程模型（M1 单进程）：
- 存储、BM25 缓存、进程内任务运行器都是**进程内**的 → 必须 `uvicorn --workers 1`
- 换 Postgres 后任务表可支持多进程认领（tasks.owner 字段已留好），但 M1 不做
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Request
from langgraph.checkpoint.base import BaseCheckpointSaver

from ..core.config import Settings
from ..embedding.bge import BgeOnnxEmbedder
from ..generation.base import LlmProvider
from ..services.answer_service import AnswerService
from ..services.ingest_service import IngestService
from ..services.reindex_service import ReindexService
from ..services.retrieval_service import RetrievalService
from ..store import Store


@dataclass
class AppContext:
    """一次进程生命周期的全部共享对象。"""

    settings: Settings
    store: Store
    embedder: BgeOnnxEmbedder
    provider: LlmProvider
    retrieval: RetrievalService
    answer: AnswerService
    ingest: IngestService
    #: 索引灰度重建（进程内任务，断点写 tasks 表）
    reindex: ReindexService
    #: LangGraph 检查点（M3）：interrupt 恢复与断点续跑都在这里落盘
    checkpointer: BaseCheckpointSaver | None = None
    #: 摄取图（M3）：POST /v1/documents 与重建都走它（reindex 走 ReindexService 内部持有的同一实例）
    ingest_graph: object | None = None
    #: Agent 运行服务（M3）：/v1/agent/* —— planner + 工具（含 MCP）+ HITL
    agent: object | None = None
    #: 进程内任务（评测 / 摄取 / Agent 续跑）：不引队列框架，状态写 tasks 表
    tasks: dict[str, asyncio.Task] = field(default_factory=dict)

    def spawn(self, task_id: str, coro) -> None:
        """登记一个后台任务（重复 id 直接覆盖，避免泄漏悬挂任务）。"""
        task = asyncio.create_task(coro)
        self.tasks[task_id] = task

        def _cleanup(_: asyncio.Task) -> None:
            self.tasks.pop(task_id, None)

        task.add_done_callback(_cleanup)


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def get_settings(request: Request) -> Settings:
    return request.app.state.ctx.settings


def data_dir(ctx: AppContext) -> Path:
    return Path(ctx.settings.data_dir)
