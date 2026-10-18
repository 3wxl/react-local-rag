"""LangGraph 检查点后端（M3）。

选型：官方 `AsyncSqliteSaver`（langgraph-checkpoint-sqlite），不用自研。
理由：检查点语义（pending writes / checkpoint_ns / 恢复点选择）直接决定
`interrupt` 能否续跑，自研的坑远多于收益；而官方 sqlite saver 无需额外服务。

两个刻意决定：
- **Postgres 存储后端也用这个 SQLite 文件**：部署形态是单副本
  （compose 固定 `--workers 1`），检查点与业务库分离不引入新的运维对象。
- **必须在 lifespan 内构造**：AsyncSqliteSaver 绑定当前事件循环（aiosqlite 线程），
  模块导入时构造会在错误的循环上建连接 —— 见 app/main.py。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ..core.config import Settings


@asynccontextmanager
async def open_checkpointer(settings: Settings) -> AsyncIterator[BaseCheckpointSaver]:
    """打开检查点后端：`memory` 供测试，`sqlite` 为默认（落 DATA_DIR）。"""
    if settings.checkpoint_backend == "memory":
        yield InMemorySaver()
        return

    path = settings.checkpoint_db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        yield saver
