"""存储选择：默认本地 SQLite，配置 DATABASE_URL(postgres) 时切 Postgres/pgvector。

为什么两种：路线图的生产选型是 Postgres+pgvector（单库、事务一致、ACL 可在召回层过滤），
但"本机/CI 没有 Docker 与 Postgres"是常态 —— 默认走 SQLite 才能保证这套东西**随时跑得起来**，
否则摄取、问答、双端对比都成了"只能在文档里存在"的能力。两种后端由同一套契约测试约束。
"""

from __future__ import annotations

from pathlib import Path

from .base import Store
from .sqlite_store import SqliteStore


def make_store(database_url: str, data_dir: Path) -> Store:
    """按 DATABASE_URL 选择存储实现。"""
    url = (database_url or "").strip()
    if url.startswith("postgres://") or url.startswith("postgresql://"):
        try:
            from .pg_store import PgStore
        except ImportError as err:  # pragma: no cover —— 取决于是否装了 [pg] 额外依赖
            raise RuntimeError(
                "DATABASE_URL 指向 Postgres，但缺少依赖："
                "先执行 `uv sync --extra pg`（安装 asyncpg + pgvector）"
            ) from err
        return PgStore(url)
    return SqliteStore(Path(data_dir) / "rag.sqlite3")
