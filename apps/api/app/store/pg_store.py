"""Postgres + pgvector 存储（生产路径）。

与 SQLite 实现**行为一致**（同一套契约测试跑两遍），差别只在落地方式：
- 向量存 `vector(512)` 列，走 pgvector 的**二进制编解码**（`register_vector`）。
  绝不用文本 I/O：pgvector 的文本输出只有约 6 位有效数字，会悄悄改掉余弦值，
  而"两端余弦一致"正是双端对比的前提。
- 余弦检索可以下推给 SQL（`<=>` 即 cosine distance）；但**排序、融合、门限仍留在 Python**，
  与 TS 完全同源（见 app/store/base.py 的说明）。
- ACL 过滤（M2）用 `doc_ids` 走 SQL WHERE，属于召回层过滤，不是取回后再筛。

需要 `apps/api[pg]` 额外依赖（asyncpg + pgvector）。没装或没配 DATABASE_URL 时不会加载本模块。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .base import (
    AnswerRecord,
    ChunkRecord,
    DocumentRecord,
    EvalRunRecord,
    Filters,
    IndexVersionRecord,
    TaskRecord,
    TraceRecord,
)

SCHEMA_PATH = Path(__file__).with_name("schema_pg.sql")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


class PgStore:
    """asyncpg 实现（连接池 + pgvector 二进制编解码）。"""

    backend = "postgres"

    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 10) -> None:
        self.dsn = dsn
        self._pool = None
        self._min_size = min_size
        self._max_size = max_size

    async def init(self) -> None:
        import asyncpg
        from pgvector.asyncpg import register_vector

        self._pool = await asyncpg.create_pool(
            self.dsn,
            min_size=self._min_size,
            max_size=self._max_size,
            init=register_vector,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # ---------------- 索引版本 ----------------

    async def get_active_version(self) -> IndexVersionRecord | None:
        row = await self._pool.fetchrow(
            "SELECT * FROM index_version WHERE state='ready' "
            "ORDER BY activated_at DESC NULLS LAST LIMIT 1"
        )
        return _version_of(row) if row else None

    async def upsert_version(self, record: IndexVersionRecord) -> None:
        await self._pool.execute(
            """INSERT INTO index_version (version, state, chunk_params, embedding_model,
                   prefix_scheme, created_at, activated_at)
               VALUES ($1,$2,$3::jsonb,$4,$5,$6,$7)
               ON CONFLICT (version) DO UPDATE SET
                   state=EXCLUDED.state, chunk_params=EXCLUDED.chunk_params,
                   embedding_model=EXCLUDED.embedding_model,
                   prefix_scheme=EXCLUDED.prefix_scheme,
                   activated_at=EXCLUDED.activated_at""",
            record.version,
            record.state,
            _json(record.chunk_params),
            record.embedding_model,
            record.prefix_scheme,
            _parse_ts(record.created_at) or datetime.now(timezone.utc),
            _parse_ts(record.activated_at),
        )

    async def activate_version(self, version: str) -> None:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE index_version SET state='stale' "
                    "WHERE state='ready' AND version<>$1",
                    version,
                )
                await conn.execute(
                    "UPDATE index_version SET state='ready', activated_at=now() "
                    "WHERE version=$1",
                    version,
                )

    # ---------------- 文档与块 ----------------

    async def upsert_document(self, record: DocumentRecord) -> None:
        await self._pool.execute(
            """INSERT INTO documents (doc_id, doc_name, content_hash, index_version, tags,
                   raw_path, size_bytes, page_count, created_at, deleted_at)
               VALUES ($1,$2,$3,$4,$5::jsonb,$6,$7,$8,$9,$10)
               ON CONFLICT (doc_id) DO UPDATE SET
                   doc_name=EXCLUDED.doc_name, content_hash=EXCLUDED.content_hash,
                   index_version=EXCLUDED.index_version, tags=EXCLUDED.tags,
                   raw_path=EXCLUDED.raw_path, size_bytes=EXCLUDED.size_bytes,
                   page_count=EXCLUDED.page_count, deleted_at=EXCLUDED.deleted_at""",
            record.doc_id,
            record.doc_name,
            record.content_hash,
            record.index_version,
            _json(record.tags),
            record.raw_path,
            record.size_bytes,
            record.page_count,
            _parse_ts(record.created_at) or datetime.now(timezone.utc),
            _parse_ts(record.deleted_at),
        )

    async def replace_doc_chunks(self, doc_id: str, chunks: list[ChunkRecord]) -> None:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM chunk_vectors WHERE chunk_id IN "
                    "(SELECT chunk_id FROM chunks WHERE doc_id=$1)",
                    doc_id,
                )
                await conn.execute("DELETE FROM chunks WHERE doc_id=$1", doc_id)
                if chunks:
                    await conn.executemany(
                        """INSERT INTO chunks (chunk_id, doc_id, index_version, ord, content,
                               content_hash, heading_path, page_start, page_end, char_start,
                               char_end, token_count, parent_id, prev_id, next_id)
                           VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8,$9,$10,$11,$12,$13,$14,$15)""",
                        [
                            (
                                c.chunk_id,
                                c.doc_id,
                                c.index_version,
                                c.ord,
                                c.content,
                                c.content_hash,
                                _json(c.heading_path),
                                c.page_start,
                                c.page_end,
                                c.char_start,
                                c.char_end,
                                c.token_count,
                                c.parent_id,
                                c.prev_id,
                                c.next_id,
                            )
                            for c in chunks
                        ],
                    )

    async def soft_delete_doc(self, doc_id: str) -> None:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE documents SET deleted_at=now() WHERE doc_id=$1", doc_id
                )
                # 召回层立刻生效：软删除后不再有向量可召回
                await conn.execute(
                    "DELETE FROM chunk_vectors WHERE chunk_id IN "
                    "(SELECT chunk_id FROM chunks WHERE doc_id=$1)",
                    doc_id,
                )

    async def list_documents(self, *, include_deleted: bool = False) -> list[DocumentRecord]:
        sql = "SELECT * FROM documents"
        if not include_deleted:
            sql += " WHERE deleted_at IS NULL"
        sql += " ORDER BY created_at"
        return [_document_of(r) for r in await self._pool.fetch(sql)]

    async def load_chunks(
        self, version: str, *, filters: Filters | None = None
    ) -> list[ChunkRecord]:
        sql = (
            "SELECT c.*, d.doc_name AS doc_name FROM chunks c "
            "JOIN documents d ON d.doc_id=c.doc_id "
            "WHERE c.index_version=$1 AND d.deleted_at IS NULL"
        )
        params: list[Any] = [version]
        if filters and filters.doc_ids:
            sql += " AND c.doc_id = ANY($2::text[])"
            params.append(list(filters.doc_ids))
        sql += " ORDER BY c.doc_id, c.ord"
        return [_chunk_of(r) for r in await self._pool.fetch(sql, *params)]

    # ---------------- 向量 ----------------

    async def put_vectors(self, version: str, rows: list[tuple[str, np.ndarray]]) -> None:
        if not rows:
            return
        await self._pool.executemany(
            """INSERT INTO chunk_vectors (chunk_id, index_version, embedding)
               VALUES ($1,$2,$3)
               ON CONFLICT (chunk_id) DO UPDATE SET
                   index_version=EXCLUDED.index_version, embedding=EXCLUDED.embedding""",
            [
                (chunk_id, version, np.asarray(vec, dtype=np.float32))
                for chunk_id, vec in rows
            ],
        )

    async def cosine_all(
        self, version: str, query_vec: np.ndarray, *, filters: Filters | None = None
    ) -> list[tuple[str, float]]:
        """精确扫描（评测与对齐口径必须用它）。

        与 SQLite 版顺序一致（doc_id, ord）：两端做同样的候选顺序，融合与并列处理才不会分叉。
        """
        sql = (
            "SELECT v.chunk_id AS chunk_id, 1 - (v.embedding <=> $2) AS score "
            "FROM chunk_vectors v JOIN chunks c ON c.chunk_id=v.chunk_id "
            "JOIN documents d ON d.doc_id=c.doc_id "
            "WHERE v.index_version=$1 AND d.deleted_at IS NULL"
        )
        params: list[Any] = [version, np.asarray(query_vec, dtype=np.float32)]
        if filters and filters.doc_ids:
            sql += " AND c.doc_id = ANY($3::text[])"
            params.append(list(filters.doc_ids))
        sql += " ORDER BY c.doc_id, c.ord"
        rows = await self._pool.fetch(sql, *params)
        return [(r["chunk_id"], float(r["score"])) for r in rows]

    async def load_vectors(
        self, version: str, *, filters: Filters | None = None
    ) -> dict[str, np.ndarray]:
        """一次取回该索引版本的全部向量（顺序与 load_chunks 一致：doc_id, ord）。"""
        sql = (
            "SELECT v.chunk_id, v.embedding FROM chunk_vectors v "
            "JOIN chunks c ON c.chunk_id=v.chunk_id "
            "JOIN documents d ON d.doc_id=c.doc_id "
            "WHERE v.index_version=$1 AND d.deleted_at IS NULL"
        )
        params: list[Any] = [version]
        if filters and filters.doc_ids:
            sql += " AND c.doc_id = ANY($2::text[])"
            params.append(list(filters.doc_ids))
        sql += " ORDER BY c.doc_id, c.ord"
        rows = await self._pool.fetch(sql, *params)
        return {
            r["chunk_id"]: np.asarray(r["embedding"], dtype=np.float32) for r in rows
        }

    async def get_vector(self, chunk_id: str) -> np.ndarray | None:
        row = await self._pool.fetchrow(
            "SELECT embedding FROM chunk_vectors WHERE chunk_id=$1", chunk_id
        )
        return np.asarray(row["embedding"], dtype=np.float32) if row else None

    # ---------------- 答案与轨迹 ----------------

    async def save_answer(self, record: AnswerRecord) -> None:
        await self._pool.execute(
            """INSERT INTO answers (answer_id, trace_id, question, answer, verdict, citations,
                   refusal, verification, evidence_snapshot, cost, config, created_at)
               VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7::jsonb,$8::jsonb,$9::jsonb,$10::jsonb,$11::jsonb,$12)""",
            record.answer_id,
            record.trace_id,
            record.question,
            record.answer,
            record.verdict,
            _json(record.citations),
            None if record.refusal is None else _json(record.refusal),
            _json(record.verification),
            _json(record.evidence_snapshot),
            _json(record.cost),
            _json(record.config),
            _parse_ts(record.created_at) or datetime.now(timezone.utc),
        )

    async def get_answer(self, answer_id: str) -> AnswerRecord | None:
        row = await self._pool.fetchrow(
            "SELECT * FROM answers WHERE answer_id=$1", answer_id
        )
        if not row:
            return None
        return AnswerRecord(
            answer_id=row["answer_id"],
            trace_id=row["trace_id"],
            question=row["question"],
            answer=row["answer"],
            verdict=row["verdict"],
            citations=json.loads(row["citations"]),
            refusal=json.loads(row["refusal"]) if row["refusal"] else None,
            verification=json.loads(row["verification"]),
            evidence_snapshot=json.loads(row["evidence_snapshot"]),
            cost=json.loads(row["cost"]),
            config=json.loads(row["config"]),
            created_at=_iso(row["created_at"]),
        )

    async def save_trace(self, record: TraceRecord) -> None:
        await self._pool.execute(
            """INSERT INTO traces (trace_id, answer_id, question, nodes, final_verdict,
                   evidence_snapshot, cost, created_at)
               VALUES ($1,$2,$3,$4::jsonb,$5,$6::jsonb,$7::jsonb,$8)
               ON CONFLICT (trace_id) DO UPDATE SET
                   nodes=EXCLUDED.nodes, final_verdict=EXCLUDED.final_verdict,
                   evidence_snapshot=EXCLUDED.evidence_snapshot, cost=EXCLUDED.cost""",
            record.trace_id,
            record.answer_id,
            record.question,
            _json(record.nodes),
            record.final_verdict,
            _json(record.evidence_snapshot),
            _json(record.cost),
            _parse_ts(record.created_at) or datetime.now(timezone.utc),
        )

    async def get_trace(self, trace_id: str) -> TraceRecord | None:
        row = await self._pool.fetchrow(
            "SELECT * FROM traces WHERE trace_id=$1", trace_id
        )
        if not row:
            return None
        return TraceRecord(
            trace_id=row["trace_id"],
            answer_id=row["answer_id"],
            question=row["question"],
            nodes=json.loads(row["nodes"]),
            final_verdict=row["final_verdict"],
            evidence_snapshot=json.loads(row["evidence_snapshot"]),
            cost=json.loads(row["cost"]),
            created_at=_iso(row["created_at"]),
        )

    # ---------------- 评测运行 ----------------

    async def save_eval_run(self, record: EvalRunRecord) -> None:
        await self._pool.execute(
            """INSERT INTO eval_runs (run_id, status, suite, golden_version, config, metrics,
                   gates_passed, failed_gates, error, created_at, finished_at)
               VALUES ($1,$2,$3,$4,$5::jsonb,$6::jsonb,$7,$8::jsonb,$9,$10,$11)
               ON CONFLICT (run_id) DO UPDATE SET
                   status=EXCLUDED.status, metrics=EXCLUDED.metrics,
                   gates_passed=EXCLUDED.gates_passed, failed_gates=EXCLUDED.failed_gates,
                   error=EXCLUDED.error, finished_at=EXCLUDED.finished_at""",
            record.run_id,
            record.status,
            record.suite,
            record.golden_version,
            _json(record.config),
            None if record.metrics is None else _json(record.metrics),
            record.gates_passed,
            _json(record.failed_gates),
            record.error,
            _parse_ts(record.created_at) or datetime.now(timezone.utc),
            _parse_ts(record.finished_at),
        )

    async def update_eval_run(self, run_id: str, **fields) -> None:
        allowed = {
            "status",
            "metrics",
            "gates_passed",
            "failed_gates",
            "error",
            "finished_at",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"不支持更新的字段：{sorted(unknown)}")
        sets, params = [], []
        for i, (key, value) in enumerate(fields.items(), start=1):
            if key in ("metrics", "failed_gates"):
                sets.append(f"{key}=${i}::jsonb")
                params.append(_json(value))
            elif key == "finished_at":
                sets.append(f"{key}=${i}")
                params.append(_parse_ts(value))
            else:
                sets.append(f"{key}=${i}")
                params.append(value)
        params.append(run_id)
        await self._pool.execute(
            f"UPDATE eval_runs SET {','.join(sets)} WHERE run_id=${len(params)}", *params
        )

    async def get_eval_run(self, run_id: str) -> EvalRunRecord | None:
        row = await self._pool.fetchrow(
            "SELECT * FROM eval_runs WHERE run_id=$1", run_id
        )
        if not row:
            return None
        return EvalRunRecord(
            run_id=row["run_id"],
            status=row["status"],
            suite=row["suite"],
            golden_version=row["golden_version"],
            config=json.loads(row["config"]),
            metrics=json.loads(row["metrics"]) if row["metrics"] else None,
            gates_passed=row["gates_passed"],
            failed_gates=json.loads(row["failed_gates"]),
            error=row["error"],
            created_at=_iso(row["created_at"]),
            finished_at=_iso(row["finished_at"]),
        )

    # ---------------- 任务状态表 ----------------

    async def save_task(self, record: TaskRecord) -> None:
        await self._pool.execute(
            """INSERT INTO tasks (task_id, kind, status, progress, checkpoint, owner, error,
                   created_at, updated_at)
               VALUES ($1,$2,$3,$4::jsonb,$5::jsonb,$6,$7,$8,$9)
               ON CONFLICT (task_id) DO UPDATE SET
                   status=EXCLUDED.status, progress=EXCLUDED.progress,
                   checkpoint=EXCLUDED.checkpoint, owner=EXCLUDED.owner,
                   error=EXCLUDED.error, updated_at=EXCLUDED.updated_at""",
            record.task_id,
            record.kind,
            record.status,
            _json(record.progress),
            _json(record.checkpoint),
            record.owner,
            record.error,
            _parse_ts(record.created_at) or datetime.now(timezone.utc),
            _parse_ts(record.updated_at) or datetime.now(timezone.utc),
        )

    async def update_task(self, task_id: str, **fields) -> None:
        allowed = {"status", "progress", "checkpoint", "owner", "error"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"不支持更新的字段：{sorted(unknown)}")
        sets, params = [], []
        for i, (key, value) in enumerate(fields.items(), start=1):
            if key in ("progress", "checkpoint"):
                sets.append(f"{key}=${i}::jsonb")
                params.append(_json(value))
            else:
                sets.append(f"{key}=${i}")
                params.append(value)
        params.append(task_id)
        await self._pool.execute(
            f"UPDATE tasks SET {','.join(sets)}, updated_at=now() "
            f"WHERE task_id=${len(params)}",
            *params,
        )

    async def get_task(self, task_id: str) -> TaskRecord | None:
        row = await self._pool.fetchrow("SELECT * FROM tasks WHERE task_id=$1", task_id)
        if not row:
            return None
        return TaskRecord(
            task_id=row["task_id"],
            kind=row["kind"],
            status=row["status"],
            progress=json.loads(row["progress"]),
            checkpoint=json.loads(row["checkpoint"]),
            owner=row["owner"],
            error=row["error"],
            created_at=_iso(row["created_at"]),
            updated_at=_iso(row["updated_at"]),
        )


# ---------------- 行 → 记录 ----------------


def _iso(value) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None


def _parse_ts(value: str | None):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _version_of(row) -> IndexVersionRecord:
    return IndexVersionRecord(
        version=row["version"],
        state=row["state"],
        chunk_params=json.loads(row["chunk_params"]),
        embedding_model=row["embedding_model"],
        prefix_scheme=row["prefix_scheme"],
        created_at=_iso(row["created_at"]),
        activated_at=_iso(row["activated_at"]),
    )


def _document_of(row) -> DocumentRecord:
    return DocumentRecord(
        doc_id=row["doc_id"],
        doc_name=row["doc_name"],
        content_hash=row["content_hash"],
        index_version=row["index_version"],
        tags=json.loads(row["tags"]),
        raw_path=row["raw_path"],
        size_bytes=row["size_bytes"],
        page_count=row["page_count"],
        created_at=_iso(row["created_at"]),
        deleted_at=_iso(row["deleted_at"]),
    )


def _chunk_of(row) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        doc_name=row["doc_name"] if "doc_name" in row else "",
        index_version=row["index_version"],
        ord=row["ord"],
        content=row["content"],
        content_hash=row["content_hash"],
        heading_path=json.loads(row["heading_path"]),
        page_start=row["page_start"],
        page_end=row["page_end"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        token_count=row["token_count"],
        parent_id=row["parent_id"],
        prev_id=row["prev_id"],
        next_id=row["next_id"],
    )
