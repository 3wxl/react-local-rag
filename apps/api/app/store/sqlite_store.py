"""本地默认存储：SQLite（单文件，零外部依赖）。

为什么默认是它：本机/CI 常常没有 Postgres 与 Docker，而"能不能跑起来"决定了
这套东西能不能被验证。向量以 float32 小端 BLOB 存入，余弦在 numpy 里算 ——
与 Postgres/pgvector 路径**共用同一份检索后处理**，两种后端行为一致（有契约测试钉住）。

生产路径仍推荐 Postgres/pgvector（见 pg_store.py 与 compose.yaml）：
单库、事务一致、ACL 可以在召回层用 SQL 过滤、向量索引可换 HNSW。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from dataclasses import asdict
from pathlib import Path

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

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS index_version (
  version TEXT PRIMARY KEY,
  state TEXT NOT NULL,
  chunk_params TEXT NOT NULL,
  embedding_model TEXT NOT NULL,
  prefix_scheme TEXT NOT NULL,
  created_at TEXT NOT NULL,
  activated_at TEXT
);

CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY,
  doc_name TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  index_version TEXT NOT NULL,
  tags TEXT NOT NULL DEFAULT '[]',
  raw_path TEXT,
  size_bytes INTEGER NOT NULL DEFAULT 0,
  page_count INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL,
  index_version TEXT NOT NULL,
  ord INTEGER NOT NULL,
  content TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  heading_path TEXT NOT NULL DEFAULT '[]',
  page_start INTEGER NOT NULL,
  page_end INTEGER NOT NULL,
  char_start INTEGER NOT NULL,
  char_end INTEGER NOT NULL,
  token_count INTEGER NOT NULL,
  parent_id TEXT NOT NULL DEFAULT '',
  prev_id TEXT NOT NULL DEFAULT '',
  next_id TEXT NOT NULL DEFAULT '',
  UNIQUE (index_version, doc_id, ord)
);
CREATE INDEX IF NOT EXISTS chunks_ver_doc ON chunks (index_version, doc_id);

CREATE TABLE IF NOT EXISTS chunk_vectors (
  chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  index_version TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vec BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunk_vectors_ver ON chunk_vectors (index_version);

CREATE TABLE IF NOT EXISTS answers (
  answer_id TEXT PRIMARY KEY,
  trace_id TEXT NOT NULL,
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  verdict TEXT NOT NULL,
  citations TEXT NOT NULL,
  refusal TEXT,
  verification TEXT NOT NULL,
  evidence_snapshot TEXT NOT NULL,
  cost TEXT NOT NULL,
  config TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS traces (
  trace_id TEXT PRIMARY KEY,
  answer_id TEXT NOT NULL,
  question TEXT NOT NULL,
  nodes TEXT NOT NULL,
  final_verdict TEXT NOT NULL,
  evidence_snapshot TEXT NOT NULL,
  cost TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
  run_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  suite TEXT NOT NULL,
  golden_version TEXT NOT NULL,
  config TEXT NOT NULL,
  metrics TEXT,
  gates_passed INTEGER,
  failed_gates TEXT NOT NULL DEFAULT '[]',
  error TEXT,
  created_at TEXT NOT NULL,
  finished_at TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
  task_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  progress TEXT NOT NULL DEFAULT '{}',
  checkpoint TEXT NOT NULL DEFAULT '{}',
  owner TEXT,
  error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_blob(vec: np.ndarray) -> bytes:
    """float32 小端二进制（与 TS Float32Array 同布局）。"""
    return np.asarray(vec, dtype="<f4").tobytes()


def _from_blob(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4", count=dim).copy()


class SqliteStore:
    """sqlite3 是同步 API；用 asyncio.to_thread 包一层，避免阻塞事件循环。"""

    backend = "sqlite"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    # ---------------- 生命周期 ----------------

    async def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("store 未初始化（先 await store.init()）")
        return self._conn

    async def _run(self, fn, *args):
        return await asyncio.to_thread(fn, *args)

    # ---------------- 索引版本 ----------------

    async def get_active_version(self) -> IndexVersionRecord | None:
        def query() -> IndexVersionRecord | None:
            row = self._db().execute(
                "SELECT * FROM index_version WHERE state = 'ready' ORDER BY activated_at DESC LIMIT 1"
            ).fetchone()
            return _version_of(row) if row else None

        return await self._run(query)

    async def upsert_version(self, record: IndexVersionRecord) -> None:
        def write() -> None:
            conn = self._db()
            conn.execute(
                """INSERT INTO index_version (version, state, chunk_params, embedding_model,
                       prefix_scheme, created_at, activated_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(version) DO UPDATE SET
                       state=excluded.state, chunk_params=excluded.chunk_params,
                       embedding_model=excluded.embedding_model,
                       prefix_scheme=excluded.prefix_scheme,
                       activated_at=excluded.activated_at""",
                (
                    record.version,
                    record.state,
                    json.dumps(record.chunk_params, ensure_ascii=False),
                    record.embedding_model,
                    record.prefix_scheme,
                    record.created_at or _now(),
                    record.activated_at,
                ),
            )
            conn.commit()

        await self._run(write)

    async def activate_version(self, version: str) -> None:
        def write() -> None:
            conn = self._db()
            # 旧版本转 stale（保留数据，便于回滚与灰度对比）
            conn.execute(
                "UPDATE index_version SET state='stale' WHERE state='ready' AND version<>?",
                (version,),
            )
            conn.execute(
                "UPDATE index_version SET state='ready', activated_at=? WHERE version=?",
                (_now(), version),
            )
            conn.commit()

        await self._run(write)

    # ---------------- 文档与块 ----------------

    async def upsert_document(self, record: DocumentRecord) -> None:
        def write() -> None:
            conn = self._db()
            conn.execute(
                """INSERT INTO documents (doc_id, doc_name, content_hash, index_version, tags,
                       raw_path, size_bytes, page_count, created_at, deleted_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(doc_id) DO UPDATE SET
                       doc_name=excluded.doc_name, content_hash=excluded.content_hash,
                       index_version=excluded.index_version, tags=excluded.tags,
                       raw_path=excluded.raw_path, size_bytes=excluded.size_bytes,
                       page_count=excluded.page_count, deleted_at=excluded.deleted_at""",
                (
                    record.doc_id,
                    record.doc_name,
                    record.content_hash,
                    record.index_version,
                    json.dumps(record.tags, ensure_ascii=False),
                    record.raw_path,
                    record.size_bytes,
                    record.page_count,
                    record.created_at or _now(),
                    record.deleted_at,
                ),
            )
            conn.commit()

        await self._run(write)

    async def replace_doc_chunks(self, doc_id: str, chunks: list[ChunkRecord]) -> None:
        def write() -> None:
            conn = self._db()
            old_ids = [
                r["chunk_id"]
                for r in conn.execute(
                    "SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,)
                ).fetchall()
            ]
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            for chunk_id in old_ids:
                conn.execute("DELETE FROM chunk_vectors WHERE chunk_id=?", (chunk_id,))
            conn.executemany(
                """INSERT INTO chunks (chunk_id, doc_id, index_version, ord, content, content_hash,
                       heading_path, page_start, page_end, char_start, char_end, token_count,
                       parent_id, prev_id, next_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        c.chunk_id,
                        c.doc_id,
                        c.index_version,
                        c.ord,
                        c.content,
                        c.content_hash,
                        json.dumps(c.heading_path, ensure_ascii=False),
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
            conn.commit()

        await self._run(write)

    async def soft_delete_doc(self, doc_id: str) -> None:
        def write() -> None:
            conn = self._db()
            conn.execute(
                "UPDATE documents SET deleted_at=? WHERE doc_id=?", (_now(), doc_id)
            )
            # 软删除后立刻从检索面移除（召回层过滤，不是取回后再筛）
            conn.execute("DELETE FROM chunk_vectors WHERE chunk_id IN "
                         "(SELECT chunk_id FROM chunks WHERE doc_id=?)", (doc_id,))
            conn.commit()

        await self._run(write)

    async def list_documents(self, *, include_deleted: bool = False) -> list[DocumentRecord]:
        def query() -> list[DocumentRecord]:
            sql = "SELECT * FROM documents"
            if not include_deleted:
                sql += " WHERE deleted_at IS NULL"
            sql += " ORDER BY created_at"
            return [_document_of(r) for r in self._db().execute(sql).fetchall()]

        return await self._run(query)

    async def load_chunks(
        self, version: str, *, filters: Filters | None = None
    ) -> list[ChunkRecord]:
        def query() -> list[ChunkRecord]:
            sql = (
                "SELECT c.*, d.doc_name AS doc_name FROM chunks c "
                "JOIN documents d ON d.doc_id = c.doc_id "
                "WHERE c.index_version = ? AND d.deleted_at IS NULL"
            )
            params: list[object] = [version]
            if filters and filters.doc_ids:
                sql += f" AND c.doc_id IN ({','.join('?' * len(filters.doc_ids))})"
                params.extend(filters.doc_ids)
            sql += " ORDER BY c.doc_id, c.ord"
            return [_chunk_of(r) for r in self._db().execute(sql, params).fetchall()]

        return await self._run(query)

    # ---------------- 向量 ----------------

    async def put_vectors(
        self, version: str, rows: list[tuple[str, np.ndarray]]
    ) -> None:
        def write() -> None:
            conn = self._db()
            payload = []
            for chunk_id, vec in rows:
                arr = np.asarray(vec, dtype="<f4")
                payload.append((chunk_id, version, int(arr.shape[0]), arr.tobytes()))
            conn.executemany(
                """INSERT INTO chunk_vectors (chunk_id, index_version, dim, vec)
                   VALUES (?,?,?,?)
                   ON CONFLICT(chunk_id) DO UPDATE SET
                       index_version=excluded.index_version, dim=excluded.dim, vec=excluded.vec""",
                payload,
            )
            conn.commit()

        await self._run(write)

    async def cosine_all(
        self, version: str, query_vec: np.ndarray, *, filters: Filters | None = None
    ) -> list[tuple[str, float]]:
        def query() -> list[tuple[str, float]]:
            sql = (
                "SELECT v.chunk_id, v.dim, v.vec FROM chunk_vectors v "
                "JOIN chunks c ON c.chunk_id = v.chunk_id "
                "JOIN documents d ON d.doc_id = c.doc_id "
                "WHERE v.index_version = ? AND d.deleted_at IS NULL"
            )
            params: list[object] = [version]
            if filters and filters.doc_ids:
                sql += f" AND c.doc_id IN ({','.join('?' * len(filters.doc_ids))})"
                params.extend(filters.doc_ids)
            sql += " ORDER BY c.doc_id, c.ord"
            rows = self._db().execute(sql, params).fetchall()
            if not rows:
                return []
            dim = rows[0]["dim"]
            matrix = np.vstack([np.frombuffer(r["vec"], dtype="<f4", count=dim) for r in rows])
            # 与 TS 同一口径：float32 元素、float64 累加（numpy 默认即为 float64 累加）
            scores = matrix.astype(np.float64) @ query_vec.astype(np.float64)
            return [(rows[i]["chunk_id"], float(scores[i])) for i in range(len(rows))]

        return await self._run(query)

    async def load_vectors(
        self, version: str, *, filters: Filters | None = None
    ) -> dict[str, np.ndarray]:
        """一次取回该索引版本的全部向量（顺序与 load_chunks 一致：doc_id, ord）。

        检索要在同一次请求里同时用到「向量分」与「MMR 的相似度」，所以整批取回，
        而不是"先算分再按需取向量"（后者会两次查询且顺序容易对不上）。
        """

        def query() -> dict[str, np.ndarray]:
            sql = (
                "SELECT v.chunk_id, v.dim, v.vec FROM chunk_vectors v "
                "JOIN chunks c ON c.chunk_id = v.chunk_id "
                "JOIN documents d ON d.doc_id = c.doc_id "
                "WHERE v.index_version = ? AND d.deleted_at IS NULL"
            )
            params: list[object] = [version]
            if filters and filters.doc_ids:
                sql += f" AND c.doc_id IN ({','.join('?' * len(filters.doc_ids))})"
                params.extend(filters.doc_ids)
            sql += " ORDER BY c.doc_id, c.ord"
            out: dict[str, np.ndarray] = {}
            for row in self._db().execute(sql, params).fetchall():
                out[row["chunk_id"]] = _from_blob(row["vec"], row["dim"])
            return out

        return await self._run(query)

    async def get_vector(self, chunk_id: str) -> np.ndarray | None:
        def query() -> np.ndarray | None:
            row = self._db().execute(
                "SELECT dim, vec FROM chunk_vectors WHERE chunk_id=?", (chunk_id,)
            ).fetchone()
            return _from_blob(row["vec"], row["dim"]) if row else None

        return await self._run(query)

    # ---------------- 答案与轨迹 ----------------

    async def save_answer(self, record: AnswerRecord) -> None:
        def write() -> None:
            conn = self._db()
            conn.execute(
                """INSERT INTO answers (answer_id, trace_id, question, answer, verdict, citations,
                       refusal, verification, evidence_snapshot, cost, config, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.answer_id,
                    record.trace_id,
                    record.question,
                    record.answer,
                    record.verdict,
                    _dump(record.citations),
                    _dump(record.refusal),
                    _dump(record.verification),
                    _dump(record.evidence_snapshot),
                    _dump(record.cost),
                    _dump(record.config),
                    record.created_at or _now(),
                ),
            )
            conn.commit()

        await self._run(write)

    async def get_answer(self, answer_id: str) -> AnswerRecord | None:
        def query() -> AnswerRecord | None:
            row = self._db().execute(
                "SELECT * FROM answers WHERE answer_id=?", (answer_id,)
            ).fetchone()
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
                created_at=row["created_at"],
            )

        return await self._run(query)

    async def save_trace(self, record: TraceRecord) -> None:
        def write() -> None:
            conn = self._db()
            conn.execute(
                """INSERT INTO traces (trace_id, answer_id, question, nodes, final_verdict,
                       evidence_snapshot, cost, created_at)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(trace_id) DO UPDATE SET
                       nodes=excluded.nodes, final_verdict=excluded.final_verdict,
                       evidence_snapshot=excluded.evidence_snapshot, cost=excluded.cost""",
                (
                    record.trace_id,
                    record.answer_id,
                    record.question,
                    _dump(record.nodes),
                    record.final_verdict,
                    _dump(record.evidence_snapshot),
                    _dump(record.cost),
                    record.created_at or _now(),
                ),
            )
            conn.commit()

        await self._run(write)

    async def get_trace(self, trace_id: str) -> TraceRecord | None:
        """trace 必须**自包含**（不依赖 answers 行仍存在），所以这里不 join answers。"""

        def query() -> TraceRecord | None:
            row = self._db().execute(
                "SELECT * FROM traces WHERE trace_id=?", (trace_id,)
            ).fetchone()
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
                created_at=row["created_at"],
            )

        return await self._run(query)

    # ---------------- 评测运行 ----------------

    async def save_eval_run(self, record: EvalRunRecord) -> None:
        def write() -> None:
            conn = self._db()
            conn.execute(
                """INSERT INTO eval_runs (run_id, status, suite, golden_version, config, metrics,
                       gates_passed, failed_gates, error, created_at, finished_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(run_id) DO UPDATE SET
                       status=excluded.status, metrics=excluded.metrics,
                       gates_passed=excluded.gates_passed, failed_gates=excluded.failed_gates,
                       error=excluded.error, finished_at=excluded.finished_at""",
                (
                    record.run_id,
                    record.status,
                    record.suite,
                    record.golden_version,
                    _dump(record.config),
                    _dump(record.metrics),
                    None if record.gates_passed is None else int(record.gates_passed),
                    _dump(record.failed_gates),
                    record.error,
                    record.created_at or _now(),
                    record.finished_at,
                ),
            )
            conn.commit()

        await self._run(write)

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

        def write() -> None:
            conn = self._db()
            sets, params = [], []
            for key, value in fields.items():
                sets.append(f"{key}=?")
                if key in ("metrics", "failed_gates"):
                    params.append(_dump(value))
                elif key == "gates_passed":
                    params.append(None if value is None else int(value))
                else:
                    params.append(value)
            params.append(run_id)
            conn.execute(f"UPDATE eval_runs SET {','.join(sets)} WHERE run_id=?", params)
            conn.commit()

        await self._run(write)

    async def get_eval_run(self, run_id: str) -> EvalRunRecord | None:
        def query() -> EvalRunRecord | None:
            row = self._db().execute(
                "SELECT * FROM eval_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if not row:
                return None
            return EvalRunRecord(
                run_id=row["run_id"],
                status=row["status"],
                suite=row["suite"],
                golden_version=row["golden_version"],
                config=json.loads(row["config"]),
                metrics=json.loads(row["metrics"]) if row["metrics"] else None,
                gates_passed=None
                if row["gates_passed"] is None
                else bool(row["gates_passed"]),
                failed_gates=json.loads(row["failed_gates"]),
                error=row["error"],
                created_at=row["created_at"],
                finished_at=row["finished_at"],
            )

        return await self._run(query)

    # ---------------- 进程内任务状态表 ----------------

    async def save_task(self, record: TaskRecord) -> None:
        def write() -> None:
            conn = self._db()
            conn.execute(
                """INSERT INTO tasks (task_id, kind, status, progress, checkpoint, owner, error,
                       created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(task_id) DO UPDATE SET
                       status=excluded.status, progress=excluded.progress,
                       checkpoint=excluded.checkpoint, owner=excluded.owner,
                       error=excluded.error, updated_at=excluded.updated_at""",
                (
                    record.task_id,
                    record.kind,
                    record.status,
                    _dump(record.progress),
                    _dump(record.checkpoint),
                    record.owner,
                    record.error,
                    record.created_at or _now(),
                    record.updated_at or _now(),
                ),
            )
            conn.commit()

        await self._run(write)

    async def update_task(self, task_id: str, **fields) -> None:
        allowed = {"status", "progress", "checkpoint", "owner", "error"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"不支持更新的字段：{sorted(unknown)}")

        def write() -> None:
            conn = self._db()
            sets, params = [], []
            for key, value in fields.items():
                sets.append(f"{key}=?")
                params.append(_dump(value) if isinstance(value, (dict, list)) else value)
            sets.append("updated_at=?")
            params.append(_now())
            params.append(task_id)
            conn.execute(f"UPDATE tasks SET {','.join(sets)} WHERE task_id=?", params)
            conn.commit()

        await self._run(write)

    async def get_task(self, task_id: str) -> TaskRecord | None:
        def query() -> TaskRecord | None:
            row = self._db().execute(
                "SELECT * FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
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
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

        return await self._run(query)


def _dump(value) -> str | None:
    """JSON 文本（中文不转义，便于直接看库里的快照）。"""
    return None if value is None else json.dumps(value, ensure_ascii=False)


# ---------------- 行 → 记录 ----------------


def _version_of(row: sqlite3.Row) -> IndexVersionRecord:
    return IndexVersionRecord(
        version=row["version"],
        state=row["state"],
        chunk_params=json.loads(row["chunk_params"]),
        embedding_model=row["embedding_model"],
        prefix_scheme=row["prefix_scheme"],
        created_at=row["created_at"],
        activated_at=row["activated_at"],
    )


def _document_of(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        doc_id=row["doc_id"],
        doc_name=row["doc_name"],
        content_hash=row["content_hash"],
        index_version=row["index_version"],
        tags=json.loads(row["tags"]),
        raw_path=row["raw_path"],
        size_bytes=row["size_bytes"],
        page_count=row["page_count"],
        created_at=row["created_at"],
        deleted_at=row["deleted_at"],
    )


def _chunk_of(row: sqlite3.Row) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        doc_name=row["doc_name"] if "doc_name" in row.keys() else "",
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
