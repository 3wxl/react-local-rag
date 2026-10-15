-- Postgres + pgvector（生产路径）。单库起步：元数据与向量同库，事务一致、少一个运维对象。
-- 与 SQLite 版（sqlite_store.py 里的 SCHEMA）保持同样的表与字段，两种后端行为一致。

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS index_version (
  version TEXT PRIMARY KEY,                       -- fnv1a32 十六进制，与 TS CURRENT_INDEX_VERSION 同值
  state TEXT NOT NULL,                            -- building | ready | stale | failed
  chunk_params JSONB NOT NULL,
  embedding_model TEXT NOT NULL,
  prefix_scheme TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  activated_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY,
  doc_name TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  index_version TEXT NOT NULL REFERENCES index_version(version),
  tags JSONB NOT NULL DEFAULT '[]',
  raw_path TEXT,
  size_bytes INTEGER NOT NULL DEFAULT 0,
  page_count INTEGER NOT NULL DEFAULT 1,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  deleted_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  index_version TEXT NOT NULL REFERENCES index_version(version),
  ord INTEGER NOT NULL,
  content TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  heading_path JSONB NOT NULL DEFAULT '[]',
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
CREATE INDEX IF NOT EXISTS chunks_ver_hash ON chunks (index_version, content_hash);

-- 向量单独一张表：重建索引/换模型时只重写这张表，块与文档不动。
-- 维度按 bge-small-zh（512）写死：换维度等于换模型，本来就要重建。
CREATE TABLE IF NOT EXISTS chunk_vectors (
  chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  index_version TEXT NOT NULL,
  embedding vector(512) NOT NULL
);
-- HNSW 供"只取 topN"的规模路径用；评测与对齐口径走精确扫描（见 pg_store.cosine_all）
CREATE INDEX IF NOT EXISTS chunk_vectors_hnsw
  ON chunk_vectors USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS answers (
  answer_id TEXT PRIMARY KEY,
  trace_id TEXT NOT NULL,
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  verdict TEXT NOT NULL,
  citations JSONB NOT NULL,
  refusal JSONB,
  verification JSONB NOT NULL,
  evidence_snapshot JSONB NOT NULL,
  cost JSONB NOT NULL,
  config JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 冗余存储 evidence_snapshot：trace 必须自包含（answers 行被清理后仍要能回放）
CREATE TABLE IF NOT EXISTS traces (
  trace_id TEXT PRIMARY KEY,
  answer_id TEXT NOT NULL,
  question TEXT NOT NULL,
  nodes JSONB NOT NULL,
  final_verdict TEXT NOT NULL,
  evidence_snapshot JSONB NOT NULL,
  cost JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_runs (
  run_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  suite TEXT NOT NULL,
  golden_version TEXT NOT NULL,
  config JSONB NOT NULL,
  metrics JSONB,
  gates_passed BOOLEAN,
  failed_gates JSONB NOT NULL DEFAULT '[]',
  error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at TIMESTAMPTZ
);

-- 进程内任务状态表（不引 Celery/Redis）：断点续跑靠 checkpoint 字段
CREATE TABLE IF NOT EXISTS tasks (
  task_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,                             -- ingest | eval
  status TEXT NOT NULL,                           -- queued | running | succeeded | failed | interrupted
  progress JSONB NOT NULL DEFAULT '{}',
  checkpoint JSONB NOT NULL DEFAULT '{}',
  owner TEXT,
  error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
