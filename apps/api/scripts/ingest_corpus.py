#!/usr/bin/env python
"""摄取命令行：把目录下的文档灌进服务端索引（一条命令准备知识库）。

用法（在 apps/api 下）：
    uv run python scripts/ingest_corpus.py ../../packages/eval/corpus
    uv run python scripts/ingest_corpus.py ./docs --delete           # 先软删除同名文档再灌
    uv run python scripts/ingest_corpus.py ./docs --no-vectors       # 只建库不向量化（无模型环境）

为什么先有 CLI 再有 HTTP 上传：验证链路时不该依赖浏览器 ——
"能不能一条命令把语料灌进去"是判断服务端是否真的可用的第一道门槛。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.embedding.bge import get_embedder  # noqa: E402
from app.services.ingest_service import IngestService  # noqa: E402
from app.store import make_store  # noqa: E402


async def run(args: argparse.Namespace) -> int:
    store = make_store(settings.database_url, settings.data_dir)
    await store.init()
    # 无向量模式：显式关掉向量路（服务仍能建库，检索走纯 BM25 降级路径）
    embedder = get_embedder(settings.bge_model_dir)
    if args.no_vectors:
        from app.embedding.bge import BgeOnnxEmbedder

        embedder = BgeOnnxEmbedder(settings.bge_model_dir, enabled=False)
    service = IngestService(store, embedder, data_dir=settings.data_dir)

    directory = Path(args.path)
    if not directory.exists():
        print(f"[ingest] 目录不存在：{directory}", file=sys.stderr)
        return 1

    if args.delete:
        for doc in await store.list_documents():
            await service.delete_document(doc.doc_id)
            print(f"[ingest] 已软删除 {doc.doc_name}")

    results = await service.ingest_directory(directory)
    ok = [r for r in results if not isinstance(r, Exception)]
    failed = [r for r in results if isinstance(r, Exception)]
    for result in ok:
        flag = "（已存在，跳过）" if result.deduped else ""
        print(
            f"[ingest] OK {result.doc_name}  块={result.chunk_count}  "
            f"版本={result.index_version}  {result.latency_ms}ms {flag}"
        )
    for err in failed:
        print(f"[ingest] FAIL {err}", file=sys.stderr)

    chunks = await store.load_chunks((await store.get_active_version()).version)
    print(
        f"[ingest] 完成：文档 {len(await store.list_documents())} 篇 / 块 {len(chunks)} 个"
        f"（失败 {len(failed)} 篇）"
    )
    await store.close()
    return 0 if not failed else 1


def _use_utf8_console() -> None:
    """Windows 控制台默认 GBK：不兜底的话一个符号就会让整条摄取流程崩掉。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    _use_utf8_console()
    parser = argparse.ArgumentParser(description="把目录文档摄取进服务端索引")
    parser.add_argument("path", help="文档目录（.md/.txt/.docx/.pdf）")
    parser.add_argument("--delete", action="store_true", help="先软删除已有文档再摄取")
    parser.add_argument(
        "--no-vectors",
        action="store_true",
        help="跳过向量化（无模型环境；此时检索走纯 BM25 降级路径）",
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
