"""Ingestion Graph（M3）：断点续跑、单文档失败隔离、轨迹行。

用**计数假 embedder** 判断"有没有真的重跑"（同 test_incremental_index 的口径）：
断点续跑不是看进度条数字，而是看"已完成的文档有没有被重新向量化"。

与 M2 的关系：机制（异步 + 断点 + 失败隔离）保留，编排换成图 ——
所以这里的断言刻意与 test_incremental_index 的 `tasks.checkpoint["done"]` 口径一致，
证明前端轮询与既有运维方式不受影响。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.core.config import settings as app_settings
from app.graphs.ingest_graph import build_ingest_graph, invoke_ingest, reindex_seed
from app.graphs.ingest_nodes import IngestDeps
from app.services.ingest_service import IngestService
from app.services.reindex_service import ReindexService
from app.store import SqliteStore, TaskRecord

from .test_incremental_index import CountingEmbedder


def _write_doc(path: Path, index: int) -> None:
    paragraphs = "\n\n".join(
        f"第 {j} 段：这是文档 {index} 的第 {j} 段内容，"
        + "用来把文档撑到多块的长度，确保切块器不会把整篇合成一块。" * 2
        for j in range(1, 21)
    )
    path.write_text(f"# 文档 {index}\n\n{paragraphs}\n", encoding="utf-8")


@pytest.fixture
async def env(tmp_path: Path):
    store = SqliteStore(tmp_path / "rag.sqlite3")
    await store.init()
    embedder = CountingEmbedder()
    ingest = IngestService(store, embedder, data_dir=tmp_path)
    # 带检查点编译：轨迹行（trace_nodes）才读得回来，也顺带覆盖检查点这条路径
    graph = build_ingest_graph(
        IngestDeps(store=store, ingest=ingest, settings=app_settings),
        InMemorySaver(),
    )
    reindex = ReindexService(store, ingest, graph=graph)
    for i in range(1, 5):
        _write_doc(tmp_path / f"doc{i}.md", i)
        await ingest.ingest_path(tmp_path / f"doc{i}.md")
    try:
        yield store, ingest, reindex, graph, embedder, tmp_path
    finally:
        await store.close()


async def _new_task(store: SqliteStore, *, done: list[str] | None = None) -> str:
    task_id = "tsk_" + str(int(time.time() * 1000) % 10**8)
    await store.save_task(
        TaskRecord(
            task_id=task_id,
            kind="reindex",
            status="interrupted" if done else "queued",
            checkpoint={"done": done or []},
        )
    )
    return task_id


async def test_rebuild_resumes_and_isolates_failure(env) -> None:
    """断点续跑 + 失败隔离：跳过的文档不重算，坏文档只进 failed[] 不中断整批。"""
    store, ingest, reindex, graph, embedder, tmp_path = env

    docs = await store.list_documents()
    done = sorted(d.doc_id for d in docs)[:2]
    # 制造一篇"原文已丢失"的文档（真实场景：挂载卷丢失 / 手工删过文件）。
    # 必须从**待处理**的文档里挑：挑了断点里已完成的那篇就等于跳过了它，失败隔离就测不到。
    broken = next(d for d in docs if d.doc_id not in done)
    Path(broken.raw_path or "").unlink()

    task_id = await _new_task(store, done=done)
    before = embedder.encoded_count
    await reindex.rebuild(task_id)

    task = await store.get_task(task_id)
    assert task is not None and task.status == "succeeded", task
    progress = task.progress
    # 4 篇都"处理过"（2 篇来自断点 + 1 篇成功 + 1 篇失败隔离）
    assert progress["done_docs"] == 4
    assert [f["doc_id"] for f in progress["failed"]] == [broken.doc_id]
    assert "原文不存在" in progress["failed"][0]["error"]
    # 断点里的两篇没有被重新向量化；失败那篇也没编出向量
    assert embedder.encoded_count == before, "已完成的文档不该被重新向量化"
    assert progress["embedded_chunks"] == 0
    assert progress["phase"] == "done"
    assert "latency_ms" in progress

    # 轨迹行：**处理过的**每篇一行 + 自检一行，seq 连续；失败行带 error 字段。
    # 断点里已完成的两篇不产生行（它们这次根本没被处理）—— 行数对上"本次实际做了什么"。
    rows = await _trace_rows(store, graph, task_id)
    assert [r["seq"] for r in rows] == list(range(1, len(rows) + 1))
    assert [r["name"] for r in rows[:-1]] == ["ingest_doc"] * (len(docs) - len(done))
    assert rows[-1]["name"] == "selfcheck"
    failed_rows = [r for r in rows if r["status"] == "failed"]
    assert len(failed_rows) == 1 and "原文不存在" in failed_rows[0]["error"]


async def _trace_rows(store, graph, task_id: str) -> list[dict]:
    """从图检查点里取这次任务的轨迹行（重跑一次检查点状态不会变，读回即可）。"""
    state = await graph.aget_state({"configurable": {"thread_id": task_id}})
    return list(state.values.get("trace_nodes") or [])


async def test_same_thread_rerun_does_not_reprocess(env) -> None:
    """同 thread 重跑（进程重启后的续跑路径）：不重复向量化，进度不倒退。"""
    store, ingest, reindex, graph, embedder, tmp_path = env

    task_id = await _new_task(store)
    docs = await store.list_documents()
    version = (await store.list_ready_versions())[0].version
    await invoke_ingest(
        graph,
        reindex_seed(task_id=task_id, version=version, ratio=1.0),
        thread_id=task_id,
        doc_count=len(docs),
        settings=app_settings,
    )
    after_first = embedder.encoded_count

    task = await store.get_task(task_id)
    done = list((task.checkpoint or {}).get("done", [])) if task else []
    assert len(done) == len(docs)

    # 模拟"进程被杀后重新拉起"：用同一 thread_id、以 checkpoint 为种子再跑一次
    await invoke_ingest(
        graph,
        reindex_seed(
            task_id=task_id,
            version=(await store.list_ready_versions())[0].version,
            ratio=1.0,
            done=done,
        ),
        thread_id=task_id,
        doc_count=len(docs),
        settings=app_settings,
    )
    assert embedder.encoded_count == after_first, "续跑必须跳过已完成的文档"


async def test_grayscale_ratio_keeps_old_version_serving(env) -> None:
    """灰度：ratio<1 时新版本 ready 但只接一部分流量，旧版本继续服务。"""
    from app.services.reindex_service import select_index_version

    store, ingest, reindex, graph, embedder, tmp_path = env
    task_id = await _new_task(store)

    await reindex.rebuild(task_id, ratio=0.2)

    versions = await store.list_ready_versions()
    newest = versions[0]
    assert newest.traffic_ratio == 0.2
    # 确定性分桶：同一 query 永远落同一版本（引用回跳不会指向另一份块）
    picks = {select_index_version(versions, f"qry_{i}").version for i in range(50)}
    assert picks == {v.version for v in versions}, picks
