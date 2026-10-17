"""Ingestion Graph（M3）：用图编排替代队列框架，保留"异步 + 断点 + 失败隔离"机制。

形状：`scan` →（条件边循环）`process_doc` → … → `activate` → `selfcheck`。

两个刻意的取舍：
1. **顺序处理，不用 Send 扇出**：现状就是顺序的（瓶颈在 CPU 向量化），
   而 `done` 必须保持确定性顺序（sorted），tasks.checkpoint 才能与 M2 的格式继续兼容。
   扇出留作后续演进（同一 State schema 下可替换）。
2. **图不写终态**：`succeeded/failed` 由调用方（路由的 run() 或 ReindexService）写 ——
   图只维护 progress/checkpoint，这样"任务状态由谁负责"不会有两处真相。

断点续跑：`thread_id = task_id`；重跑时从 `tasks.checkpoint["done"]` 播种已完成清单，
LangGraph 自己的检查点同时保留（M4 的轨迹回放按 thread 查）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from .ingest_nodes import IngestDeps, build_ingest_nodes
from .state import IngestState

if TYPE_CHECKING:
    from ..core.config import Settings


def build_ingest_graph(deps: IngestDeps, checkpointer: BaseCheckpointSaver | None = None):
    """编译摄取图（进程内一次；一次任务一次 invoke）。"""
    nodes = build_ingest_nodes(deps)

    graph = StateGraph(IngestState)
    graph.add_node("scan", nodes["scan"])
    graph.add_node("process_doc", nodes["process_doc"])
    graph.add_node("activate", nodes["activate"])
    graph.add_node("selfcheck", nodes["selfcheck"])

    graph.add_edge(START, "scan")
    graph.add_conditional_edges(
        "scan",
        nodes["route_more"],
        {"process": "process_doc", "activate": "activate"},
    )
    # 逐文档循环：每处理完一篇回到条件边，清单耗尽才去 activate
    graph.add_conditional_edges(
        "process_doc",
        nodes["route_more"],
        {"process": "process_doc", "activate": "activate"},
    )
    graph.add_edge("activate", "selfcheck")
    graph.add_edge("selfcheck", END)

    return graph.compile(checkpointer=checkpointer)


def upload_seed(
    *, task_id: str, path: str, doc_name: str, acl: dict, done: list[str] | None = None
) -> dict:
    """单文档上传的初始 State（`POST /v1/documents` 用）。"""
    return {
        "task_id": task_id,
        "mode": "upload",
        "version": "",
        "ratio": 1.0,
        "upload_path": path,
        "upload_name": doc_name,
        "acl": acl,
        "done": list(done or []),
        "pending": [],
        "failed": [],
        "results": [],
        "stats": {},
        "trace_nodes": [],
    }


def reindex_seed(
    *, task_id: str, version: str, ratio: float, done: list[str] | None = None
) -> dict:
    """全量重建的初始 State（`POST /v1/index/rebuild` 用）。"""
    return {
        "task_id": task_id,
        "mode": "reindex",
        "version": version,
        "ratio": ratio,
        "acl": {},
        "done": list(done or []),
        "pending": [],
        "failed": [],
        "results": [],
        "stats": {},
        "trace_nodes": [],
    }


async def invoke_ingest(graph, seed: dict, *, thread_id: str, doc_count: int, settings: "Settings") -> dict:
    """跑一次摄取图。

    `recursion_limit` 按文档数算（一篇文档一个超级步 + scan/activate/selfcheck 与余量）——
    摄入是批量任务，用单请求的 `graph_max_steps` 会把大语料直接截断；步数上限只取作下限。
    """
    limit = doc_count + 8
    return await graph.ainvoke(
        seed,
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max(limit, settings.graph_max_steps),
        },
    )
