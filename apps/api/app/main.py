"""FastAPI 应用入口。

本地启动（在 apps/api 目录下）：
    uv run uvicorn app.main:app --reload --port 8000

进程模型：**单进程**（`--workers 1`）。存储、BM25 缓存、进程内任务运行器都在进程内；
换 Postgres 后任务表已具备多进程认领的字段（tasks.owner），但 M1 不做。

依赖加载策略：模型**惰性加载**，启动时不下载、不加载权重 ——
没有模型的机器照样能跑检索、摄取（无向量）、Harness 与跨语言一致性测试，
只有 `/v1/answer` 的生成分支会返回 503 model_unavailable。
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.deps import AppContext
from .api.errors import ApiError, api_error_handler
from .api.router import api_router, v1_router
from .core.config import settings
from .embedding.bge import get_embedder
from .generation.factory import build_provider
from .graphs import open_checkpointer
from .graphs.ingest_graph import build_ingest_graph
from .graphs.ingest_nodes import IngestDeps
from .graphs.qa_graph import build_qa_graph
from .graphs.qa_nodes import QaDeps
from .mcp import McpStdioClient
from .services.agent_service import AgentService, mount_mcp_tools
from .services.answer_service import AnswerService
from .services.ingest_service import IngestService
from .services.reindex_service import ReindexService
from .services.retrieval_service import RetrievalService
from .store import make_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = make_store(settings.database_url, settings.data_dir)
    await store.init()
    embedder = get_embedder(settings.bge_model_dir)
    provider = build_provider(
        settings.llm_provider,
        onnx_dir=settings.qwen_model_dir,
        ollama_model=settings.ollama_model,
        ollama_base_url=settings.ollama_base_url,
    )
    retrieval = RetrievalService(store, embedder)
    ingest = IngestService(
        store,
        embedder,
        data_dir=settings.data_dir,
        # 摄取完成后 BM25 缓存必须失效，否则检索到的还是旧语料
        on_index_changed=lambda version=None: retrieval.invalidate(version),
    )
    # 检查点必须在这里打开（而不是模块导入时）：AsyncSqliteSaver 绑定当前事件循环
    async with open_checkpointer(settings) as checkpointer:
        # 三张图在进程内编译一次，依赖用闭包注入（State 只放 JSON 原生类型，见 app/graphs/state.py）
        qa_graph = build_qa_graph(
            QaDeps(
                store=store,
                retrieval=retrieval,
                embedder=embedder,
                provider=provider,
                settings=settings,
            ),
            checkpointer,
        )
        # 摄取图与重建服务共享同一个图实例（编译有成本，检查点后端也必须一致）
        ingest_graph = build_ingest_graph(
            IngestDeps(store=store, ingest=ingest, settings=settings), checkpointer
        )
        reindex = ReindexService(store, ingest, graph=ingest_graph)

        # MCP 工具（可选）：配置了启动命令才起子进程；默认关闭，CI 不依赖外部进程
        mcp_client = None
        mounted_tools: list = []
        if settings.mcp_server_command:
            mcp_client = McpStdioClient(
                settings.mcp_server_command,
                allowlist=settings.mcp_allowlist,
                timeout_s=settings.mcp_timeout_s,
            )
            try:
                await mcp_client.start()
                mounted_tools = await mount_mcp_tools(mcp_client)
            except Exception as err:  # noqa: BLE001 —— 外部工具挂了不该让服务起不来
                print(f"[mcp] 挂载失败，本轮以无 MCP 工具运行：{err}")
                await mcp_client.close()
                mcp_client = None
                mounted_tools = []

        agent_service = AgentService(
            store=store,
            retrieval=retrieval,
            embedder=embedder,
            provider=provider,
            settings=settings,
            checkpointer=checkpointer,
            mounted_tools=mounted_tools,
        )

        ctx = AppContext(
            settings=settings,
            store=store,
            embedder=embedder,
            provider=provider,
            retrieval=retrieval,
            answer=AnswerService(qa_graph, settings=settings),
            ingest=ingest,
            reindex=reindex,
            checkpointer=checkpointer,
            ingest_graph=ingest_graph,
            agent=agent_service,
        )
        app.state.ctx = ctx
        try:
            yield
        finally:
            for task in list(ctx.tasks.values()):
                task.cancel()
            if mcp_client is not None:
                await mcp_client.close()
            await store.close()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_exception_handler(ApiError, api_error_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
# 契约路径：前端服务器模式与评测都走这里
app.include_router(v1_router, prefix="/v1")
