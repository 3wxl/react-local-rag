"""健康检查：把"依赖是否就绪"讲清楚，而不是只回一个 ok。

前端服务器模式靠它显示连通状态；运维靠它区分
"服务没起来"（连不上）与"服务起来了但没有索引/没有模型"（503 + 具体原因）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...parity.chunk_params import (
    CURRENT_INDEX_VERSION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PREFIX_SCHEME,
)
from ..deps import AppContext, get_ctx

router = APIRouter()


@router.get("/health")
async def health(ctx: AppContext = Depends(get_ctx)) -> dict:
    ctx_ = ctx
    active = await ctx_.store.get_active_version()
    documents = await ctx_.store.list_documents()
    chunks = (
        await ctx_.store.load_chunks(active.version) if active is not None else []
    )
    return {
        "status": "ok",
        "storage": {
            "backend": ctx_.store.backend,
            "documents": len(documents),
            "chunks": len(chunks),
        },
        "index": {
            "version": f"fnv1a32:{CURRENT_INDEX_VERSION}",
            "state": active.state if active is not None else "empty",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "prefix_scheme": EMBEDDING_PREFIX_SCHEME,
        },
        "models": {
            "embed": "ready" if ctx_.embedder.available else "missing",
            "llm": "ready" if ctx_.provider.available() else "unavailable",
            "llm_provider": ctx_.provider.name,
            "llm_model": ctx_.provider.model,
        },
    }
