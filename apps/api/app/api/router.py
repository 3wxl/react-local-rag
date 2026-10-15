"""API 路由汇总。

- `/api/*`：历史路径（保留 `/api/health`，monorepo 拆分时的健康检查）
- `/v1/*`：契约路径（docs/api-contract.md），前端服务器模式直连这里
"""

from fastapi import APIRouter

from .routes import answer, documents, eval as eval_routes, health, retrieve

api_router = APIRouter()


@api_router.get("/health")
async def legacy_health() -> dict[str, str]:
    """历史健康检查（`GET /api/health`）。新代码请用 `/v1/health`（含依赖就绪详情）。"""
    return {"status": "ok"}


v1_router = APIRouter()
v1_router.include_router(health.router, tags=["health"])
v1_router.include_router(retrieve.router, tags=["retrieve"])
v1_router.include_router(answer.router, tags=["answer"])
v1_router.include_router(documents.router, tags=["documents"])
v1_router.include_router(eval_routes.router, tags=["eval"])
