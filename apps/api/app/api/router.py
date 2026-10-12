"""API 路由汇总。

后续在这里挂载 /chat、/ingest 等业务路由。
"""

from fastapi import APIRouter

api_router = APIRouter()


@api_router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
