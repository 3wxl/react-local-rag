"""FastAPI 应用入口。

本地启动（在 apps/api 目录下）：
    uv run uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings

app = FastAPI(title=settings.app_name, version="0.1.0")
app.include_router(api_router, prefix="/api")
