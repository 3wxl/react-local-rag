# apps/api — RAG 平台后端

FastAPI + LangGraph + pgvector。

## 环境准备

依赖用 [uv](https://docs.astral.sh/uv/) 管理（首次会创建 `apps/api/.venv`）：

```bash
cd apps/api
uv sync
```

## 启动开发服务器

```bash
uv run uvicorn app.main:app --reload --port 8000
```

健康检查：`GET http://localhost:8000/api/health` → `{"status": "ok"}`
接口文档：`http://localhost:8000/docs`

## 测试

```bash
uv run pytest
```

## 配置

数据库连接等通过环境变量或 `.env` 文件覆盖（见 `app/core/config.py`），默认：

- `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag`
- `CORS_ORIGINS=["http://localhost:5173"]`（vite dev server）

## 目录结构

```
app/
├── main.py        # FastAPI 入口
├── core/          # 配置、数据库连接
├── api/           # 路由（router.py 汇总，后续加 /chat、/ingest）
└── graphs/        # LangGraph 图定义
tests/             # pytest
```
