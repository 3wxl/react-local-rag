# 服务端镜像：Python 3.12 + uv（与 pyproject 的 requires-python 对齐）
#
# 模型权重**不打进镜像**（上百 MB，且会随量化方案变化）：运行时以只读卷挂到 /models，
# 与浏览器端共用 apps/web/public/models 下同一份文件。
FROM python:3.12-slim AS base

# uv：依赖解析与安装（锁文件进镜像，保证可复现）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /srv
ENV UV_PROJECT_ENVIRONMENT=/srv/.venv \
    PYTHONUNBUFFERED=1

COPY apps/api/pyproject.toml apps/api/uv.lock ./
# pg 额外依赖：生产走 Postgres+pgvector（默认 SQLite 路径不需要）
RUN uv sync --frozen --no-dev --extra pg

COPY apps/api/app ./app
COPY apps/api/scripts ./scripts

ENV PATH="/srv/.venv/bin:$PATH" \
    DATA_DIR=/data \
    MODELS_DIR=/models \
    EVAL_DIR=/eval

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
