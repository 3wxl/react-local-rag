"""应用配置。

通过环境变量或 apps/api/.env 覆盖（.env 已被 gitignore，勿提交密钥）。
默认值刻意指向**仓库内已有的资产**，让 `uv run uvicorn` 开箱即用：
- 模型权重复用浏览器端下载的 apps/web/public/models（同一份文件，不重复下载）
- 评测资产放在 packages/eval（与 apps/web、Harness 共用唯一真源）
- 存储默认本地 SQLite；配 DATABASE_URL=postgresql://… 即切 Postgres+pgvector
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# apps/api/app/core/config.py → core → app → apps/api → apps → <repo>
REPO_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "rag-api"

    # ---------------- 存储 ----------------
    #: 空 = 本地 SQLite（DATA_DIR 下）；postgresql://… = Postgres+pgvector
    database_url: str = ""
    data_dir: Path = REPO_ROOT / "apps" / "api" / ".data"

    # ---------------- 资产路径 ----------------
    eval_dir: Path = REPO_ROOT / "packages" / "eval"
    models_dir: Path = REPO_ROOT / "apps" / "web" / "public" / "models"

    # ---------------- 模型通道 ----------------
    #: onnx | ollama | fake（fake 供测试用：确定性输出，不下载权重）
    llm_provider: str = "onnx"
    ollama_model: str = "qwen2.5:0.5b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    #: 生成上限（与浏览器端 eval-answers 一致的 512；贪心解码，可复现）
    max_new_tokens: int = 512
    #: 嵌入后端：onnx | fixed（fixed 读 packages/eval/embeddings.json，仅供 Harness/CI）
    embed_backend: str = "onnx"

    # ---------------- 服务 ----------------
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    @property
    def bge_model_dir(self) -> Path:
        return self.models_dir / "Xenova" / "bge-small-zh-v1.5"

    @property
    def qwen_model_dir(self) -> Path:
        return self.models_dir / "onnx-community" / "Qwen2.5-0.5B-Instruct"

    @property
    def fixed_embeddings_path(self) -> Path:
        return self.eval_dir / "embeddings.json"


settings = Settings()
