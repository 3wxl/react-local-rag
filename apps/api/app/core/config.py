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

    # ---------------- 图编排（M3） ----------------
    #: 检查点后端：sqlite（默认，落 checkpoint_db_path）| memory（测试 / 无盘环境）
    checkpoint_backend: str = "sqlite"
    #: 单请求图深度上限（同时作为 recursion_limit 兜底；友好原因由 budget 模块给出）
    graph_max_steps: int = 12
    #: 单请求 token 预算（预检 + 逐步记账，超限 422 budget_exceeded）
    request_token_budget: int = 4096
    #: 单请求成本上限（本地模型按 0 计；接云端通道时配合下面的价目）
    request_cost_limit_usd: float = 0.5
    #: 价目（每 1K token，美元）：默认 0 = 本地口径，estimated_cost_usd 随之保持 0
    cost_per_1k_prompt_usd: float = 0.0
    cost_per_1k_completion_usd: float = 0.0
    #: Agent 无进展检测：连续 N 次没有新增证据即提前收敛（多跳何时停）
    agent_no_progress_limit: int = 2
    #: Agent 工具检索的默认条数（工具参数缺省时用）
    graph_tool_top_k: int = 5
    #: MCP 工具服务（stdio 启动命令，空 = 禁用；CI 与默认部署都不起外部进程）
    mcp_server_command: str = ""
    #: MCP 工具 allowlist（空 = 全部放行的前提是显式配置了服务；见 app/mcp/client.py）
    mcp_allowlist: list[str] = []
    mcp_timeout_s: float = 20.0
    #: 每次 MCP 调用写审计日志（JSONL）：默认开；路径空 = DATA_DIR/mcp_audit.jsonl
    mcp_audit_enabled: bool = True
    mcp_audit_path: str = ""

    # ---------------- Skills（M5） ----------------
    #: 是否启用 Skills 注册表（内置技能在 app/skills/builtin，可追加外部目录）
    skills_enabled: bool = True
    #: 额外技能目录（每个子目录一个 SKILL.md）；内置目录总是被扫描
    extra_skills_dirs: list[Path] = []

    @property
    def checkpoint_db_path(self) -> Path:
        """检查点库：与业务库分离的 SQLite 文件（单副本部署，见 docs/ROADMAP-FULLSTACK.md §5）。"""
        return Path(self.data_dir) / "graph_checkpoints.sqlite3"

    @property
    def mcp_audit_log_path(self) -> Path:
        return Path(self.mcp_audit_path) if self.mcp_audit_path else Path(self.data_dir) / "mcp_audit.jsonl"

    @property
    def builtin_skills_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "skills" / "builtin"

    @property
    def skill_dirs(self) -> list[Path]:
        return [self.builtin_skills_dir, *self.extra_skills_dirs]

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
