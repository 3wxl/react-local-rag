"""应用配置。

通过环境变量或 apps/api/.env 文件覆盖（.env 已被 gitignore，勿提交密钥）。
例如：DATABASE_URL=postgresql://user:pass@localhost:5432/rag
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "rag-api"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/rag"
    cors_origins: list[str] = ["http://localhost:5173"]


settings = Settings()
