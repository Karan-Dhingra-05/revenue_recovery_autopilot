from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or a .env file.

    All values have sensible defaults for local development with Docker Compose.
    In production / CI, override via environment variables.
    """

    database_url: str = "postgresql://postgres:postgres@localhost:5432/revenue_recovery"
    redis_url: str = "redis://localhost:6379/0"

    # Razorpay Test Mode — populated in Phase 2
    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""
    razorpay_webhook_secret: str = ""

    # LLM — populated in Phase 4
    llm_api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Module-level singleton — import this everywhere.
settings = Settings()
