from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_NAME: str = "Document RAG Backend"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "app_db"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    DATABASE_URL: str = ""

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_URL: str = ""
    CHAT_MEMORY_TTL_SECONDS: int = 86400
    MAX_CHAT_TURNS: int = 10

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "documents"
    QDRANT_VECTOR_SIZE: int = 768

    OPENAI_API_KEY: str = "mock-key"
    OPENAI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    OPENAI_MODEL: str = "gemini-2.0-flash"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-004"

    VLLM_BASE_URL: str = "http://host.docker.internal:8002/v1"
    LOCAL_VLLM_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"

    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 100
    MAX_FILE_SIZE_BYTES: int = 10485760
    SIMILARITY_THRESHOLD: float = 0.3
    MAX_HISTORY_TOKENS: int = 2000

    # Guardrail settings
    GUARDRAILS_ENABLED: bool = True
    GUARDRAILS_MODE: str = "enforce"  # "enforce" or "monitor"
    GUARDRAILS_DEBUG: bool = False
    ALLOW_PII_FROM_CONTEXT: bool = False
    MAX_MESSAGE_CHARS: int = 2000
    # Threshold overrides (0 = use value from policy.toml)
    INJECTION_BLOCK_THRESHOLD: float = 0.0
    GROUNDEDNESS_MIN_CLAIM_SUPPORT: float = 0.0

    @model_validator(mode="after")
    def assemble_urls(self) -> "Settings":
        if not self.DATABASE_URL:
            self.DATABASE_URL = (
                f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
                f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            )
        if not self.REDIS_URL:
            auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
            self.REDIS_URL = f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"
        return self


settings = Settings()
