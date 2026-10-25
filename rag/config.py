from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(ROOT_DIR / ".env", ROOT_DIR / ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    openrouter_api_key: SecretStr = Field(..., alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    openrouter_app_title: str = Field(default="b1-rag", alias="OPENROUTER_APP_TITLE")

    llm_model: str = Field(default="openai/gpt-4.1-mini", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_timeout_seconds: float = Field(default=60.0, alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")
    llm_structured_method: Literal["function_calling", "json_schema", "json_mode"] = Field(
        default="function_calling",
        alias="LLM_STRUCTURED_METHOD",
    )

    embedding_model: str = Field(default="openai/text-embedding-3-large", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=3072, alias="EMBEDDING_DIM")
    embedding_batch_size: int = Field(default=64, alias="EMBEDDING_BATCH_SIZE")

    rerank_model: str = Field(default="cohere/rerank-v3.5", alias="RERANK_MODEL")
    rerank_candidates: int = Field(default=30, ge=1, alias="RERANK_CANDIDATES")
    rerank_top_n: int = Field(default=5, ge=1, alias="RERANK_TOP_N")
    rerank_min_score: float = Field(default=0.1, ge=0.0, le=1.0, alias="RERANK_MIN_SCORE")
    rerank_timeout_seconds: float = Field(default=30.0, alias="RERANK_TIMEOUT_SECONDS")

    search_fusion: Literal["rrf", "dbsf"] = Field(default="rrf", alias="SEARCH_FUSION")
    search_dense_limit: int = Field(default=30, ge=1, alias="SEARCH_DENSE_LIMIT")
    search_sparse_limit: int = Field(default=30, ge=1, alias="SEARCH_SPARSE_LIMIT")
    bm25_language: str = Field(default="russian", alias="BM25_LANGUAGE")
    bm25_avg_len: float = Field(default=150.0, alias="BM25_AVG_LEN")

    chunk_size: int = Field(default=1000, ge=100, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, ge=0, alias="CHUNK_OVERLAP")

    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: SecretStr | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="chunks", alias="QDRANT_COLLECTION")
    qdrant_timeout_seconds: int = Field(default=30, alias="QDRANT_TIMEOUT_SECONDS")
    qdrant_upsert_batch_size: int = Field(default=64, alias="QDRANT_UPSERT_BATCH_SIZE")

    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    task_status_ttl_seconds: int = Field(default=7 * 24 * 3600, alias="TASK_STATUS_TTL_SECONDS")

    storage_dir: Path = Field(default=ROOT_DIR / "data" / "uploads", alias="STORAGE_DIR")
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, ge=1, alias="MAX_UPLOAD_BYTES")

    retry_attempts: int = Field(default=4, ge=1, alias="RETRY_ATTEMPTS")
    retry_max_wait_seconds: float = Field(default=10.0, alias="RETRY_MAX_WAIT_SECONDS")


def load_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
