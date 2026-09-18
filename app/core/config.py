from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "EvalForge"
    app_version: str = "0.1.0"
    app_env: str = "development"
    log_level: str = "INFO"

    llm_provider: str = "deterministic"
    llm_model: str = "deterministic-test"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_structured_output_mode: str = "json_schema"

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    chunk_size: int = 120
    chunk_overlap: int = 30
    retrieval_top_k: int = 5
    reranking_enabled: bool = False
    reranker_candidate_k: int = 10
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    langfuse_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None

    database_url: str = (
        "postgresql+psycopg://evalforge:evalforge@localhost:5432/evalforge"
    )

    hybrid_retrieval_enabled: bool = False
    lexical_candidate_k: int = 10

    chroma_host: str = "localhost"
    chroma_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
