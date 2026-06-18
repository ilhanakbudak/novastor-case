"""Centralised, type-safe configuration."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # --- LLM ---
    llm_provider: str = "openai"            # openai | anthropic | ollama
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # --- Embeddings ---
    embedding_provider: str = "openai"      # openai | sentence-transformers
    embedding_model: str = "text-embedding-3-small"

    # --- Retrieval ---
    vector_store_dir: str = "./.vectorstore"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    # --- NovaStor data sources ---
    knowledge_base_dir: str = "./data/knowledge_base"
    customers_path: str = "./data/customers.json"
    shipments_path: str = "./data/shipments.json"

    # --- Ops ---
    log_level: str = "INFO"


settings = Settings()
