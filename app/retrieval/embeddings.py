"""Embeddings.

Returns a LangChain `Embeddings` object based on config, so the rest of the
app never hard-codes a provider. Two backends:

  * "openai"               -> cloud, high quality, needs an API key (default)
  * "sentence-transformers"-> local, runs offline, no key (exam-day fallback)

The backend libraries are imported *lazily* (inside the function) so this
module loads even when only one backend is installed.

RULE: index and query MUST use the same embedder. Different models produce
vectors in different spaces and silently break similarity search.
"""
from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings

from app.config import settings
from app.exceptions import ConfigError, EmbeddingError

logger = logging.getLogger(__name__)

# Sensible local default if the configured model is a cloud model name.
_DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedder() -> Embeddings:
    """Build the embedder chosen by settings.embedding_provider."""
    provider = settings.embedding_provider.lower()

    try:
        if provider == "openai":
            if not settings.openai_api_key:
                raise ConfigError(
                    "embedding_provider=openai but OPENAI_API_KEY is not set",
                    detail="Set the key in .env or switch to sentence-transformers.",
                )
            from langchain_openai import OpenAIEmbeddings

            logger.info("Embeddings: OpenAI '%s'", settings.embedding_model)
            return OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.openai_api_key,
            )

        if provider in {"sentence-transformers", "huggingface", "local"}:
            from langchain_huggingface import HuggingFaceEmbeddings

            # If the configured model looks like a cloud name, use the local default.
            model = settings.embedding_model if "/" in settings.embedding_model else _DEFAULT_LOCAL_MODEL
            logger.info("Embeddings: local HuggingFace '%s'", model)
            return HuggingFaceEmbeddings(model_name=model)

        raise ConfigError(f"Unknown embedding_provider: '{provider}'")

    except ConfigError:
        raise
    except ImportError as exc:
        raise EmbeddingError(
            f"Embedding backend for '{provider}' is not installed",
            detail=str(exc),
        ) from exc
