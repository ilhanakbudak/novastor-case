"""LLM factory.

Returns a LangChain chat model based on config. Shared by the RAG pipeline
(Day 2) and the agent (Day 3), so provider choice lives in exactly one place.

Backends (all imported lazily):
  * "openai"     -> cloud (needs OPENAI_API_KEY)        [default]
  * "anthropic"  -> cloud (needs ANTHROPIC_API_KEY)
  * "ollama"     -> local models, fully offline         [exam-day fallback]
"""
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from app.config import settings
from app.exceptions import ConfigError

logger = logging.getLogger(__name__)


def get_llm() -> BaseChatModel:
    """Build the chat model chosen by settings.llm_provider."""
    provider = settings.llm_provider.lower()

    try:
        if provider == "openai":
            if not settings.openai_api_key:
                raise ConfigError("llm_provider=openai but OPENAI_API_KEY is not set")
            from langchain_openai import ChatOpenAI

            logger.info("LLM: OpenAI '%s' (temp=%s)", settings.llm_model, settings.llm_temperature)
            return ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                api_key=settings.openai_api_key,
            )

        if provider == "anthropic":
            if not settings.anthropic_api_key:
                raise ConfigError("llm_provider=anthropic but ANTHROPIC_API_KEY is not set")
            from langchain_anthropic import ChatAnthropic

            logger.info("LLM: Anthropic '%s' (temp=%s)", settings.llm_model, settings.llm_temperature)
            return ChatAnthropic(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                api_key=settings.anthropic_api_key,
            )

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            logger.info("LLM: Ollama '%s' (temp=%s)", settings.llm_model, settings.llm_temperature)
            return ChatOllama(model=settings.llm_model, temperature=settings.llm_temperature)

        raise ConfigError(f"Unknown llm_provider: '{provider}'")

    except ConfigError:
        raise
    except ImportError as exc:
        raise ConfigError(
            f"LLM backend for '{provider}' is not installed",
            detail=str(exc),
        ) from exc
