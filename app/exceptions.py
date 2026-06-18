"""Custom exception hierarchy.

A small, typed exception tree lets you (a) catch app errors specifically
without swallowing real bugs like KeyError, and (b) map errors to HTTP status
codes cleanly in the API layer (Day 4). Each error carries an http_status so
the FastAPI handler can translate it with zero guesswork.
"""
from __future__ import annotations


class AppError(Exception):
    """Base class for all expected application errors."""

    http_status: int = 500

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class ConfigError(AppError):
    """Missing or invalid configuration (e.g. no API key when one is needed)."""
    http_status = 500


class IngestionError(AppError):
    """A document could not be loaded, parsed, or chunked."""
    http_status = 422


class EmbeddingError(AppError):
    """The embedding model/provider failed."""
    http_status = 502


class VectorStoreError(AppError):
    """The vector store could not be read from or written to."""
    http_status = 500


class RetrievalError(AppError):
    """Retrieval failed (no index, query error, etc.)."""
    http_status = 500


class AgentError(AppError):
    """The agent / workflow failed to produce a result."""
    http_status = 500


class ToolError(AppError):
    """A tool invoked by the agent failed."""
    http_status = 502


class GenerationError(AppError):
    """The LLM failed to produce an answer."""
    http_status = 502


class NotFoundError(AppError):
    """A requested account, shipment, or record does not exist (or isn't visible)."""
    http_status = 404


class DataSourceError(AppError):
    """A backing data file could not be read or parsed."""
    http_status = 500
