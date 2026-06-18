"""API request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    customer_id: str = Field(..., min_length=1, description="Authenticated customer id.")
    message: str = Field(..., min_length=1, description="The customer's message.")


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)


class IngestRequest(BaseModel):
    source: str | None = Field(default=None, description="Override knowledge-base path.")


class IngestResponse(BaseModel):
    chunks_indexed: int
    source: str


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    customers_loaded: int
