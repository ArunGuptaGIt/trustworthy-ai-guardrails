from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.booking import BookingState


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, description="Unique conversation session identifier")
    message: str = Field(min_length=1, description="User query or message")
    document_ids: list[str] | None = Field(default=None, description="Optional document ID filter")


class SourceChunk(BaseModel):
    document_id: str
    chunk_index: int


class GuardrailsSummarySchema(BaseModel):
    """Compact guardrail status included in every ChatResponse."""

    status: str  # passed, masked, flagged, blocked, safe_response, redacted, regenerated, fallback
    flags: list[str] = Field(default_factory=list)
    # detail is only populated when GUARDRAILS_DEBUG=true
    detail: dict[str, Any] | None = None
    policy_version: str = ""


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    booking: BookingState | None = None
    # Optional guardrails summary - always present when guardrails are enabled
    guardrails: GuardrailsSummarySchema | None = None
