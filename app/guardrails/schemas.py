"""
Output schemas for the RAG JSON response format (Phase 3.1).

The LLM is instructed to return ONLY:
{
  "answer": str,
  "answerable": bool,
  "citations": [{"document_id": str, "chunk_index": int}],
  "confidence": float  (0.0 to 1.0)
}

Strict Pydantic v2 validation is used: no type coercion, so the
string "false" is correctly rejected as a non-bool value.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class Citation(BaseModel):
    """A single citation pointing to a retrieved chunk."""

    model_config = {"strict": True}

    document_id: str = Field(min_length=1, max_length=256)
    chunk_index: int = Field(ge=0)


class RAGResponseSchema(BaseModel):
    """
    Strict schema for the LLM's RAG JSON response.

    No coercion: booleans must be actual JSON booleans (true/false),
    not strings.  Confidence must be a float in [0, 1].
    """

    model_config = {"strict": True}

    answer: str = Field(min_length=1, max_length=16000)
    answerable: bool
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("citations", mode="before")
    @classmethod
    def citations_list(cls, v: object) -> object:
        if v is None:
            return []
        return v

    @model_validator(mode="after")
    def semantic_checks(self) -> RAGResponseSchema:
        # answerable=true requires at least one citation
        if self.answerable and not self.citations:
            raise ValueError("answerable=true requires at least one citation")
        # answerable=false: answer should indicate uncertainty
        if not self.answerable:
            lower = self.answer.lower()
            uncertainty_phrases = [
                "do not contain",
                "don't contain",
                "cannot find",
                "not found",
                "no information",
                "unable to find",
                "not in the documents",
                "not available",
                "no relevant",
            ]
            if not any(p in lower for p in uncertainty_phrases):
                # Soft check: don't raise, just note. Some valid refusals use different words.
                pass
        return self


class BookingExtractStrict(BaseModel):
    """
    Strict booking extraction schema (fixes the bool("false") bug).

    Strict mode means JSON "false" maps to Python False (correct),
    and the string "false" raises a ValidationError.
    """

    model_config = {"strict": True}

    name: str | None = Field(default=None, max_length=200)
    email: str | None = Field(default=None)
    date: str | None = Field(default=None)
    time: str | None = Field(default=None)
    cancel: bool = Field(default=False)
    confirm: bool = Field(default=False)

    @field_validator("date", mode="before")
    @classmethod
    def validate_date_format(cls, v: object) -> object:
        if v is None:
            return None
        import re
        if not isinstance(v, str):
            return None
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            return None
        return v

    @field_validator("time", mode="before")
    @classmethod
    def validate_time_format(cls, v: object) -> object:
        if v is None:
            return None
        import re
        if not isinstance(v, str):
            return None
        if not re.match(r"^\d{1,2}:\d{2}$", v):
            return None
        return v
