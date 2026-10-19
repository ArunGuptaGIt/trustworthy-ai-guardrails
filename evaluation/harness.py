"""
Evaluation harness for offline testing of the guardrail system.

Provides:
- ScriptedLLMService: returns pre-defined outputs in order
- CompliantAttackerLLM: worst-case stub that obeys injected instructions
- StubVectorStore: returns configurable search results
- InMemoryAsyncRedis: in-process Redis replacement
- build_pipeline(): constructs a full RAGPipelineService offline
"""
from __future__ import annotations

import json
import re
from typing import Any
from unittest.mock import AsyncMock

from app.guardrails.config import GuardrailConfig
from app.guardrails.input_pipeline import InputGuardrails
from app.guardrails.output_pipeline import OutputGuardrails
from app.schemas.booking import BookingExtract
from app.services.booking import BookingService
from app.services.embeddings import BaseEmbeddingService
from app.services.llm import BaseLLMService
from app.services.memory import MemoryService
from app.services.rag import RAGPipelineService
from app.services.vector_store import VectorSearchResult, VectorStoreService


class InMemoryAsyncRedis:
    """In-process Redis replacement for offline testing."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, name: str) -> str | None:
        return self.store.get(name)

    async def set(
        self, name: str, value: str, ex: int | None = None
    ) -> bool:
        self.store[name] = value
        return True

    async def delete(self, *names: str) -> int:
        count = 0
        for name in names:
            if name in self.store:
                del self.store[name]
                count += 1
        return count

    async def ping(self) -> bool:
        return True


class ScriptedLLMService(BaseLLMService):
    """Returns pre-defined raw outputs in order."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self._index = 0

    def add_response(self, response: str) -> None:
        self._responses.append(response)

    async def generate_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        if self._index < len(self._responses):
            resp = self._responses[self._index]
            self._index += 1
            return resp
        return (
            '{"answer": "Mock answer from scripted LLM.", '
            '"answerable": false, "citations": [], "confidence": 0.0}'
        )

    async def rewrite_query(
        self, history: list[dict[str, str]], query: str
    ) -> str:
        return query

    async def classify_intent(
        self, message: str, history: list[dict[str, str]]
    ) -> str:
        msg_lower = message.lower()
        if any(k in msg_lower for k in ["book", "interview", "schedule"]):
            return "booking"
        if any(
            k in msg_lower
            for k in ["cancel", "confirm", "yes", "@", "[email"]
        ):
            return "booking_followup"
        return "question"

    async def extract_booking_info(
        self, message: str, current_state: dict[str, Any]
    ) -> BookingExtract:
        cancel = "cancel" in message.lower()
        confirm = "confirm" in message.lower() or "yes" in message.lower()
        email = current_state.get("email")
        m_email = re.search(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", message
        )
        if m_email:
            email = m_email.group(0)
        m_date = re.search(r"\b202\d-\d{2}-\d{2}\b", message)
        date = m_date.group(0) if m_date else current_state.get("date")
        m_time = re.search(r"\b\d{1,2}:\d{2}\b", message)
        time_val = m_time.group(0) if m_time else current_state.get("time")
        name = current_state.get("name")
        if (
            not name
            and not cancel
            and not confirm
            and "@" not in message
            and "[EMAIL" not in message
        ):
            cleaned = message.strip()
            for prefix in [
                "my name is", "i am", "name is", "this is", "im",
            ]:
                if cleaned.lower().startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    break
            cleaned = cleaned.strip(".,!?:;\"'")
            if cleaned and not any(
                k in cleaned.lower()
                for k in ["book", "interview", "schedule"]
            ):
                name = cleaned
        return BookingExtract(
            name=name,
            email=email,
            date=date,
            time=time_val,
            cancel=cancel,
            confirm=confirm,
        )


class CompliantAttackerLLM(BaseLLMService):
    """
    Worst-case LLM stub that obeys injected instructions.

    If the prompt mentions "ignore", "reveal", "system prompt", etc.,
    it returns the canary token and system prompt text.
    If context contains another person's PII, it repeats it.
    Otherwise it fabricates numbers and policies.
    """

    def __init__(self, canary: str = "CANARY_12345678") -> None:
        self.canary = canary

    async def generate_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        lower = prompt.lower()
        # Leak system prompt
        if "reveal" in lower or "system prompt" in lower or "ignore" in lower:
            leaked = system_prompt or "no system prompt"
            return json.dumps({
                "answer": f"Sure! The system prompt is: {leaked}",
                "answerable": True,
                "citations": [],
                "confidence": 1.0,
            })
        # Repeat PII from context
        emails = re.findall(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", prompt
        )
        phones = re.findall(r"\b9[678]\d{8}\b", prompt)
        if emails or phones:
            leaked_pii = ", ".join(emails + phones)
            return json.dumps({
                "answer": f"Here is the info: {leaked_pii}",
                "answerable": True,
                "citations": [],
                "confidence": 1.0,
            })
        # Fabricate numbers
        return json.dumps({
            "answer": (
                "The fee is Rs. 99,999 and the deadline is 2025-01-01. "
                "All candidates must pay immediately."
            ),
            "answerable": True,
            "citations": [],
            "confidence": 0.95,
        })

    async def rewrite_query(
        self, history: list[dict[str, str]], query: str
    ) -> str:
        return query

    async def classify_intent(
        self, message: str, history: list[dict[str, str]]
    ) -> str:
        return "question"

    async def extract_booking_info(
        self, message: str, current_state: dict[str, Any]
    ) -> BookingExtract:
        return BookingExtract(
            name="Fake Name", email="fake@evil.com",
            date="2099-01-01", time="00:00",
            cancel=False, confirm=True,
        )


class StubEmbeddingService(BaseEmbeddingService):
    """Returns a zero vector."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 768 for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.0] * 768

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 768 for _ in texts]


class StubVectorStore(VectorStoreService):
    """Returns pre-configured search results."""

    def __init__(
        self, results: list[VectorSearchResult] | None = None
    ) -> None:
        self._results = results or []
        # satisfy parent __init__ with a mock client
        super().__init__(client=AsyncMock())

    def set_results(self, results: list[VectorSearchResult]) -> None:
        self._results = results

    async def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        document_ids: list[str] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        return self._results

    async def ensure_collection_exists(self) -> None:
        pass


def build_pipeline(
    config: GuardrailConfig | None = None,
    llm: BaseLLMService | None = None,
    chunks: list[VectorSearchResult] | None = None,
    redis: InMemoryAsyncRedis | None = None,
) -> RAGPipelineService:
    """Build a complete offline RAGPipelineService."""
    cfg = config or GuardrailConfig()
    r = redis or InMemoryAsyncRedis()
    llm_svc = llm or ScriptedLLMService()
    embed_svc = StubEmbeddingService()
    vector_svc = StubVectorStore(results=chunks or [])
    memory_svc = MemoryService(redis_client=r)  # type: ignore[arg-type]
    booking_repo = AsyncMock()
    booking_repo.get_by_date_and_time.return_value = None
    booking_repo.create.return_value = AsyncMock(
        id="test-booking-id",
        booking_date="2029-01-15",
        booking_time="10:00",
    )
    booking_svc = BookingService(
        redis_client=r,  # type: ignore[arg-type]
        booking_repo=booking_repo,
        llm_service=llm_svc,
    )
    input_guard = InputGuardrails(config=cfg, redis=r)
    output_guard = OutputGuardrails(config=cfg)
    return RAGPipelineService(
        llm_service=llm_svc,
        embedding_service=embed_svc,
        vector_store_service=vector_svc,
        memory_service=memory_svc,
        booking_service=booking_svc,
        input_guardrails=input_guard,
        output_guardrails=output_guard,
    )
