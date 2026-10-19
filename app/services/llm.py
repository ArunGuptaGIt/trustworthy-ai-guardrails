import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from openai import AsyncOpenAI, RateLimitError
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.exceptions import LLMProviderError
from app.schemas.booking import BookingExtract

logger = logging.getLogger(__name__)


class BaseLLMService(ABC):
    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate text completion from prompt and context."""
        pass

    @abstractmethod
    async def rewrite_query(self, history: list[dict[str, str]], query: str) -> str:
        """Rewrite follow-up query into a standalone search query."""
        pass

    @abstractmethod
    async def classify_intent(self, message: str, history: list[dict[str, str]]) -> str:
        """Classify incoming user message into: 'question', 'booking', or 'booking_followup'."""
        pass

    @abstractmethod
    async def extract_booking_info(self, message: str, current_state: dict[str, Any]) -> BookingExtract:
        """Extract booking fields (name, email, date, time, cancel, confirm) using structured LLM call."""
        pass


class OpenAILLMService(BaseLLMService):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url or settings.OPENAI_BASE_URL
        self.model = model or settings.OPENAI_MODEL
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)

    @retry(
        retry=retry_if_not_exception_type(RateLimitError),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )
    async def generate_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.2,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except RateLimitError:
            logger.error("Rate limit exceeded, skipping retry")
            raise
        except Exception as e:
            logger.error(f"OpenAI chat completion error: {e}")
            raise LLMProviderError(f"LLM completion error: {str(e)}") from e

    async def rewrite_query(self, history: list[dict[str, str]], query: str) -> str:
        if not history:
            return query

        system_prompt = (
            "Given a chat history and the latest user question which might reference context in the history, "
            "formulate a standalone question which can be understood without the conversation history. "
            "Do NOT answer the question, just reformulate it if needed, or return it verbatim if already clear. "
            "Return ONLY the rewritten question text without any explanations, prefixes, or quotes."
        )

        formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])
        prompt = f"Chat History:\n{formatted_history}\n\nLatest Question: {query}\n\nStandalone Question:"

        try:
            standalone = await self.generate_response(prompt, system_prompt=system_prompt)
            if standalone:
                standalone = standalone.strip(" \"'\t\n")
                for prefix in ["Standalone Question:", "Standalone question:", "Question:", "Rewritten Question:"]:
                    if standalone.startswith(prefix):
                        standalone = standalone[len(prefix) :].strip(" \"'\t\n")
            return standalone if standalone else query
        except Exception as e:
            logger.warning(f"Failed to rewrite query, using original: {e}")
            return query

    async def classify_intent(self, message: str, history: list[dict[str, str]]) -> str:
        system_prompt = (
            "- 'booking_followup': User is answering a prompt related to an ongoing booking "
            "(e.g. providing name, email, date, time, confirming, or cancelling).\n"
            "- 'question': User is asking a general question or searching for information from documents.\n"
            'Return valid JSON strictly matching schema: {"intent": "question" | "booking" | "booking_followup"}'
        )

        formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-2:]])
        user_prompt = f"History:\n{formatted_history}\nUser message: {message}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            raw_content = response.choices[0].message.content or "{}"
            data = json.loads(raw_content)
            intent = str(data.get("intent", "question")).lower()
            if intent in ("booking", "booking_followup", "question"):
                return intent
            return "question"
        except Exception as e:
            logger.warning(f"Intent classification failed, falling back to heuristic: {e}")
            lower_msg = message.lower()
            if any(k in lower_msg for k in ["book", "interview", "schedule", "appointment"]):
                return "booking"
            return "question"

    async def extract_booking_info(self, message: str, current_state: dict[str, Any]) -> BookingExtract:
        system_prompt = (
            "You are an assistant extracting interview booking details from candidate messages.\n"
            "Current saved state:\n"
            f"{json.dumps(current_state)}\n\n"
            "Extract any updated values from the user message:\n"
            "- name: full name of person booking\n"
            "- email: candidate email address\n"
            "- date: booking date formatted strictly as YYYY-MM-DD\n"
            "- time: booking time formatted strictly as HH:MM (24h format)\n"
            "- cancel: true if user says cancel, stop, start over, clear, or restart\n"
            "- confirm: true if user explicitly confirms (e.g. yes, confirm, proceed)\n\n"
            "Return ONLY valid JSON matching exact schema:\n"
            '{"name": string|null, "email": string|null, "date": string|null, '
            '"time": string|null, "cancel": boolean, "confirm": boolean}'
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            raw_content = response.choices[0].message.content or "{}"
            data = json.loads(raw_content)
            # Fix: use strict boolean check - do not use bool() on raw values
            # bool("false") == True in Python, so we explicitly check type
            raw_cancel = data.get("cancel", False)
            raw_confirm = data.get("confirm", False)
            cancel = raw_cancel is True or raw_cancel == 1
            confirm = raw_confirm is True or raw_confirm == 1
            return BookingExtract(
                name=data.get("name"),
                email=data.get("email"),
                date=data.get("date"),
                time=data.get("time"),
                cancel=cancel,
                confirm=confirm,
            )
        except Exception as e:
            logger.error(f"Failed to extract booking info via LLM: {e}")
            raise LLMProviderError(f"Booking extraction failed: {str(e)}") from e


class MockLLMService(BaseLLMService):
    """Mock LLM implementation for tests and offline development."""

    async def generate_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        # If the system prompt asks for JSON (RAG guardrail format), return valid JSON.
        # Parse chunk refs from the prompt to provide faithful citations.
        if system_prompt and '"answerable"' in (system_prompt + prompt):
            from app.guardrails.spotlight import parse_chunk_refs

            refs = parse_chunk_refs(prompt)
            if refs:
                doc_id, chunk_idx = refs[0]
                citations_json = f'[{{"document_id": "{doc_id}", "chunk_index": {chunk_idx}}}]'
            else:
                citations_json = "[]"
                return (
                    '{"answer": "The uploaded documents do not contain information to answer your question.", '
                    '"answerable": false, "citations": [], "confidence": 0.0}'
                )
            return (
                f'{{"answer": "This is a mock LLM answer generated from retrieved context.", '
                f'"answerable": true, "citations": {citations_json}, "confidence": 0.9}}'
            )
        return "This is a mock LLM answer generated from retrieved context."

    async def rewrite_query(self, history: list[dict[str, str]], query: str) -> str:
        return query

    async def classify_intent(self, message: str, history: list[dict[str, str]]) -> str:
        msg_lower = message.lower()
        if "book" in msg_lower or "interview" in msg_lower or "schedule" in msg_lower:
            return "booking"
        if any(k in msg_lower for k in ["cancel", "confirm", "yes", "@"]):
            return "booking_followup"
        return "question"

    async def extract_booking_info(self, message: str, current_state: dict[str, Any]) -> BookingExtract:
        msg_lower = message.lower()
        cancel = "cancel" in msg_lower or "stop" in msg_lower
        confirm = "yes" in msg_lower or "confirm" in msg_lower or "proceed" in msg_lower

        name = current_state.get("name")
        email = current_state.get("email")
        date = current_state.get("date")
        time = current_state.get("time")

        if "@" in message:
            for word in message.split():
                if "@" in word:
                    email = word.strip(".,!?:;")
                    break

        if "202" in message:
            import re

            m_date = re.search(r"\b202\d-\d{2}-\d{2}\b", message)
            if m_date:
                date = m_date.group(0)

        if ":" in message:
            import re

            m_time = re.search(r"\b\d{1,2}:\d{2}\b", message)
            if m_time:
                time = m_time.group(0)

        if not name and not cancel and not confirm and "@" not in message:
            cleaned = message.strip()
            for prefix in ["my name is", "i am", "name is", "this is", "it is", "im", "i'm"]:
                if cleaned.lower().startswith(prefix):
                    cleaned = cleaned[len(prefix) :].strip()
                    break
            cleaned = cleaned.strip(".,!?:;\"'")
            if cleaned and not any(k in cleaned.lower() for k in ["book", "interview", "schedule", "appointment"]):
                name = cleaned

        return BookingExtract(
            name=name,
            email=email,
            date=date,
            time=time,
            cancel=cancel,
            confirm=confirm,
        )


class FallbackLLMService(BaseLLMService):
    """LLM wrapper executing primary LLM service and falling back to secondary (vLLM) on failure."""

    def __init__(
        self,
        primary_service: BaseLLMService,
        fallback_service: BaseLLMService,
    ) -> None:
        self.primary = primary_service
        self.fallback = fallback_service

    async def generate_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        try:
            return await self.primary.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                history=history,
            )
        except Exception as e:
            logger.warning(f"Primary LLM failed ({e}). Falling back to vLLM...")
            try:
                return await self.fallback.generate_response(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history=history,
                )
            except Exception as fallback_err:
                logger.error(f"Fallback LLM also failed: {fallback_err}")
                raise LLMProviderError(
                    "All LLM providers are currently unavailable. Please try again later."
                ) from fallback_err

    async def rewrite_query(self, history: list[dict[str, str]], query: str) -> str:
        try:
            return await self.primary.rewrite_query(history, query)
        except Exception as e:
            logger.warning(f"Primary rewrite failed ({e}). Falling back to vLLM...")
            try:
                return await self.fallback.rewrite_query(history, query)
            except Exception:
                logger.warning("Fallback rewrite also failed, using original query.")
                return query

    async def classify_intent(self, message: str, history: list[dict[str, str]]) -> str:
        try:
            return await self.primary.classify_intent(message, history)
        except Exception as e:
            logger.warning(f"Primary intent classification failed ({e}). Falling back to vLLM...")
            try:
                return await self.fallback.classify_intent(message, history)
            except Exception:
                logger.warning("Fallback intent classification failed, using heuristic.")
                lower_msg = message.lower()
                if any(k in lower_msg for k in ["book", "interview", "schedule", "appointment"]):
                    return "booking"
                return "question"

    async def extract_booking_info(self, message: str, current_state: dict[str, Any]) -> BookingExtract:
        try:
            return await self.primary.extract_booking_info(message, current_state)
        except Exception as e:
            logger.warning(f"Primary booking extraction failed ({e}). Falling back to vLLM...")
            try:
                return await self.fallback.extract_booking_info(message, current_state)
            except Exception as fallback_err:
                logger.error(f"Fallback booking extraction also failed: {fallback_err}")
                raise LLMProviderError(
                    "All LLM providers are currently unavailable for booking extraction. Please try again later."
                ) from fallback_err
