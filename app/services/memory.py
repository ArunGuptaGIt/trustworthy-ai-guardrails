import json
import logging
from typing import Any

import redis.asyncio as redis

from app.core.config import settings

logger = logging.getLogger(__name__)


class MemoryService:
    def __init__(
        self,
        redis_client: redis.Redis,  # type: ignore[type-arg]
        ttl_seconds: int = settings.CHAT_MEMORY_TTL_SECONDS,
        max_turns: int = settings.MAX_CHAT_TURNS,
    ) -> None:
        self.redis: Any = redis_client
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns

    def _history_key(self, session_id: str) -> str:
        return f"chat:history:{session_id}"

    async def get_history(self, session_id: str) -> list[dict[str, str]]:
        key = self._history_key(session_id)
        try:
            raw_data = await self.redis.get(key)
            if not raw_data:
                return []
            history: list[dict[str, str]] = json.loads(raw_data)
            return history
        except Exception as e:
            logger.error(f"Failed to fetch conversation history from Redis: {e}")
            return []

    async def add_turn(self, session_id: str, user_message: str, assistant_response: str) -> None:
        key = self._history_key(session_id)
        try:
            history = await self.get_history(session_id)
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": assistant_response})

            # cap total stored turns (each turn consists of 1 user + 1 assistant message)
            max_messages = self.max_turns * 2
            if len(history) > max_messages:
                history = history[-max_messages:]

            await self.redis.set(key, json.dumps(history), ex=self.ttl_seconds)
            logger.info(f"Saved chat history turn to Redis [session_id='{session_id}', total_messages={len(history)}]")
        except Exception as e:
            logger.error(f"Failed to save conversation turn to Redis: {e}")

    async def clear_history(self, session_id: str) -> None:
        key = self._history_key(session_id)
        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.error(f"Failed to clear history from Redis: {e}")

    @staticmethod
    def cap_history_by_token_budget(
        history: list[dict[str, str]], max_tokens: int = settings.MAX_HISTORY_TOKENS
    ) -> list[dict[str, str]]:
        if not history:
            return []

        # approximate 1 token = 4 characters
        max_chars = max_tokens * 4
        capped: list[dict[str, str]] = []
        current_chars = 0

        # work backwards from most recent messages
        for msg in reversed(history):
            msg_len = len(msg.get("content", ""))
            if current_chars + msg_len > max_chars and capped:
                break
            capped.insert(0, msg)
            current_chars += msg_len

        return capped
