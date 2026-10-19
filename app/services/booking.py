import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as redis
from email_validator import EmailNotValidError, validate_email

from app.core.config import settings
from app.repositories.booking_repo import BookingRepository
from app.schemas.booking import BookingState
from app.services.llm import BaseLLMService

logger = logging.getLogger(__name__)


class BookingService:
    def __init__(
        self,
        redis_client: redis.Redis,  # type: ignore[type-arg]
        booking_repo: BookingRepository,
        llm_service: BaseLLMService,
    ) -> None:
        self.redis: Any = redis_client
        self.booking_repo = booking_repo
        self.llm_service = llm_service

    def _state_key(self, session_id: str) -> str:
        return f"booking:state:{session_id}"

    async def get_state(self, session_id: str) -> BookingState:
        key = self._state_key(session_id)
        try:
            raw = await self.redis.get(key)
            if not raw:
                return BookingState()
            data = json.loads(raw)
            return BookingState(**data)
        except Exception as e:
            logger.error(f"Failed to fetch booking state from Redis: {e}")
            return BookingState()

    async def save_state(self, session_id: str, state: BookingState) -> None:
        key = self._state_key(session_id)
        try:
            await self.redis.set(key, json.dumps(state.model_dump()), ex=settings.CHAT_MEMORY_TTL_SECONDS)
            logger.info(f"Updated booking state in Redis [session_id='{session_id}']")
        except Exception as e:
            logger.error(f"Failed to save booking state to Redis: {e}")

    async def clear_state(self, session_id: str) -> None:
        key = self._state_key(session_id)
        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.error(f"Failed to clear booking state from Redis: {e}")

    @staticmethod
    def is_valid_email(email: str) -> bool:
        try:
            validate_email(email, check_deliverability=False)
            return True
        except EmailNotValidError:
            return False

    @staticmethod
    def is_valid_date(date_str: str) -> bool:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            today = datetime.now(UTC).date()
            return target_date >= today
        except ValueError:
            return False

    @staticmethod
    def is_valid_time(time_str: str) -> bool:
        # validate HH:MM format
        if not re.match(r"^(?:[01]\d|2[0-3]):[0-5]\d$", time_str):
            return False
        return True

    async def process_booking_turn(self, session_id: str, message: str) -> tuple[str, BookingState | None]:
        state = await self.get_state(session_id)
        state.in_progress = True

        # extract structured fields from user message using LLM
        extract = await self.llm_service.extract_booking_info(message, state.model_dump())

        if extract.cancel:
            await self.clear_state(session_id)
            return (
                "Interview booking process has been cancelled. Let me know if you need help with anything else.",
                None,
            )

        # merge extracted fields into state
        if extract.name and extract.name.strip():
            state.name = extract.name.strip()

        if extract.email and extract.email.strip():
            email_candidate = extract.email.strip()
            if self.is_valid_email(email_candidate):
                state.email = email_candidate
            else:
                await self.save_state(session_id, state)
                return (
                    f"'{email_candidate}' is not a valid email address. Please provide a valid email.",
                    state,
                )

        if extract.date and extract.date.strip():
            date_candidate = extract.date.strip()
            if self.is_valid_date(date_candidate):
                state.date = date_candidate
            else:
                await self.save_state(session_id, state)
                today_str = datetime.now(UTC).strftime("%Y-%m-%d")
                return (
                    f"'{date_candidate}' is invalid or in the past. "
                    f"Please specify a date on or after {today_str} (format: YYYY-MM-DD).",
                    state,
                )

        if extract.time and extract.time.strip():
            time_candidate = extract.time.strip()
            if self.is_valid_time(time_candidate):
                state.time = time_candidate
            else:
                await self.save_state(session_id, state)
                return (
                    f"'{time_candidate}' is not a valid time. Please use 24h format HH:MM (e.g., 14:30).",
                    state,
                )

        # prompt for missing fields in priority order
        if not state.name:
            await self.save_state(session_id, state)
            return ("Sure! Let's schedule an interview. What is your full name?", state)

        if not state.email:
            await self.save_state(session_id, state)
            return (f"Thanks {state.name}. What is your email address?", state)

        if not state.date:
            await self.save_state(session_id, state)
            today_str = datetime.now(UTC).strftime("%Y-%m-%d")
            return (
                f"What date would you like to schedule? (Use YYYY-MM-DD format, e.g., {today_str})",
                state,
            )

        if not state.time:
            await self.save_state(session_id, state)
            return (
                "What time would you like to book? (Please use 24h format HH:MM, e.g., 10:00 or 14:30)",
                state,
            )

        # all 4 fields are collected and validated, now check slot availability in PostgreSQL
        existing = await self.booking_repo.get_by_date_and_time(state.date, state.time)
        if existing:
            conflict_date, conflict_time = state.date, state.time
            state.time = None  # reset time to let user select another slot
            await self.save_state(session_id, state)
            return (
                f"The slot on {conflict_date} at {conflict_time} is already booked. Please choose a different time.",
                state,
            )

        # prompt for explicit user confirmation before committing booking
        if not state.awaiting_confirmation:
            state.awaiting_confirmation = True
            await self.save_state(session_id, state)
            return (
                f"Please review your interview details:\n"
                f"- Name: {state.name}\n"
                f"- Email: {state.email}\n"
                f"- Date: {state.date}\n"
                f"- Time: {state.time}\n\n"
                f"Reply 'confirm' or 'yes' to finalize, or 'cancel' to restart.",
                state,
            )

        # check for confirmation response
        msg_lower = message.lower().strip()
        if extract.confirm or msg_lower in (
            "yes",
            "confirm",
            "proceed",
            "yup",
            "sure",
            "ok",
            "correct",
        ):
            db_booking = await self.booking_repo.create(
                session_id=session_id,
                name=state.name,
                email=state.email,
                booking_date=state.date,
                booking_time=state.time,
            )
            await self.clear_state(session_id)
            return (
                f"Your interview has been booked for {db_booking.booking_date} at {db_booking.booking_time}! "
                f"Booking ID: {db_booking.id}",
                None,
            )
        else:
            return (
                f"Please reply 'confirm' to finalize your booking for {state.date} at {state.time}, "
                "or 'cancel' to stop.",
                state,
            )
