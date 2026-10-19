from unittest.mock import AsyncMock

import pytest

from app.schemas.booking import BookingState
from app.services.booking import BookingService
from app.services.llm import MockLLMService


def test_email_validation() -> None:
    assert BookingService.is_valid_email("user@example.com") is True
    assert BookingService.is_valid_email("alice.smith+tech@domain.co.uk") is True
    assert BookingService.is_valid_email("invalid-email") is False
    assert BookingService.is_valid_email("@domain.com") is False


def test_date_validation() -> None:
    assert BookingService.is_valid_date("2029-12-31") is True
    assert BookingService.is_valid_date("2020-01-01") is False  # past date
    assert BookingService.is_valid_date("invalid-date") is False


def test_time_validation() -> None:
    assert BookingService.is_valid_time("14:30") is True
    assert BookingService.is_valid_time("09:00") is True
    assert BookingService.is_valid_time("25:00") is False  # invalid hour
    assert BookingService.is_valid_time("14:60") is False  # invalid minute
    assert BookingService.is_valid_time("2pm") is False


@pytest.mark.asyncio
async def test_booking_cancellation_flow(mock_redis: any) -> None:
    mock_repo = AsyncMock()
    llm_service = MockLLMService()
    booking_service = BookingService(
        redis_client=mock_redis,
        booking_repo=mock_repo,
        llm_service=llm_service,
    )

    session_id = "test-session-cancel"
    state = BookingState(name="John Doe", email="john@example.com")
    await booking_service.save_state(session_id, state)

    response_text, new_state = await booking_service.process_booking_turn(session_id, "Please cancel the booking")

    assert "cancelled" in response_text.lower()
    assert new_state is None

    cleared_state = await booking_service.get_state(session_id)
    assert cleared_state.name is None


@pytest.mark.asyncio
async def test_booking_slot_conflict(mock_redis: any) -> None:
    mock_repo = AsyncMock()
    # simulate existing booking on 2028-10-10 at 10:00
    mock_existing_booking = AsyncMock()
    mock_repo.get_by_date_and_time.return_value = mock_existing_booking

    llm_service = MockLLMService()
    booking_service = BookingService(
        redis_client=mock_redis,
        booking_repo=mock_repo,
        llm_service=llm_service,
    )

    session_id = "test-session-conflict"
    state = BookingState(
        name="John Doe",
        email="john@example.com",
        date="2028-10-10",
        time="10:00",
    )
    await booking_service.save_state(session_id, state)

    response_text, new_state = await booking_service.process_booking_turn(session_id, "Schedule on 2028-10-10 at 10:00")

    assert "already booked" in response_text.lower()
    assert new_state is not None
    assert new_state.time is None  # time reset to ask for another slot


@pytest.mark.asyncio
async def test_multi_turn_booking_in_progress(mock_redis: any) -> None:
    mock_repo = AsyncMock()
    mock_repo.get_by_date_and_time.return_value = None

    llm_service = MockLLMService()
    booking_service = BookingService(
        redis_client=mock_redis,
        booking_repo=mock_repo,
        llm_service=llm_service,
    )

    session_id = "test-multi-turn-progress"

    # Turn 1: User says book interview
    text1, state1 = await booking_service.process_booking_turn(session_id, "I want to book an interview")
    assert "full name" in text1.lower()
    assert state1 is not None
    assert state1.in_progress is True

    # Turn 2: User provides name
    text2, state2 = await booking_service.process_booking_turn(session_id, "Alice Smith")
    assert "email address" in text2.lower()
    assert state2 is not None
    assert state2.name == "Alice Smith"
    assert state2.in_progress is True
