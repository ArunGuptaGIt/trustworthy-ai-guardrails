import pytest

from app.services.memory import MemoryService


@pytest.mark.asyncio
async def test_memory_add_and_get_turn(memory_service: MemoryService) -> None:
    session_id = "mem-test-session"
    await memory_service.add_turn(session_id, "Hello", "Hi there!")

    history = await memory_service.get_history(session_id)
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hello"}
    assert history[1] == {"role": "assistant", "content": "Hi there!"}


@pytest.mark.asyncio
async def test_memory_clear(memory_service: MemoryService) -> None:
    session_id = "mem-test-clear"
    await memory_service.add_turn(session_id, "Ping", "Pong")
    await memory_service.clear_history(session_id)

    history = await memory_service.get_history(session_id)
    assert len(history) == 0


def test_cap_history_by_token_budget() -> None:
    history = [
        {"role": "user", "content": "Message " + "A" * 400},
        {"role": "assistant", "content": "Response " + "B" * 400},
        {"role": "user", "content": "Message " + "C" * 400},
        {"role": "assistant", "content": "Response " + "D" * 400},
    ]

    # max_tokens=250 => max_chars=1000.
    # Most recent 2 messages (C & D) total ~800 chars, adding message B would exceed 1000 chars.
    capped = MemoryService.cap_history_by_token_budget(history, max_tokens=250)

    assert len(capped) == 2
    assert capped[0]["content"].startswith("Message C")
    assert capped[1]["content"].startswith("Response D")
