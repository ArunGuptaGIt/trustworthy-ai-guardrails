import pytest
from fastapi.testclient import TestClient

from app.api.deps import (
    get_embedding_service,
    get_llm_service,
    get_qdrant_client,
    get_redis_client,
)
from app.db.session import get_db_session
from app.main import app
from app.services.embeddings import MockEmbeddingService
from app.services.llm import MockLLMService
from app.services.memory import MemoryService


class InMemoryAsyncRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, name: str) -> str | None:
        return self.store.get(name)

    async def set(self, name: str, value: str, ex: int | None = None) -> bool:
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


@pytest.fixture
def mock_redis() -> InMemoryAsyncRedis:
    return InMemoryAsyncRedis()


@pytest.fixture
def memory_service(mock_redis: InMemoryAsyncRedis) -> MemoryService:
    return MemoryService(redis_client=mock_redis)  # type: ignore[arg-type]


@pytest.fixture
def mock_llm_service() -> MockLLMService:
    return MockLLMService()


@pytest.fixture
def mock_embedding_service() -> MockEmbeddingService:
    return MockEmbeddingService()


@pytest.fixture
def client(mock_redis: InMemoryAsyncRedis) -> TestClient:
    from unittest.mock import AsyncMock

    mock_db = AsyncMock()
    mock_qdrant = AsyncMock()

    app.dependency_overrides[get_redis_client] = lambda: mock_redis
    app.dependency_overrides[get_embedding_service] = lambda: MockEmbeddingService()
    app.dependency_overrides[get_llm_service] = lambda: MockLLMService()
    app.dependency_overrides[get_db_session] = lambda: mock_db
    app.dependency_overrides[get_qdrant_client] = lambda: mock_qdrant

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
