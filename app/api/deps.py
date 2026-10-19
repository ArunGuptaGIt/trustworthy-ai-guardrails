import redis.asyncio as redis
from fastapi import Depends
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db_session
from app.guardrails.config import GuardrailConfig
from app.guardrails.input_pipeline import InputGuardrails
from app.guardrails.output_pipeline import OutputGuardrails
from app.repositories.booking_repo import BookingRepository
from app.repositories.document_repo import DocumentRepository
from app.services.booking import BookingService
from app.services.document_loader import DocumentLoaderService
from app.services.embeddings import (
    BaseEmbeddingService,
    MockEmbeddingService,
    OpenAIEmbeddingService,
)
from app.services.llm import BaseLLMService, FallbackLLMService, MockLLMService, OpenAILLMService
from app.services.memory import MemoryService
from app.services.rag import RAGPipelineService
from app.services.vector_store import VectorStoreService

# module-level singletons for async connection pools
_redis_pool: redis.Redis | None = None  # type: ignore[type-arg]
_qdrant_client: AsyncQdrantClient | None = None


def get_redis_client() -> redis.Redis:  # type: ignore[type-arg]
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_pool


def get_qdrant_client() -> AsyncQdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    return _qdrant_client


_MOCK_API_KEYS = {
    "mock-key",
    "",
    "your_openai_api_key_here",
    "your_api_key_here",
    "your_gemini_api_key",
}


def get_embedding_service() -> BaseEmbeddingService:
    if settings.OPENAI_API_KEY in _MOCK_API_KEYS:
        return MockEmbeddingService()
    return OpenAIEmbeddingService()


def get_llm_service() -> BaseLLMService:
    if settings.OPENAI_API_KEY in _MOCK_API_KEYS:
        return MockLLMService()

    primary = OpenAILLMService(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        model=settings.OPENAI_MODEL,
    )
    fallback = OpenAILLMService(
        api_key="EMPTY",
        base_url=settings.VLLM_BASE_URL,
        model=settings.LOCAL_VLLM_MODEL,
    )
    return FallbackLLMService(primary_service=primary, fallback_service=fallback)


def get_vector_store_service(
    client: AsyncQdrantClient = Depends(get_qdrant_client),
) -> VectorStoreService:
    return VectorStoreService(client=client)


def get_document_loader_service() -> DocumentLoaderService:
    return DocumentLoaderService()


def get_document_repo(session: AsyncSession = Depends(get_db_session)) -> DocumentRepository:
    return DocumentRepository(session)


def get_booking_repo(session: AsyncSession = Depends(get_db_session)) -> BookingRepository:
    return BookingRepository(session)


def get_memory_service(redis_client: redis.Redis = Depends(get_redis_client)) -> MemoryService:  # type: ignore[type-arg]
    return MemoryService(redis_client=redis_client)


def get_booking_service(
    redis_client: redis.Redis = Depends(get_redis_client),  # type: ignore[type-arg]
    booking_repo: BookingRepository = Depends(get_booking_repo),
    llm_service: BaseLLMService = Depends(get_llm_service),
) -> BookingService:
    return BookingService(
        redis_client=redis_client,
        booking_repo=booking_repo,
        llm_service=llm_service,
    )


def get_guardrail_config() -> GuardrailConfig:
    return GuardrailConfig.from_settings()


def get_input_guardrails(
    redis_client: redis.Redis = Depends(get_redis_client),  # type: ignore[type-arg]
    config: GuardrailConfig = Depends(get_guardrail_config),
) -> InputGuardrails:
    return InputGuardrails(config=config, redis=redis_client)


def get_output_guardrails(
    config: GuardrailConfig = Depends(get_guardrail_config),
) -> OutputGuardrails:
    return OutputGuardrails(config=config)


def get_rag_service(
    llm_service: BaseLLMService = Depends(get_llm_service),
    embedding_service: BaseEmbeddingService = Depends(get_embedding_service),
    vector_store_service: VectorStoreService = Depends(get_vector_store_service),
    memory_service: MemoryService = Depends(get_memory_service),
    booking_service: BookingService = Depends(get_booking_service),
    input_guardrails: InputGuardrails = Depends(get_input_guardrails),
    output_guardrails: OutputGuardrails = Depends(get_output_guardrails),
) -> RAGPipelineService:
    return RAGPipelineService(
        llm_service=llm_service,
        embedding_service=embedding_service,
        vector_store_service=vector_store_service,
        memory_service=memory_service,
        booking_service=booking_service,
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails,
    )
