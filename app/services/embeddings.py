import logging
from abc import ABC, abstractmethod

from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.exceptions import LLMProviderError

logger = logging.getLogger(__name__)


class BaseEmbeddingService(ABC):
    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings in batches."""
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        pass


class OpenAIEmbeddingService(BaseEmbeddingService):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url or settings.OPENAI_BASE_URL
        self.model = model or settings.OPENAI_EMBEDDING_MODEL
        self.batch_size = batch_size
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        # process texts in batches to respect provider payload limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = await self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        res = await self.embed_texts([text])
        if not res:
            raise LLMProviderError("Empty embedding returned for query")
        return res[0]

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            # sort vectors by index if present to preserve input order across providers
            items_with_idx = list(enumerate(response.data))
            items_with_idx.sort(
                key=lambda pair: pair[1].index if getattr(pair[1], "index", None) is not None else pair[0]
            )
            return [item.embedding for _, item in items_with_idx]
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise LLMProviderError(f"Embedding service request failed: {str(e)}") from e


class MockEmbeddingService(BaseEmbeddingService):
    """Fallback embedding provider for offline testing and development."""

    def __init__(self, vector_size: int = settings.QDRANT_VECTOR_SIZE) -> None:
        self.vector_size = vector_size

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            # generate deterministic mock vector based on text hash
            val = float(hash(text) % 1000) / 1000.0
            vec = [val] * self.vector_size
            embeddings.append(vec)
        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        res = await self.embed_texts([text])
        return res[0]
