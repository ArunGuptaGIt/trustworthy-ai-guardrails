import logging
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest_models

from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.schemas.ingestion import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    document_id: str
    chunk_index: int
    text: str
    score: float


class VectorStoreService:
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
        vector_size: int | None = None,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.vector_size = vector_size or settings.QDRANT_VECTOR_SIZE

        if client is not None:
            self.client = client
        else:
            self.client = AsyncQdrantClient(host=self.host, port=self.port)

    async def ensure_collection_exists(self) -> None:
        try:
            exists = await self.client.collection_exists(self.collection_name)
            if not exists:
                logger.info(f"Creating Qdrant collection '{self.collection_name}' with size {self.vector_size}")
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest_models.VectorParams(
                        size=self.vector_size,
                        distance=rest_models.Distance.COSINE,
                    ),
                )
            else:
                collection_info = await self.client.get_collection(self.collection_name)
                current_size = getattr(collection_info.config.params.vectors, "size", None)
                if current_size and current_size != self.vector_size:
                    logger.warning(
                        f"Qdrant collection vector size mismatch (existing: {current_size}, "
                        f"configured: {self.vector_size}). Recreating collection '{self.collection_name}'..."
                    )
                    await self.client.delete_collection(self.collection_name)
                    await self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=rest_models.VectorParams(
                            size=self.vector_size,
                            distance=rest_models.Distance.COSINE,
                        ),
                    )
        except Exception as e:
            logger.error(f"Failed to check/create Qdrant collection: {e}")
            raise VectorStoreError(f"Qdrant collection initialization failed: {str(e)}") from e

    async def upsert_chunks(
        self,
        document_id: str,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks or not embeddings:
            return

        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings lists must be of equal length")

        points: list[rest_models.PointStruct] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            point_id = str(uuid.uuid4())
            payload: dict[str, Any] = {
                "document_id": document_id,
                "chunk_index": chunk.index,
                "text": chunk.text,
            }
            points.append(
                rest_models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        try:
            await self.ensure_collection_exists()
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
        except Exception as e:
            logger.error(f"Failed to upsert points to Qdrant: {e}")
            raise VectorStoreError(f"Qdrant upsert failed: {str(e)}") from e

    async def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        document_ids: list[str] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        await self.ensure_collection_exists()

        query_filter: rest_models.Filter | None = None
        if document_ids:
            query_filter = rest_models.Filter(
                must=[
                    rest_models.FieldCondition(
                        key="document_id",
                        match=rest_models.MatchAny(any=document_ids),
                    )
                ]
            )

        try:
            results = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
            )

            searchResults: list[VectorSearchResult] = []
            for hit in results.points:
                payload = hit.payload or {}
                searchResults.append(
                    VectorSearchResult(
                        document_id=str(payload.get("document_id", "")),
                        chunk_index=int(payload.get("chunk_index", 0)),
                        text=str(payload.get("text", "")),
                        score=float(hit.score),
                    )
                )

            return searchResults
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e):
                logger.info(
                    f"Collection '{self.collection_name}' not found or empty during search, returning empty list."
                )
                return []
            logger.error(f"Failed to search Qdrant: {e}")
            raise VectorStoreError(f"Qdrant search failed: {str(e)}") from e

    async def delete_document_vectors(self, document_id: str) -> None:
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest_models.FilterSelector(
                    filter=rest_models.Filter(
                        must=[
                            rest_models.FieldCondition(
                                key="document_id",
                                match=rest_models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
        except Exception as e:
            logger.error(f"Failed to delete document vectors from Qdrant: {e}")
            raise VectorStoreError(f"Qdrant deletion failed: {str(e)}") from e

    async def has_document_vectors(self, document_id: str) -> bool:
        await self.ensure_collection_exists()
        try:
            res = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=rest_models.Filter(
                    must=[
                        rest_models.FieldCondition(
                            key="document_id",
                            match=rest_models.MatchValue(value=document_id),
                        )
                    ]
                ),
                limit=1,
            )
            return len(res[0]) > 0
        except Exception as e:
            logger.warning(f"Failed to check vectors for document '{document_id}': {e}")
            return False
