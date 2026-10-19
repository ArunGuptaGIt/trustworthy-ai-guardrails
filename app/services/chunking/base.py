from abc import ABC, abstractmethod

from app.schemas.ingestion import TextChunk


class BaseChunker(ABC):
    @abstractmethod
    def chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
        """Split text into structured chunks adhering to chunk_size and chunk_overlap."""
        pass
