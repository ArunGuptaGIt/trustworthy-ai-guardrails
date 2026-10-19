from app.schemas.ingestion import ChunkingStrategy
from app.services.chunking.base import BaseChunker
from app.services.chunking.fixed_size import FixedSizeChunker
from app.services.chunking.recursive import RecursiveChunker


def get_chunker(strategy: ChunkingStrategy) -> BaseChunker:
    if strategy == ChunkingStrategy.FIXED:
        return FixedSizeChunker()
    elif strategy == ChunkingStrategy.RECURSIVE:
        return RecursiveChunker()
    raise ValueError(f"Unknown chunking strategy: {strategy}")


__all__ = ["BaseChunker", "FixedSizeChunker", "RecursiveChunker", "get_chunker"]
