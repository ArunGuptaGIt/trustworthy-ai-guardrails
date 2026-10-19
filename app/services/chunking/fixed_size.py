from app.schemas.ingestion import TextChunk
from app.services.chunking.base import BaseChunker


class FixedSizeChunker(BaseChunker):
    def chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
        if not text:
            return []

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and strictly smaller than chunk_size")

        step = chunk_size - chunk_overlap
        chunks: list[TextChunk] = []
        text_len = len(text)
        start = 0
        idx = 0

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_str = text[start:end].strip()
            if chunk_str:
                chunks.append(TextChunk(index=idx, text=chunk_str))
                idx += 1
            if end == text_len:
                break
            start += step

        return chunks
