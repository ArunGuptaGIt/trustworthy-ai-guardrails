from app.schemas.ingestion import TextChunk
from app.services.chunking.base import BaseChunker


class RecursiveChunker(BaseChunker):
    def __init__(self, separators: list[str] | None = None) -> None:
        # separators listed in order of hierarchy preference
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
        if not text:
            return []

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and strictly smaller than chunk_size")

        raw_splits = self._split_text(text, self.separators, chunk_size)
        merged_chunks = self._merge_splits(raw_splits, chunk_size, chunk_overlap)

        return [TextChunk(index=i, text=chunk_str) for i, chunk_str in enumerate(merged_chunks)]

    def _split_text(self, text: str, separators: list[str], chunk_size: int) -> list[str]:
        if not separators:
            return [text]

        separator = separators[0]
        new_separators = separators[1:]

        if separator == "":
            # fallback character split
            return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        if separator in text:
            splits = text.split(separator)
        else:
            return self._split_text(text, new_separators, chunk_size)

        final_splits: list[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) <= chunk_size:
                final_splits.append(s)
            else:
                # piece is too large, recurse with next separator
                sub_splits = self._split_text(s, new_separators, chunk_size)
                final_splits.extend(sub_splits)

        return final_splits

    def _merge_splits(self, splits: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
        chunks: list[str] = []
        current_doc: list[str] = []
        current_len = 0

        for split in splits:
            split_len = len(split)

            if current_len + split_len + (1 if current_len > 0 else 0) > chunk_size:
                if current_doc:
                    chunk_text = " ".join(current_doc).strip()
                    if chunk_text:
                        chunks.append(chunk_text)

                    # pop items from current_doc to maintain overlap limit
                    while current_doc and current_len > chunk_overlap:
                        popped = current_doc.pop(0)
                        current_len -= len(popped) + 1

                current_doc.append(split)
                current_len += split_len + (1 if current_len > 1 else 0)
            else:
                current_doc.append(split)
                current_len += split_len + (1 if current_len > 1 else 0)

        if current_doc:
            final_chunk = " ".join(current_doc).strip()
            if final_chunk:
                chunks.append(final_chunk)

        return chunks
