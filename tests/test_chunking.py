import pytest

from app.schemas.ingestion import ChunkingStrategy
from app.services.chunking import FixedSizeChunker, RecursiveChunker, get_chunker


def test_chunker_factory() -> None:
    fixed_chunker = get_chunker(ChunkingStrategy.FIXED)
    assert isinstance(fixed_chunker, FixedSizeChunker)

    recursive_chunker = get_chunker(ChunkingStrategy.RECURSIVE)
    assert isinstance(recursive_chunker, RecursiveChunker)


def test_fixed_size_chunking() -> None:
    chunker = FixedSizeChunker()
    text = "abcdefghijklmnopqrstuvwxyz"
    # chunk_size=10, overlap=2 => step=8
    # 0..10 ("abcdefghij"), 8..18 ("ijklmnopqr"), 16..26 ("qrstuvwxyz")
    chunks = chunker.chunk_text(text, chunk_size=10, chunk_overlap=2)

    assert len(chunks) == 3
    assert chunks[0].index == 0
    assert chunks[0].text == "abcdefghij"
    assert chunks[1].text == "ijklmnopqr"
    assert chunks[2].text == "qrstuvwxyz"


def test_fixed_size_chunking_invalid_parameters() -> None:
    chunker = FixedSizeChunker()
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunker.chunk_text("test", chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="strictly smaller than chunk_size"):
        chunker.chunk_text("test", chunk_size=10, chunk_overlap=10)


def test_recursive_chunking_paragraphs() -> None:
    chunker = RecursiveChunker()
    text = "First paragraph content here.\n\nSecond paragraph content here.\n\nThird paragraph content here."
    chunks = chunker.chunk_text(text, chunk_size=40, chunk_overlap=10)

    assert len(chunks) >= 3
    assert "First paragraph" in chunks[0].text
    assert "Second paragraph" in chunks[1].text


def test_recursive_chunking_sentences() -> None:
    chunker = RecursiveChunker()
    text = "Sentence one is brief. Sentence two is also short. Sentence three concludes."
    chunks = chunker.chunk_text(text, chunk_size=30, chunk_overlap=5)

    assert len(chunks) >= 2
    assert "Sentence one" in chunks[0].text


def test_recursive_chunking_invalid_parameters() -> None:
    chunker = RecursiveChunker()
    with pytest.raises(ValueError):
        chunker.chunk_text("test", chunk_size=-5, chunk_overlap=0)

    with pytest.raises(ValueError):
        chunker.chunk_text("test", chunk_size=20, chunk_overlap=25)
