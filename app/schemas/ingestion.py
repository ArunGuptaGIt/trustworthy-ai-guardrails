from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"


class TextChunk(BaseModel):
    index: int
    text: str


class DocumentCreate(BaseModel):
    filename: str
    file_type: str
    file_hash: str
    chunking_strategy: ChunkingStrategy
    chunk_count: int
    file_size: int


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_type: str
    file_hash: str
    chunking_strategy: ChunkingStrategy
    chunk_count: int
    file_size: int
    created_at: datetime


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int
    limit: int
    offset: int = Field(ge=0)
