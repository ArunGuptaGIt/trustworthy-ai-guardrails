from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentModel
from app.schemas.ingestion import ChunkingStrategy, DocumentCreate


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, document_id: str) -> DocumentModel | None:
        stmt = select(DocumentModel).where(DocumentModel.id == document_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_hash_and_strategy(self, file_hash: str, strategy: ChunkingStrategy) -> DocumentModel | None:
        stmt = select(DocumentModel).where(
            DocumentModel.file_hash == file_hash,
            DocumentModel.chunking_strategy == strategy.value,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(self, doc_in: DocumentCreate) -> DocumentModel:
        db_doc = DocumentModel(
            filename=doc_in.filename,
            file_type=doc_in.file_type,
            file_hash=doc_in.file_hash,
            chunking_strategy=doc_in.chunking_strategy.value,
            chunk_count=doc_in.chunk_count,
            file_size=doc_in.file_size,
        )
        self.session.add(db_doc)
        await self.session.flush()
        await self.session.refresh(db_doc)
        return db_doc

    async def list_paginated(self, limit: int = 10, offset: int = 0) -> tuple[list[DocumentModel], int]:
        count_stmt = select(func.count()).select_from(DocumentModel)
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        list_stmt = select(DocumentModel).order_by(DocumentModel.created_at.desc()).offset(offset).limit(limit)
        items_result = await self.session.execute(list_stmt)
        items = list(items_result.scalars().all())

        return items, total

    async def delete(self, document_id: str) -> bool:
        doc = await self.get_by_id(document_id)
        if not doc:
            return False
        await self.session.delete(doc)
        return True
