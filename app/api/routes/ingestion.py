import logging

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.api.deps import (
    get_document_loader_service,
    get_document_repo,
    get_embedding_service,
    get_vector_store_service,
)
from app.core.config import settings
from app.core.exceptions import DocumentNotFoundError, InvalidFileTypeError
from app.repositories.document_repo import DocumentRepository
from app.schemas.ingestion import (
    ChunkingStrategy,
    DocumentCreate,
    DocumentListResponse,
    DocumentResponse,
)
from app.services.chunking import get_chunker
from app.services.document_loader import DocumentLoaderService
from app.services.embeddings import BaseEmbeddingService
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    chunking_strategy: ChunkingStrategy = Form(ChunkingStrategy.RECURSIVE),
    chunk_size: int = Form(settings.DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(settings.DEFAULT_CHUNK_OVERLAP),
    loader_service: DocumentLoaderService = Depends(get_document_loader_service),
    document_repo: DocumentRepository = Depends(get_document_repo),
    embedding_service: BaseEmbeddingService = Depends(get_embedding_service),
    vector_store_service: VectorStoreService = Depends(get_vector_store_service),
) -> DocumentResponse:
    filename = file.filename or "uploaded_document"
    content_bytes = await file.read()

    if not content_bytes:
        raise InvalidFileTypeError("Uploaded file is empty")

    processed_doc = loader_service.process_bytes(filename, content_bytes)

    # deduplication check: return existing metadata if hash, strategy, and vectors match
    existing_doc = await document_repo.get_by_hash_and_strategy(
        file_hash=processed_doc.file_hash,
        strategy=chunking_strategy,
    )
    if existing_doc:
        has_vectors = await vector_store_service.has_document_vectors(existing_doc.id)
        if has_vectors:
            logger.info(f"Duplicate document detected (hash={processed_doc.file_hash[:8]}). Returning existing record.")
            return DocumentResponse.model_validate(existing_doc)
        logger.warning(f"Existing metadata found for {existing_doc.id}, but vectors missing in Qdrant. Re-indexing...")

    chunker = get_chunker(chunking_strategy)
    chunks = chunker.chunk_text(
        text=processed_doc.content_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunk_texts = [c.text for c in chunks]
    embeddings = await embedding_service.embed_texts(chunk_texts)

    if existing_doc:
        target_doc = existing_doc
    else:
        doc_create = DocumentCreate(
            filename=processed_doc.filename,
            file_type=processed_doc.file_type,
            file_hash=processed_doc.file_hash,
            chunking_strategy=chunking_strategy,
            chunk_count=len(chunks),
            file_size=processed_doc.file_size,
        )
        target_doc = await document_repo.create(doc_create)

    # store vectors in qdrant with document payload
    await vector_store_service.upsert_chunks(
        document_id=target_doc.id,
        chunks=chunks,
        embeddings=embeddings,
    )

    return DocumentResponse.model_validate(target_doc)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    limit: int = 10,
    offset: int = 0,
    document_repo: DocumentRepository = Depends(get_document_repo),
) -> DocumentListResponse:
    items, total = await document_repo.list_paginated(limit=limit, offset=offset)
    return DocumentListResponse(
        items=[DocumentResponse.model_validate(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    document_repo: DocumentRepository = Depends(get_document_repo),
) -> DocumentResponse:
    doc = await document_repo.get_by_id(document_id)
    if not doc:
        raise DocumentNotFoundError(document_id)
    return DocumentResponse.model_validate(doc)


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
async def delete_document(
    document_id: str,
    document_repo: DocumentRepository = Depends(get_document_repo),
    vector_store_service: VectorStoreService = Depends(get_vector_store_service),
) -> dict[str, str]:
    deleted = await document_repo.delete(document_id)
    if not deleted:
        raise DocumentNotFoundError(document_id)

    # cleanup associated qdrant vectors
    await vector_store_service.delete_document_vectors(document_id)

    return {"status": "success", "message": f"Document '{document_id}' and vectors deleted"}
