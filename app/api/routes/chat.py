import logging

from fastapi import APIRouter, Depends, status

from app.api.deps import get_rag_service
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag import RAGPipelineService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(
    request: ChatRequest,
    rag_service: RAGPipelineService = Depends(get_rag_service),
) -> ChatResponse:
    response = await rag_service.answer_chat(
        session_id=request.session_id,
        message=request.message,
        document_ids=request.document_ids,
    )
    return response
