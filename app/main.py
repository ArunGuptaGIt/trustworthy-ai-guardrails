import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import get_qdrant_client, get_vector_store_service
from app.api.routes.booking import router as booking_router
from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.ingestion import router as ingestion_router
from app.core.config import settings
from app.core.exceptions import AppException, app_exception_handler, generic_exception_handler
from app.core.logging import setup_logging
from app.core.middleware import RequestIDMiddleware

setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Initializing application resources...")
    try:
        qdrant_client = get_qdrant_client()
        vector_store_service = get_vector_store_service(client=qdrant_client)
        await vector_store_service.ensure_collection_exists()
    except Exception as e:
        logger.warning(f"Qdrant collection startup initialization deferred: {e}")
    yield
    logger.info("Shutting down application resources...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-grade RAG backend with Document Ingestion and Multi-Turn Conversational Booking",
    version="0.1.0",
    lifespan=lifespan,
)

# middleware registration
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# global exception handlers
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# include API routers
app.include_router(health_router)
app.include_router(ingestion_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(booking_router, prefix="/api/v1")
