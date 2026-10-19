import logging
from typing import Any

import redis.asyncio as redis
from fastapi import APIRouter, Depends, Response, status
from qdrant_client import AsyncQdrantClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_qdrant_client, get_redis_client
from app.db.session import get_db_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(
    response: Response,
    session: AsyncSession = Depends(get_db_session),
    redis_client: redis.Redis = Depends(get_redis_client),  # type: ignore[type-arg]
    qdrant_client: AsyncQdrantClient = Depends(get_qdrant_client),
) -> dict[str, Any]:
    checks: dict[str, str] = {}
    is_healthy = True

    # check postgresql
    try:
        await session.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        logger.error(f"Postgres healthcheck failed: {e}")
        checks["postgres"] = f"unhealthy: {str(e)}"
        is_healthy = False

    # check redis
    try:
        await redis_client.ping()
        checks["redis"] = "ok"
    except Exception as e:
        logger.error(f"Redis healthcheck failed: {e}")
        checks["redis"] = f"unhealthy: {str(e)}"
        is_healthy = False

    # check qdrant
    try:
        await qdrant_client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        logger.error(f"Qdrant healthcheck failed: {e}")
        checks["qdrant"] = f"unhealthy: {str(e)}"
        is_healthy = False

    if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ok" if is_healthy else "unhealthy",
        "checks": checks,
    }
