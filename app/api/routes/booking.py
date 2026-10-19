from fastapi import APIRouter, Depends

from app.api.deps import get_booking_repo
from app.repositories.booking_repo import BookingRepository
from app.schemas.booking import BookingListResponse, BookingResponse

router = APIRouter(prefix="/bookings", tags=["bookings"])


@router.get("", response_model=BookingListResponse)
async def list_bookings(
    limit: int = 10,
    offset: int = 0,
    booking_repo: BookingRepository = Depends(get_booking_repo),
) -> BookingListResponse:
    items, total = await booking_repo.list_paginated(limit=limit, offset=offset)
    return BookingListResponse(
        items=[BookingResponse.model_validate(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )
