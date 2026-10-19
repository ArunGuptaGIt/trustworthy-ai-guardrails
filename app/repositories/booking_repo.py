from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import BookingModel


class BookingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_date_and_time(self, booking_date: str, booking_time: str) -> BookingModel | None:
        stmt = select(BookingModel).where(
            BookingModel.booking_date == booking_date,
            BookingModel.booking_time == booking_time,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(
        self,
        session_id: str,
        name: str,
        email: str,
        booking_date: str,
        booking_time: str,
    ) -> BookingModel:
        db_booking = BookingModel(
            session_id=session_id,
            name=name,
            email=email,
            booking_date=booking_date,
            booking_time=booking_time,
        )
        self.session.add(db_booking)
        await self.session.flush()
        await self.session.refresh(db_booking)
        return db_booking

    async def list_paginated(self, limit: int = 10, offset: int = 0) -> tuple[list[BookingModel], int]:
        count_stmt = select(func.count()).select_from(BookingModel)
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        list_stmt = select(BookingModel).order_by(BookingModel.created_at.desc()).offset(offset).limit(limit)
        items_result = await self.session.execute(list_stmt)
        items = list(items_result.scalars().all())

        return items, total
