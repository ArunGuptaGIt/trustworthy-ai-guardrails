from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class BookingState(BaseModel):
    in_progress: bool = False
    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None
    awaiting_confirmation: bool = False


class BookingExtract(BaseModel):
    name: str | None = Field(default=None, description="Extracted candidate full name")
    email: str | None = Field(default=None, description="Extracted candidate email address")
    date: str | None = Field(default=None, description="Extracted date in YYYY-MM-DD format")
    time: str | None = Field(default=None, description="Extracted time in HH:MM format")
    cancel: bool = Field(default=False, description="True if the user requested to cancel or restart booking")
    confirm: bool = Field(default=False, description="True if user explicitly confirmed the proposed slot")


class BookingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    name: str
    email: str
    booking_date: str
    booking_time: str
    created_at: datetime


class BookingListResponse(BaseModel):
    items: list[BookingResponse]
    total: int
    limit: int
    offset: int
