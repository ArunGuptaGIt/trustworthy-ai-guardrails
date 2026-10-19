from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse


class AppException(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class DocumentNotFoundError(AppException):
    def __init__(self, document_id: str) -> None:
        super().__init__(
            message=f"Document with id '{document_id}' was not found",
            status_code=status.HTTP_404_NOT_FOUND,
        )


class InvalidFileTypeError(AppException):
    def __init__(self, message: str = "Invalid file type. Only PDF and TXT files are accepted.") -> None:
        super().__init__(message=message, status_code=status.HTTP_400_BAD_REQUEST)


class FileTooLargeError(AppException):
    def __init__(self, max_size_mb: float) -> None:
        super().__init__(
            message=f"File exceeds maximum allowed size of {max_size_mb:.1f} MB",
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class EmptyDocumentError(AppException):
    def __init__(self, message: str = "Document contains no extractable text") -> None:
        super().__init__(message=message, status_code=status.HTTP_400_BAD_REQUEST)


class BookingConflictError(AppException):
    def __init__(self, booking_date: str, booking_time: str) -> None:
        super().__init__(
            message=f"Slot {booking_date} at {booking_time} is already booked. Please choose another time.",
            status_code=status.HTTP_409_CONFLICT,
        )


class BookingValidationError(AppException):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, status_code=status.HTTP_400_BAD_REQUEST)


class LLMProviderError(AppException):
    def __init__(self, message: str = "LLM provider service failure") -> None:
        super().__init__(message=message, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


class VectorStoreError(AppException):
    def __init__(self, message: str = "Vector database operational error") -> None:
        super().__init__(message=message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


async def app_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, AppException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.message,
                    "type": exc.__class__.__name__,
                    "details": exc.details,
                }
            },
        )
    return await generic_exception_handler(request, exc)


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": "An unexpected internal error occurred",
                "type": "InternalServerError",
            }
        },
    )
