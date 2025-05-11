from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api.exceptions.base import AppException


def register_exception_handlers(app: FastAPI) -> None:
    """Register exception handlers for the application"""

    @app.exception_handler(AppException)
    async def handle_app_exception(request: Request, exc: AppException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.message,
                    "code": exc.status_code,
                    "details": exc.details,
                    "path": request.url.path,
                }
            },
        )

    # Add handlers for validation errors, unexpected errors, etc.
    @app.exception_handler(Exception)
    async def handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        # Log the exception here
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "message": "An unexpected error occurred",
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "path": request.url.path,
                }
            },
        )
