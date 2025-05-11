from fastapi import status

from app.api.exceptions.base import AppException


class BackgroundTasksNotFoundException(AppException):
    def __init__(self):
        super().__init__(
            message="Background tasks are required to add chunks",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"background_tasks": "Background tasks are required to add chunks"},
        )
