from fastapi import status

from app.api.exceptions.base import AppException


class LibraryNotFoundException(AppException):
    """Exception raised when a library is not found"""

    def __init__(self, library_id: str):
        super().__init__(
            message=f"Library with ID '{library_id}' not found",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"library_id": library_id},
        )


class LibraryAlreadyExistsException(AppException):
    """
    Exception raised when trying to create a library
    with a name that already exists
    """

    def __init__(self, name: str):
        super().__init__(
            message=f"Library with name '{name}' already exists",
            status_code=status.HTTP_409_CONFLICT,
            details={"name": name},
        )


class IndexerNotFoundException(AppException):
    """Exception raised when an indexer type is not found"""

    def __init__(self, indexer_type: str):
        super().__init__(
            message=f"Indexer type '{indexer_type}' not found",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"indexer_type": indexer_type},
        )


class InvalidDistanceMetricException(AppException):
    """Exception raised when an invalid distance metric is used"""

    def __init__(self, distance_metric: str):
        super().__init__(
            message=f"Invalid distance metric: {distance_metric}",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"distance_metric": distance_metric},
        )
