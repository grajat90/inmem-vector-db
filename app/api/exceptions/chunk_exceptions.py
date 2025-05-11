from fastapi import status

from app.api.exceptions.base import AppException


class ChunkNotFoundException(AppException):
    def __init__(self, chunk_id: str):
        super().__init__(
            f"Chunk with ID {chunk_id} not found",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"chunk_id": chunk_id},
        )
