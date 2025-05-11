from fastapi import status

from app.api.exceptions.base import AppException


class DocumentNotFoundException(AppException):
    def __init__(self, document_id: str):
        super().__init__(
            message=f"Document with ID {document_id} not found",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"document_id": document_id},
        )
