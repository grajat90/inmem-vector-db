from typing import Any, Optional

from pydantic import BaseModel, Field

from app.api.schemas.chunk import CreateChunkRequest
from app.core.models.document import DocumentMetadata


class CreateDocumentRequest(BaseModel):
    title: str = Field(description="The title of the document")
    description: str = Field(description="A description of the document")
    metadata: Optional[DocumentMetadata] = Field(
        default=None, description="Metadata for the document"
    )
    chunks: Optional[list[CreateChunkRequest]] = Field(
        default=None, description="Chunks to include in this document"
    )


class DocumentResponse(BaseModel):
    document_id: str = Field(description="The ID of the document")
    title: str = Field(description="The title of the document")
    chunk_count: int = Field(description="The number of chunks in the document")
    library_id: Optional[str] = Field(
        default=None, description="The ID of the library the document belongs to"
    )
    description: Optional[str] = Field(
        default=None, description="A description of the document"
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Metadata for the document"
    )
    message: Optional[str] = Field(
        default=None, description="A message indicating the success of the operation"
    )


class DeleteDocumentResponse(BaseModel):
    document_id: str = Field(description="The ID of the document")
    library_id: str = Field(description="The ID of the library the document belongs to")
    message: str = Field(description="A message indicating the success of the operation")
