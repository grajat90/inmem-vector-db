from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.api.schemas.document import CreateDocumentRequest
from app.core.models.library import LibraryMetadata


class IndexerType(str, Enum):
    FLAT = "flat"
    HNSW = "hnsw"
    LSH = "lsh"


class CreateLibraryRequest(BaseModel):
    name: str = Field(description="The name of the library")
    metadata: Optional[LibraryMetadata] = Field(
        default=None, description="Metadata for the library"
    )
    indexer: Optional[IndexerType] = Field(
        default=IndexerType.FLAT, description="Indexer to use for the library"
    )
    documents: Optional[list[CreateDocumentRequest]] = Field(
        default=None, description="Documents to include in the library"
    )


class LibraryResponse(BaseModel):
    library_id: str = Field(description="The ID of the library")
    name: str = Field(description="The name of the library")
    document_count: int = Field(description="The number of documents in the library")
    chunk_count: int = Field(description="The number of chunks in the library")
    message: Optional[str] = Field(
        default=None, description="A message indicating the success of the operation"
    )


class LibraryDetailsResponse(LibraryResponse):
    metadata: dict[str, Any] = Field(description="The metadata of the library")
    documents: list[str] = Field(description="The IDs of the documents in the library")
    last_updated: str = Field(description="The last updated date of the library")


class DeleteLibraryResponse(BaseModel):
    library_id: str = Field(description="The ID of the library")
    chunk_count: int = Field(description="The number of chunks in the library")
    document_count: int = Field(description="The number of documents in the library")
    message: str = Field(description="A message indicating the success of the operation")
