from typing import Any, Optional

from pydantic import BaseModel, Field

from app.core.models.chunk import ChunkMetadata


class CreateChunkRequest(BaseModel):
    text: str = Field(description="The text content of the chunk")
    document_id: Optional[str] = Field(
        default=None, description="ID of the document this chunk belongs to"
    )
    metadata: Optional[ChunkMetadata] = Field(
        default=None, description="Metadata for the chunk"
    )


class CreateChunkResponse(BaseModel):
    chunk_id: str = Field(description="The ID of the chunk")
    document_id: str = Field(description="ID of the document this chunk belongs to")
    library_id: str = Field(description="ID of the library this chunk belongs to")
    message: str = Field(description="A message indicating the success of the operation")


class CreateChunkBatchResponse(BaseModel):
    library_id: str = Field(description="ID of the library this chunk belongs to")
    added_chunks: list[str] = Field(description="A list of chunk IDs that were added")
    count: int = Field(description="The number of chunks added")
    message: str = Field(description="A message indicating the success of the operation")


class BasicChunkDetails(BaseModel):
    chunk_id: str = Field(description="The ID of the chunk")
    document_id: str = Field(description="ID of the document this chunk belongs to")
    text: str = Field(description="The text content of the chunk")
    metadata: Optional[ChunkMetadata | dict[str, Any]] = Field(
        default=None, description="Metadata for the chunk"
    )


class DeleteChunkResponse(BaseModel):
    library_id: str = Field(description="ID of the library this chunk belongs to")
    chunk_id: str = Field(description="ID of the chunk to delete")
    message: str = Field(description="A message indicating the success of the operation")
