from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks

from app.api.dependencies import get_libraries_dependency
from app.api.schemas.chunk import (
    BasicChunkDetails,
    CreateChunkBatchResponse,
    CreateChunkRequest,
    CreateChunkResponse,
    DeleteChunkResponse,
)
from app.api.services.chunk_service import ChunkService

router = APIRouter(prefix="/libraries", tags=["chunks"])


@router.post("/{library_id}/chunks")
async def add_chunk(
    library_id: str,
    chunk_request: CreateChunkRequest,
    libraries: get_libraries_dependency,
) -> CreateChunkResponse:
    """Add a chunk to a library and document with metadata"""
    chunk_service = ChunkService(libraries, library_id, chunk_request.document_id)
    inserted_chunk, _ = await chunk_service.add_chunk(chunk_request)

    return CreateChunkResponse(
        chunk_id=inserted_chunk.id,
        document_id=inserted_chunk.document_id,
        library_id=library_id,
        message="Chunk added successfully",
    )


@router.post("/{library_id}/chunks/batch")
async def add_chunks_batch(
    library_id: str,
    chunk_requests: List[CreateChunkRequest],
    background_tasks: BackgroundTasks,
    libraries: get_libraries_dependency,
) -> CreateChunkBatchResponse:
    """Add multiple chunks to a library in a single batch operation"""
    chunk_service = ChunkService(libraries, library_id, background_tasks=background_tasks)
    chunks, library = await chunk_service.add_chunks_batch(chunk_requests)

    return CreateChunkBatchResponse(
        library_id=library.id,
        added_chunks=[chunk.id for chunk in chunks],
        count=len(chunks),
        message="Chunks added successfully",
    )


@router.get("/{library_id}/chunks")
async def list_chunks(
    library_id: str,
    libraries: get_libraries_dependency,
    document_id: Optional[str] = None,
) -> List[BasicChunkDetails]:
    """List chunks in a library, optionally filtered by document"""

    chunk_service = ChunkService(libraries, library_id, document_id)
    chunks = chunk_service.list_chunks()

    return [
        BasicChunkDetails(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            text=chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
        )
        for chunk in chunks
    ]


@router.get("/{library_id}/chunks/{chunk_id}")
async def get_chunk(
    library_id: str,
    chunk_id: str,
    libraries: get_libraries_dependency,
) -> BasicChunkDetails:
    """Get a specific chunk"""
    chunk_service = ChunkService(libraries, library_id)
    chunk = chunk_service.get_chunk(chunk_id)

    return BasicChunkDetails(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        text=chunk.text,
        metadata=chunk.metadata.model_dump(),
    )


@router.put("/{library_id}/chunks/{chunk_id}")
async def update_chunk(
    library_id: str,
    chunk_id: str,
    chunk_request: CreateChunkRequest,
    libraries: get_libraries_dependency,
) -> CreateChunkResponse:
    """Update a chunk with metadata"""
    chunk_service = ChunkService(libraries, library_id)
    updated_chunk = await chunk_service.update_chunk(chunk_id, chunk_request)

    return CreateChunkResponse(
        chunk_id=updated_chunk.id,
        document_id=updated_chunk.document_id,
        library_id=library_id,
        message="Chunk updated successfully",
    )


@router.delete("/{library_id}/chunks/{chunk_id}")
async def delete_chunk(
    library_id: str,
    chunk_id: str,
    libraries: get_libraries_dependency,
) -> DeleteChunkResponse:
    """Delete a chunk"""
    chunk_service = ChunkService(libraries, library_id)
    chunk_deleted = chunk_service.delete_chunk(chunk_id)

    return DeleteChunkResponse(
        library_id=library_id,
        chunk_id=chunk_deleted.id,
        message="Chunk deleted successfully",
    )
