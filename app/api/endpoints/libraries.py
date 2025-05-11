from typing import Annotated, Dict

from fastapi import APIRouter, BackgroundTasks, Depends

from app.api.dependencies import get_libraries_dependency, get_reindex_lock
from app.api.exceptions.library_exceptions import LibraryNotFoundException
from app.api.schemas.library import (
    CreateLibraryRequest,
    DeleteLibraryResponse,
    LibraryDetailsResponse,
    LibraryResponse,
)
from app.api.services.library_service import LibraryService

router = APIRouter(prefix="/libraries", tags=["libraries"])


@router.post("")
async def create_library(
    library_request: CreateLibraryRequest,
    background_tasks: BackgroundTasks,
    libraries: get_libraries_dependency,
) -> LibraryResponse:
    """
    Create a new library with optional documents and chunks.

    This endpoint allows creating a complete library structure in a single request,
    including nested documents and their chunks.
    """
    library_service = LibraryService(libraries, background_tasks=background_tasks)

    library = await library_service.add_library(library_request)

    return LibraryResponse(
        library_id=library.id,
        name=library.name,
        document_count=len(library.documents),
        chunk_count=len(library.chunks),
        message="Library created successfully with all documents and chunks",
    )


@router.get("")
async def list_libraries(
    libraries: get_libraries_dependency,
) -> list[LibraryResponse]:
    """List all libraries"""
    return [
        LibraryResponse(
            library_id=library.id,
            name=library.name,
            document_count=len(library.documents),
            chunk_count=len(library.chunks),
        )
        for library in libraries.values()
    ]


@router.get("/{library_id}")
async def get_library(
    library_id: str,
    libraries: get_libraries_dependency,
) -> LibraryDetailsResponse:
    """Get details for a specific library"""
    library_service = LibraryService(libraries, library_id=library_id)
    library = library_service.get_library()

    return LibraryDetailsResponse(
        library_id=library.id,
        name=library.name,
        document_count=len(library.documents),
        chunk_count=len(library.chunks),
        metadata=library.metadata.model_dump(),
        documents=[doc.id for doc in library.documents.values()],
        last_updated=library.indexer.last_updated.isoformat(),
    )


@router.put("/{library_id}")
async def update_library(
    library_id: str,
    library_request: CreateLibraryRequest,
    libraries: get_libraries_dependency,
) -> LibraryResponse:
    """Update library metadata"""
    if library_id not in libraries:
        raise LibraryNotFoundException(library_id)

    library_service = LibraryService(libraries, library_id=library_id)
    updated_library = library_service.update_library(library_request)
    return LibraryResponse(
        library_id=updated_library.id,
        name=updated_library.name,
        message="Library updated successfully",
        document_count=len(updated_library.documents),
        chunk_count=len(updated_library.chunks),
    )


@router.delete("/{library_id}")
async def delete_library(
    library_id: str,
    libraries: get_libraries_dependency,
    reindex_locks: Annotated[Dict[str, object], Depends(get_reindex_lock)],
) -> DeleteLibraryResponse:
    """Delete a library"""
    if library_id not in libraries:
        raise LibraryNotFoundException(library_id)

    library_service = LibraryService(
        libraries, library_id=library_id, reindex_locks=reindex_locks
    )
    deleted_library = library_service.delete_library()

    return DeleteLibraryResponse(
        library_id=deleted_library.id,
        chunk_count=len(deleted_library.chunks),
        document_count=len(deleted_library.documents),
        message="Library deleted successfully",
    )
