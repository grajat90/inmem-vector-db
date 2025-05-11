from fastapi import APIRouter

from app.api.dependencies import get_libraries_dependency
from app.api.schemas.search import SearchRequest, SearchResponse
from app.api.services.library_service import LibraryService

router = APIRouter(tags=["search"])


@router.post("/libraries/{library_id}/search")
async def search_library(
    library_id: str,
    search_request: SearchRequest,
    libraries: get_libraries_dependency,
) -> SearchResponse:
    """
    Search for chunks in a library with enhanced options.

    This endpoint performs a semantic search on the chunks in a library,
    with options for different distance metrics, filtering, and response customization.
    """
    library_service = LibraryService(libraries, library_id)
    chunk_results, documents_info = await library_service.search_library(search_request)

    return SearchResponse(
        query=search_request.query,
        results=chunk_results,
        documents=documents_info,
        total_results=len(chunk_results),
    )
