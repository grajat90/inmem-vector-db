from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum

from models.library import Library
from indexers.indexer import DistanceMetric
from dependencies import get_libraries
from core.embedding import get_embedding

router = APIRouter(
    tags=["search"],
    responses={404: {"description": "Library not found"}},
)

# Search request and response schemas
class SearchRequest(BaseModel):
    query: str = Field(description="The search query text")
    k: int = Field(default=5, description="Number of results to return")
    distance_metric: Optional[str] = Field(
        default="euclidean", 
        description="Distance metric to use for similarity search. Options: euclidean, cosine, dot_product, manhattan"
    )
    include_metadata: bool = Field(
        default=True, 
        description="Whether to include metadata in the response"
    )
    include_embeddings: bool = Field(
        default=False, 
        description="Whether to include embeddings in the response"
    )
    filter_by_tags: Optional[List[str]] = Field(
        default=None, 
        description="Filter results by these tags (all tags must match)"
    )

class ChunkSearchResult(BaseModel):
    id: str = Field(description="Chunk ID")
    document_id: str = Field(description="Document ID the chunk belongs to")
    text: str = Field(description="Text content of the chunk")
    score: float = Field(description="Similarity score (lower is better for distance metrics)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Chunk metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Chunk embedding vector")

class DocumentInfo(BaseModel):
    id: str = Field(description="Document ID")
    title: str = Field(description="Document title")
    description: str = Field(description="Document description")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")
    
class SearchResponse(BaseModel):
    query: str = Field(description="The original query")
    results: List[ChunkSearchResult] = Field(description="Matching chunks")
    documents: Dict[str, DocumentInfo] = Field(description="Information about documents containing matches")
    total_results: int = Field(description="Total number of results")

@router.post("/libraries/{library_id}/search", response_model=SearchResponse)
async def search_library(
    library_id: str,
    search_request: SearchRequest,
    libraries: Dict[str, Library] = Depends(get_libraries),
):
    """
    Search for chunks in a library with enhanced options.
    
    This endpoint performs a semantic search on the chunks in a library,
    with options for different distance metrics, filtering, and response customization.
    """
    if library_id not in libraries:
        raise HTTPException(status_code=404, detail="Library not found")
    
    library = libraries[library_id]
    
    # Get embedding for query
    query_embedding = await get_embedding(search_request.query)
    
    # Convert distance metric string to enum
    try:
        distance_metric = DistanceMetric(search_request.distance_metric)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid distance metric: {search_request.distance_metric}. Valid options are: {', '.join([m.value for m in DistanceMetric])}")
    
    # Search library with specified metric
    results = library.search(
        query_embedding, 
        k=search_request.k
    )
    
    # Apply tag filtering if requested
    if search_request.filter_by_tags:
        filtered_results = []
        for chunk in results:
            # Check if all requested tags are in the chunk's tags
            if all(tag in chunk.metadata.tags for tag in search_request.filter_by_tags):
                filtered_results.append(chunk)
        results = filtered_results[:search_request.k]  # Keep only up to k results
    
    # Calculate distances to get scores
    scores = []
    for chunk in results:
        score = library.indexer._calculate_distance(
            query_embedding, 
            chunk.embedding, 
            distance_metric
        )
        scores.append(score)
    
    # Collect unique document IDs
    document_ids = list(set(chunk.document_id for chunk in results))
    documents_info = {}
    
    # Get document information
    for doc_id in document_ids:
        if doc_id in library.documents:
            doc = library.documents[doc_id]
            documents_info[doc_id] = DocumentInfo(
                id=doc.id,
                title=doc.title,
                description=doc.description,
                metadata=doc.metadata.model_dump() if search_request.include_metadata else None
            )
    
    # Format chunk results
    chunk_results = []
    for i, (chunk, score) in enumerate(zip(results, scores)):
        result = ChunkSearchResult(
            id=chunk.id,
            document_id=chunk.document_id,
            text=chunk.text,
            score=float(score),  # Convert numpy float to Python float
            metadata=chunk.metadata.model_dump() if search_request.include_metadata else None,
            embedding=chunk.embedding.tolist() if search_request.include_embeddings else None
        )
        chunk_results.append(result)
    
    return SearchResponse(
        query=search_request.query,
        results=chunk_results,
        documents=documents_info,
        total_results=len(chunk_results)
    ) 