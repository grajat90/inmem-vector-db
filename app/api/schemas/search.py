from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from app.api.exceptions.library_exceptions import InvalidDistanceMetricException
from app.api.exceptions.validation_exceptions import InvalidHyperparameterException
from app.api.schemas.document import DocumentResponse
from app.core.indexers.indexer import DistanceMetric


class SearchRequest(BaseModel):
    query: str = Field(description="The search query text")
    k: int = Field(default=5, description="Number of results to return")
    distance_metric: Optional[str] = Field(
        default="euclidean",
        description="""
        Distance metric to use for similarity search.
        Options: euclidean, cosine, dot_product, manhattan
        """,
    )
    include_metadata: bool = Field(
        default=True, description="Whether to include metadata in the response"
    )
    include_embeddings: bool = Field(
        default=False, description="Whether to include embeddings in the response"
    )
    filter_by_tags: Optional[List[str]] = Field(
        default=None, description="Filter results by these tags (all tags must match)"
    )

    @field_validator("distance_metric")
    @classmethod
    def validate_distance_metric(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in [e.value for e in DistanceMetric]:
            raise InvalidDistanceMetricException(v)
        return v

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        if v <= 0:
            raise InvalidHyperparameterException("k", v)
        return v


class ChunkSearchResult(BaseModel):
    id: str = Field(description="Chunk ID")
    document_id: str = Field(description="Document ID the chunk belongs to")
    text: str = Field(description="Text content of the chunk")
    score: float = Field(description="Similarity score (lower is better for distance metrics)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Chunk metadata")
    embedding: Optional[List[float]] = Field(
        default=None, description="Chunk embedding vector"
    )


class SearchResponse(BaseModel):
    query: str = Field(description="The original query")
    results: List[ChunkSearchResult] = Field(description="Matching chunks")
    documents: Dict[str, DocumentResponse] = Field(
        description="Information about documents containing matches"
    )
    total_results: int = Field(description="Total number of results")
