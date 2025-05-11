import uuid
from datetime import datetime
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator


class ChunkMetadata(BaseModel):
    source: str = Field(description="The source of the chunk")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="The date and time the chunk was created",
    )
    tags: list[str] = Field(default_factory=list, description="The tags of the chunk")
    author: Optional[str] = Field(default=None, description="The author of the chunk")
    title: Optional[str] = Field(default=None, description="The title of the chunk")
    description: Optional[str] = Field(
        default=None, description="The description of the chunk"
    )
    page: Optional[int] = Field(default=None, ge=0, description="The page number of the chunk")

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Ensure tags are unique and non-empty strings"""
        if not all(isinstance(tag, str) for tag in v):
            raise TypeError("All tags must be strings")
        if any(not tag.strip() for tag in v):
            raise ValueError("Tags cannot be empty strings")
        # Return unique tags
        return list(set(v))

    @field_validator("page")
    @classmethod
    def validate_page(cls, v: Optional[int]) -> Optional[int]:
        """Ensure page number is non-negative if provided"""
        if v is not None and v < 0:
            raise ValueError("Page number must be non-negative")
        return v


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = Field(description="The id of the document that this chunk belongs to")
    text: str = Field(description="The text of the chunk")
    embedding: np.ndarray = Field(description="The embedding of the chunk")
    metadata: ChunkMetadata = Field(description="The metadata of the chunk")
    model_config = {"arbitrary_types_allowed": True}

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Ensure text is not empty"""
        if not v.strip():
            raise ValueError("Chunk text cannot be empty")
        return v

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: np.ndarray) -> np.ndarray:
        """Validate embedding is a non-empty numpy array with proper dimensions"""
        if v.ndim != 1:
            raise ValueError("Embedding must be a 1-dimensional array")
        if v.size <= 0:
            raise ValueError("Embedding cannot be empty")
        return v
