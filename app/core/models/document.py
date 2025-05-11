import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class DocumentMetadata(BaseModel):
    source: str = Field(description="The source of the document")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="The date and time the document was created",
    )
    tags: list[str] = Field(
        default_factory=list, description="The tags of the document"
    )
    author: Optional[str] = Field(
        default=None, description="The author of the document"
    )
    title: Optional[str] = Field(default=None, description="The title of the document")
    description: Optional[str] = Field(
        default=None, description="The description of the document"
    )

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


class Document(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the document",
    )
    title: str = Field(description="The title of the document")
    description: str = Field(description="The description of the document")
    chunks: list[str] = Field(description="The chunks in the document")
    metadata: DocumentMetadata = Field(description="The metadata of the document")

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Ensure title is not empty"""
        if not v.strip():
            raise ValueError("Document title cannot be empty")
        return v
