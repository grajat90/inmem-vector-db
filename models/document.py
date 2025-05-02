from datetime import datetime
from pydantic import BaseModel, Field
import uuid
from typing import Optional


class DocumentMetadata(BaseModel):
    source: str = Field(description="The source of the document")
    created_at: datetime = Field(default_factory=datetime.now, description="The date and time the document was created")
    tags: list[str] = Field(default_factory=list, description="The tags of the document")
    author: Optional[str] = Field(default=None, description="The author of the document")
    title: Optional[str] = Field(default=None, description="The title of the document")
    description: Optional[str] = Field(default=None, description="The description of the document")


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the document")
    title: str = Field(description="The title of the document")
    description: str = Field(description="The description of the document")
    chunks: list[str] = Field(description="The chunks in the document")
    metadata: DocumentMetadata = Field(description="The metadata of the document")
