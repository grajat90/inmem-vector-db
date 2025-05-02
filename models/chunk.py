from datetime import datetime
import uuid
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np

class ChunkMetadata(BaseModel):
    source: str = Field(description="The source of the chunk")
    created_at: datetime = Field(default_factory=datetime.now, description="The date and time the chunk was created")
    tags: list[str] = Field(default_factory=list, description="The tags of the chunk")
    author: Optional[str] = Field(default=None, description="The author of the chunk")
    title: Optional[str] = Field(default=None, description="The title of the chunk")
    description: Optional[str] = Field(default=None, description="The description of the chunk")
    page: Optional[int] = Field(default=None, description="The page number of the chunk")


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = Field(description="The id of the document that this chunk belongs to")
    text: str = Field(description="The text of the chunk")
    embedding: np.ndarray = Field(description="The embedding of the chunk")
    metadata: ChunkMetadata = Field(description="The metadata of the chunk")
    model_config = {
        "arbitrary_types_allowed": True
    }
