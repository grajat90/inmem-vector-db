import os
import pickle
import uuid
from datetime import datetime
from threading import Lock
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator

from app.config.project_settings import get_project_settings
from app.core.indexers.flat_index import FlatIndexer
from app.core.indexers.indexer import Indexer
from app.core.models.chunk import Chunk
from app.core.models.document import Document

config = get_project_settings()


class LibraryMetadata(BaseModel):
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="The date and time the library was created",
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="The date and time the library was last updated",
    )
    tags: list[str] = Field(default_factory=list, description="The tags of the library")


class Library(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the library",
    )
    name: str = Field(description="The name of the library")
    documents: dict[str, Document] = Field(description="The documents in the library")
    chunks: dict[str, Chunk] = Field(description="The chunks in the library")
    metadata: LibraryMetadata = Field(description="The metadata of the library")
    indexer: Indexer = Field(
        default_factory=FlatIndexer, description="The indexer of the library"
    )

    @field_validator("name")
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Library name cannot be empty")
        return v

    # Thread safety lock for concurrent operations
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_dir = os.path.join(
            os.path.dirname(__file__), "../../..", config.data_directory
        )
        os.makedirs(data_dir, exist_ok=True)
        self._save_path = os.path.join(data_dir, f"{self.name}.pkl")
        self._lock = Lock()
        with self._lock:
            self.indexer.build(list(self.chunks.values()))

        if os.path.exists(self._save_path):
            self.load(self._save_path)

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        return [self.chunks[chunk_id] for chunk_id in chunk_ids]

    def get_chunk(self, chunk_id: str) -> Chunk:
        return self.chunks[chunk_id]

    def get_documents(self, document_ids: list[str]) -> list[Document]:
        return [self.documents[document_id] for document_id in document_ids]

    def get_document(self, document_id: str) -> Document:
        return self.documents[document_id]

    def add_chunk(self, chunk: Chunk, document: Optional[Document] = None):
        document_id = chunk.document_id
        with self._lock:
            if document_id not in self.documents:
                if document is None:
                    raise ValueError(
                        """
                        Document does not exist in library.
                        Please provide a document to add the chunk to.
                        """
                    )
                self.documents[document_id] = document
            # Only append to document's chunks if not already there
            if chunk.id not in self.documents[document_id].chunks:
                self.documents[document_id].chunks.append(chunk.id)
            self.chunks[chunk.id] = chunk
            self.indexer.add(chunk.id, chunk.embedding)
        self.save(self._save_path)

    def add_chunks(self, chunks: list[Chunk]):
        document_ids = {chunk.document_id for chunk in chunks}
        for document_id in document_ids:
            if document_id not in self.documents:
                raise ValueError(
                    f"""
                    Document with id {document_id} does not exist in library.
                    Please provide a document to add the chunk to.
                    """
                )
            self.documents[document_id].chunks.extend([chunk.id for chunk in chunks])
        for chunk in chunks:
            self.chunks[chunk.id] = chunk

        self.rebuild_index()

    def rebuild_index(self):
        self.indexer.build(list(self.chunks.values()))

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[Chunk]:
        chunk_ids = self.indexer.search(query_embedding, k)
        return self.get_chunks(chunk_ids)

    def save(self, path: str):
        self_data = {
            "id": self.id,
            "name": self.name,
            "documents": self.documents,
            "chunks": self.chunks,
            "metadata": self.metadata,
            "indexer": self.indexer.get_dict_repr(),
        }

        with self._lock:
            with open(path, "wb") as f:
                pickle.dump(self_data, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self_data = pickle.load(f)

        with self._lock:
            self.id = self_data["id"]
            self.name = self_data["name"]
            self.documents = self_data["documents"]
            self.chunks = self_data["chunks"]
            self.metadata = self_data["metadata"]
            self.indexer.load_from_dict(self_data["indexer"])
