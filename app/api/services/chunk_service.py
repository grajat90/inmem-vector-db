from typing import Dict, Optional

from fastapi import BackgroundTasks

from app.api.exceptions.chunk_exceptions import ChunkNotFoundException
from app.api.exceptions.common_exceptions import BackgroundTasksNotFoundException
from app.api.exceptions.document_exceptions import DocumentNotFoundException
from app.api.exceptions.library_exceptions import LibraryNotFoundException
from app.api.schemas.chunk import CreateChunkRequest
from app.core.embedding import get_embedding, get_embeddings
from app.core.models.chunk import Chunk, ChunkMetadata
from app.core.models.library import Library


class ChunkService:
    def __init__(
        self,
        libraries: Dict[str, Library],
        library_id: str,
        document_id: Optional[str] = None,
        background_tasks: Optional[BackgroundTasks] = None,
    ):
        self.libraries = libraries
        if library_id not in libraries:
            raise LibraryNotFoundException(library_id)

        self.library = libraries[library_id]

        if document_id is not None:
            if self.library is None:
                raise LibraryNotFoundException(library_id)
            if document_id not in self.library.documents:
                raise DocumentNotFoundException(document_id)
            self.document = self.library.documents[document_id]
        else:
            self.document = None

        self.background_tasks = background_tasks

    async def add_chunk(
        self, chunk_request: CreateChunkRequest
    ) -> tuple[Chunk, Library]:
        if not self.library:
            raise LibraryNotFoundException(self.library_id)

        if self.document is None:
            raise DocumentNotFoundException(self.document_id)

        # Get embedding asynchronously
        embedding: list[float] = await get_embedding(chunk_request.text)

        # Create chunk with metadata
        chunk_metadata: ChunkMetadata = chunk_request.metadata or ChunkMetadata(
            source="api"
        )
        chunk = Chunk(
            text=chunk_request.text,
            embedding=embedding,
            document_id=self.document.id,
            metadata=chunk_metadata,
        )

        # Add chunk to library and document
        self.library.add_chunk(chunk, self.document)

        return (chunk, self.library)

    async def add_chunks_batch(
        self, chunk_requests: list[CreateChunkRequest]
    ) -> tuple[list[Chunk], Library]:
        if self.library is None:
            raise LibraryNotFoundException(self.library_id)

        if self.background_tasks is None:
            raise BackgroundTasksNotFoundException()

        # Validate all document IDs exist before processing
        document_ids: set[str] = {
            chunk_request.document_id for chunk_request in chunk_requests
        }
        for document_id in document_ids:
            if document_id not in self.library.documents:
                raise DocumentNotFoundException(document_id)

        # Collect all chunk texts for batch embedding
        chunk_texts: list[str] = [
            chunk_request.text for chunk_request in chunk_requests
        ]

        # Get embeddings in batch
        embeddings: list[list[float]] = await get_embeddings(chunk_texts)

        chunks: list[Chunk] = []

        # Create and add each chunk
        for chunk_request, embedding in zip(chunk_requests, embeddings):
            document_id = chunk_request.document_id

            # Create chunk with metadata
            chunk_metadata: ChunkMetadata = chunk_request.metadata or ChunkMetadata(
                source="api"
            )
            chunk: Chunk = Chunk(
                text=chunk_request.text,
                embedding=embedding,
                document_id=document_id,
                metadata=chunk_metadata,
            )

            # Add chunk to library and document
            chunks.append(chunk)

        self.background_tasks.add_task(self.library.add_chunks, chunks)

        return (chunks, self.library)

    def list_chunks(self) -> list[Chunk]:
        if self.library is None:
            raise LibraryNotFoundException(self.library_id)

        # If document_id provided, filter chunks by document
        if self.document is not None:
            chunk_ids = self.document.chunks
            chunks = [
                self.library.chunks[chunk_id]
                for chunk_id in chunk_ids
                if chunk_id in self.library.chunks
            ]
        else:
            chunks = list(self.library.chunks.values())

        return chunks

    def get_chunk(self, chunk_id: str) -> Chunk:
        if self.library is None:
            raise LibraryNotFoundException(self.library_id)

        if chunk_id not in self.library.chunks:
            raise ChunkNotFoundException(chunk_id)

        return self.library.chunks[chunk_id]

    async def update_chunk(
        self, chunk_id: str, chunk_request: CreateChunkRequest
    ) -> Chunk:
        if self.library is None:
            raise LibraryNotFoundException(self.library_id)

        if chunk_id not in self.library.chunks:
            raise ChunkNotFoundException(chunk_id)

        chunk = self.library.chunks[chunk_id]

        # Check if document exists
        if self.document is None:
            raise DocumentNotFoundException(chunk.document_id)
        # Get new embedding if text changed
        if chunk.text != chunk_request.text:
            embedding = await get_embedding(chunk_request.text)
            chunk.embedding = embedding
            chunk.text = chunk_request.text

            # Update embedding in indexer
            self.library.indexer.update(chunk_id, embedding)

        # If document changed, update references
        if chunk.document_id != self.document.id:
            old_document = self.library.documents[chunk.document_id]
            new_document = self.library.documents[self.document.id]

            # Remove from old document
            if chunk_id in old_document.chunks:
                old_document.chunks.remove(chunk_id)

            # Add to new document if not already there
            if chunk_id not in new_document.chunks:
                new_document.chunks.append(chunk_id)

            # Update chunk document reference
            chunk.document_id = self.document.id

        # Update metadata if provided
        if chunk_request.metadata:
            # Update metadata fields while preserving created_at
            created_at = chunk.metadata.created_at
            chunk.metadata = chunk_request.metadata
            chunk.metadata.created_at = created_at

        self.library.chunks[chunk_id] = chunk

        return chunk

    def delete_chunk(self, chunk_id: str) -> Chunk:
        if self.library is None:
            raise LibraryNotFoundException(self.library_id)

        if chunk_id not in self.library.chunks:
            raise ChunkNotFoundException(chunk_id)

        chunk = self.library.chunks[chunk_id]

        chunk_deleted = chunk.model_copy()

        # Remove chunk reference from document
        document_id = chunk.document_id
        if document_id in self.library.documents:
            document = self.library.documents[document_id]
            if chunk_id in document.chunks:
                document.chunks.remove(chunk_id)

        # Remove from index
        self.library.indexer.delete(chunk_id)

        # Remove from chunks dict
        del self.library.chunks[chunk_id]

        return chunk_deleted
