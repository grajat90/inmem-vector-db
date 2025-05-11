from typing import Dict, Optional

from fastapi import BackgroundTasks

from app.api.exceptions.document_exceptions import DocumentNotFoundException
from app.api.exceptions.library_exceptions import LibraryNotFoundException
from app.api.schemas.document import CreateDocumentRequest
from app.api.services.background_tasks import reindex_library
from app.core.embedding import get_embeddings
from app.core.models.chunk import Chunk, ChunkMetadata
from app.core.models.document import Document, DocumentMetadata
from app.core.models.library import Library


class DocumentService:
    def __init__(
        self,
        libraries: Dict[str, Library],
        library_id: str,
        background_tasks: Optional[BackgroundTasks] = None,
    ):
        self.libraries = libraries
        self.library_id = library_id
        if library_id not in libraries:
            raise LibraryNotFoundException(library_id)

        self.library = libraries[library_id]
        self.background_tasks = background_tasks

    async def add_document(self, document_request: CreateDocumentRequest) -> Document:
        doc_metadata = document_request.metadata or DocumentMetadata(source="api")

        # Create document with metadata
        document = Document(
            title=document_request.title,
            description=document_request.description,
            chunks=[],
            metadata=doc_metadata,
        )

        self.library.documents[document.id] = document

        # Process chunks if provided
        if document_request.chunks:
            # Collect all chunk texts for batch embedding
            chunk_texts = [
                chunk_request.text for chunk_request in document_request.chunks
            ]
            chunk_metadatas = [
                chunk_request.metadata or ChunkMetadata(source="api")
                for chunk_request in document_request.chunks
            ]

            # Get embeddings in batch
            embeddings = await get_embeddings(chunk_texts)

            # Create and add each chunk
            for text, embedding, metadata in zip(
                chunk_texts, embeddings, chunk_metadatas
            ):
                chunk = Chunk(
                    text=text,
                    embedding=embedding,
                    document_id=document.id,
                    metadata=metadata,
                )

                # Add chunk to document and library
                self.library.add_chunk(chunk, document)

        return document

    def list_documents(self) -> list[Document]:
        return list(self.library.documents.values())

    def get_document(self, document_id: str) -> Document:
        if document_id not in self.library.documents:
            raise DocumentNotFoundException(document_id)

        return self.library.documents[document_id]

    def update_document(
        self, document_id: str, document_request: CreateDocumentRequest
    ) -> Document:
        if document_id not in self.library.documents:
            raise DocumentNotFoundException(document_id)

        document = self.library.documents[document_id]
        document.title = document_request.title
        document.description = document_request.description

        if document_request.metadata:
            # Update metadata fields while preserving created_at
            created_at = document.metadata.created_at
            document.metadata = document_request.metadata
            document.metadata.created_at = created_at

        # Do not process chunks for update

        self.library.documents[document_id] = document

        return document

    def delete_document(self, document_id: str) -> Document:
        if document_id not in self.library.documents:
            raise DocumentNotFoundException(document_id)
        # Get document to delete
        document = self.library.documents[document_id]

        # Remove all chunks in document from the library
        chunk_ids = document.chunks.copy()
        for chunk_id in chunk_ids:
            if chunk_id in self.library.chunks:
                # Remove from chunks dict
                del self.library.chunks[chunk_id]

        # Remove document from library
        del self.library.documents[document_id]

        # Trigger reindex in background
        self.background_tasks.add_task(reindex_library, self.library_id)

        return document
