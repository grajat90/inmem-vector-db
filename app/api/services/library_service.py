import asyncio
from datetime import datetime
from typing import Dict, Optional

from fastapi import BackgroundTasks

from app.api.exceptions.library_exceptions import (
    InvalidDistanceMetricException,
    LibraryNotFoundException,
)
from app.api.schemas.document import DocumentResponse
from app.api.schemas.library import CreateLibraryRequest, IndexerType
from app.api.schemas.search import ChunkSearchResult, SearchRequest
from app.api.services.background_tasks import reindex_library
from app.config.indexer_hparams import (
    HNSWIndexerHParams,
    LSHIndexerHParams,
    get_default_hparams,
)
from app.core.embedding import get_embedding, get_embeddings
from app.core.indexers.flat_index import FlatIndexer
from app.core.indexers.hnsw import HNSWIndexer
from app.core.indexers.indexer import DistanceMetric, Indexer
from app.core.indexers.lsh import LSHIndexer
from app.core.models.chunk import Chunk, ChunkMetadata
from app.core.models.document import Document, DocumentMetadata
from app.core.models.library import Library, LibraryMetadata


class LibraryService:
    def __init__(
        self,
        libraries: Dict[str, Library],
        library_id: Optional[str] = None,
        background_tasks: Optional[BackgroundTasks] = None,
        reindex_locks: Optional[Dict[str, asyncio.Lock]] = None,
    ):
        self.libraries = libraries
        if library_id is not None:
            if library_id not in libraries:
                raise LibraryNotFoundException(library_id)
            self.library = libraries[library_id]
        else:
            self.library = None

        self.background_tasks = background_tasks
        self.reindex_locks = reindex_locks

    async def add_library(self, library_request: CreateLibraryRequest) -> Library:
        """
        Add a new library to the in-memory store.

        This method creates a new library with the given name and optional metadata.
        It also allows for adding documents and their chunks to the library simultaneously.
        """

        if self.background_tasks is None:
            raise AttributeError("Background tasks are required to add a library")

        metadata = library_request.metadata or LibraryMetadata(tags=[])

        indexer: Indexer = None

        if library_request.indexer == IndexerType.HNSW:
            hparams: HNSWIndexerHParams = get_default_hparams(library_request.indexer)
            indexer = HNSWIndexer(
                m=hparams.m,
                ef_construction=hparams.ef_construction,
                max_level=hparams.max_level,
                level_mult=hparams.level_mult,
            )
        elif library_request.indexer == IndexerType.LSH:
            hparams: LSHIndexerHParams = get_default_hparams(library_request.indexer)
            indexer = LSHIndexer(hash_size=hparams.hash_size, num_tables=hparams.num_tables)
        else:
            indexer = FlatIndexer()

        # Create the library
        library = Library(
            name=library_request.name,
            documents={},
            chunks={},
            metadata=metadata,
            indexer=indexer,
        )

        # Add documents and chunks if provided
        if library_request.documents:
            # Collect all chunk texts for batch embedding
            all_chunks_data = []

            # First, gather all chunks to get embeddings in batch
            for doc_request in library_request.documents:
                doc_metadata = doc_request.metadata or DocumentMetadata(source="api")

                # Create document
                document = Document(
                    title=doc_request.title,
                    description=doc_request.description,
                    chunks=[],
                    metadata=doc_metadata,
                )

                library.documents[document.id] = document

                # Process chunks if any
                if doc_request.chunks:
                    for chunk_request in doc_request.chunks:
                        all_chunks_data.append(
                            {
                                "text": chunk_request.text,
                                "document_id": document.id,
                                "metadata": chunk_request.metadata
                                or ChunkMetadata(source="api"),
                            }
                        )

            # Get all embeddings at once if there are chunks
            if all_chunks_data:
                all_texts = [chunk_data["text"] for chunk_data in all_chunks_data]
                embeddings = await get_embeddings(all_texts)

                # Create chunks with embeddings
                for i, chunk_data in enumerate(all_chunks_data):
                    chunk_metadata = chunk_data["metadata"]

                    # Create chunk
                    chunk = Chunk(
                        text=chunk_data["text"],
                        embedding=embeddings[i],
                        document_id=chunk_data["document_id"],
                        metadata=chunk_metadata,
                    )

                    # Add chunk to document
                    library.documents[chunk_data["document_id"]].chunks.append(chunk.id)
                    library.chunks[chunk.id] = chunk

        # Store the library
        self.libraries[library.id] = library

        # Build the index in the background
        self.background_tasks.add_task(reindex_library, library.id)

        return library

    def get_library(self) -> Library:
        if self.library is None:
            raise LibraryNotFoundException(self.library_id)

        return self.library

    def update_library(self, library_request: CreateLibraryRequest) -> Library:
        if self.library is None:
            raise LibraryNotFoundException(self.library_id)

        self.library.name = library_request.name

        if library_request.metadata:
            # Update metadata fields while preserving created_at
            created_at = self.library.metadata.created_at
            self.library.metadata = library_request.metadata
            self.library.metadata.created_at = created_at
            self.library.metadata.updated_at = datetime.now()  # Update timestamp

        return self.library

    def delete_library(self) -> Library:
        if self.library is None:
            raise LibraryNotFoundException(self.library_id)

        deleted_library = self.library.model_copy()

        # Remove the library
        del self.libraries[self.library.id]

        # Also remove the reindex lock if it exists
        if self.reindex_locks is not None and self.library.id in self.reindex_locks:
            del self.reindex_locks[self.library.id]

        return deleted_library

    async def search_library(
        self, search_request: SearchRequest
    ) -> tuple[list[ChunkSearchResult], dict[str, DocumentResponse]]:
        if self.library is None:
            raise LibraryNotFoundException(self.library_id)

        # Get embedding for query
        query_embedding: list[float] = await get_embedding(search_request.query)

        # Convert distance metric string to enum
        try:
            distance_metric: DistanceMetric = DistanceMetric(search_request.distance_metric)
        except ValueError:
            raise InvalidDistanceMetricException(search_request.distance_metric)

        # Search library with specified metric
        results: list[Chunk] = self.library.search(query_embedding, k=search_request.k)

        # Apply tag filtering if requested
        if search_request.filter_by_tags:
            filtered_results: list[Chunk] = []
            for chunk in results:
                # Check if all requested tags are in the chunk's tags
                if all(tag in chunk.metadata.tags for tag in search_request.filter_by_tags):
                    filtered_results.append(chunk)
            results = filtered_results[: search_request.k]  # Keep only up to k results

        # Calculate distances to get scores
        scores: list[float] = []
        for chunk in results:
            score = self.library.indexer._calculate_distance(
                query_embedding, chunk.embedding, distance_metric
            )
            scores.append(score)

        # Collect unique document IDs
        document_ids = list(set(chunk.document_id for chunk in results))
        documents_info: dict[str, DocumentResponse] = {}

        # Get document information
        for doc_id in document_ids:
            if doc_id in self.library.documents:
                doc = self.library.documents[doc_id]
                documents_info[doc_id] = DocumentResponse(
                    document_id=doc.id,
                    title=doc.title,
                    description=doc.description,
                    metadata=doc.metadata.model_dump()
                    if search_request.include_metadata
                    else None,
                    chunk_count=len(doc.chunks),
                )

        # Format chunk results
        chunk_results: list[ChunkSearchResult] = []
        for chunk, score in zip(results, scores):
            result = ChunkSearchResult(
                id=chunk.id,
                document_id=chunk.document_id,
                text=chunk.text,
                score=float(score),  # Convert numpy float to Python float
                metadata=chunk.metadata.model_dump()
                if search_request.include_metadata
                else None,
                embedding=chunk.embedding.tolist()
                if search_request.include_embeddings
                else None,
            )
            chunk_results.append(result)

        return chunk_results, documents_info
