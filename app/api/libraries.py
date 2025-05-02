from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

from models.library import Library, LibraryMetadata
from models.document import Document, DocumentMetadata
from models.chunk import Chunk, ChunkMetadata
from app.dependencies import get_libraries, get_reindex_lock
from app.core.background_tasks import reindex_library
from app.core.embedding import get_embeddings

router = APIRouter(
    prefix="/libraries",
    tags=["libraries"],
    responses={404: {"description": "Library not found"}},
)

# Request schemas for nested creation
class ChunkRequest(BaseModel):
    text: str = Field(description="The text content of the chunk")
    metadata: Optional[ChunkMetadata] = Field(default=None, description="Metadata for the chunk")

class DocumentWithChunksRequest(BaseModel):
    title: str = Field(description="The title of the document")
    description: str = Field(description="A description of the document")
    metadata: Optional[DocumentMetadata] = Field(default=None, description="Metadata for the document")
    chunks: Optional[List[ChunkRequest]] = Field(default=None, description="Chunks to include in this document")

class LibraryRequest(BaseModel):
    name: str = Field(description="The name of the library")
    metadata: Optional[LibraryMetadata] = Field(default=None, description="Metadata for the library")
    documents: Optional[List[DocumentWithChunksRequest]] = Field(default=None, description="Documents to include in the library")


@router.post("", response_model=dict)
async def create_library(
    library_request: LibraryRequest,
    background_tasks: BackgroundTasks,
    libraries: Dict[str, Library] = Depends(get_libraries),
):
    """
    Create a new library with optional documents and chunks.
    
    This endpoint allows creating a complete library structure in a single request,
    including nested documents and their chunks.
    """
    metadata = library_request.metadata or LibraryMetadata(tags=[])
    
    # Create the library
    library = Library(
        name=library_request.name,
        documents={},
        chunks={},
        metadata=metadata
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
                metadata=doc_metadata
            )
            
            library.documents[document.id] = document
            
            # Process chunks if any
            if doc_request.chunks:
                for chunk_request in doc_request.chunks:
                    all_chunks_data.append({
                        "text": chunk_request.text,
                        "document_id": document.id,
                        "metadata": chunk_request.metadata or ChunkMetadata(source="api")
                    })
        
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
                    metadata=chunk_metadata
                )
                
                # Add chunk to document
                library.documents[chunk_data["document_id"]].chunks.append(chunk.id)
                library.chunks[chunk.id] = chunk
    
    # Store the library
    libraries[library.id] = library
    
    # Build the index in the background
    background_tasks.add_task(reindex_library, library.id)
    
    return {
        "library_id": library.id,
        "name": library.name,
        "document_count": len(library.documents),
        "chunk_count": len(library.chunks),
        "message": "Library created successfully with all documents and chunks"
    }


@router.get("", response_model=List[dict])
async def list_libraries(
    libraries: Dict[str, Library] = Depends(get_libraries),
):
    """List all libraries"""
    return [
        {
            "id": library.id,
            "name": library.name,
            "document_count": len(library.documents),
            "chunk_count": len(library.chunks)
        }
        for library in libraries.values()
    ]


@router.get("/{library_id}", response_model=dict)
async def get_library(
    library_id: str,
    libraries: Dict[str, Library] = Depends(get_libraries),
):
    """Get details for a specific library"""
    if library_id not in libraries:
        raise HTTPException(status_code=404, detail="Library not found")
    
    library = libraries[library_id]
    
    return {
        "id": library.id,
        "name": library.name,
        "document_count": len(library.documents),
        "chunk_count": len(library.chunks),
        "metadata": library.metadata.model_dump(),
        "documents": [doc.id for doc in library.documents.values()],
        "last_updated": library.indexer.last_updated.isoformat()
    }


@router.put("/{library_id}", response_model=dict)
async def update_library(
    library_id: str,
    library_request: LibraryRequest,
    libraries: Dict[str, Library] = Depends(get_libraries),
):
    """Update library metadata"""
    if library_id not in libraries:
        raise HTTPException(status_code=404, detail="Library not found")
    
    library = libraries[library_id]
    library.name = library_request.name
    
    if library_request.metadata:
        # Update metadata fields while preserving created_at
        created_at = library.metadata.created_at
        library.metadata = library_request.metadata
        library.metadata.created_at = created_at
        library.metadata.updated_at = LibraryMetadata().updated_at  # Update timestamp
    
    return {
        "id": library.id,
        "name": library.name,
        "message": "Library updated successfully"
    }


@router.delete("/{library_id}", response_model=dict)
async def delete_library(
    library_id: str,
    libraries: Dict[str, Library] = Depends(get_libraries),
    reindex_locks: Dict[str, object] = Depends(get_reindex_lock),
):
    """Delete a library"""
    if library_id not in libraries:
        raise HTTPException(status_code=404, detail="Library not found")
    
    # Remove the library
    del libraries[library_id]
    
    # Also remove the reindex lock if it exists
    if library_id in reindex_locks:
        del reindex_locks[library_id]
    
    return {"message": "Library deleted successfully"} 