from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

from models.chunk import Chunk, ChunkMetadata
from models.library import Library
from dependencies import get_libraries
from core.embedding import get_embedding, get_embeddings

router = APIRouter(
    tags=["chunks"],
    responses={404: {"description": "Chunk not found"}},
)

# Chunk request
class ChunkRequest(BaseModel):
    text: str = Field(description="The text content of the chunk")
    document_id: str = Field(description="ID of the document this chunk belongs to")
    metadata: Optional[ChunkMetadata] = Field(default=None, description="Metadata for the chunk")


@router.post("/libraries/{library_id}/chunks", response_model=dict)
async def add_chunk(
    library_id: str,
    chunk_request: ChunkRequest,
    libraries: Dict[str, Library] = Depends(get_libraries),
):
    """Add a chunk to a library and document with metadata"""
    if library_id not in libraries:
        raise HTTPException(status_code=404, detail="Library not found")
    
    library = libraries[library_id]
    
    document_id = chunk_request.document_id
    if document_id not in library.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = library.documents[document_id]
    
    # Get embedding asynchronously
    embedding = await get_embedding(chunk_request.text)
    
    # Create chunk with metadata
    chunk_metadata = chunk_request.metadata or ChunkMetadata(source="api")
    chunk = Chunk(
        text=chunk_request.text,
        embedding=embedding,
        document_id=document_id,
        metadata=chunk_metadata
    )
    
    # Add chunk to library and document
    library.add_chunk(chunk, document)
    
    return {
        "chunk_id": chunk.id,
        "document_id": document_id,
        "library_id": library_id,
        "message": "Chunk added successfully"
    }

@router.post("/libraries/{library_id}/chunks/batch", response_model=dict)
async def add_chunks_batch(
    library_id: str,
    chunk_requests: List[ChunkRequest],
    background_tasks: BackgroundTasks,
    libraries: Dict[str, Library] = Depends(get_libraries),
):
    """Add multiple chunks to a library in a single batch operation"""
    if library_id not in libraries:
        raise HTTPException(status_code=404, detail="Library not found")
    
    library = libraries[library_id]
    
    # Validate all document IDs exist before processing
    document_ids = {chunk_request.document_id for chunk_request in chunk_requests}
    for document_id in document_ids:
        if document_id not in library.documents:
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
    
    # Collect all chunk texts for batch embedding
    chunk_texts = [chunk_request.text for chunk_request in chunk_requests]
    
    # Get embeddings in batch
    embeddings = await get_embeddings(chunk_texts)
    
    chunks: list[Chunk] = []
    
    # Create and add each chunk
    for i, (chunk_request, embedding) in enumerate(zip(chunk_requests, embeddings)):
        document_id = chunk_request.document_id
        
        # Create chunk with metadata
        chunk_metadata = chunk_request.metadata or ChunkMetadata(source="api")
        chunk = Chunk(
            text=chunk_request.text,
            embedding=embedding,
            document_id=document_id,
            metadata=chunk_metadata
        )
        
        # Add chunk to library and document
        chunks.append(chunk)
    
    background_tasks.add_task(library.add_chunks, chunks)
    
    return {
        "library_id": library_id,
        "added_chunks": [chunk.id for chunk in chunks],
        "count": len(chunks),
        "message": "Chunks added successfully"
    }


@router.get("/libraries/{library_id}/chunks", response_model=List[dict])
async def list_chunks(
    library_id: str,
    document_id: Optional[str] = None,
    libraries: Dict[str, Library] = Depends(get_libraries),
):
    """List chunks in a library, optionally filtered by document"""
    if library_id not in libraries:
        raise HTTPException(status_code=404, detail="Library not found")
    
    library = libraries[library_id]
    
    # If document_id provided, filter chunks by document
    if document_id:
        if document_id not in library.documents:
            raise HTTPException(status_code=404, detail="Document not found")
        
        document = library.documents[document_id]
        chunk_ids = document.chunks
        chunks = [library.chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in library.chunks]
    else:
        chunks = list(library.chunks.values())
    
    return [
        {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "text": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
        }
        for chunk in chunks
    ]


@router.get("/libraries/{library_id}/chunks/{chunk_id}", response_model=dict)
async def get_chunk(
    library_id: str,
    chunk_id: str,
    libraries: Dict[str, Library] = Depends(get_libraries),
):
    """Get a specific chunk"""
    if library_id not in libraries:
        raise HTTPException(status_code=404, detail="Library not found")
    
    library = libraries[library_id]
    
    if chunk_id not in library.chunks:
        raise HTTPException(status_code=404, detail="Chunk not found")
    
    chunk = library.chunks[chunk_id]
    
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "metadata": chunk.metadata.model_dump()
    }


@router.put("/libraries/{library_id}/chunks/{chunk_id}", response_model=dict)
async def update_chunk(
    library_id: str,
    chunk_id: str,
    chunk_request: ChunkRequest,
    libraries: Dict[str, Library] = Depends(get_libraries),
):
    """Update a chunk with metadata"""
    if library_id not in libraries:
        raise HTTPException(status_code=404, detail="Library not found")
    
    library = libraries[library_id]
    
    if chunk_id not in library.chunks:
        raise HTTPException(status_code=404, detail="Chunk not found")
    
    chunk = library.chunks[chunk_id]
    
    # Check if document exists
    document_id = chunk_request.document_id
    if document_id not in library.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get new embedding if text changed
    if chunk.text != chunk_request.text:
        embedding = await get_embedding(chunk_request.text)
        chunk.embedding = embedding
        chunk.text = chunk_request.text
        
        # Update embedding in indexer
        library.indexer.update(chunk_id, embedding)
    
    # If document changed, update references
    if chunk.document_id != document_id:
        old_document = library.documents[chunk.document_id]
        new_document = library.documents[document_id]
        
        # Remove from old document
        if chunk_id in old_document.chunks:
            old_document.chunks.remove(chunk_id)
        
        # Add to new document if not already there
        if chunk_id not in new_document.chunks:
            new_document.chunks.append(chunk_id)
        
        # Update chunk document reference
        chunk.document_id = document_id
    
    # Update metadata if provided
    if chunk_request.metadata:
        # Update metadata fields while preserving created_at
        created_at = chunk.metadata.created_at
        chunk.metadata = chunk_request.metadata
        chunk.metadata.created_at = created_at
    
    library.chunks[chunk_id] = chunk
    
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "library_id": library_id,
        "message": "Chunk updated successfully"
    }


@router.delete("/libraries/{library_id}/chunks/{chunk_id}", response_model=dict)
async def delete_chunk(
    library_id: str,
    chunk_id: str,
    libraries: Dict[str, Library] = Depends(get_libraries),
):
    """Delete a chunk"""
    if library_id not in libraries:
        raise HTTPException(status_code=404, detail="Library not found")
    
    library = libraries[library_id]
    
    if chunk_id not in library.chunks:
        raise HTTPException(status_code=404, detail="Chunk not found")
    
    chunk = library.chunks[chunk_id]
    
    # Remove chunk reference from document
    document_id = chunk.document_id
    if document_id in library.documents:
        document = library.documents[document_id]
        if chunk_id in document.chunks:
            document.chunks.remove(chunk_id)
    
    # Remove from index
    library.indexer.delete(chunk_id)
    
    # Remove from chunks dict
    del library.chunks[chunk_id]
    
    return {
        "library_id": library_id,
        "chunk_id": chunk_id,
        "message": "Chunk deleted successfully"
    } 