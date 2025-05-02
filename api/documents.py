from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

from models.document import Document, DocumentMetadata
from models.chunk import Chunk, ChunkMetadata
from models.library import Library
from dependencies import get_libraries
from core.background_tasks import reindex_library
from core.embedding import get_embeddings

# Import the chunk request model
from api.libraries import ChunkRequest

router = APIRouter(
    tags=["documents"],
    responses={404: {"description": "Document not found"}},
)

# Document request with optional chunks
class DocumentRequest(BaseModel):
    title: str = Field(description="The title of the document")
    description: str = Field(description="A description of the document")
    metadata: Optional[DocumentMetadata] = Field(default=None, description="Metadata for the document")
    chunks: Optional[List[ChunkRequest]] = Field(default=None, description="Chunks to include in this document")


@router.post("/libraries/{library_id}/documents", response_model=dict)
async def add_document(
    library_id: str,
    document_request: DocumentRequest,
    libraries: Dict[str, Library] = Depends(get_libraries),
):
    """
    Add a document to a library, optionally with chunks.
    
    This endpoint allows creating a document with its chunks in a single request.
    """
    if library_id not in libraries:
        raise HTTPException(status_code=404, detail="Library not found")
    
    doc_metadata = document_request.metadata or DocumentMetadata(source="api")
    
    # Create document with metadata
    document = Document(
        title=document_request.title,
        description=document_request.description,
        chunks=[],
        metadata=doc_metadata
    )
    
    library = libraries[library_id]
    library.documents[document.id] = document
    
    # Process chunks if provided
    if document_request.chunks:
        # Collect all chunk texts for batch embedding
        chunk_texts = [chunk_request.text for chunk_request in document_request.chunks]
        chunk_metadatas = [chunk_request.metadata or ChunkMetadata(source="api") for chunk_request in document_request.chunks]
        
        # Get embeddings in batch
        embeddings = await get_embeddings(chunk_texts)
        
        # Create and add each chunk
        for i, (text, embedding, metadata) in enumerate(zip(chunk_texts, embeddings, chunk_metadatas)):
            chunk = Chunk(
                text=text,
                embedding=embedding,
                document_id=document.id,
                metadata=metadata
            )
            
            # Add chunk to document and library
            library.add_chunk(chunk, document)
    
    return {
        "document_id": document.id,
        "library_id": library_id,
        "title": document.title,
        "chunk_count": len(document.chunks),
        "message": "Document added successfully"
    }


@router.get("/libraries/{library_id}/documents", response_model=List[dict])
async def list_documents(
    library_id: str,
    libraries: Dict[str, Library] = Depends(get_libraries),
):
    """List documents in a library"""
    if library_id not in libraries:
        raise HTTPException(status_code=404, detail="Library not found")
    
    library = libraries[library_id]
    
    return [
        {
            "id": doc.id,
            "title": doc.title,
            "description": doc.description,
            "chunk_count": len(doc.chunks)
        }
        for doc in library.documents.values()
    ]


@router.get("/libraries/{library_id}/documents/{document_id}", response_model=dict)
async def get_document(
    library_id: str,
    document_id: str,
    libraries: Dict[str, Library] = Depends(get_libraries),
):
    """Get a specific document"""
    if library_id not in libraries:
        raise HTTPException(status_code=404, detail="Library not found")
    
    library = libraries[library_id]
    
    if document_id not in library.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = library.documents[document_id]
    
    return {
        "id": document.id,
        "title": document.title,
        "description": document.description,
        "metadata": document.metadata.model_dump(),
        "chunks": document.chunks
    }


@router.put("/libraries/{library_id}/documents/{document_id}", response_model=dict)
async def update_document(
    library_id: str,
    document_id: str,
    document_request: DocumentRequest,
    libraries: Dict[str, Library] = Depends(get_libraries),
):
    """Update a document and optionally add new chunks"""
    if library_id not in libraries:
        raise HTTPException(status_code=404, detail="Library not found")
    
    library = libraries[library_id]
    
    if document_id not in library.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = library.documents[document_id]
    document.title = document_request.title
    document.description = document_request.description
    
    if document_request.metadata:
        # Update metadata fields while preserving created_at
        created_at = document.metadata.created_at
        document.metadata = document_request.metadata
        document.metadata.created_at = created_at
    
    # Do not process chunks for update

    library.documents[document_id] = document
    
    return {
        "id": document.id,
        "library_id": library_id,
        "chunk_count": len(document.chunks),
        "message": "Document updated successfully"
    }


@router.delete("/libraries/{library_id}/documents/{document_id}", response_model=dict)
async def delete_document(
    library_id: str,
    document_id: str,
    background_tasks: BackgroundTasks,
    libraries: Dict[str, Library] = Depends(get_libraries),
):
    """Delete a document and its chunks"""
    if library_id not in libraries:
        raise HTTPException(status_code=404, detail="Library not found")
    
    library = libraries[library_id]
    
    if document_id not in library.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get document to delete
    document = library.documents[document_id]
    
    # Remove all chunks in document from the library
    chunk_ids = document.chunks.copy()
    for chunk_id in chunk_ids:
        if chunk_id in library.chunks:
            # Remove from chunks dict
            del library.chunks[chunk_id]
    
    # Remove document from library
    del library.documents[document_id]
    
    # Trigger reindex in background
    background_tasks.add_task(reindex_library, library_id)
    
    return {
        "library_id": library_id,
        "document_id": document_id,
        "message": "Document and associated chunks deleted successfully"
    } 