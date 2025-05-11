from typing import List

from fastapi import APIRouter, BackgroundTasks

from app.api.dependencies import get_libraries_dependency
from app.api.schemas.document import (
    CreateDocumentRequest,
    DeleteDocumentResponse,
    DocumentResponse,
)
from app.api.services.document_service import DocumentService

router = APIRouter(tags=["documents"])


@router.post("/libraries/{library_id}/documents")
async def add_document(
    library_id: str,
    document_request: CreateDocumentRequest,
    libraries: get_libraries_dependency,
) -> DocumentResponse:
    """
    Add a document to a library, optionally with chunks.

    This endpoint allows creating a document with its chunks in a single request.
    """
    document_service = DocumentService(libraries, library_id)
    document = await document_service.add_document(document_request)

    return DocumentResponse(
        document_id=document.id,
        library_id=library_id,
        title=document.title,
        chunk_count=len(document.chunks),
        message="Document added successfully",
    )


@router.get("/libraries/{library_id}/documents")
async def list_documents(
    library_id: str,
    libraries: get_libraries_dependency,
) -> List[DocumentResponse]:
    """List documents in a library"""
    document_service = DocumentService(libraries, library_id)
    documents = document_service.list_documents()

    return [
        DocumentResponse(
            document_id=doc.id,
            title=doc.title,
            description=doc.description,
            chunk_count=len(doc.chunks),
        )
        for doc in documents
    ]


@router.get("/libraries/{library_id}/documents/{document_id}")
async def get_document(
    library_id: str,
    document_id: str,
    libraries: get_libraries_dependency,
) -> DocumentResponse:
    """Get a specific document"""
    document_service = DocumentService(libraries, library_id)
    document = document_service.get_document(document_id)

    return DocumentResponse(
        document_id=document.id,
        title=document.title,
        description=document.description,
        metadata=document.metadata.model_dump(),
        chunk_count=len(document.chunks),
    )


@router.put("/libraries/{library_id}/documents/{document_id}")
async def update_document(
    library_id: str,
    document_id: str,
    document_request: CreateDocumentRequest,
    libraries: get_libraries_dependency,
) -> DocumentResponse:
    """Update a document and optionally add new chunks"""
    document_service = DocumentService(libraries, library_id)
    document = document_service.update_document(document_id, document_request)

    return DocumentResponse(
        document_id=document.id,
        library_id=library_id,
        title=document.title,
        chunk_count=len(document.chunks),
        message="Document updated successfully",
    )


@router.delete("/libraries/{library_id}/documents/{document_id}")
async def delete_document(
    library_id: str,
    document_id: str,
    background_tasks: BackgroundTasks,
    libraries: get_libraries_dependency,
) -> DeleteDocumentResponse:
    """Delete a document and its chunks"""
    document_service = DocumentService(libraries, library_id, background_tasks)
    document = document_service.delete_document(document_id)

    return DeleteDocumentResponse(
        document_id=document.id,
        library_id=library_id,
        message="Document and associated chunks deleted successfully",
    )
