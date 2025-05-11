from fastapi import APIRouter

from app.api.endpoints import chunks, documents, libraries, search

router = APIRouter()

router.include_router(libraries.router)
router.include_router(documents.router)
router.include_router(chunks.router)
router.include_router(search.router)
