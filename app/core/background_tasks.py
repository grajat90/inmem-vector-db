import asyncio

# In-memory storage
from app.dependencies import get_libraries, get_reindex_lock

async def reindex_library(library_id: str):
    """Reindex a library in the background"""
    # Get references to the global stores
    libraries = get_libraries()
    
    if library_id not in libraries:
        return
    
    lock = await get_reindex_lock(library_id)
    
    # Use the lock to prevent concurrent reindexing
    async with lock:
        library = libraries[library_id]
        # Rebuild the index
        chunks = list(library.chunks.values())
        library.indexer.build(chunks) 