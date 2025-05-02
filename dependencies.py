import asyncio
from typing import Dict

from models.library import Library

# In-memory storage
libraries: Dict[str, Library] = {}

# Reindex lock to prevent multiple concurrent reindexing operations
reindex_locks: Dict[str, asyncio.Lock] = {}

def get_libraries() -> Dict[str, Library]:
    """Get the in-memory libraries store"""
    return libraries

async def get_reindex_lock(library_id: str) -> asyncio.Lock:
    """Get lock for reindexing a library"""
    if library_id not in reindex_locks:
        reindex_locks[library_id] = asyncio.Lock()
    return reindex_locks[library_id] 