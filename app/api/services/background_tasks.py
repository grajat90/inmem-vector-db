from app.api.dependencies import get_libraries, get_reindex_lock


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
        library.rebuild_index()
