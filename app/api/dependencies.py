import asyncio
import os
import pickle
from typing import Annotated, Dict

from fastapi import Depends

from app.config.project_settings import get_project_settings
from app.core.models.library import Library

config = get_project_settings()

# In-memory storage
libraries: Dict[str, Library] = {}

# Reindex lock to prevent multiple concurrent reindexing operations
reindex_locks: Dict[str, asyncio.Lock] = {}


def get_libraries() -> Dict[str, Library]:
    """Get the in-memory libraries store"""
    return libraries


get_libraries_dependency = Annotated[Dict[str, Library], Depends(get_libraries)]


async def get_reindex_lock(library_id: str) -> asyncio.Lock:
    """Get lock for reindexing a library"""
    if library_id not in reindex_locks:
        reindex_locks[library_id] = asyncio.Lock()
    return reindex_locks[library_id]


# Load libraries from data folder on startup
def load_libraries_from_disk() -> None:
    """Load all libraries from the data folder if any exist"""
    data_dir = os.path.join(os.path.dirname(__file__), "../..", config.data_directory)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return

    for filename in os.listdir(data_dir):
        if filename.endswith(".pkl"):
            try:
                # Load the library directly from the pickle file
                library_path = os.path.join(data_dir, filename)
                with open(library_path, "rb") as f:
                    library_data = pickle.load(f)

                # Create a library with all required fields
                library = Library(
                    id=library_data["id"],
                    name=library_data["name"],
                    documents=library_data["documents"],
                    chunks=library_data["chunks"],
                    metadata=library_data["metadata"],
                )

                # Load the indexer data
                library.load(library_path)

                # Add to in-memory storage
                libraries[library.id] = library
                print(f"Loaded library: {library.name} (ID: {library.id})")
            except Exception as e:
                print(f"Error loading library from {filename}: {str(e)}")
