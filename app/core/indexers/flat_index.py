import warnings
from datetime import datetime
from threading import Lock

import numpy as np
from pydantic import BaseModel

from app.core.indexers.indexer import DistanceMetric, Indexer
from app.core.models.chunk import Chunk


class FlatIndexer(Indexer):
    """
    Flat Indexer

    This indexer is a simple linear search index.

    Time complexity:
    - Add: O(1)
    - Search: O(n)

    Space complexity:
    - O(n)

    Generally fine for smaller datasets. Not recommended for larger datasets.

    """

    def __init__(self, name: str = "flat_indexer"):
        super().__init__(name=name, embeddings={})
        self._lock = Lock()  # For thread safety

    def build(self, chunks: list[Chunk]):
        """Build the index from a list of chunks"""
        with self._lock:
            self.embeddings = {}
            for chunk in chunks:
                self.embeddings[chunk.id] = np.array(chunk.embedding, dtype=np.float32)
            self.last_updated = datetime.now()

    def add(self, chunk_id: str, embedding: np.ndarray):
        """Add a new chunk embedding to the index"""
        with self._lock:
            self.embeddings[chunk_id] = embedding
            self.last_updated = datetime.now()

    def delete(self, chunk_id: str):
        """Delete a chunk from the index"""
        with self._lock:
            if chunk_id in self.embeddings:
                del self.embeddings[chunk_id]
                self.last_updated = datetime.now()
            else:
                warnings.warn(
                    f"Chunk {chunk_id} not found in the index. Skipping.", UserWarning
                )

    def update(self, chunk_id: str, embedding: np.ndarray):
        """Update an existing chunk's embedding"""
        with self._lock:
            if chunk_id in self.embeddings:
                self.embeddings[chunk_id] = embedding
                self.last_updated = datetime.now()
            else:
                raise ValueError(f"Chunk {chunk_id} not found in the index. Cannot update.")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
    ) -> list[str]:
        """Search the index for the k nearest neighbors to the query embedding"""

        # Convert embedding to numpy array if needed
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding, dtype=np.float32)

        # Compute distances using the specified metric
        with self._lock:
            # Prepare for distance calculations
            distances = {}

            # Calculate distances using the specified metric
            for chunk_id, embedding in self.embeddings.items():
                distance = self._calculate_distance(
                    query_embedding, embedding, distance_metric
                )
                distances[chunk_id] = distance

        # Sort by distance (ascending) and take top-k
        nearest_ids = sorted(distances.keys(), key=lambda k: distances[k])[:k]

        if len(nearest_ids) == 0:
            warnings.warn(
                "No nearest neighbors found for query embedding. Returning empty list.",
                UserWarning,
            )
        elif len(nearest_ids) < k:
            warnings.warn(
                f"""
                Only {len(nearest_ids)} nearest neighbors found for query embedding.
                Returning {len(nearest_ids)} nearest neighbors.
                """,
                UserWarning,
            )

        return nearest_ids

    def get_dict_repr(self) -> dict:
        """Get a complete dictionary representation of the indexer for serialization"""
        return {
            "name": self.name,
            "created": self.created.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "embeddings": {k: v.tolist() for k, v in self.embeddings.items()},
        }

    def load_from_dict(self, dict_repr: dict):
        """Restore the indexer from a dictionary representation"""
        try:
            # Validate with Pydantic
            config = FlatIndexConfig(**dict_repr)

            # Apply validated data
            self.name = config.name
            self.created = config.created
            self.last_updated = config.last_updated

            # Convert embedding lists back to numpy arrays
            self.embeddings = {k: np.array(v) for k, v in config.embeddings.items()}

        except Exception as e:
            raise ValueError(f"Failed to load Flat indexer from dictionary: {str(e)}") from e


class FlatIndexConfig(BaseModel):
    name: str
    created: datetime
    last_updated: datetime
    embeddings: dict[str, list[float]]
