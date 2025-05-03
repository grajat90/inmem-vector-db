import numpy as np
from datetime import datetime
from threading import Lock
import pickle

from indexers.indexer import DistanceMetric, Indexer
from models.chunk import Chunk



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

    def update(self, chunk_id: str, embedding: np.ndarray):
        """Update an existing chunk's embedding"""
        with self._lock:
            if chunk_id in self.embeddings:
                self.embeddings[chunk_id] = embedding
                self.last_updated = datetime.now()

    def search(
        self, 
        query_embedding: np.ndarray,
        k: int = 5, 
        distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    ) -> list[str]:
        
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
                    query_embedding, 
                    embedding, 
                    distance_metric
                )
                distances[chunk_id] = distance
        
        # Sort by distance (ascending) and take top-k
        nearest_ids = sorted(distances.keys(), key=lambda k: distances[k])[:k]
        
        return nearest_ids

    def save(self, path: str):
        """Save the index to disk using pickle"""
        with self._lock:
            data = {
                'embeddings': self.embeddings,
                'created': self.created,
                'last_updated': self.last_updated,
                'name': self.name
            }
            
            with open(path, 'wb') as f:
                pickle.dump(data, f)

    def load(self, path: str):
        """Load the index from disk using pickle"""
        with self._lock:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.embeddings = data['embeddings']
            self.created = data['created']
            self.last_updated = data['last_updated']
            self.name = data['name']
