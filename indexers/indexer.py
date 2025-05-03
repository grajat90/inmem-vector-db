# Basic model for any indexer:
#     - add, delete, update
#     - search/query with knn
#       - metrics: time taken, memory usage, etc
#     - save, load
#     - build/re-build index
#     - metadata: name, created, last updated
#     - mechanisms for race conditions (either locks semaphores, etc)
#     - async and multithreading where applicable with thread safety

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import numpy as np
from models.chunk import Chunk

class DistanceMetric(str, Enum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class Indexer(BaseModel, ABC):
    name: str = Field(description="Name of the indexer")
    created: datetime = Field(default_factory=datetime.now, description="Date and time when the indexer was created")
    last_updated: datetime = Field(default_factory=datetime.now, description="Date and time when the indexer was last updated")
    embeddings: dict[str, np.ndarray] = Field(description="Embeddings of the indexer")
    model_config = {
        "arbitrary_types_allowed": True
    }

    @abstractmethod
    def build(self, chunks: list[Chunk]):
        pass

    @abstractmethod
    def add(self, chunk_id: str, embedding: np.ndarray):
        pass

    @abstractmethod
    def delete(self, chunk_id: str):
        pass

    @abstractmethod
    def update(self, chunk_id: str, embedding: np.ndarray):
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5, distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN) -> list[str]:
        """
        Search for k-nearest neighbors using the query embedding
        
        Args:
            query_embedding: The query embedding vector
            k: Number of nearest neighbors to return (default: 5)
            distance_metric: Distance metric to use for similarity calculation
                - euclidean: Euclidean distance (L2 norm)
                - cosine: Cosine distance (1 - cosine similarity)
                - dot_product: Negative dot product (higher is more similar)
                - manhattan: Manhattan distance (L1 norm)
            
        Returns:
            List of k chunk ids that are nearest to the query embedding
        """
        pass

    @abstractmethod
    def get_dict_repr(self) -> dict:
        pass

    @abstractmethod
    def load_from_dict(self, dict_repr: dict):
        pass

    def _calculate_distance(
        self, 
        query_embedding: np.ndarray, 
        vector_embedding: np.ndarray, 
        distance_metric: DistanceMetric
    ) -> float:
        """
        Calculate distance between two embeddings using the specified metric
        
        Args:
            query_embedding: Query vector
            vector_embedding: Vector to compare against
            distance_metric: Distance metric to use
            
        Returns:
            Distance value (lower is more similar except for dot_product)
        """
        if distance_metric == DistanceMetric.EUCLIDEAN:
            # Euclidean distance (L2 norm)
            return np.linalg.norm(query_embedding - vector_embedding)
            
        elif distance_metric == DistanceMetric.COSINE:
            # Cosine distance = 1 - cosine similarity
            dot_product = np.dot(query_embedding, vector_embedding)
            query_norm = np.linalg.norm(query_embedding)
            vector_norm = np.linalg.norm(vector_embedding)
            
            # Avoid division by zero
            magnitude = query_norm * vector_norm
            if magnitude == 0:
                return 1.0  # Maximum distance
                
            similarity = dot_product / magnitude
            return 1.0 - similarity
            
        elif distance_metric == DistanceMetric.DOT_PRODUCT:
            # Negative dot product (to sort in ascending order)
            return -np.dot(query_embedding, vector_embedding)
            
        elif distance_metric == DistanceMetric.MANHATTAN:
            # Manhattan distance (L1 norm)
            return np.sum(np.abs(query_embedding - vector_embedding))
