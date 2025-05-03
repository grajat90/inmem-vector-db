import numpy as np
from datetime import datetime
from threading import Lock
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Tuple, Set, Optional
import math

from indexers.indexer import DistanceMetric, Indexer
from models.chunk import Chunk


class LSHIndexer(Indexer):
    """
    Locality Sensitive Hashing (LSH) Indexer
    
    This indexer uses random hyperplane projections to hash vectors into buckets.
    Vectors that are similar are likely to be hashed to the same buckets.
    
    Time complexity:
    - Add: O(k*L) where k is hash size and L is number of hash tables
    - Search: O(k*L + n') where n' is the total number of candidates from all buckets
    
    Space complexity:
    - O(n*L) where n is number of items and L is number of hash tables
    
    Good for approximate nearest neighbor search in high-dimensional spaces.
    """
    
    def __init__(
        self,
        name: str = "lsh_indexer",
        hash_size: int = 8,      # Size of each hash
        num_tables: int = 10,    # Number of hash tables
        seed: int = 42          # Random seed for reproducibility
    ):
        super().__init__(name=name, embeddings={})
        
        # LSH parameters
        self._hash_size = hash_size
        self._num_tables = num_tables
        self._seed = seed
        
        # Initialize hash tables and their hyperplanes
        self._hash_tables = [{} for _ in range(num_tables)]
        self._hyperplanes = None
        
        # Dimensionality of embeddings (set when first vector is added)
        self._dim = None
        
        # Lock for thread safety
        self._lock = Lock()
        
    def build(self, chunks: list[Chunk]):
        """Build index from scratch with provided chunks"""
        with self._lock:
            # Reset the hash tables
            self._hash_tables = [{} for _ in range(self._num_tables)]
            self.embeddings = {}
            
            # Set the dimensionality if chunks are provided
            if chunks and hasattr(chunks[0], 'embedding') and chunks[0].embedding is not None:
                self._dim = len(chunks[0].embedding)
                self._initialize_hyperplanes()
            
            # Add chunks to the index
            for chunk in chunks:
                self.add(chunk.id, chunk.embedding, _skip_lock=True)
            
            self.last_updated = datetime.now()
    
    def add(self, chunk_id: str, embedding: np.ndarray, _skip_lock: bool = False):
        """Add a new item to the index
        
        Args:
            chunk_id: Unique identifier for the chunk
            embedding: Vector embedding for the chunk
            _skip_lock: Internal parameter. Set to True when called from a method that already holds the lock
        """
        if not _skip_lock:
            with self._lock:
                return self.add(chunk_id, embedding, _skip_lock=True)
        
        # Convert to numpy array if needed
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        # Initialize dimensionality and hyperplanes if this is the first vector
        if self._dim is None:
            self._dim = embedding.shape[0]
            self._initialize_hyperplanes()
        
        # Store the embedding
        self.embeddings[chunk_id] = embedding
        
        # Add to hash tables
        for table_idx, hash_table in enumerate(self._hash_tables):
            hash_value = self._hash_vector(embedding, table_idx)
            if hash_value not in hash_table:
                hash_table[hash_value] = set()
            hash_table[hash_value].add(chunk_id)
        
        self.last_updated = datetime.now()
    
    def delete(self, chunk_id: str, _skip_lock: bool = False):
        """Remove an item from the index
        
        Args:
            chunk_id: Unique identifier for the chunk to remove
            _skip_lock: Internal parameter. Set to True when called from a method that already holds the lock
        """
        if not _skip_lock:
            with self._lock:
                return self.delete(chunk_id, _skip_lock=True)
        
        if chunk_id not in self.embeddings:
            return
        
        embedding = self.embeddings[chunk_id]
        
        # Remove from hash tables
        for table_idx, hash_table in enumerate(self._hash_tables):
            hash_value = self._hash_vector(embedding, table_idx)
            if hash_value in hash_table and chunk_id in hash_table[hash_value]:
                hash_table[hash_value].remove(chunk_id)
                # Clean up empty buckets
                if len(hash_table[hash_value]) == 0:
                    del hash_table[hash_value]
        
        # Remove the embedding
        del self.embeddings[chunk_id]
        self.last_updated = datetime.now()
    
    def update(self, chunk_id: str, embedding: np.ndarray):
        """Update an existing chunk's embedding
        
        Args:
            chunk_id: Unique identifier for the chunk
            embedding: New vector embedding for the chunk
        """
        with self._lock:
            # If the chunk exists, delete it first
            if chunk_id in self.embeddings:
                self.delete(chunk_id, _skip_lock=True)
            
            # Then add it with the new embedding
            self.add(chunk_id, embedding, _skip_lock=True)
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5, 
        distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    ) -> list[str]:
        """Search for k-nearest neighbors using the query embedding
        
        Args:
            query_embedding: The query embedding vector
            k: Number of nearest neighbors to return (default: 5)
            distance_metric: Distance metric to use for similarity calculation
            
        Returns:
            List of k chunk ids that are nearest to the query embedding
        """
        # Convert to numpy array if needed
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        
        with self._lock:
            # If the index is empty, return empty list
            if not self.embeddings:
                return []
            
            # Find candidate chunks from hash tables
            candidates = set()
            for table_idx, hash_table in enumerate(self._hash_tables):
                hash_value = self._hash_vector(query_embedding, table_idx)
                if hash_value in hash_table:
                    candidates.update(hash_table[hash_value])
            
            # For very small datasets or bad hash functions, we might not have enough candidates
            # In that case, use all chunks as candidates
            if len(candidates) < k:
                candidates = set(self.embeddings.keys())
            
            # Calculate distances for all candidates
            distances = {}
            for chunk_id in candidates:
                distance = self._calculate_distance(
                    query_embedding, 
                    self.embeddings[chunk_id], 
                    distance_metric
                )
                distances[chunk_id] = distance
        
        # Sort by distance (ascending) and take top-k
        nearest_ids = sorted(distances.keys(), key=lambda x: distances[x])[:k]
        
        return nearest_ids
    
    def get_dict_repr(self) -> dict:
        """Get a complete dictionary representation of the indexer for serialization"""
        with self._lock:
            # Convert hash tables to a serializable format
            serialized_tables = []
            for table in self._hash_tables:
                serialized_table = {}
                for hash_val, chunk_ids in table.items():
                    serialized_table[hash_val] = list(chunk_ids)
                serialized_tables.append(serialized_table)
            
            # Properly serialize hyperplanes
            hyperplanes_serialized = None
            if self._hyperplanes is not None:
                hyperplanes_serialized = []
                for table_planes in self._hyperplanes:
                    serialized_planes = []
                    for plane in table_planes:
                        # Make sure we're calling tolist() only on numpy arrays
                        if isinstance(plane, np.ndarray):
                            serialized_planes.append(plane.tolist())
                        else:
                            serialized_planes.append(plane)
                    hyperplanes_serialized.append(serialized_planes)
            
            return {
                "name": self.name,
                "created": self.created.isoformat(),
                "last_updated": self.last_updated.isoformat(),
                "embeddings": {k: v.tolist() for k, v in self.embeddings.items()},
                "hash_size": self._hash_size,
                "num_tables": self._num_tables,
                "seed": self._seed,
                "hash_tables": serialized_tables,
                "hyperplanes": hyperplanes_serialized,
                "dim": self._dim
            }
    
    def load_from_dict(self, dict_repr: dict):
        """Restore the indexer from a dictionary representation"""
        try:
            # Validate with Pydantic
            config = LSHConfig(**dict_repr)
            
            # Apply validated data
            self.name = config.name
            self.created = datetime.fromisoformat(config.created)
            self.last_updated = datetime.fromisoformat(config.last_updated)
            
            # Convert embedding lists back to numpy arrays
            self.embeddings = {k: np.array(v, dtype=np.float32) for k, v in config.embeddings.items()}
            
            # Load LSH-specific parameters
            self._hash_size = config.hash_size
            self._num_tables = config.num_tables
            self._seed = config.seed
            self._dim = config.dim
            
            # Reconstruct hash tables
            self._hash_tables = []
            for table_dict in config.hash_tables:
                hash_table = {}
                for hash_val, chunk_ids in table_dict.items():
                    hash_table[hash_val] = set(chunk_ids)
                self._hash_tables.append(hash_table)
            
            # Reconstruct hyperplanes
            if config.hyperplanes:
                self._hyperplanes = []
                for table_planes in config.hyperplanes:
                    planes = [np.array(plane, dtype=np.float32) for plane in table_planes]
                    self._hyperplanes.append(planes)
            
        except Exception as e:
            raise ValueError(f"Failed to load LSH indexer from dictionary: {str(e)}")
    
    def _initialize_hyperplanes(self):
        """Initialize random hyperplanes for hashing"""
        np.random.seed(self._seed)
        # Create one set of random hyperplanes for each hash table
        # Each hyperplane is a unit vector in the embedding space
        self._hyperplanes = []
        for _ in range(self._num_tables):
            # Create hash_size random hyperplanes for this table
            planes = []
            for _ in range(self._hash_size):
                # Random hyperplane as a unit vector
                plane = np.random.randn(self._dim)
                # Normalize to unit length
                plane = plane / np.linalg.norm(plane)
                planes.append(plane)
            self._hyperplanes.append(planes)
    
    def _hash_vector(self, vec: np.ndarray, table_idx: int) -> str:
        """Hash a vector to a binary string using the specified hash table's hyperplanes
        
        Args:
            vec: Vector to hash
            table_idx: Index of the hash table to use
            
        Returns:
            Binary hash string
        """
        # Normalize the vector for cosine similarity
        vec_norm = vec / np.linalg.norm(vec)
        
        # Get the hyperplanes for this table
        planes = self._hyperplanes[table_idx]
        
        # Create hash: 1 if vec is on positive side of hyperplane, 0 otherwise
        hash_bits = []
        for plane in planes:
            # Dot product determines which side of hyperplane the vector is on
            bit = 1 if np.dot(vec_norm, plane) >= 0 else 0
            hash_bits.append(str(bit))
        
        # Join bits to form hash string
        return ''.join(hash_bits)
    
    def _multi_probe(self, hash_value: str, num_probes: int = 2) -> List[str]:
        """Generate additional hash values to probe by flipping bits
        
        Args:
            hash_value: Original hash value
            num_probes: Number of bit positions to flip
            
        Returns:
            List of hash values to probe
        """
        probe_hashes = [hash_value]
        
        # Only flip up to num_probes bits
        for i in range(min(num_probes, len(hash_value))):
            # Flip each bit position
            for pos in range(len(hash_value)):
                new_hash = list(hash_value)
                # Flip 0 to 1 or 1 to 0
                new_hash[pos] = '1' if new_hash[pos] == '0' else '0'
                probe_hashes.append(''.join(new_hash))
        
        return probe_hashes


class LSHConfig(BaseModel):
    name: str = Field(..., description="Name of the indexer")
    created: str = Field(..., description="Creation timestamp as ISO format string")
    last_updated: str = Field(..., description="Last updated timestamp as ISO format string")
    embeddings: Dict[str, List[float]] = Field(..., description="Embeddings dictionary")
    hash_size: int = Field(..., description="Size of each hash")
    num_tables: int = Field(..., description="Number of hash tables")
    seed: int = Field(..., description="Random seed for reproducibility")
    hash_tables: List[Dict[str, List[str]]] = Field(..., description="Hash tables")
    hyperplanes: Optional[List[List[List[float]]]] = Field(None, description="Hyperplanes for hashing")
    dim: Optional[int] = Field(None, description="Dimensionality of vectors")
    
    @field_validator('created', 'last_updated')
    @classmethod
    def validate_datetime(cls, v):
        """Validate that the timestamps are in ISO format"""
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid datetime format: {v}")
    
    @field_validator('hash_tables')
    @classmethod
    def validate_hash_tables(cls, v):
        """Validate hash tables structure"""
        if not isinstance(v, list):
            raise ValueError("Hash tables must be a list")
        
        for table in v:
            if not isinstance(table, dict):
                raise ValueError("Each hash table must be a dictionary")
                
            for hash_val, chunk_ids in table.items():
                if not isinstance(hash_val, str):
                    raise ValueError(f"Hash value must be a string: {hash_val}")
                    
                if not isinstance(chunk_ids, list):
                    raise ValueError(f"Chunk IDs must be a list: {chunk_ids}")
        
        return v
