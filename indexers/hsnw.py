from pydantic import BaseModel, Field, field_validator
from indexers.indexer import DistanceMetric, Indexer
from models.chunk import Chunk
import numpy as np
import math
import random
from datetime import datetime
from typing import List, Optional, Tuple
from threading import Lock


class HSNWIndexer(Indexer):
    """
    Hierarchical Navigable Small World (HSNW) Indexer

    This indexer is basically a skip list of NSW graphs with full graph at the lowest level.

    Time complexity:
    - Add: O(n log(n))
    - Search: O(log(n))

    Worst case space complexity:
    - O(n log(n))

    Generally recommended for very large datasets. Might be overkill and actually inefficient
    for smaller datasets.
    """

    
    def __init__(self, 
                 name: str = "hsnw_indexer", 
                 m: int = 16,           # Maximum number of connections per node
                 ef_construction: int = 200,  # Size of dynamic candidate list during construction
                 max_level: int = 4,    # Maximum layer in the graph
                 level_mult: float = 1/math.log(2)):  # Level multiplier
        super().__init__(name=name, embeddings={})
        
        # HNSW parameters
        self._m = m # Maximum number of connections per node
        self._m_max0 = m * 2  # Max connections for ground layer (typically 2*m)
        self._ef_construction = ef_construction # Size of dynamic candidate list during construction
        self._max_level = max_level # Maximum layer in the graph
        self._level_mult = level_mult # Level multiplier
        
        # Graph structure: Dict of dicts for each level
        # {level: {node_id: set(neighbor_ids)}}
        self._graph = {level: {} for level in range(max_level + 1)}
        
        # Entry point to the graph (node with highest level)
        self._entry_point = None
        self._max_layer = 0
        
        # Node levels: node_id -> level
        self._node_levels = {}
        
        # Lock for thread safety
        self._lock = Lock()
        
        # Dimensionality of vectors (initialized when first vector is added)
        self._dim = None
        
    def build(self, chunks: list[Chunk]):
        """Build index from scratch with provided chunks"""
        with self._lock:
            # Reset the graph
            self._graph = {level: {} for level in range(self._max_level + 1)}
            self._entry_point = None
            self._max_layer = 0
            self._node_levels = {}
            self.embeddings = {}
            
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
        
        # Skip if already added
        if chunk_id in self.embeddings:
            return
        
        # Initialize dimensionality if this is the first vector
        if self._dim is None:
            self._dim = embedding.shape[0]
        
        # Normalize and store the embedding
        normalized_embedding = embedding / np.linalg.norm(embedding)
        self.embeddings[chunk_id] = normalized_embedding
        
        # Randomly select the level for the new element
        level = self._random_level()
        self._node_levels[chunk_id] = level
        
        # Update max level and entry point if needed
        if level > self._max_layer:
            self._max_layer = level
            self._entry_point = chunk_id
            
        # If this is the first element, it becomes the entry point and we're done
        if len(self.embeddings) == 1:
            # Initialize empty connection sets for all levels
            self._entry_point = chunk_id
            for l in range(level + 1):
                self._graph[l][chunk_id] = set()
            return
        
        # Find nearest neighbors to insert the new element in the graph
        curr_node = self._entry_point
        curr_dist = self._calculate_distance_raw(normalized_embedding, self.embeddings[curr_node], DistanceMetric.EUCLIDEAN)
        
        # Traverse from top to level l+1, searching for closest node
        for lc in range(self._max_layer, level, -1):
            changed = True
            while changed:
                changed = False
                
                # Check neighbors at this level
                neighbors = self._graph[lc][curr_node]
                for neighbor in neighbors:
                    dist = self._calculate_distance_raw(normalized_embedding, self.embeddings[neighbor], DistanceMetric.EUCLIDEAN)
                    if dist < curr_dist:
                        curr_dist = dist
                        curr_node = neighbor
                        changed = True
        
        # For each level from 'level' down to 0
        for lc in range(min(level, self._max_layer), -1, -1):
            # Find ef_construction nearest neighbors at level lc
            neighbors = self._search_level(normalized_embedding, curr_node, self._ef_construction, lc)
            
            # Initialize connections for the new node at this level
            if chunk_id not in self._graph[lc]:
                self._graph[lc][chunk_id] = set()
            
            # Select M nearest neighbors
            neighbors_to_connect = self._select_neighbors(normalized_embedding, neighbors, self._m if lc > 0 else self._m_max0)
            
            # Add bidirectional connections
            for neighbor_id, _ in neighbors_to_connect:
                self._graph[lc][chunk_id].add(neighbor_id)
                
                # Initialize if neighbor doesn't have connections at this level
                if neighbor_id not in self._graph[lc]:
                    self._graph[lc][neighbor_id] = set()
                
                self._graph[lc][neighbor_id].add(chunk_id)
                
                # Ensure nodes don't exceed max connections
                max_m = self._m if lc > 0 else self._m_max0
                if len(self._graph[lc][neighbor_id]) > max_m:
                    # Prune connections to keep only m/m_max0 closest
                    self._graph[lc][neighbor_id] = set(
                        [node_id for node_id, _ in self._select_neighbors(
                            self.embeddings[neighbor_id], 
                            [(nid, self._calculate_distance_raw(
                                self.embeddings[neighbor_id], 
                                self.embeddings[nid], 
                                DistanceMetric.EUCLIDEAN
                            )) for nid in self._graph[lc][neighbor_id]], 
                            max_m
                        )]
                    )
        
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
        
        # Get the level of the node to be deleted
        level = self._node_levels[chunk_id]
        
        # Remove connections at each level
        for l in range(level + 1):
            if l in self._graph and chunk_id in self._graph[l]:
                # Get all neighbors at this level
                neighbors = self._graph[l][chunk_id]
                
                # Remove bidirectional connections
                for neighbor in neighbors:
                    if neighbor in self._graph[l]:
                        self._graph[l][neighbor].discard(chunk_id)
                
                # Delete node from this level
                del self._graph[l][chunk_id]
        
        # Update entry point if needed
        if chunk_id == self._entry_point:
            if len(self.embeddings) > 1:
                # Find a new entry point (node with the highest level)
                max_level = -1
                new_entry = None
                for node_id, level in self._node_levels.items():
                    if node_id != chunk_id and level > max_level:
                        max_level = level
                        new_entry = node_id
                
                self._entry_point = new_entry
                self._max_layer = max_level
            else:
                # No more nodes in the graph
                self._entry_point = None
                self._max_layer = 0
        
        # Remove from storage
        del self.embeddings[chunk_id]
        del self._node_levels[chunk_id]
        
        self.last_updated = datetime.now()
    
    def update(self, chunk_id: str, embedding: np.ndarray):
        """Update an existing item in the index"""
        with self._lock:
            # Simple implementation: delete and add again
            self.delete(chunk_id, _skip_lock=True)
            self.add(chunk_id, embedding, _skip_lock=True)
    
    def search(self, query_embedding: np.ndarray, k: int = 5, distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN) -> list[str]:
        """Search for k-nearest neighbors using the query embedding"""
        with self._lock:
            if not self.embeddings or self._entry_point is None:
                return []
            
            if len(self.embeddings) <= k:
                # If we have fewer items than requested, return all
                return list(self.embeddings.keys())
            
            # Normalize query embedding
            query_normalized = query_embedding / np.linalg.norm(query_embedding)
            
            # Start from the entry point
            curr_node = self._entry_point
            curr_dist = self._calculate_distance_raw(query_normalized, self.embeddings[curr_node], distance_metric)
            
            # Traverse the graph from top level to bottom
            for level in range(self._max_layer, 0, -1):
                changed = True
                while changed:
                    changed = False
                    
                    # Check neighbors at this level
                    neighbors = self._graph[level][curr_node] if curr_node in self._graph[level] else set()
                    for neighbor in neighbors:
                        dist = self._calculate_distance_raw(query_normalized, self.embeddings[neighbor], distance_metric)
                        if dist < curr_dist:
                            curr_dist = dist
                            curr_node = neighbor
                            changed = True
            
            # Search for ef_construction closest nodes on the bottom layer
            ef_search = max(k, self._ef_construction)
            nearest = self._search_level(query_normalized, curr_node, ef_search, 0, distance_metric)
            
            # Return k closest nodes
            return [node_id for node_id, _ in nearest[:k]]
    
    def get_dict_repr(self) -> dict:
        """Get a complete dictionary representation of the indexer for serialization"""
        return {
            "name": self.name,
            "created": self.created.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "embeddings": {k: v.tolist() for k, v in self.embeddings.items()},
            "graph": self._graph,
            "entry_point": self._entry_point,
            "max_layer": self._max_layer,
            "node_levels": self._node_levels,
            "m": self._m,
            "m_max0": self._m_max0,
            "ef_construction": self._ef_construction,
            "max_level": self._max_level,
            "level_mult": self._level_mult,
            "dim": self._dim
        }
    
    def load_from_dict(self, dict_repr: dict):
        """Restore the indexer from a dictionary representation"""

        try:
            # Validate with Pydantic
            config = HSNWConfig(**dict_repr)
            
            # Apply validated data
            self.name = config.name
            self.created = config.created
            self.last_updated = config.last_updated
            
            # Convert embedding lists back to numpy arrays
            self.embeddings = {k: np.array(v) for k, v in config.embeddings.items()}
            
            self._graph = config.graph
            self._entry_point = config.entry_point
            self._max_layer = config.max_layer
            self._node_levels = config.node_levels
            self._m = config.m
            self._m_max0 = config.m_max0
            self._ef_construction = config.ef_construction
            self._max_level = config.max_level
            self._level_mult = config.level_mult
            self._dim = config.dim
            
        except Exception as e:
            raise ValueError(f"Failed to load HSNW indexer from dictionary: {str(e)}")
        
        
    # Internal helper methods
    
    def _random_level(self) -> int:
        """Generate a random level for a new node based on the level distribution"""
        # Probability distribution based on level_mult parameter
        # Higher level_mult = more levels
        r = random.random()
        level = int(-math.log(r) * self._level_mult)
        
        # Ensure level doesn't exceed max_level
        return min(level, self._max_level)
    
    def _search_level(self, 
                     query_vector: np.ndarray, 
                     entry_point: str, 
                     ef: int, 
                     level: int,
                     distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN) -> List[Tuple[str, float]]:
        """Search for nearest neighbors at a specific level"""
        # Initialize visited set
        visited = set([entry_point])
        
        # Distance from query to entry point
        distance_to_entry = self._calculate_distance_raw(query_vector, self.embeddings[entry_point], distance_metric)
        
        # Initialize candidate and nearest neighbor sets
        # Structure: [(node_id, distance)]
        candidates = [(entry_point, distance_to_entry)]  # Min heap for candidates (closest first)
        nearest = [(entry_point, distance_to_entry)]     # Max heap for results (farthest first)
        
        # Get furthest distance in result set
        def __get_furthest_distance():
            if not nearest:
                return float('inf')
            return max(d for _, d in nearest)
        
        while candidates:
            # Get closest candidate
            candidates.sort(key=lambda x: x[1])  # Sort by distance (closest first)
            curr_node, curr_dist = candidates.pop(0)
            
            # If current distance is greater than the furthest in our result set, we're done
            furthest_distance = __get_furthest_distance()
            if curr_dist > furthest_distance and len(nearest) >= ef:
                break
            
            # Check all neighbors of the current node at this level
            if curr_node in self._graph[level]:
                for neighbor in self._graph[level][curr_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        
                        # Calculate distance to neighbor
                        dist = self._calculate_distance_raw(query_vector, self.embeddings[neighbor], distance_metric)
                        
                        # If we haven't found ef neighbors yet or if the neighbor is closer than our furthest neighbor
                        if len(nearest) < ef or dist < furthest_distance:
                            # Add to candidates
                            candidates.append((neighbor, dist))
                            
                            # Add to result set
                            nearest.append((neighbor, dist))
                            
                            # If we have too many results, remove the furthest one
                            if len(nearest) > ef:
                                nearest.sort(key=lambda x: x[1])  # Sort by distance (closest first)
                                nearest = nearest[:ef]              # Keep only ef closest
        
        # Sort by distance (closest first) and return
        nearest.sort(key=lambda x: x[1])
        return nearest
    
    def _select_neighbors(self, 
                         query_vector: np.ndarray, 
                         candidates: List[Tuple[str, float]], 
                         m: int) -> List[Tuple[str, float]]:
        """Select m nearest neighbors from candidates"""
        # Simple implementation: just select m closest
        candidates.sort(key=lambda x: x[1])  # Sort by distance (closest first)
        return candidates[:m]
    
    def _calculate_distance_raw(self, 
                              vec1: np.ndarray, 
                              vec2: np.ndarray, 
                              distance_metric: DistanceMetric) -> float:
        """Calculate distance between two vectors using the specified metric"""
        return super()._calculate_distance(vec1, vec2, distance_metric)




class HSNWConfig(BaseModel):
    name: str = Field(..., description="Name of the indexer")
    created: str = Field(..., description="Creation timestamp as ISO format string")
    last_updated: str = Field(..., description="Last updated timestamp as ISO format string")
    embeddings: dict[str, list[float]] = Field(..., description="Embeddings dictionary")
    graph: dict[int, dict[str, set[str]]] = Field(..., description="Graph structure")
    entry_point: Optional[str] = Field(None, description="Entry point node ID")
    max_layer: int = Field(..., description="Maximum layer in the graph")
    node_levels: dict[str, int] = Field(..., description="Node levels mapping")
    m: int = Field(..., description="Maximum number of connections per node")
    m_max0: int = Field(..., description="Maximum connections for ground layer")
    ef_construction: int = Field(..., description="Size of dynamic candidate list during construction")
    max_level: int = Field(..., description="Maximum layer in the graph")
    level_mult: float = Field(..., description="Level multiplier")
    dim: Optional[int] = Field(None, description="Dimensionality of vectors")
    
    @field_validator('created', 'last_updated')
    @classmethod
    def validate_datetime(cls, v):
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid datetime format: {v}")
    
    @field_validator('graph')
    @classmethod
    def validate_graph(cls, v):
        for level, nodes in v.items():
            if not isinstance(level, int) or level < 0:
                raise ValueError(f"Invalid level {level} in graph")
            
            for node_id, neighbors in nodes.items():
                if not isinstance(node_id, str):
                    raise ValueError(f"Node ID must be a string, got {type(node_id)}")
                
                if not isinstance(neighbors, set) and not isinstance(neighbors, list):
                    raise ValueError(f"Neighbors must be a set or list, got {type(neighbors)}")
                
                # Convert lists to sets if needed
                if isinstance(neighbors, list):
                    v[level][node_id] = set(neighbors)
        return v
    
    @field_validator('embeddings')
    @classmethod
    def validate_embeddings(cls, v):
        if not v:
            return v
            
        # Check consistency of embedding dimensions
        dims = [len(emb) for emb in v.values()]
        if len(set(dims)) > 1:
            raise ValueError(f"Inconsistent embedding dimensions: {set(dims)}")
        return v