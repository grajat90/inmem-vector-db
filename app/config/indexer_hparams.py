import math
from functools import lru_cache

from pydantic import BaseModel, Field

from app.api.schemas.library import IndexerType


class HNSWIndexerHParams(BaseModel):
    m: int = Field(default=16, description="Maximum number of connections per node")
    ef_construction: int = Field(
        default=200, description="Size of dynamic candidate list during construction"
    )
    max_level: int = Field(default=4, description="Maximum layer in the graph")
    level_mult: float = Field(default=1 / math.log(2), description="Level multiplier")


class LSHIndexerHParams(BaseModel):
    hash_size: int = Field(default=8, description="Size of each hash")
    num_tables: int = Field(default=10, description="Number of hash tables")


@lru_cache
def get_default_hparams(indexer_type: IndexerType) -> BaseModel:
    if indexer_type == IndexerType.HNSW:
        return HNSWIndexerHParams()
    elif indexer_type == IndexerType.LSH:
        return LSHIndexerHParams()
    else:
        raise ValueError(f"Invalid indexer type: {indexer_type}")
