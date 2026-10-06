from enum import Enum

from pydantic import BaseModel

from enterprise_rag.config import (
    DEFAULT_PREFETCH_LIMIT,
    DEFAULT_RRF_LIMIT,
    DEFAULT_TOP_K,
)


class QueryModeEnum(str, Enum):
    dense = "dense"
    sparse = "sparse"
    hybrid = "hybrid"
    hybrid_rerank = "hybrid_rerank"


class QueryRequest(BaseModel):
    query: str
    prefetch_limit: int = DEFAULT_PREFETCH_LIMIT
    rrf_limit: int = DEFAULT_RRF_LIMIT
    top_k: int = DEFAULT_TOP_K
    mode: QueryModeEnum = QueryModeEnum.hybrid_rerank


# obj.model_dump() transforms the Model in a dict with keys the fields and values the values
