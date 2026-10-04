from config import DEFAULT_RRF_LIMIT, DEFAULT_TOP_K, DEFAULT_PREFETCH_LIMIT
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    prefetch_limit: int = DEFAULT_PREFETCH_LIMIT
    rrf_limit: int = DEFAULT_RRF_LIMIT
    top_k: int = DEFAULT_TOP_K


# obj.model_dump() transforms the Model in a dict with keys the fields and values the values
