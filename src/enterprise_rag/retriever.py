from typing import Any

from config import (
    CACHE_DIR,
    COLLECTION_NAME,
    CROSS_ENCODER_MODEL_NAME,
    DEFAULT_RRF_LIMIT,
    DEFAULT_TOP_K,
    DEFAULT_PREFETCH_LIMIT,
    DENSE_MODEL,
    QDRANT_URL,
    SPARSE_MODEL,
)
from flashrank import Ranker, RerankRequest
from qdrant_client import QdrantClient
from qdrant_client.models import Document as QDocument
from qdrant_client.models import Fusion, FusionQuery, Prefetch, QueryResponse


class CrossEncoderReranker:
    def __init__(self):
        """Initialize the ranker. The cache dir specified is the root. You will find at cache_dir + 'rerank_models'."""
        self.ranker = Ranker(model_name=CROSS_ENCODER_MODEL_NAME, cache_dir=CACHE_DIR)

    def rerank(
        self, query: str, results: QueryResponse, top_k: int = DEFAULT_TOP_K
    ) -> list[dict[str, Any]]:
        """Takes results of the hybrid search and the new ranking."""
        passages = [
            {
                "id": response.id,
                "text": response.payload["page_content"],
                "meta": response.payload["metadata"],  # this is a dictionary
            }
            for response in results.points
        ]

        rerankrequest = RerankRequest(query=query, passages=passages)
        reranked_results = self.ranker.rerank(rerankrequest)

        return reranked_results[
            :top_k
        ]  # context {"id": str, "text": str, "meta": dict, "score": np.float32}


class Retriever:
    """Abstract the logic of Retrieval in RAG."""

    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)
        self.collection_name = COLLECTION_NAME
        self.dense_model = DENSE_MODEL
        self.sparse_model = SPARSE_MODEL

    def retrieve(
        self,
        query: str,
        prefetch_limit: int = DEFAULT_PREFETCH_LIMIT,
        rrf_limit: int = DEFAULT_RRF_LIMIT,
    ) -> QueryResponse:
        """Retrieve points by cosine similarity with the query."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=QDocument(text=query, model=self.dense_model),
                    using="dense_vector",
                    limit=prefetch_limit,
                ),
                Prefetch(
                    query=QDocument(text=query, model=self.sparse_model),
                    using="bm25_sparse_vector",
                    limit=prefetch_limit,
                ),
            ],  # score here is COSINE similarity
            query=FusionQuery(fusion=Fusion.RRF),  # score here is ranking
            limit=rrf_limit,
            with_payload=True,
        )
        return results
