# chiavi queries.jsonl: qid, type[semantick, keyword], query, gold_chunk_id, source, start_idx

import json
from eval.eval_config import EVAL_FOLDER
from enterprise_rag.pydantic_models import QueryModeEnum
from enterprise_rag.config import COLLECTION_NAME, QDRANT_URL
from qdrant_client import QdrantClient
from enterprise_rag.retriever import Retriever, CrossEncoderReranker
from eval.metrics import mrr_at_k, recall_at_k
from eval.eval_config import TEST_RRF_LIMIT, TEST_PREFETCH_LIMIT, METRIC_FILE, METRIC_RESULT_FILE
import pandas as pd
import time
import argparse

if __name__ == "__main__": 
    # Enable csv regeneration from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--regen', 
        action='store_true', 
        help="Abilita la modalità solo test"
    )

    args = parser.parse_args()
    
    if args.regen: 

        # Open connection
        client = QdrantClient(url=QDRANT_URL)

        # Count how many chunks
        collection_chunks = client.count( 
            collection_name=COLLECTION_NAME, 
            exact=True
        ).count

        # Fetch all chunks
        content = client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=True,
            with_vectors=False,
            # Default limit of the API is 10, so write the limit as the exact number of chunks to fetch all
            limit=collection_chunks
        )[0] # -> ([records], None)
        
        # Check all golden_chunk_id are existing + collecting queries
        ids = set([record.id for record in content])
        queries = []
        with open(EVAL_FOLDER / "queries.jsonl", "r") as f: 
            for row in f: 
                dict_row = json.loads(row)
                assert dict_row["gold_chunk_id"] in ids, "Gold_chunks_id not present. Inconsistency between queries and data"
                queries.append(dict_row)

        # Build statistics for metric computation
        metric_data = []

        modes = list(QueryModeEnum) # Take all QueryModeEnum objects 
        retriever = Retriever()
        reranker = CrossEncoderReranker()

        def find_rank(retrieved_ids:list[str], gold_chunk_id:str) -> int | None: 
            for i, idx in enumerate(retrieved_ids): 
                if idx == gold_chunk_id: return i + 1 # rank 1-based --- if 10th elem then it has rank 10 - more interpretable
            return None

        # Warm up the system to measure latency without loading
        for mode in modes: 
            query_response = retriever.retrieve("hello world", mode, TEST_PREFETCH_LIMIT, TEST_RRF_LIMIT)
            if mode == QueryModeEnum.hybrid_rerank: rerank_res = reranker.rerank("hello world", query_response, top_k=TEST_RRF_LIMIT)

        for mode in modes: 
            for query in queries: 
                start = time.time()
                query_response = retriever.retrieve(query["query"], mode, TEST_PREFETCH_LIMIT, TEST_RRF_LIMIT)

                retrieved_ids = [response.id for response in query_response.points]

                if mode == QueryModeEnum.hybrid_rerank: 
                    rerank_res = reranker.rerank(
                        query["query"], 
                        query_response, 
                        top_k=TEST_RRF_LIMIT    # this makes comparable the order for hybrid with and without rerank
                    ) # context {"id": str, "text": str, "meta": dict, "score": np.float32}
                    # These two are equals, which means ids are kept invariant, what changes is the order
                    # print(retrieved_ids)
                    # print([el["id"] for el in rerank_res])
                    
                    retrieved_ids = [el["id"] for el in rerank_res]
                end = time.time()

                metric_record = {
                    "qid": query["qid"],
                    "type": query["type"],
                    "mode": mode, 
                    "rank": find_rank(retrieved_ids, query["gold_chunk_id"]), 
                    "latency_ms": (end - start) * 1000, 
                    "retrieved_ids": retrieved_ids
                }

                metric_data.append(metric_record)

        metric_df = pd.DataFrame(metric_data)
        metric_df.to_csv(METRIC_FILE, index=False)

    # Metric computation 
    metric_df = pd.read_csv(METRIC_FILE)
    metric_res = []
    for (typ, mode), group in metric_df.groupby(["type", "mode"]):
        
        ranks:pd.Series = group["rank"]

        recall_at_5 = recall_at_k(ranks, 5)
        recall_at_10 = recall_at_k(ranks, 10)
        mrr_at_5 = mrr_at_k(ranks, 5)
        mrr_at_10 = mrr_at_k(ranks, 10)

        metric_res_record = {
            "type": typ,
            "mode": mode,
            "recall@5": recall_at_5, 
            "recall@10": recall_at_10,
            "mrr@5": mrr_at_5,
            "mrr@10":mrr_at_10
        }
        metric_res.append(metric_res_record)
    metric_res_df = pd.DataFrame(metric_res)
    metric_res_df.to_csv(METRIC_RESULT_FILE, index=False)


