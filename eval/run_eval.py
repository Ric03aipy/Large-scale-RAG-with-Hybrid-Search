# chiavi queries.jsonl: qid, type[semantick, keyword], query, gold_chunk_id, source, start_idx

import json
from eval.eval_config import EVAL_FOLDER
from enterprise_rag.pydantic_models import QueryModeEnum
from enterprise_rag.config import COLLECTION_NAME, QDRANT_URL
from qdrant_client import QdrantClient
from enterprise_rag.retriever import Retriever, CrossEncoderReranker
from eval.metrics import mrr_at_k, recall_at_k
from eval.eval_config import TEST_RRF_LIMIT, TEST_PREFETCH_LIMIT, METRIC_FILE, METRIC_RESULT_FILE, LATENCY_RESULT_FILE
import pandas as pd
import time
import argparse



def load_queries(): 
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
    return queries



def find_rank(retrieved_ids:list[str], gold_chunk_id:str) -> int | None: 
    for i, idx in enumerate(retrieved_ids): 
        if idx == gold_chunk_id: return i + 1 # rank 1-based --- if 10th elem then it has rank 10 - more interpretable
    return None



def run_retrieval(queries: list[dict], modes:list, retriever:Retriever, reranker:CrossEncoderReranker):
    """ Build statistics for metric computation by running the retrieval for each mode and over each query."""
    metric_data = []
    for mode in modes: 
        for query in queries: 
            start = time.perf_counter()
            query_response = retriever.retrieve(query["query"], mode, TEST_PREFETCH_LIMIT, TEST_RRF_LIMIT)

            retrieved_ids = [response.id for response in query_response.points]

            if mode == QueryModeEnum.hybrid: 
                # Append for hybrid
                end = time.perf_counter()
                metric_record = {
                    "qid": query["qid"],
                    "type": query["type"],
                    "mode": mode, 
                    "rank": find_rank(retrieved_ids, query["gold_chunk_id"]), 
                    "latency_ms": (end - start) * 1000, 
                    "retrieved_ids": retrieved_ids
                }
                metric_data.append(metric_record)

                # Apply reranker as well
                rerank_res = reranker.rerank(
                    query["query"], 
                    query_response, 
                    top_k=TEST_RRF_LIMIT    # this makes comparable the order for hybrid with and without rerank
                ) # context {"id": str, "text": str, "meta": dict, "score": np.float32}

                # Ensure ids are the same for hybrid and hybrid + rerank
                assert set(retrieved_ids) == {el["id"] for el in rerank_res}
                
                retrieved_ids = [el["id"] for el in rerank_res]
            end = time.perf_counter()

            metric_record = {
                "qid": query["qid"],
                "type": query["type"],
                "mode": mode if mode != QueryModeEnum.hybrid else QueryModeEnum.hybrid_rerank, 
                "rank": find_rank(retrieved_ids, query["gold_chunk_id"]), 
                "latency_ms": (end - start) * 1000, 
                "retrieved_ids": retrieved_ids
            }
            metric_data.append(metric_record)
    return metric_data



def compute_metrics(metric_df:pd.DataFrame):
    metric_res = []
    for (typ, mode), group in metric_df.groupby(["type", "mode"]):
        
        ranks:pd.Series = group["rank"]

        recall_at_5, hits_at_5 = recall_at_k(ranks, 5)
        recall_at_10, hits_at_10 = recall_at_k(ranks, 10)
        mrr_at_5 = mrr_at_k(ranks, 5)
        mrr_at_10 = mrr_at_k(ranks, 10)


        metric_res_record = {
            "n_queries": len(ranks),
            "type": typ,
            "mode": mode,
            "recall@5": recall_at_5, 
            "hits@5": hits_at_5,
            "hits@10": hits_at_10,
            "recall@10": recall_at_10,
            "mrr@5": mrr_at_5,
            "mrr@10":mrr_at_10
        }
        metric_res.append(metric_res_record)
    return metric_res



def compute_latency(metric_df:pd.DataFrame):
    latency_res = []
    latency_serie = metric_df.groupby("mode")["latency_ms"].quantile([0.5, 0.95])
    for mode in metric_df["mode"].unique():
        latency_res.append({
            "mode": mode, 
            "p50": latency_serie.loc[(mode, 0.50)],
            "p95": latency_serie.loc[(mode, 0.95)]
        })
    return latency_res


if __name__ == "__main__": 
    # Enable csv regeneration from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--regen', 
        action='store_true', 
        help="Regenerate csv of raw metrics."
    )
    args = parser.parse_args()
    
    if args.regen: 

        queries = load_queries()

        modes = [QueryModeEnum.sparse, QueryModeEnum.dense, QueryModeEnum.hybrid] # Hybrid is tested with and without reranker in the same step to avoid duouble runs
        retriever = Retriever()
        reranker = CrossEncoderReranker()

        # Warm up the system to measure latency without loading
        for mode in modes: 
            query_response = retriever.retrieve("hello world", mode, TEST_PREFETCH_LIMIT, TEST_RRF_LIMIT)
            if mode == QueryModeEnum.hybrid: rerank_res = reranker.rerank("hello world", query_response, top_k=TEST_RRF_LIMIT)

        metric_data = run_retrieval(queries, modes, retriever, reranker)
        
        metric_df = pd.DataFrame(metric_data)
        metric_df.to_csv(METRIC_FILE, index=False)

    # Metric computation 
    metric_df = pd.read_csv(METRIC_FILE)
    metric_res = compute_metrics(metric_df)

    metric_res_df = pd.DataFrame(metric_res)
    metric_res_df.to_csv(METRIC_RESULT_FILE, index=False)

    latency_res = compute_latency(metric_df)
    latency_df = pd.DataFrame(latency_res)
    latency_df.to_csv(LATENCY_RESULT_FILE, index=False)


