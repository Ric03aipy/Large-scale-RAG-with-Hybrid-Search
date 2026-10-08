import pandas as pd

def recall_at_k(ranks:list|pd.Series, k: int) -> float: 
    """Returns the hit rate, the fraction of questions with rank lower or equal than k."""
    ranks = ranks.tolist()
    if not ranks: 
        raise ValueError("Unexpected empty rank list. Check for bugs in the retrieval process.")
    count = 0
    for r in ranks: 
        if r <= k: count += 1
    return count / len(ranks)


def mrr_at_k(ranks:list|pd.Series, k: int) -> float:
    """
    For each question compute the ratio 1/rank for success (rank <= k) and consider 0 for failures. Then average.
    MRR (mean reciprocal rank) is a fundamental metric for retrieval since llm suffers from position bias, that is they give more importance to elements at the beginning.
    """
    ranks = ranks.tolist()
    if not ranks: 
        raise ValueError("Unexpected empty rank list. Check for bugs in the retrieval process.")
    tot = 0
    for r in ranks: 
        if r <= k: tot += (1 / r)
    return tot / len(ranks)


