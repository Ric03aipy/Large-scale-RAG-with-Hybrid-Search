import pandas as pd

def recall_at_k(ranks:list|pd.Series, k: int) -> tuple[float, int]: 
    """Returns the hit rate, the fraction of questions with rank lower or equal than k."""
    ranks = list(ranks)
    if not ranks:
        raise ValueError("Unexpected empty rank list. Check for bugs in the retrieval process.")
    count = 0
    for r in ranks: 
        if not pd.isna(r) and r <= k: count += 1
    return count / len(ranks), count


def mrr_at_k(ranks:list|pd.Series, k: int) -> float:
    """For each question compute the ratio 1/rank for success (rank <= k) and consider 0 for failures. Then average."""
    ranks = list(ranks)
    if not ranks: 
        raise ValueError("Unexpected empty rank list. Check for bugs in the retrieval process.")
    tot = 0
    for r in ranks: 
        if not pd.isna(r) and r <= k: tot += (1 / r)
    return tot / len(ranks)


