from pathlib import Path

import pandas as pd


def chunk_key(source: str, start_index: int) -> str:
    """Portable chunk identifier, e.g. 'pep-0555@20614'.

    It depends only on the file name and on the position of the chunk in the file,
    not on the absolute path where the file was ingested nor on the Qdrant point id.
    """
    clean_source = str(source).replace("\\", "/")
    return f"{Path(clean_source).stem}@{int(start_index)}"


def find_rank(retrieved_keys: list[str], gold_key: str) -> int | None:
    """1-based position of the gold chunk in the retrieved list, None if it is absent."""
    for i, key in enumerate(retrieved_keys):
        if key == gold_key:
            return i + 1
    return None


def recall_at_k(ranks: list | pd.Series, k: int) -> tuple[float, int]:
    """Returns the hit rate, the fraction of questions with rank lower or equal than k."""
    ranks = list(ranks)
    if not ranks:
        raise ValueError(
            "Unexpected empty rank list. Check for bugs in the retrieval process."
        )
    count = 0
    for r in ranks:
        if not pd.isna(r) and r <= k:
            count += 1
    return count / len(ranks), count


def mrr_at_k(ranks: list | pd.Series, k: int) -> float:
    """For each question compute the ratio 1/rank for success (rank <= k) and consider 0 for failures. Then average."""
    ranks = list(ranks)
    if not ranks:
        raise ValueError(
            "Unexpected empty rank list. Check for bugs in the retrieval process."
        )
    tot = 0
    for r in ranks:
        if not pd.isna(r) and r <= k:
            tot += 1 / r
    return tot / len(ranks)
