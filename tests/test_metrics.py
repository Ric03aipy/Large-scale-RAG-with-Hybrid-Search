import pandas as pd
import pytest

from eval.metrics import chunk_key, find_rank, mrr_at_k, recall_at_k

EPSILON = 10e-3


def test_metrics():
    serie = [1, 3, None, 6, 2]
    # Expected Recall@5 = 0,6, MRR@5 ≈ 0,367, Recall@10 = 0,8, MRR@10 = 0,4;

    # Test on values
    assert abs(recall_at_k(serie, 5)[0] - 0.6) < EPSILON
    assert abs(recall_at_k(serie, 10)[0] - 0.8) < EPSILON
    assert abs(mrr_at_k(serie, 5) - 0.367) < EPSILON
    assert abs(mrr_at_k(serie, 10) - 0.4) < EPSILON

    # Test on count
    assert recall_at_k(serie, 5)[1] == 3
    assert recall_at_k(serie, 10)[1] == 4

    # Test on types
    serie = pd.Series(serie)
    assert abs(recall_at_k(serie, 10)[0] - 0.8) < EPSILON
    assert abs(mrr_at_k(serie, 5) - 0.367) < EPSILON

    # Rank exactly = k
    serie = [5]
    assert recall_at_k(serie, 5)[0] == 1
    assert recall_at_k(serie, 5)[1] == 1

    # Empty list
    with pytest.raises(ValueError):
        recall_at_k([], 5)


def test_find_rank():
    ids = list("abcdghi")
    existing_golden_id = "c"
    non_existing_golden_id = "z"
    first = "a"
    last = "i"
    assert find_rank(ids, existing_golden_id) == 3
    assert find_rank(ids, first) == 1
    assert find_rank(ids, last) == len(ids)
    assert find_rank(ids, non_existing_golden_id) is None


def test_chunk_key():
    # same key whatever the absolute path where the file was ingested
    assert chunk_key("/home/a/data/pep-0555.rst", 20614) == "pep-0555@20614"
    assert chunk_key("/other/place/pep-0555.rst", 20614) == "pep-0555@20614"
    # the stem stored in queries.jsonl gives the same key
    assert chunk_key("pep-0555", 20614) == "pep-0555@20614"
    # Windows-style path
    assert chunk_key("C:\\data\\pep-0555.rst", 1) == "pep-0555@1"
    # different position -> different key
    assert chunk_key("pep-0555", 1) != chunk_key("pep-0555", 2)
