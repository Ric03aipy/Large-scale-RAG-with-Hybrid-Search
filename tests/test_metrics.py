from eval.metrics import recall_at_k, mrr_at_k
from eval.run_eval import find_rank
import pandas as pd
import pytest

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