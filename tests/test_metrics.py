import pytest

from euregsearch.evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k

A, B, C = ("32014L0065", "25"), ("32014R1286", "8"), ("32017R0565", "3")


def test_perfect_ranking_scores_one():
    assert ndcg_at_k([A, B], {A, B}, 10) == pytest.approx(1.0)


def test_nothing_relevant_retrieved_scores_zero():
    assert ndcg_at_k([C], {A}, 10) == 0.0
    assert recall_at_k([C], {A}, 10) == 0.0
    assert mrr_at_k([C], {A}, 10) == 0.0


def test_ndcg_rewards_higher_placement():
    assert ndcg_at_k([A, C], {A}, 10) > ndcg_at_k([C, A], {A}, 10)


def test_recall_counts_relevant_found():
    assert recall_at_k([A, C], {A, B}, 10) == pytest.approx(0.5)


def test_recall_respects_the_cutoff():
    assert recall_at_k([C, A], {A}, 1) == 0.0


def test_mrr_is_the_reciprocal_of_the_first_hit():
    assert mrr_at_k([C, A], {A}, 10) == pytest.approx(0.5)


def test_mrr_of_a_first_place_hit_is_one():
    assert mrr_at_k([A, C], {A}, 10) == pytest.approx(1.0)


def test_empty_retrieval_scores_zero():
    assert ndcg_at_k([], {A}, 10) == 0.0
    assert recall_at_k([], {A}, 10) == 0.0


def test_empty_relevant_set_scores_zero_not_crash():
    assert ndcg_at_k([A], set(), 10) == 0.0
    assert recall_at_k([A], set(), 10) == 0.0
