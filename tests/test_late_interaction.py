import numpy as np

from euregsearch.index.late_interaction import LateInteractionStore, maxsim
from euregsearch.provenance import ArticleRef


def unit(rows):
    matrix = np.array(rows, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def ref(article: str) -> ArticleRef:
    return ArticleRef(celex="32014L0065", article=article, language="en",
                      version="original", retrieved="2026-08-25", text="x")


def test_identical_tokens_score_one_each():
    q = unit([[1, 0], [0, 1]])
    assert maxsim(q, q) == 2.0


def test_each_query_token_takes_its_best_match():
    q = unit([[1, 0]])
    d = unit([[0, 1], [1, 0], [0, 1]])
    assert maxsim(q, d) == 1.0


def test_orthogonal_document_scores_zero():
    assert maxsim(unit([[1, 0]]), unit([[0, 1]])) == 0.0


def test_a_single_matching_token_is_not_averaged_away():
    """The point of late interaction: one strong match survives a long document."""
    q = unit([[1, 0]])
    short = unit([[1, 0]])
    padded = unit([[1, 0]] + [[0, 1]] * 200)
    assert maxsim(q, short) == maxsim(q, padded)


def test_empty_document_scores_zero():
    assert maxsim(unit([[1, 0]]), np.empty((0, 2), dtype=np.float32)) == 0.0


def test_store_ranks_by_score():
    store = LateInteractionStore()
    store.add([ref("1"), ref("2")], [unit([[0, 1]]), unit([[1, 0]])])
    hits = store.search(unit([[1, 0]]), k=2)
    assert [r.article for r, _ in hits] == ["2", "1"]


def test_store_respects_k():
    store = LateInteractionStore()
    store.add([ref("1"), ref("2")], [unit([[1, 0]]), unit([[1, 0]])])
    assert len(store.search(unit([[1, 0]]), k=1)) == 1


def test_mismatched_inputs_are_rejected():
    store = LateInteractionStore()
    try:
        store.add([ref("1")], [])
    except ValueError:
        return
    raise AssertionError("expected ValueError")
