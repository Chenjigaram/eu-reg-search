import numpy as np
import pytest

from euregsearch.index.store import VectorStore, truncate_and_normalise
from euregsearch.provenance import ArticleRef


def ref(article: str) -> ArticleRef:
    return ArticleRef(celex="32014L0065", article=article, language="en",
                      version="consolidated", retrieved="2026-08-21", text=f"Article {article}")


def test_search_returns_the_nearest_vector_first():
    store = VectorStore()
    store.add([ref("1"), ref("2")], np.array([[1.0, 0.0], [0.0, 1.0]]))
    hits = store.search(np.array([0.9, 0.1]), k=2)
    assert hits[0][0].article == "1"


def test_search_returns_k_results():
    store = VectorStore()
    store.add([ref("1"), ref("2"), ref("3")], np.eye(3))
    assert len(store.search(np.array([1.0, 0.0, 0.0]), k=2)) == 2


def test_scores_are_descending():
    store = VectorStore()
    store.add([ref("1"), ref("2"), ref("3")], np.eye(3))
    scores = [s for _r, s in store.search(np.array([0.6, 0.5, 0.4]), k=3)]
    assert scores == sorted(scores, reverse=True)


def test_length_reflects_what_was_added():
    store = VectorStore()
    store.add([ref("1"), ref("2")], np.eye(2))
    assert len(store) == 2


def test_mismatched_counts_are_rejected():
    store = VectorStore()
    with pytest.raises(ValueError):
        store.add([ref("1")], np.eye(2))


def test_search_on_empty_store_returns_nothing():
    assert VectorStore().search(np.array([1.0, 0.0]), k=5) == []


def test_truncation_reduces_dimensions_and_renormalises():
    vectors = np.random.default_rng(0).normal(size=(4, 8))
    reduced = truncate_and_normalise(vectors, 4)
    assert reduced.shape == (4, 4)
    assert np.allclose(np.linalg.norm(reduced, axis=1), 1.0)


def test_truncation_to_full_width_is_a_no_op_in_shape():
    vectors = np.random.default_rng(1).normal(size=(3, 6))
    assert truncate_and_normalise(vectors, 6).shape == (3, 6)
