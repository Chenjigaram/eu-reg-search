import numpy as np

from euregsearch.index.cache import VectorCache


def test_cold_cache_reports_everything_as_missing():
    cache = VectorCache.empty("model-a")
    known, missing = cache.split(["one", "two"])
    assert known == {}
    assert missing == ["one", "two"]


def test_stored_vectors_come_back():
    cache = VectorCache.empty("model-a")
    cache.store(["one", "two"], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    known, missing = cache.split(["two", "one"])
    assert missing == []
    assert np.allclose(known["one"], [1.0, 0.0])
    assert np.allclose(known["two"], [0.0, 1.0])


def test_partial_hit_only_reports_new_text_as_missing():
    cache = VectorCache.empty("model-a")
    cache.store(["one"], np.array([[1.0, 0.0]], dtype=np.float32))
    known, missing = cache.split(["one", "three"])
    assert list(known) == ["one"]
    assert missing == ["three"]


def test_a_different_model_does_not_share_vectors(tmp_path):
    path = tmp_path / "vectors.npz"
    a = VectorCache(path, "model-a")
    a.store(["one"], np.array([[1.0, 0.0]], dtype=np.float32))
    a.save()
    b = VectorCache(path, "model-b")
    b.load()
    _known, missing = b.split(["one"])
    assert missing == ["one"]


def test_round_trip_through_disk(tmp_path):
    path = tmp_path / "vectors.npz"
    a = VectorCache(path, "model-a")
    a.store(["one", "two"], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    a.save()
    b = VectorCache(path, "model-a")
    b.load()
    known, missing = b.split(["one", "two"])
    assert missing == []
    assert np.allclose(known["one"], [1.0, 0.0])


def test_loading_a_missing_file_is_not_an_error(tmp_path):
    cache = VectorCache(tmp_path / "absent.npz", "model-a")
    cache.load()
    assert cache.split(["one"])[1] == ["one"]
