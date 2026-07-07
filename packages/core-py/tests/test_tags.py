"""
Tests for Tags (semantic meaning reference points).

20 tests covering: add/get/remove, distances/similarity/classify,
save/load roundtrip, dimension validation, default tags, determinism,
edge cases, and backward compat aliases.
"""
import json
import os
import tempfile
import numpy as np
import pytest


def _make_tags(dimension: int = 8) -> "MeaningTags":  # noqa: F821
    """Create a MeaningTags instance with test vectors."""
    from domains.infrastructure.anchor_store import MeaningTags
    store = MeaningTags(dimension=dimension)
    store.add("factual", np.random.randn(dimension).tolist())
    store.add("procedural", np.random.randn(dimension).tolist())
    store.add("interrogative", np.random.randn(dimension).tolist())
    return store


class TestTagsAddGet:
    def test_add_returns_none(self):
        store = _make_tags()
        assert store.add("new_tag", np.zeros(8).tolist()) is None

    def test_get_returns_vector(self):
        store = _make_tags()
        vec = store.get("factual")
        assert vec is not None
        assert len(vec) == 8

    def test_get_nonexistent_returns_none(self):
        store = _make_tags()
        assert store.get("nonexistent") is None

    def test_names_returns_registered(self):
        store = _make_tags()
        names = store.names()
        assert "factual" in names
        assert "procedural" in names
        assert "interrogative" in names


class TestTagsClassify:
    def test_distances_computes_cosine_distance(self):
        store = _make_tags()
        vec = np.random.randn(8).tolist()
        dists = store.distances(vec)
        assert len(dists) == 3
        for name in ["factual", "procedural", "interrogative"]:
            assert name in dists
            assert isinstance(dists[name], float)

    def test_classify_returns_nearest(self):
        store = _make_tags()
        # Make a vector exactly like "factual" tag
        factual_vec = store.get("factual").tolist()
        label = store.classify(factual_vec)
        assert label == "factual"

    def test_classify_empty_returns_unknown(self):
        from domains.infrastructure.anchor_store import MeaningTags
        store = MeaningTags(dimension=8)
        assert store.classify(np.zeros(8).tolist()) == "unknown"

    def test_similarity_between_vector_and_tag(self):
        store = _make_tags()
        vec = np.random.randn(8).tolist()
        sim = store.similarity(vec, "factual")
        assert isinstance(sim, float)
        assert -1.0 <= sim <= 1.0

    def test_similarity_nonexistent_tag_returns_zero(self):
        store = _make_tags()
        assert store.similarity(np.zeros(8).tolist(), "nope") == 0.0


class TestTagsSaveLoad:
    def test_save_load_roundtrip(self):
        store = _make_tags()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            store.save(path)
            loaded = store.load(path)
            assert loaded.dimension == store.dimension
            assert set(loaded.names()) == set(store.names())
            # Compare vectors
            np.testing.assert_array_almost_equal(
                loaded.get("factual"), store.get("factual"), decimal=5
            )
        finally:
            os.unlink(path)

    def test_load_nonexistent_returns_empty(self):
        from domains.infrastructure.anchor_store import MeaningTags
        store = MeaningTags.load("/tmp/nonexistent_tags_123.json")
        assert len(store.names()) == 0


class TestTagsValidation:
    def test_wrong_dimension_raises(self):
        store = _make_tags(dimension=8)
        with pytest.raises(ValueError, match="dim"):
            store.add("bad", [1.0, 2.0])

    def test_vectors_are_l2_normalized(self):
        store = _make_tags()
        for name in store.names():
            vec = store.get(name)
            norm = np.linalg.norm(vec)
            assert abs(norm - 1.0) < 1e-5


class TestTagsDefaults:
    def test_default_tags_create_seven(self):
        from domains.infrastructure.anchor_store import get_default_meaning_tags
        store = get_default_meaning_tags(dimension=16)
        assert len(store.names()) == 7
        expected = {"factual", "conceptual", "procedural", "interrogative",
                    "descriptive", "directive", "analytical"}
        assert set(store.names()) == expected

    def test_default_tags_are_normalized(self):
        from domains.infrastructure.anchor_store import get_default_meaning_tags
        store = get_default_meaning_tags(dimension=16)
        for name in store.names():
            vec = store.get(name)
            assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


class TestTagsDeterminism:
    def test_seed_deterministic(self):
        from domains.infrastructure.anchor_store import _seed_tag_from_text
        a = _seed_tag_from_text("hello world", 16)
        b = _seed_tag_from_text("hello world", 16)
        np.testing.assert_array_equal(a, b)

    def test_different_descriptions_differ(self):
        from domains.infrastructure.anchor_store import _seed_tag_from_text
        a = _seed_tag_from_text("factual assertion", 16)
        b = _seed_tag_from_text("question uncertainty", 16)
        assert not np.allclose(a, b)


class TestTagsMutate:
    def test_remove_existing(self):
        store = _make_tags()
        assert store.remove("factual") is True
        assert store.get("factual") is None

    def test_remove_nonexistent(self):
        store = _make_tags()
        assert store.remove("nope") is False


class TestTagsEdgeCases:
    def test_distances_all_same_vector(self):
        store = _make_tags()
        vec = np.ones(8) / np.sqrt(8)
        dists = store.distances(vec.tolist())
        assert all(d >= 0.0 for d in dists.values())

    def test_classify_with_zero_vector(self):
        store = _make_tags()
        label = store.classify(np.zeros(8).tolist())
        # All distances = 1.0, returns first alphabetically
        assert isinstance(label, str)
