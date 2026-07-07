"""
Tests for MeaningTags — semantic meaning reference vectors.

MeaningTags define the coordinate system for embedding space.
A text's meaning is determined by which meaning tag it's nearest to.
"""
import json
import os
import tempfile
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tags(dimension: int = 8) -> "MeaningTags":  # noqa: F821
    """Create a MeaningTags instance with test vectors."""
    from domains.infrastructure.anchor_store import MeaningTags
    store = MeaningTags(dimension=dimension)
    # Add tags along orthogonal axes
    vec_a = np.zeros(dimension, dtype=np.float32)
    vec_a[0] = 1.0
    store.add("factual", vec_a.tolist())

    vec_b = np.zeros(dimension, dtype=np.float32)
    vec_b[1] = 1.0
    store.add("interrogative", vec_b.tolist())

    vec_c = np.zeros(dimension, dtype=np.float32)
    vec_c[2] = 1.0
    store.add("procedural", vec_c.tolist())

    return store


# ---------------------------------------------------------------------------
# Add / Get
# ---------------------------------------------------------------------------

class TestMeaningTagsAddGet:
    def test_add_returns_none(self):
        store = _make_tags()
        assert store.add("new", [0.0] * 8) is None

    def test_get_returns_vector(self):
        store = _make_tags()
        vec = store.get("factual")
        assert vec is not None
        assert len(vec) == 8
        assert vec[0] == pytest.approx(1.0, abs=1e-6)

    def test_get_nonexistent_returns_none(self):
        store = _make_tags()
        assert store.get("nonexistent") is None

    def test_names_returns_registered(self):
        store = _make_tags()
        names = store.names()
        assert len(names) == 3
        assert "factual" in names
        assert "interrogative" in names
        assert "procedural" in names

    def test_add_wrong_dimension_raises(self):
        from domains.infrastructure.anchor_store import MeaningTags
        store = MeaningTags(dimension=8)
        with pytest.raises(ValueError, match="dim 4"):
            store.add("bad", [1.0, 0.0, 0.0, 0.0])

    def test_vectors_are_l2_normalized(self):
        store = _make_tags()
        for name in store.names():
            vec = store.get(name)
            norm = np.linalg.norm(vec)
            assert norm == pytest.approx(1.0, abs=1e-5), f"{name} not normalized"


# ---------------------------------------------------------------------------
# Distances / Classify / Similarity
# ---------------------------------------------------------------------------

class TestMeaningTagsClassify:
    def test_distances_computes_cosine_distance(self):
        store = _make_tags()
        # Vector aligned with factual (along axis 0)
        vec = [1.0] + [0.0] * 7
        dists = store.distances(vec)
        assert dists["factual"] == pytest.approx(0.0, abs=1e-5)
        assert dists["interrogative"] == pytest.approx(1.0, abs=1e-5)

    def test_classify_returns_nearest(self):
        store = _make_tags()
        # Vector near factual (axis 0)
        vec = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert store.classify(vec) == "factual"

    def test_classify_empty_returns_unknown(self):
        from domains.infrastructure.anchor_store import MeaningTags
        store = MeaningTags(dimension=8)
        assert store.classify([0.0] * 8) == "unknown"

    def test_classify_with_zero_vector(self):
        store = _make_tags()
        result = store.classify([0.0] * 8)
        # Zero vector: all distances equal, first one wins
        assert result in store.names()

    def test_similarity_between_vector_and_tag(self):
        store = _make_tags()
        vec = [1.0] + [0.0] * 7
        sim = store.similarity(vec, "factual")
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_similarity_nonexistent_returns_zero(self):
        store = _make_tags()
        vec = [1.0] + [0.0] * 7
        assert store.similarity(vec, "nonexistent") == 0.0


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

class TestMeaningTagsSaveLoad:
    def test_save_load_roundtrip(self):
        from domains.infrastructure.anchor_store import MeaningTags
        store = _make_tags()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            store.save(path)
            loaded = MeaningTags.load(path)
            assert loaded.dimension == store.dimension
            assert set(loaded.names()) == set(store.names())
            for name in store.names():
                assert np.allclose(loaded.get(name), store.get(name), atol=1e-6)
        finally:
            os.unlink(path)

    def test_load_nonexistent_returns_empty(self):
        from domains.infrastructure.anchor_store import MeaningTags
        loaded = MeaningTags.load("/tmp/nonexistent_meaning_tags_123.json")
        assert loaded.dimension == 128
        assert loaded.names() == []


# ---------------------------------------------------------------------------
# Remove / Mutate
# ---------------------------------------------------------------------------

class TestMeaningTagsMutate:
    def test_remove_existing(self):
        store = _make_tags()
        assert store.remove("factual") is True
        assert store.get("factual") is None
        assert len(store.names()) == 2

    def test_remove_nonexistent(self):
        store = _make_tags()
        assert store.remove("nonexistent") is False
        assert len(store.names()) == 3


# ---------------------------------------------------------------------------
# Default Tags
# ---------------------------------------------------------------------------

class TestMeaningTagsDefaults:
    def test_default_tags_create_seven(self):
        from domains.infrastructure.anchor_store import get_default_meaning_tags
        store = get_default_meaning_tags(dimension=16)
        assert len(store.names()) == 7

    def test_default_tags_are_normalized(self):
        from domains.infrastructure.anchor_store import get_default_meaning_tags
        store = get_default_meaning_tags(dimension=16)
        for name in store.names():
            vec = store.get(name)
            norm = np.linalg.norm(vec)
            assert norm == pytest.approx(1.0, abs=1e-5), f"{name} not normalized"

    def test_default_tags_include_expected_names(self):
        from domains.infrastructure.anchor_store import get_default_meaning_tags
        store = get_default_meaning_tags(dimension=16)
        expected = {"factual", "conceptual", "procedural", "interrogative",
                    "descriptive", "directive", "analytical"}
        assert set(store.names()) == expected


# ---------------------------------------------------------------------------
# Seed Determinism
# ---------------------------------------------------------------------------

class TestMeaningTagsDeterminism:
    def test_seed_deterministic(self):
        from domains.infrastructure.anchor_store import _seed_tag_from_text
        v1 = _seed_tag_from_text("test description", dimension=16)
        v2 = _seed_tag_from_text("test description", dimension=16)
        assert np.allclose(v1, v2, atol=1e-6)

    def test_different_descriptions_differ(self):
        from domains.infrastructure.anchor_store import _seed_tag_from_text
        v1 = _seed_tag_from_text("factual assertion", dimension=16)
        v2 = _seed_tag_from_text("interrogative question", dimension=16)
        assert not np.allclose(v1, v2, atol=1e-3)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestMeaningTagsEdgeCases:
    def test_distances_all_same_vector(self):
        store = _make_tags()
        # Vector identical to factual
        vec = [1.0] + [0.0] * 7
        dists = store.distances(vec)
        assert dists["factual"] == pytest.approx(0.0, abs=1e-5)
        # Other distances should be 1.0 (orthogonal)
        assert dists["interrogative"] == pytest.approx(1.0, abs=1e-5)
        assert dists["procedural"] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Integration: _label_by_meaning
# ---------------------------------------------------------------------------

class TestLabelByMeaning:
    def test_label_factual_statement(self):
        from domains.inference.slo_embedder import _label_by_meaning
        from domains.infrastructure.anchor_store import get_default_meaning_tags
        store = get_default_meaning_tags(dimension=128)
        label = _label_by_meaning("The sky is blue today", store)
        assert label in store.names()

    def test_label_question(self):
        from domains.inference.slo_embedder import _label_by_meaning
        from domains.infrastructure.anchor_store import get_default_meaning_tags
        store = get_default_meaning_tags(dimension=128)
        label = _label_by_meaning("What is the meaning of life?", store)
        assert label in store.names()

    def test_label_empty_text(self):
        from domains.inference.slo_embedder import _label_by_meaning
        from domains.infrastructure.anchor_store import get_default_meaning_tags
        store = get_default_meaning_tags(dimension=128)
        label = _label_by_meaning("", store)
        # Empty text may return None or a default label
        assert label is None or label in store.names()

    def test_label_none_store(self):
        from domains.inference.slo_embedder import _label_by_meaning
        label = _label_by_meaning("Hello world", None)
        assert label is None
