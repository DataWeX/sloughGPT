"""Tests for MeaningTags — fixed semantic reference vectors (anchor store)."""

import json
import os

import numpy as np
import pytest

from domains.infrastructure.anchor_store import (
    DEFAULT_MEANING_TAGS,
    MeaningTags,
    get_default_meaning_tags,
)


def _unit_vector(dim, seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return (v / np.linalg.norm(v)).tolist()


class TestMeaningTagsAddGet:
    def test_add_and_get(self):
        s = MeaningTags(dimension=4)
        vec = _unit_vector(4, 1)
        s.add("factual", vec)
        got = s.get("factual")
        assert got is not None
        assert np.allclose(got, vec)

    def test_get_missing_returns_none(self):
        s = MeaningTags()
        assert s.get("nope") is None

    def test_add_wrong_dimension_raises(self):
        s = MeaningTags(dimension=8)
        with pytest.raises(ValueError, match="dim"):
            s.add("factual", [0.1, 0.2])

    def test_add_zero_vector_keeps_zero(self):
        s = MeaningTags(dimension=4)
        s.add("zero", [0.0, 0.0, 0.0, 0.0])
        assert np.allclose(s.get("zero"), 0.0)

    def test_add_normalizes(self):
        s = MeaningTags(dimension=4)
        v = [3.0, 4.0, 0.0, 0.0]
        s.add("factual", v)
        assert np.linalg.norm(s.get("factual")) == pytest.approx(1.0)

    def test_names(self):
        s = MeaningTags(dimension=4)
        s.add("a", _unit_vector(4, 1))
        s.add("b", _unit_vector(4, 2))
        assert set(s.names()) == {"a", "b"}


class TestMeaningTagsDistances:
    def test_identical_direction_zero_distance(self):
        s = MeaningTags(dimension=8)
        v = _unit_vector(8, 1)
        s.add("t", v)
        assert s.distances(v)["t"] == pytest.approx(0.0, abs=1e-6)

    def test_opposite_direction_distance_two(self):
        s = MeaningTags(dimension=8)
        v = _unit_vector(8, 1)
        s.add("t", v)
        assert s.distances([-x for x in v])["t"] == pytest.approx(2.0, abs=1e-6)

    def test_empty_store_returns_empty(self):
        s = MeaningTags()
        assert s.distances(_unit_vector(8, 3)) == {}

    def test_all_tags_present_in_distances(self):
        s = get_default_meaning_tags()
        d = s.distances(_unit_vector(128, 7))
        assert set(d.keys()) == set(DEFAULT_MEANING_TAGS.keys())
        assert all(0.0 <= v <= 2.0 for v in d.values())


class TestMeaningTagsClassify:
    def test_classify_nearest(self):
        s = MeaningTags(dimension=8)
        a = _unit_vector(8, 1)
        b = _unit_vector(8, 2)
        s.add("a", a)
        s.add("b", b)
        assert s.classify(a) == "a"
        assert s.classify(b) == "b"

    def test_classify_empty_store_unknown(self):
        s = MeaningTags()
        assert s.classify(_unit_vector(8, 5)) == "unknown"

    def test_classify_closer_to_copied_vector(self):
        s = MeaningTags(dimension=8)
        base = _unit_vector(8, 1)
        near = _unit_vector(8, 2)
        far = _unit_vector(8, 3)
        s.add("near", near)
        s.add("far", far)
        blended = np.array(base) * 0.0 + np.array(near) * 0.99 + np.array(far) * 0.01
        assert s.classify(blended.tolist()) == "near"


class TestMeaningTagsSimilarity:
    def test_self_similarity_one(self):
        s = MeaningTags(dimension=8)
        v = _unit_vector(8, 1)
        s.add("t", v)
        assert s.similarity(v, "t") == pytest.approx(1.0, abs=1e-6)

    def test_missing_tag_zero(self):
        s = MeaningTags()
        assert s.similarity(_unit_vector(8, 1), "missing") == 0.0

    def test_similarity_symmetric_to_distance(self):
        s = MeaningTags(dimension=8)
        v1 = _unit_vector(8, 1)
        v2 = _unit_vector(8, 2)
        s.add("t", v1)
        sim = s.similarity(v2, "t")
        dist = s.distances(v2)["t"]
        assert sim == pytest.approx(1.0 - dist)


class TestMeaningTagsRemove:
    def test_remove_existing(self):
        s = MeaningTags(dimension=8)
        s.add("t", _unit_vector(8, 1))
        assert s.remove("t") is True
        assert s.get("t") is None

    def test_remove_missing_returns_false(self):
        s = MeaningTags()
        assert s.remove("t") is False


class TestMeaningTagsSaveLoad:
    def test_roundtrip(self, tmp_path):
        s = MeaningTags(dimension=8)
        s.add("a", _unit_vector(8, 1))
        s.add("b", _unit_vector(8, 2))
        path = str(tmp_path / "tags.json")
        s.save(path)
        loaded = MeaningTags.load(path)
        assert loaded.dimension == 8
        assert set(loaded.names()) == {"a", "b"}
        assert np.allclose(loaded.get("a"), s.get("a"))
        assert np.allclose(loaded.get("b"), s.get("b"))

    def test_save_file_is_json(self, tmp_path):
        s = MeaningTags(dimension=8)
        s.add("t", _unit_vector(8, 1))
        path = str(tmp_path / "tags.json")
        s.save(path)
        with open(path) as f:
            data = json.load(f)
        assert data["dimension"] == 8
        assert "t" in data["tags"]

    def test_load_missing_file_returns_empty(self, tmp_path):
        loaded = MeaningTags.load(str(tmp_path / "nope.json"))
        assert loaded.names() == []

    def test_save_creates_parent_dirs(self, tmp_path):
        s = MeaningTags(dimension=8)
        s.add("t", _unit_vector(8, 1))
        path = str(tmp_path / "deep" / "nested" / "tags.json")
        s.save(path)
        assert os.path.exists(path)


class TestMeaningTagsRefine:
    def test_refine_returns_used_counts(self):
        s = MeaningTags(dimension=8)
        s.add("a", _unit_vector(8, 1))
        s.add("b", _unit_vector(8, 2))
        rng = np.random.default_rng(0)
        embeddings = np.vstack(
            [
                rng.normal(s.get("a"), 0.01) for _ in range(5)
            ] + [
                rng.normal(s.get("b"), 0.01) for _ in range(5)
            ]
        )
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        texts = [f"text{i}" for i in range(10)]
        refined = s.refine(texts, embeddings, lr=0.5, min_samples=3)
        assert set(refined.keys()) == {"a", "b"}
        assert refined["a"] >= 3
        assert refined["b"] >= 3

    def test_refine_moves_tag_toward_centroid(self):
        s = MeaningTags(dimension=8)
        s.add("a", _unit_vector(8, 1))
        before = s.get("a").copy()
        rng = np.random.default_rng(99)
        offset = rng.standard_normal(8)
        offset /= np.linalg.norm(offset)
        embeddings = np.stack([before + 0.2 * offset for _ in range(5)])
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        s.refine(["x"] * 5, embeddings, lr=1.0, min_samples=3)
        after = s.get("a")
        assert not np.allclose(before, after)
        assert np.linalg.norm(after) == pytest.approx(1.0, abs=1e-6)

    def test_refine_empty_embeddings_returns_empty(self):
        s = MeaningTags(dimension=8)
        s.add("a", _unit_vector(8, 1))
        assert s.refine([], np.zeros((0, 8))) == {}

    def test_refine_below_min_samples_skips(self):
        s = MeaningTags(dimension=8)
        s.add("a", _unit_vector(8, 1))
        embeddings = np.stack([s.get("a") for _ in range(2)])
        assert s.refine(["x", "y"], embeddings, min_samples=3) == {}

    def test_refine_lr_zero_no_change(self):
        s = MeaningTags(dimension=8)
        s.add("a", _unit_vector(8, 1))
        before = s.get("a").copy()
        embeddings = np.stack([s.get("a") for _ in range(3)])
        refined = s.refine(["x"] * 3, embeddings, lr=0.0, min_samples=3)
        assert refined == {"a": 3}
        assert np.allclose(s.get("a"), before)

    def test_refine_skips_unknown_label(self):
        s = MeaningTags(dimension=8)
        embeddings = np.ones((3, 8))
        assert s.refine(["x"] * 3, embeddings, min_samples=2) == {}


class TestDefaultMeaningTags:
    def test_has_all_default_tags(self):
        s = get_default_meaning_tags()
        assert set(s.names()) == set(DEFAULT_MEANING_TAGS.keys())

    def test_custom_dimension(self):
        s = get_default_meaning_tags(dimension=64)
        assert s.dimension == 64
        assert all(v.shape == (64,) for v in s._tags.values())

    def test_default_dimension_128(self):
        s = get_default_meaning_tags()
        assert s.dimension == 128
        assert all(v.shape == (128,) for v in s._tags.values())

    def test_tags_are_normalized(self):
        s = get_default_meaning_tags()
        for name in s.names():
            assert np.linalg.norm(s.get(name)) == pytest.approx(1.0, abs=1e-6)

    def test_classification_is_deterministic(self):
        s = get_default_meaning_tags()
        v = _unit_vector(128, 42)
        assert s.classify(v) == s.classify(v)

    def test_tags_distinct(self):
        s = get_default_meaning_tags()
        names = s.names()
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                sim = s.similarity(s.get(a).tolist(), b)
                assert sim < 0.99, f"tags {a} and {b} too similar"
