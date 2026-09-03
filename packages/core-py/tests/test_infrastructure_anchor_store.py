"""Tests for MeaningTags — semantic reference vector store."""
from __future__ import annotations

import numpy as np
import pytest

from domains.infrastructure.anchor_store import MeaningTags


@pytest.fixture()
def store() -> MeaningTags:
    return MeaningTags(dimension=4)


@pytest.fixture()
def populated_store() -> MeaningTags:
    s = MeaningTags(dimension=3)
    s.add("factual", [1.0, 0.0, 0.0])
    s.add("emotional", [0.0, 1.0, 0.0])
    s.add("procedural", [0.0, 0.0, 1.0])
    return s


class TestAdd:
    def test_add_tag(self, store: MeaningTags):
        store.add("test", [1.0, 0.0, 0.0, 0.0])
        assert "test" in store.names()

    def test_add_normalizes(self, store: MeaningTags):
        store.add("test", [3.0, 0.0, 0.0, 0.0])
        vec = store.get("test")
        assert vec is not None
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-6

    def test_add_wrong_dim_raises(self, store: MeaningTags):
        with pytest.raises(ValueError, match="dim"):
            store.add("bad", [1.0, 0.0])

    def test_add_zero_vector(self, store: MeaningTags):
        store.add("zero", [0.0, 0.0, 0.0, 0.0])
        vec = store.get("zero")
        assert vec is not None
        assert np.linalg.norm(vec) == 0.0


class TestGet:
    def test_get_existing(self, populated_store: MeaningTags):
        vec = populated_store.get("factual")
        assert vec is not None
        assert len(vec) == 3

    def test_get_nonexistent(self, populated_store: MeaningTags):
        assert populated_store.get("nope") is None


class TestNames:
    def test_names(self, populated_store: MeaningTags):
        names = populated_store.names()
        assert set(names) == {"factual", "emotional", "procedural"}

    def test_names_empty(self, store: MeaningTags):
        assert store.names() == []


class TestDistances:
    def test_identical_direction(self, populated_store: MeaningTags):
        dists = populated_store.distances([1.0, 0.0, 0.0])
        assert dists["factual"] == pytest.approx(0.0, abs=1e-6)

    def test_opposite_direction(self, populated_store: MeaningTags):
        dists = populated_store.distances([-1.0, 0.0, 0.0])
        assert dists["factual"] == pytest.approx(2.0, abs=1e-6)

    def test_perpendicular(self, populated_store: MeaningTags):
        dists = populated_store.distances([0.0, 1.0, 0.0])
        assert dists["factual"] == pytest.approx(1.0, abs=1e-6)

    def test_returns_all_tags(self, populated_store: MeaningTags):
        dists = populated_store.distances([1.0, 0.0, 0.0])
        assert len(dists) == 3


class TestClassify:
    def test_classify_nearest(self, populated_store: MeaningTags):
        result = populated_store.classify([1.0, 0.1, 0.0])
        assert result == "factual"

    def test_classify_exact(self, populated_store: MeaningTags):
        result = populated_store.classify([0.0, 0.0, 1.0])
        assert result == "procedural"
