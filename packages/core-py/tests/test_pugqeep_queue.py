"""Tests for domains.infrastructure.pugqeep.queue — ModelQueue."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from domains.infrastructure.pugqeep.queue import ModelQueue
from domains.infrastructure.pugqeep.model_tree import ModelTree
from domains.infrastructure.pugqeep.library import PointLibrary
from domains.infrastructure.pugqeep.config import QueueConfig, TreeConfig


@pytest.fixture
def queue():
    return ModelQueue()


@pytest.fixture
def queue_with_storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = QueueConfig(storage_dir=Path(tmpdir))
        q = ModelQueue(config=cfg)
        yield q


def _make_tree(name="t1", n_clusters=4):
    lib = PointLibrary(name=f"{name}_lib")
    tree = ModelTree(name, lib, n_clusters=n_clusters)
    return tree


class TestModelQueueAddRemove:
    def test_add_tree(self, queue):
        tree = _make_tree()
        added = queue.add_tree("t1", tree=tree)
        assert added is tree
        assert queue.get_tree("t1") is tree

    def test_add_tree_creates_from_config(self, queue):
        tree = queue.add_tree("t2")
        assert isinstance(tree, ModelTree)
        assert queue.get_tree("t2") is tree

    def test_list_trees(self, queue):
        queue.add_tree("a")
        queue.add_tree("b")
        assert set(queue.list_trees()) == {"a", "b"}

    def test_remove_tree(self, queue):
        queue.add_tree("a")
        assert queue.remove_tree("a") is True
        assert queue.get_tree("a") is None

    def test_remove_tree_missing(self, queue):
        assert queue.remove_tree("nope") is False

    def test_max_trees_exceeded(self):
        cfg = QueueConfig(max_trees=2)
        q = ModelQueue(config=cfg)
        q.add_tree("a")
        q.add_tree("b")
        with pytest.raises(ValueError, match="Queue full"):
            q.add_tree("c")


class TestModelQueueSharedLibrary:
    def test_shared_library_used(self, queue_with_storage):
        queue_with_storage.add_tree("a")
        queue_with_storage.add_tree("b")
        stats = queue_with_storage.stats()
        assert stats["shared_library"] is True

    def test_no_shared_library(self, queue):
        stats = queue.stats()
        assert stats["shared_library"] is False


class TestModelQueueCompress:
    def test_compress_and_store(self, queue):
        tree = queue.add_tree("t1")
        weights = np.random.randn(128)
        tree.library.compress_and_store(weights, identity="layer1")
        assert tree.library.has("layer1")

    def test_stats(self, queue):
        tree = queue.add_tree("t1")
        weights = np.random.randn(128)
        tree.library.compress_and_store(weights, identity="w1")
        stats = queue.stats()
        assert stats["num_trees"] == 1
        assert stats["total_points"] >= 1
        assert stats["total_raw_bytes"] > 0

    def test_stats_empty(self, queue):
        stats = queue.stats()
        assert stats["num_trees"] == 0
        assert stats["total_points"] == 0


class TestModelQueueDedup:
    def test_dedup_disabled(self):
        cfg = QueueConfig(dedup=False)
        q = ModelQueue(config=cfg)
        result = q.deduplicate()
        assert result["merged"] == 0

    def test_dedup_enabled(self, queue):
        queue.add_tree("a")
        queue.add_tree("b")
        result = queue.deduplicate()
        assert result["merged"] >= 0  # no error


class TestModelQueuePersistence:
    def test_save_all_and_load_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = QueueConfig(storage_dir=Path(tmpdir))
            q1 = ModelQueue(config=cfg)
            tree = q1.add_tree("m1")
            weights = np.random.randn(128)
            tree.library.compress_and_store(weights, identity="w1")

            q1.save_all(Path(tmpdir))

            q2 = ModelQueue(config=cfg)
            q2.load_all(Path(tmpdir))
            assert q2.get_tree("m1") is not None
            assert q2.get_tree("m1").library.has("w1")
