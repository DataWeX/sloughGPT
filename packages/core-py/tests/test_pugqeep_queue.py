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

    def test_add_tree_default_config(self, queue):
        tree = queue.add_tree("my_model")
        assert tree.name == "my_model"

    def test_add_tree_with_explicit_config(self, queue):
        cfg = TreeConfig(name="explicit", n_clusters=8)
        tree = queue.add_tree("t1", config=cfg)
        assert tree.n_clusters == 8

    def test_get_tree_missing(self, queue):
        assert queue.get_tree("nonexistent") is None

    def test_list_trees_empty(self, queue):
        assert queue.list_trees() == []

    def test_add_multiple_trees(self, queue):
        for i in range(5):
            queue.add_tree(f"t{i}")
        assert len(queue.list_trees()) == 5

    def test_remove_does_not_affect_others(self, queue):
        queue.add_tree("a")
        queue.add_tree("b")
        queue.remove_tree("a")
        assert queue.get_tree("b") is not None

    def test_add_tree_replaces_existing(self, queue):
        t1 = _make_tree(name="t1")
        t2 = _make_tree(name="t2")
        queue.add_tree("t1", tree=t1)
        queue.add_tree("t1", tree=t2)
        assert queue.get_tree("t1") is t2

    def test_add_tree_returns_model_tree(self, queue):
        tree = queue.add_tree("t1")
        assert isinstance(tree, ModelTree)

    def test_add_tree_name_preserved(self, queue):
        tree = queue.add_tree("my_custom_name")
        assert tree.name == "my_custom_name"

    def test_remove_tree_returns_false_twice(self, queue):
        queue.add_tree("a")
        queue.remove_tree("a")
        assert queue.remove_tree("a") is False

    def test_add_after_remove(self, queue):
        queue.add_tree("a")
        queue.remove_tree("a")
        tree = queue.add_tree("a")
        assert queue.get_tree("a") is tree

    def test_list_trees_order(self, queue):
        queue.add_tree("c")
        queue.add_tree("a")
        queue.add_tree("b")
        names = queue.list_trees()
        assert "a" in names
        assert "b" in names
        assert "c" in names

    def test_add_tree_with_none_tree(self, queue):
        tree = queue.add_tree("t1", tree=None)
        assert isinstance(tree, ModelTree)

    def test_add_tree_with_config_and_tree(self, queue):
        t = _make_tree(name="t1")
        cfg = TreeConfig(name="t1", n_clusters=8)
        tree = queue.add_tree("t1", tree=t, config=cfg)
        assert tree is t

    def test_many_trees(self, queue):
        for i in range(10):
            queue.add_tree(f"t{i}")
        assert len(queue.list_trees()) == 10


class TestModelQueueSharedLibrary:
    def test_shared_library_used(self, queue_with_storage):
        queue_with_storage.add_tree("a")
        queue_with_storage.add_tree("b")
        stats = queue_with_storage.stats()
        assert stats["shared_library"] is True

    def test_no_shared_library(self, queue):
        stats = queue.stats()
        assert stats["shared_library"] is False

    def test_shared_library_creates_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = QueueConfig(storage_dir=Path(tmpdir) / "subdir")
            q = ModelQueue(config=cfg)
            q.add_tree("a")
            assert q._shared_library is not None

    def test_shared_library_stores_points(self, queue_with_storage):
        queue_with_storage.add_tree("a")
        tree = queue_with_storage.get_tree("a")
        weights = np.random.randn(128)
        tree.library.compress_and_store(weights, identity="w1")
        assert tree.library.has("w1")

    def test_shared_library_visible_across_trees(self, queue_with_storage):
        t1 = queue_with_storage.add_tree("a")
        t2 = queue_with_storage.add_tree("b")
        assert t1.library is t2.library

    def test_shared_library_name(self, queue_with_storage):
        assert queue_with_storage._shared_library.name == "shared"


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

    def test_stats_multiple_trees(self, queue):
        for i in range(3):
            tree = queue.add_tree(f"t{i}")
            tree.library.compress_and_store(np.random.randn(64), identity=f"w{i}")
        stats = queue.stats()
        assert stats["num_trees"] == 3
        assert stats["total_points"] == 3

    def test_stats_ratio(self, queue):
        tree = queue.add_tree("t1")
        weights = np.random.randn(256)
        tree.library.compress_and_store(weights, identity="w1")
        stats = queue.stats()
        assert stats["ratio"] > 0

    def test_stats_compressed_bytes(self, queue):
        tree = queue.add_tree("t1")
        weights = np.random.randn(128)
        tree.library.compress_and_store(weights, identity="w1")
        stats = queue.stats()
        assert stats["total_compressed_bytes"] > 0

    def test_stats_trees_list(self, queue):
        queue.add_tree("a")
        queue.add_tree("b")
        stats = queue.stats()
        assert set(stats["trees"]) == {"a", "b"}

    def test_stats_multiple_weights(self, queue):
        tree = queue.add_tree("t1")
        for i in range(5):
            tree.library.compress_and_store(np.random.randn(64), identity=f"w{i}")
        stats = queue.stats()
        assert stats["total_points"] == 5

    def test_compress_stores_in_library(self, queue):
        tree = queue.add_tree("t1")
        weights = np.random.randn(128)
        point = tree.library.compress_and_store(weights, identity="w1")
        assert tree.library.get("w1") is not None

    def test_compress_cluster_method(self, queue):
        tree = queue.add_tree("t1")
        weights = np.random.randn(256)
        tree.library.compress_and_store(weights, identity="w1", method="cluster")
        assert tree.library.has("w1")

    def test_compress_function_method(self, queue):
        tree = queue.add_tree("t1")
        weights = np.linspace(0, 1, 256).astype(np.float32)
        tree.library.compress_and_store(weights, identity="w1", method="function")
        assert tree.library.has("w1")


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
        assert result["merged"] >= 0

    def test_dedup_no_trees(self, queue):
        result = queue.deduplicate()
        assert result["merged"] == 0
        assert result["bytes_saved"] == 0
        assert result["groups"] == 0

    def test_dedup_returns_dict(self, queue):
        queue.add_tree("a")
        result = queue.deduplicate()
        assert isinstance(result, dict)
        assert "merged" in result
        assert "bytes_saved" in result
        assert "groups" in result

    def test_dedup_disabled_returns_zeros(self):
        cfg = QueueConfig(dedup=False)
        q = ModelQueue(config=cfg)
        result = q.deduplicate()
        assert result["merged"] == 0
        assert result["bytes_saved"] == 0
        assert result["groups"] == 0

    def test_dedup_empty_queue(self, queue):
        result = queue.deduplicate()
        assert result["merged"] == 0
        assert result["groups"] == 0


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

    def test_save_all_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            q = ModelQueue()
            q.add_tree("t1")
            save_dir = Path(tmpdir) / "output"
            q.save_all(save_dir)
            assert save_dir.exists()

    def test_load_all_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            q = ModelQueue()
            q.load_all(Path(tmpdir))
            assert len(q.list_trees()) == 0

    def test_load_all_nonexistent_dir(self):
        q = ModelQueue()
        q.load_all(Path("/nonexistent/path/that/doesnt/exist"))
        assert len(q.list_trees()) == 0

    def test_save_load_roundtrip_multiple_trees(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            q1 = ModelQueue()
            for i in range(3):
                tree = q1.add_tree(f"m{i}")
                tree.library.compress_and_store(np.random.randn(64), identity=f"w{i}")

            q1.save_all(Path(tmpdir))

            q2 = ModelQueue()
            q2.load_all(Path(tmpdir))
            assert len(q2.list_trees()) == 3
            for i in range(3):
                assert q2.get_tree(f"m{i}") is not None

    def test_save_all_no_trees(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            q = ModelQueue()
            q.save_all(Path(tmpdir))
            assert Path(tmpdir).exists()

    def test_load_all_preserves_tree_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            q1 = ModelQueue()
            q1.add_tree("alpha")
            q1.add_tree("beta")
            q1.save_all(Path(tmpdir))

            q2 = ModelQueue()
            q2.load_all(Path(tmpdir))
            assert set(q2.list_trees()) == {"alpha", "beta"}

    def test_load_all_tree_is_loaded_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            q1 = ModelQueue()
            q1.add_tree("m1")
            q1.save_all(Path(tmpdir))

            q2 = ModelQueue()
            q2.load_all(Path(tmpdir))
            assert q2.get_tree("m1").is_loaded is True

    def test_save_load_preserves_points(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            q1 = ModelQueue()
            tree = q1.add_tree("m1")
            tree.library.compress_and_store(np.random.randn(128), identity="layer1")
            q1.save_all(Path(tmpdir))

            q2 = ModelQueue()
            q2.load_all(Path(tmpdir))
            assert q2.get_tree("m1").library.has("layer1")


class TestModelQueueDefaultConfig:
    def test_default_config(self, queue):
        assert queue.config.max_trees == 10
        assert queue.config.default_n_clusters == 16
        assert queue.config.dedup is True

    def test_custom_config(self):
        cfg = QueueConfig(max_trees=5, default_n_clusters=8, dedup=False)
        q = ModelQueue(config=cfg)
        assert q.config.max_trees == 5
        assert q.config.default_n_clusters == 8
        assert q.config.dedup is False

    def test_config_max_trees_zero(self):
        cfg = QueueConfig(max_trees=0)
        q = ModelQueue(config=cfg)
        with pytest.raises(ValueError):
            q.add_tree("a")

    def test_config_default_n_clusters(self):
        cfg = QueueConfig(default_n_clusters=32)
        q = ModelQueue(config=cfg)
        tree = q.add_tree("t1")
        assert tree.n_clusters == 32

    def test_config_storage_dir_none(self):
        q = ModelQueue()
        assert q._shared_library is None

    def test_config_dedup_true_by_default(self):
        q = ModelQueue()
        assert q.config.dedup is True


class TestModelQueueTreeInteraction:
    def test_trees_independent(self, queue):
        t1 = queue.add_tree("a")
        t2 = queue.add_tree("b")
        t1.library.compress_and_store(np.random.randn(64), identity="w1")
        assert t1.library.has("w1")
        assert not t2.library.has("w1")

    def test_get_tree_after_remove(self, queue):
        queue.add_tree("a")
        queue.remove_tree("a")
        queue.add_tree("a")
        assert queue.get_tree("a") is not None

    def test_stats_after_all_removed(self, queue):
        queue.add_tree("a")
        queue.remove_tree("a")
        stats = queue.stats()
        assert stats["num_trees"] == 0

    def test_multiple_add_remove_cycles(self, queue):
        for _ in range(5):
            queue.add_tree("a")
            queue.remove_tree("a")
        assert len(queue.list_trees()) == 0

    def test_remove_tree_decreases_count(self, queue):
        queue.add_tree("a")
        queue.add_tree("b")
        queue.remove_tree("a")
        stats = queue.stats()
        assert stats["num_trees"] == 1

    def test_add_tree_with_same_name_different_tree(self, queue):
        t1 = _make_tree(name="t1")
        t2 = _make_tree(name="t2")
        queue.add_tree("t1", tree=t1)
        queue.add_tree("t1", tree=t2)
        assert queue.get_tree("t1") is t2
        assert len(queue.list_trees()) == 1

    def test_stats_sum_across_trees(self, queue):
        t1 = queue.add_tree("a")
        t2 = queue.add_tree("b")
        t1.library.compress_and_store(np.random.randn(64), identity="w1")
        t2.library.compress_and_store(np.random.randn(64), identity="w2")
        stats = queue.stats()
        assert stats["total_points"] == 2
