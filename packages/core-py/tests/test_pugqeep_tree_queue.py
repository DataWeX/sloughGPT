"""Tests for domains/infrastructure/pugqeep/model_tree.py and queue.py."""

import base64
from pathlib import Path

import numpy as np
import pytest

import domains.infrastructure.numpy_engine as numpy_engine
import domains.infrastructure.pugqeep.model_tree as model_tree_module
import domains.infrastructure.pugqeep.tree as tree_module
from domains.infrastructure.pugqeep.model_tree import ModelTree
from domains.infrastructure.pugqeep.tree import (
    save_library,
    load_library,
    load_from_points,
    load_model_to_points,
    decompress_tree,
)
from domains.infrastructure.pugqeep.queue import ModelQueue
from domains.infrastructure.pugqeep.config import QueueConfig, TreeConfig
from domains.infrastructure.pugqeep.library import PointLibrary
from domains.infrastructure.pugqeep.point import Point


def _weights(n=256):
    rng = np.random.RandomState(0)
    return {
        "weight": rng.randn(n).astype(np.float32),
        "bias": rng.randn(4).astype(np.float32),
        "embedding": rng.randn(50, 8).astype(np.float32),
    }


# ---- ModelTree -----------------------------------------------------------


def test_tree_init_defaults():
    tree = ModelTree("m1")
    assert tree.name == "m1"
    assert tree.library.name == "m1_points"
    assert tree.n_clusters == 16
    assert tree._method == "cluster"
    assert tree._skip_embeddings is True
    assert tree._skip_biases is True
    assert tree.is_loaded is False


def test_tree_init_with_existing_library():
    lib = PointLibrary(name="shared")
    tree = ModelTree("m1", library=lib)
    assert tree.library is lib


def test_tree_init_with_config():
    cfg = TreeConfig(name="m1", n_clusters=8, method="function", skip_embeddings=False, skip_biases=False)
    tree = ModelTree("m1", n_clusters=16, config=cfg)
    assert tree.n_clusters == 8
    assert tree._method == "function"
    assert tree._skip_embeddings is False
    assert tree._skip_biases is False


def test_load_weights_skips_embeddings_and_biases():
    tree = ModelTree("m1", n_clusters=4)
    stats = tree.load_weights(_weights(n=256))
    assert stats["model"] == "m1"
    assert stats["num_weights"] == 3
    assert stats["method"] == "cluster"
    assert stats["total_raw_bytes"] > 0
    assert tree.is_loaded is True

    emb = tree.library.get("m1.embedding")
    assert emb.function_type == "raw"
    bias = tree.library.get("m1.bias")
    assert bias.function_type == "raw"
    weight = tree.library.get("m1.weight")
    assert weight.function_type == "cluster"


def test_load_weights_small_cluster_is_raw():
    tree = ModelTree("m1", n_clusters=8)
    tree.load_weights({"w": np.random.RandomState(1).randn(3).astype(np.float32)})
    assert tree.library.get("m1.w").function_type == "raw"


def test_load_weights_function_method():
    tree = ModelTree("m1", n_clusters=4)
    tree.load_weights(_weights(n=256), method="function")
    assert tree.library.get("m1.weight").function_type in ("linear", "polynomial", "periodic")


def test_get_weight_raw_round_trip():
    tree = ModelTree("m1")
    raw = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    tree.load_weights({"w": raw})
    got = tree.get_weight("w")
    np.testing.assert_allclose(got, raw, atol=1e-6)


def test_get_weight_cluster_regenerates():
    tree = ModelTree("m1", n_clusters=4)
    w = _weights(n=256)["weight"]
    tree.load_weights({"w": w})
    got = tree.get_weight("w")
    assert got.shape == (256,)
    assert np.isfinite(got).all()


def test_get_weight_missing_returns_none():
    tree = ModelTree("m1")
    assert tree.get_weight("nope") is None


def test_get_weight_uses_recorded_shape():
    tree = ModelTree("m1", n_clusters=4)
    w = np.random.RandomState(2).randn(8, 8).astype(np.float32)
    tree.load_weights({"w": w})
    got = tree.get_weight("w")
    assert got.shape == (8, 8)


def test_estimate_size():
    tree = ModelTree("m1")
    tree._weight_shapes["w"] = (4, 4)
    assert tree._estimate_size("w") == 16
    # Unknown weight returns 0 (safer than arbitrary fallback)
    assert tree._estimate_size("missing") == 0


def test_tree_stats():
    tree = ModelTree("m1")
    tree.load_weights(_weights())
    stats = tree.stats()
    assert stats["model"] == "m1"
    assert stats["loaded"] is True
    assert stats["num_weights"] == 3
    assert "library" in stats


def test_load_weights_counts_cluster_residual_bytes(monkeypatch):
    tree = ModelTree("m1", n_clusters=4)
    original = tree._compressor.compress_cluster

    def with_residual(flat, point_id, n_clusters):
        point = original(flat, point_id, n_clusters)
        point.residual = np.ones(len(flat), dtype=np.float32)
        return point

    monkeypatch.setattr(tree._compressor, "compress_cluster", with_residual)
    stats = tree.load_weights(_weights(n=256))
    cluster = tree.library.get("m1.weight")
    assert cluster.function_type == "cluster"
    assert cluster.residual is not None
    assert stats["total_compressed_bytes"] >= cluster.residual.nbytes


# ---- save/load helpers ----------------------------------------------------


def test_save_and_load_library(tmp_path):
    lib = PointLibrary(name="l")
    lib.compress_and_store(np.random.RandomState(3).randn(8).astype(np.float32), "w")
    path = save_library(lib, tmp_path / "lib.points.json")
    loaded = load_library(path)
    assert loaded.name == "l"
    assert len(loaded.list_all()) == 1


def test_load_from_points_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_from_points(str(tmp_path / "nonexistent"))


def test_load_from_points_round_trip(tmp_path):
    tree = ModelTree("mymodel")
    w = _weights(n=64)
    tree.load_weights(w)
    lib_path = tmp_path / "mymodel.points.json"
    tree.library.save(lib_path)
    (tmp_path / "mymodel.meta.json").write_text(
        '{"metadata": {"weight_shapes": {"weight": [64], "bias": [4], "embedding": [50, 8]}}}'
    )

    loaded, meta = load_from_points(str(tmp_path / "mymodel"))
    assert loaded.name == "mymodel"
    assert loaded.is_loaded is True
    assert meta == {"metadata": {"weight_shapes": {"weight": [64], "bias": [4], "embedding": [50, 8]}}}
    assert loaded._weight_shapes["weight"] == (64,)


def test_decompress_tree_round_trip():
    tree = ModelTree("m1", n_clusters=4)
    w = _weights(n=256)
    tree.load_weights(w)
    weights = decompress_tree(tree)
    assert set(weights.keys()) == {"weight", "bias", "embedding"}
    assert weights["weight"].shape == (256,)
    assert weights["bias"].shape == (4,)
    assert weights["embedding"].shape == (50, 8)
    np.testing.assert_allclose(weights["bias"], w["bias"], atol=1e-6)


def test_load_model_to_points_creates_own_library(monkeypatch):
    def fake_load_weights(model_id):
        return ({"arch": "test"}, {
            "w": np.ones(32, dtype=np.float32),
            "bias": np.zeros(4, dtype=np.float32),
        })

    monkeypatch.setattr(numpy_engine, "_load_weights", fake_load_weights)
    tree = load_model_to_points("fake", n_clusters=4)
    assert tree.name == "fake"
    assert tree.library.name == "fake"
    assert tree.library.get("fake.w").function_type == "cluster"
    assert tree.library.get("fake.bias").function_type == "raw"
    assert tree.is_loaded is True


def test_load_model_to_points_uses_provided_library(monkeypatch):
    def fake_load_weights(model_id):
        return ({"arch": "test"}, {"w": np.ones(32, dtype=np.float32)})

    monkeypatch.setattr(numpy_engine, "_load_weights", fake_load_weights)
    lib = PointLibrary(name="custom")
    tree = load_model_to_points("fake", library=lib, n_clusters=4)
    assert tree.library is lib


def test_load_from_points_nonprefixed_identity(tmp_path):
    lib = PointLibrary(name="mymodel")
    lib.compress_and_store(np.arange(16, dtype=np.float32), "foreign_w")
    lib.save(tmp_path / "mymodel.points.json")

    loaded, _ = load_from_points(str(tmp_path / "mymodel"))
    assert "foreign_w" in loaded._weight_shapes
    assert loaded._weight_shapes["foreign_w"] == ()


def test_decompress_tree_nonprefixed_identity():
    lib = PointLibrary(name="lib")
    lib.compress_and_store(np.arange(16, dtype=np.float32), "foreign_w")
    tree = ModelTree("m1", library=lib)
    tree._weight_shapes["foreign_w"] = (16,)
    weights = decompress_tree(tree)
    assert "foreign_w" in weights
    assert weights["foreign_w"].shape == (16,)


# ---- ModelQueue ------------------------------------------------------------


def test_queue_init_defaults():
    q = ModelQueue()
    assert q.config.max_trees == 10
    assert q._trees == {}
    assert q._shared_library is None


def test_queue_add_tree_creates():
    q = ModelQueue()
    tree = q.add_tree("a")
    assert q.get_tree("a") is tree
    assert tree.library.name == "a_points"


def test_queue_add_tree_explicit():
    q = ModelQueue()
    lib = PointLibrary(name="custom")
    tree = ModelTree("a", lib)
    assert q.add_tree("a", tree=tree) is tree
    assert q.get_tree("a") is tree


def test_queue_add_tree_config():
    q = ModelQueue()
    tree = q.add_tree("a", config=TreeConfig(name="a", n_clusters=6))
    assert tree.n_clusters == 6


def test_queue_max_trees_limit():
    q = ModelQueue(QueueConfig(max_trees=2))
    q.add_tree("a")
    q.add_tree("b")
    with pytest.raises(ValueError, match="Queue full"):
        q.add_tree("c")


def test_queue_storage_dir_creates_shared_library(tmp_path):
    q = ModelQueue(QueueConfig(storage_dir=tmp_path))
    assert q._shared_library is not None
    q.add_tree("a")
    assert q.get_tree("a").library is q._shared_library


def test_queue_list_and_remove_tree():
    q = ModelQueue()
    q.add_tree("a")
    q.add_tree("b")
    assert q.list_trees() == ["a", "b"]
    assert q.remove_tree("a") is True
    assert q.remove_tree("a") is False
    assert q.list_trees() == ["b"]


def test_queue_get_missing_tree():
    q = ModelQueue()
    assert q.get_tree("nope") is None


def test_queue_dedup_disabled():
    q = ModelQueue(QueueConfig(dedup=False))
    result = q.deduplicate()
    assert result == {"merged": 0, "bytes_saved": 0, "groups": 0}


def test_queue_dedup_merges_identical_points():
    q = ModelQueue()
    a = q.add_tree("a")
    b = q.add_tree("b")
    identical = np.random.RandomState(5).randn(32).astype(np.float32)
    a.library.compress_and_store(identical.copy(), "a_w")
    b.library.compress_and_store(identical.copy(), "b_w")
    result = q.deduplicate()
    assert result["merged"] == 1
    assert result["bytes_saved"] > 0


def test_queue_stats():
    q = ModelQueue()
    q.add_tree("a")
    q.get_tree("a").load_weights(_weights())
    stats = q.stats()
    assert stats["num_trees"] == 1
    assert stats["trees"] == ["a"]
    assert stats["total_points"] == 3
    assert stats["ratio"] > 0
    assert stats["shared_library"] is False


def test_queue_save_and_load_all(tmp_path):
    q = ModelQueue()
    tree = q.add_tree("a")
    tree.load_weights(_weights(n=64))
    q.save_all(tmp_path)

    q2 = ModelQueue()
    q2.load_all(tmp_path)
    assert q2.list_trees() == ["a"]
    assert len(q2.get_tree("a").library.list_all()) == 3
    assert q2.get_tree("a").is_loaded is True


def test_queue_load_model(monkeypatch):
    q = ModelQueue()
    lib = PointLibrary(name="fake_points")
    tree = ModelTree("fake", lib)
    monkeypatch.setattr(tree_module, "load_model_to_points", lambda *a, **k: tree)

    loaded = q.load_model("fake")
    assert loaded is tree
    assert q.get_tree("fake") is tree
    assert q.list_trees() == ["fake"]
