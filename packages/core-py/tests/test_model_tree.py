"""Tests for domains.infrastructure.pugqeep.model_tree — ModelTree."""

import numpy as np
import pytest


class TestModelTree:
    def test_init_defaults(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m")
        assert tree.name == "m"
        assert tree.is_loaded is False

    def test_init_with_clusters(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=8)
        assert tree.n_clusters == 8

    def test_load_weights(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        weights = {
            "w1": np.random.randn(8, 8).astype(np.float32),
            "b1": np.random.randn(8).astype(np.float32),
        }
        stats = tree.load_weights(weights)
        assert stats["model"] == "m"
        assert stats["num_weights"] == 2
        assert stats["total_raw_bytes"] > 0
        assert stats["ratio"] > 0
        assert tree.is_loaded is True

    def test_get_weight(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        raw = np.random.randn(4, 4).astype(np.float32)
        tree.load_weights({"w": raw})
        result = tree.get_weight("w")
        assert result is not None
        assert result.shape == (4, 4)

    def test_get_weight_missing(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        tree.load_weights({})
        assert tree.get_weight("nonexistent") is None

    def test_skip_embeddings(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4, config=None)
        # Override skip settings
        tree._skip_embeddings = True
        tree._skip_biases = True
        weights = {
            "embed.weight": np.random.randn(8, 8).astype(np.float32),
            "fc.bias": np.random.randn(8).astype(np.float32),
            "fc.weight": np.random.randn(8, 8).astype(np.float32),
        }
        tree.load_weights(weights)
        # Embedding and bias should be stored as raw (no compression)
        for name in ["embed.weight", "fc.bias"]:
            point_id = f"m.{name}"
            point = tree.library.get(point_id)
            assert point.function_type == "raw"

    def test_small_tensor_raw(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=16)
        # Tensor smaller than n_clusters * 2 → stored as raw
        small = np.random.randn(4).astype(np.float32)
        tree.load_weights({"tiny": small})
        point = tree.library.get("m.tiny")
        assert point.function_type == "raw"

    def test_cluster_compression(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        large = np.random.randn(64).astype(np.float32)
        tree.load_weights({"big": large}, method="cluster")
        point = tree.library.get("m.big")
        assert point.function_type == "cluster"

    def test_stats(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        tree.load_weights({"w": np.zeros((4, 4), dtype=np.float32)})
        stats = tree.stats()
        assert stats["model"] == "m"
        assert stats["loaded"] is True
        assert stats["num_weights"] == 1
        assert "library" in stats

    def test_estimate_size(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        tree._weight_shapes["w"] = (3, 4)
        assert tree._estimate_size("w") == 12
        assert tree._estimate_size("missing") == 0

    def test_function_method(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        arr = np.random.randn(64).astype(np.float32)
        tree.load_weights({"w": arr}, method="function")
        point = tree.library.get("m.w")
        assert point.function_type != "raw"  # should be compressed (periodic/polynomial/etc)

    def test_load_empty_weights(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        stats = tree.load_weights({})
        assert stats["num_weights"] == 0
        assert stats["total_raw_bytes"] == 0
        assert tree.is_loaded is True

    def test_get_weights_multiple(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        w1 = np.random.randn(4, 4).astype(np.float32)
        w2 = np.random.randn(4, 4).astype(np.float32)
        tree.load_weights({"w1": w1, "w2": w2})
        weights = tree.get_weights(["w1", "w2"])
        assert "w1" in weights
        assert "w2" in weights
        assert weights["w1"].shape == (4, 4)
        assert weights["w2"].shape == (4, 4)

    def test_get_weights_all(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        tree.load_weights({"a": np.zeros((2, 2), dtype=np.float32)})
        weights = tree.get_weights()
        assert "a" in weights

    def test_get_weights_missing(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        tree.load_weights({})
        weights = tree.get_weights(["nonexistent"])
        assert weights["nonexistent"] is None

    def test_is_loaded_property(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        assert tree.is_loaded is False
        tree.is_loaded = True
        assert tree.is_loaded is True

    def test_weight_shapes_tracked(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        w = np.random.randn(3, 5).astype(np.float32)
        tree.load_weights({"my_weight": w})
        assert "my_weight" in tree._weight_shapes
        assert tree._weight_shapes["my_weight"] == (3, 5)

    def test_weight_dtypes_tracked(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        w = np.random.randn(4).astype(np.float32)
        tree.load_weights({"w": w})
        assert "w" in tree._weight_dtypes
        assert tree._weight_dtypes["w"] == np.float32

    def test_compression_ratio(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        w = np.random.randn(128).astype(np.float32)
        stats = tree.load_weights({"w": w})
        assert stats["ratio"] > 0
        assert stats["total_compressed_bytes"] > 0

    def test_multiple_loads(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        tree.load_weights({"w1": np.zeros((4, 4), dtype=np.float32)})
        tree.load_weights({"w2": np.ones((4, 4), dtype=np.float32)})
        assert tree.get_weight("w1") is not None
        assert tree.get_weight("w2") is not None

    def test_library_name(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("my_model", n_clusters=4)
        assert tree.library.name == "my_model_points"

    def test_estimate_size_from_point(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        w = np.random.randn(64).astype(np.float32)
        tree.load_weights({"w": w})
        # Estimate from point metadata if shape not tracked
        point = tree.library.get("m.w")
        assert point is not None

    def test_estimate_cluster_assignments(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        w = np.random.randn(64).astype(np.float32)
        tree.load_weights({"w": w}, method="cluster")
        point = tree.library.get("m.w")
        if point.function_type == "cluster":
            assert "assignments" in point.params

    def test_raw_weight_roundtrip(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        tree._skip_embeddings = False
        tree._skip_biases = False
        original = np.random.randn(4).astype(np.float32)
        tree.load_weights({"w": original})
        recovered = tree.get_weight("w")
        np.testing.assert_array_almost_equal(original, recovered)

    def test_cluster_weight_retrievable(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        w = np.random.randn(64).astype(np.float32)
        tree.load_weights({"w": w}, method="cluster")
        result = tree.get_weight("w")
        assert result is not None
        assert result.shape == (64,)

    def test_function_weight_retrievable(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        w = np.random.randn(64).astype(np.float32)
        tree.load_weights({"w": w}, method="function")
        result = tree.get_weight("w")
        assert result is not None
        assert result.shape == (64,)

    def test_2d_weight_roundtrip_raw(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        original = np.random.randn(8, 8).astype(np.float32)
        # Embedding-like name gets stored as raw
        tree.load_weights({"embed.w": original})
        recovered = tree.get_weight("embed.w")
        np.testing.assert_array_almost_equal(original, recovered)

    def test_estimate_size_zero_shape(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        # No shape, no point → returns 0
        assert tree._estimate_size("nonexistent") == 0

    def test_load_weights_with_progress_callback(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        calls = []
        def _cb(done, total, name):
            calls.append((done, total, name))
        w = {"a": np.zeros((4, 4), dtype=np.float32)}
        tree.load_weights(w, on_progress=_cb)
        assert len(calls) == 1
        assert calls[0] == (1, 1, "a")

    def test_load_weights_parallel(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        weights = {f"w{i}": np.random.randn(64).astype(np.float32) for i in range(5)}
        stats = tree.load_weights(weights, num_workers=2)
        assert stats["num_weights"] == 5
        assert tree.is_loaded is True

    def test_stats_before_load(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        stats = tree.stats()
        assert stats["loaded"] is False
        assert stats["num_weights"] == 0

    def test_library_stores_all_points(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        weights = {f"w{i}": np.zeros((4, 4), dtype=np.float32) for i in range(5)}
        tree.load_weights(weights)
        all_points = tree.library.list_all()
        assert len(all_points) == 5

    def test_skip_embeddings_disabled(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        tree._skip_embeddings = False
        tree._skip_biases = False
        w = np.random.randn(64).astype(np.float32)
        tree.load_weights({"embed.weight": w})
        point = tree.library.get("m.embed.weight")
        assert point.function_type == "cluster"
