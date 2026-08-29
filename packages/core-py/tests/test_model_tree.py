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
        assert tree._estimate_size("missing") == 1000

    def test_function_method(self):
        from domains.infrastructure.pugqeep.model_tree import ModelTree
        tree = ModelTree("m", n_clusters=4)
        arr = np.random.randn(64).astype(np.float32)
        tree.load_weights({"w": arr}, method="function")
        point = tree.library.get("m.w")
        assert point.function_type != "raw"  # should be compressed (periodic/polynomial/etc)
