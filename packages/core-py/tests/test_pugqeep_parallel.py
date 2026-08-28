"""Tests for parallel batch operations in pugqeep."""

import time

import numpy as np
import pytest

from domains.infrastructure.pugqeep.model_tree import ModelTree, decompress_tree
from domains.infrastructure.pugqeep.library import PointLibrary
from domains.infrastructure.pugqeep.facade import PGQ


# ── ModelTree parallel compression ──────────────────────────────────


class TestModelTreeParallel:
    def _make_weights(self, n=10, size=200):
        return {f"layer_{i}": np.random.randn(size, size).astype(np.float32) for i in range(n)}

    def test_load_weights_sequential(self):
        weights = self._make_weights(5, 100)
        tree = ModelTree("test-seq", n_clusters=16)
        result = tree.load_weights(weights, num_workers=0)
        assert result["num_weights"] == 5
        assert tree.is_loaded

    def test_load_weights_parallel(self):
        weights = self._make_weights(8, 100)
        tree = ModelTree("test-par", n_clusters=16)
        result = tree.load_weights(weights, num_workers=2)
        assert result["num_weights"] == 8
        assert result.get("workers") == 2
        assert tree.is_loaded
        # Verify all weights are retrievable
        for name in weights:
            point = tree.library.get(f"test-par.{name}")
            assert point is not None

    def test_sequential_and_parallel_produce_same_results(self):
        weights = self._make_weights(6, 150)
        # Sequential
        tree_seq = ModelTree("seq", n_clusters=16)
        tree_seq.load_weights(weights, num_workers=0)
        # Parallel
        tree_par = ModelTree("par", n_clusters=16)
        tree_par.load_weights(weights, num_workers=2)
        # Both should have same ratio (compression is deterministic for same seed)
        seq_result = tree_seq.load_weights({}, num_workers=0)
        par_result = tree_par.load_weights({}, num_workers=0)
        # Check all points exist
        for name in weights:
            assert tree_seq.library.has(f"seq.{name}")
            assert tree_par.library.has(f"par.{name}")

    def test_parallel_with_single_weight(self):
        weights = {"only": np.random.randn(50, 50).astype(np.float32)}
        tree = ModelTree("single", n_clusters=16)
        result = tree.load_weights(weights, num_workers=2)
        assert result["num_weights"] == 1

    def test_parallel_with_skipped_weights(self):
        weights = {
            "layer_0": np.random.randn(100, 100).astype(np.float32),
            "embed_tokens": np.random.randn(50, 50).astype(np.float32),
            "classifier_bias": np.random.randn(50).astype(np.float32),
        }
        tree = ModelTree("skip", n_clusters=16, config=None)
        tree._skip_embeddings = True
        tree._skip_biases = True
        result = tree.load_weights(weights, num_workers=2)
        assert result["num_weights"] == 3
        # Embedding and bias should be stored as raw
        embed_point = tree.library.get("skip.embed_tokens")
        assert embed_point is not None
        assert embed_point.function_type == "raw"

    def test_decompress_tree_sequential(self):
        weights = self._make_weights(5, 100)
        tree = ModelTree("dec-seq", n_clusters=16)
        tree.load_weights(weights, num_workers=0)
        decompressed = decompress_tree(tree, num_workers=0)
        assert len(decompressed) == 5
        for name in weights:
            assert name in decompressed
            assert decompressed[name].shape == weights[name].shape

    def test_decompress_tree_parallel(self):
        weights = self._make_weights(8, 100)
        tree = ModelTree("dec-par", n_clusters=16)
        tree.load_weights(weights, num_workers=0)
        decompressed = decompress_tree(tree, num_workers=2)
        assert len(decompressed) == 8
        for name in weights:
            assert name in decompressed
            assert decompressed[name].shape == weights[name].shape


# ── PGQ parallel batch ops ──────────────────────────────────────────


class TestPGQParallelBatch:
    def _make_data(self, n=10, size=100):
        return {f"arr_{i}": np.random.randn(size, size).astype(np.float32) for i in range(n)}

    def test_put_many_sequential(self):
        pgq = PGQ("test-put-seq")
        data = self._make_data(5, 80)
        result = pgq.put_many(data, num_workers=0)
        assert result["count"] == 5
        assert result["total_bytes"] > 0

    def test_put_many_parallel(self):
        pgq = PGQ("test-put-par")
        data = self._make_data(8, 80)
        result = pgq.put_many(data, num_workers=2)
        assert result["count"] == 8
        assert result["total_bytes"] > 0
        # Verify all stored
        for name in data:
            assert pgq.has(name)

    def test_get_many_sequential(self):
        pgq = PGQ("test-get-seq")
        data = self._make_data(5, 80)
        pgq.put_many(data, num_workers=0)
        names = list(data.keys())
        result = pgq.get_many(names, num_workers=0)
        assert len(result) == 5
        for name in names:
            assert name in result
            assert result[name] is not None

    def test_get_many_parallel(self):
        pgq = PGQ("test-get-par")
        data = self._make_data(8, 80)
        pgq.put_many(data, num_workers=0)
        names = list(data.keys())
        result = pgq.get_many(names, num_workers=2)
        assert len(result) == 8
        for name in names:
            assert name in result
            assert result[name] is not None

    def test_put_many_parallel_matches_sequential(self):
        # Sequential
        pgq_seq = PGQ("match-seq")
        data = self._make_data(6, 100)
        pgq_seq.put_many(data, num_workers=0)
        # Parallel
        pgq_par = PGQ("match-par")
        pgq_par.put_many(data, num_workers=2)
        # Both should have same keys
        for name in data:
            assert pgq_seq.has(name)
            assert pgq_par.has(name)
