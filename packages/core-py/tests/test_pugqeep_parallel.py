"""Tests for parallel batch operations in pugqeep."""

import time

import numpy as np
import pytest

from domains.infrastructure.pugqeep.model_tree import ModelTree
from domains.infrastructure.pugqeep.tree import decompress_tree
from domains.infrastructure.pugqeep.library import PointLibrary
from domains.infrastructure.pugqeep.facade import PGQ


# -- ModelTree parallel compression --


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
        for name in weights:
            point = tree.library.get(f"test-par.{name}")
            assert point is not None

    def test_sequential_and_parallel_produce_same_results(self):
        weights = self._make_weights(6, 150)
        tree_seq = ModelTree("seq", n_clusters=16)
        tree_seq.load_weights(weights, num_workers=0)
        tree_par = ModelTree("par", n_clusters=16)
        tree_par.load_weights(weights, num_workers=2)
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

    def test_sequential_result_stats(self):
        weights = self._make_weights(4, 80)
        tree = ModelTree("stats-seq", n_clusters=16)
        result = tree.load_weights(weights, num_workers=0)
        assert "ratio" in result
        assert result["ratio"] > 0
        assert result["method"] == "cluster"

    def test_parallel_result_stats(self):
        weights = self._make_weights(4, 80)
        tree = ModelTree("stats-par", n_clusters=16)
        result = tree.load_weights(weights, num_workers=2)
        assert "ratio" in result
        assert result["ratio"] > 0
        assert result["workers"] == 2

    def test_get_weight_after_load(self):
        weights = self._make_weights(3, 60)
        tree = ModelTree("get-w", n_clusters=16)
        tree.load_weights(weights, num_workers=0)
        for name in weights:
            w = tree.get_weight(name)
            assert w is not None
            assert w.shape == weights[name].shape

    def test_get_weights_bulk(self):
        weights = self._make_weights(4, 70)
        tree = ModelTree("bulk", n_clusters=16)
        tree.load_weights(weights, num_workers=0)
        all_w = tree.get_weights()
        assert len(all_w) == 4

    def test_model_tree_stats(self):
        weights = self._make_weights(3, 50)
        tree = ModelTree("st", n_clusters=16)
        tree.load_weights(weights, num_workers=0)
        s = tree.stats()
        assert s["loaded"] is True
        assert s["num_weights"] == 3

    def test_empty_weights(self):
        tree = ModelTree("empty", n_clusters=16)
        result = tree.load_weights({}, num_workers=0)
        assert result["num_weights"] == 0

    def test_load_weights_twice_accumulates(self):
        w1 = {"a": np.random.randn(40, 40).astype(np.float32)}
        w2 = {"b": np.random.randn(40, 40).astype(np.float32)}
        tree = ModelTree("acc", n_clusters=16)
        tree.load_weights(w1, num_workers=0)
        tree.load_weights(w2, num_workers=0)
        assert tree.library.has("acc.a")
        assert tree.library.has("acc.b")

    def test_is_loaded_setter(self):
        tree = ModelTree("setter", n_clusters=16)
        assert tree.is_loaded is False
        tree.is_loaded = True
        assert tree.is_loaded is True

    def test_result_has_model_name(self):
        weights = self._make_weights(2, 50)
        tree = ModelTree("named", n_clusters=16)
        result = tree.load_weights(weights, num_workers=0)
        assert result["model"] == "named"

    def test_result_has_raw_bytes(self):
        weights = self._make_weights(2, 50)
        tree = ModelTree("bytes", n_clusters=16)
        result = tree.load_weights(weights, num_workers=0)
        assert result["total_raw_bytes"] > 0
        assert result["total_compressed_bytes"] > 0

    def test_get_weight_nonexistent(self):
        tree = ModelTree("noexist", n_clusters=16)
        tree.load_weights({}, num_workers=0)
        assert tree.get_weight("missing") is None

    def test_get_weights_subset(self):
        weights = self._make_weights(4, 50)
        tree = ModelTree("subset", n_clusters=16)
        tree.load_weights(weights, num_workers=0)
        subset = tree.get_weights(["layer_0", "layer_1"])
        assert len(subset) == 2
        assert "layer_0" in subset
        assert "layer_1" in subset

    def test_stats_before_load(self):
        tree = ModelTree("pre", n_clusters=16)
        s = tree.stats()
        assert s["loaded"] is False
        assert s["num_weights"] == 0

    def test_decompress_preserves_dtype(self):
        weights = {"w": np.random.randn(20, 20).astype(np.float64)}
        tree = ModelTree("dtype", n_clusters=16)
        tree.load_weights(weights, num_workers=0)
        decompressed = decompress_tree(tree, num_workers=0)
        assert decompressed["w"].dtype == np.float64

    def test_sequential_total_bytes_match(self):
        weights = self._make_weights(3, 40)
        tree = ModelTree("tbytes", n_clusters=16)
        result = tree.load_weights(weights, num_workers=0)
        expected_raw = sum(w.nbytes for w in weights.values())
        assert result["total_raw_bytes"] == expected_raw

    def test_library_name_matches_tree(self):
        tree = ModelTree("libcheck", n_clusters=16)
        assert tree.library.name == "libcheck_points"

    def test_parallel_total_bytes_match(self):
        weights = self._make_weights(3, 40)
        tree = ModelTree("ptbytes", n_clusters=16)
        result = tree.load_weights(weights, num_workers=2)
        expected_raw = sum(w.nbytes for w in weights.values())
        assert result["total_raw_bytes"] == expected_raw

    def test_decompress_tree_empty(self):
        tree = ModelTree("dec-empty", n_clusters=16)
        tree.load_weights({}, num_workers=0)
        decompressed = decompress_tree(tree, num_workers=0)
        assert len(decompressed) == 0

    def test_get_weight_after_parallel_load(self):
        weights = self._make_weights(3, 60)
        tree = ModelTree("get-w-par", n_clusters=16)
        tree.load_weights(weights, num_workers=2)
        for name in weights:
            w = tree.get_weight(name)
            assert w is not None
            assert w.shape == weights[name].shape

    def test_result_has_compressed_bytes(self):
        weights = self._make_weights(2, 50)
        tree = ModelTree("cbytes", n_clusters=16)
        result = tree.load_weights(weights, num_workers=0)
        assert result["total_compressed_bytes"] > 0

    def test_result_ratio_greater_than_one(self):
        weights = self._make_weights(2, 100)
        tree = ModelTree("ratio1", n_clusters=16)
        result = tree.load_weights(weights, num_workers=0)
        assert result["ratio"] >= 1.0


# -- PGQ parallel batch ops --


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
        pgq_seq = PGQ("match-seq")
        data = self._make_data(6, 100)
        pgq_seq.put_many(data, num_workers=0)
        pgq_par = PGQ("match-par")
        pgq_par.put_many(data, num_workers=2)
        for name in data:
            assert pgq_seq.has(name)
            assert pgq_par.has(name)

    def test_put_many_empty(self):
        pgq = PGQ("empty-put")
        result = pgq.put_many({}, num_workers=0)
        assert result["count"] == 0

    def test_get_many_missing_names(self):
        pgq = PGQ("missing")
        result = pgq.get_many(["nonexistent1", "nonexistent2"], num_workers=0)
        assert result["nonexistent1"] is None
        assert result["nonexistent2"] is None

    def test_exists_many(self):
        pgq = PGQ("exists")
        data = self._make_data(3, 50)
        pgq.put_many(data, num_workers=0)
        result = pgq.exists_many(list(data.keys()) + ["nope"])
        assert all(result[k] for k in data)
        assert not result["nope"]

    def test_remove_many(self):
        pgq = PGQ("rem-many")
        data = self._make_data(4, 50)
        pgq.put_many(data, num_workers=0)
        removed = pgq.remove_many(list(data.keys()))
        assert removed == 4
        for name in data:
            assert not pgq.has(name)

    def test_put_single_array(self):
        pgq = PGQ("single")
        arr = np.random.randn(50, 50).astype(np.float32)
        pgq.put("w1", arr)
        assert pgq.has("w1")
        got = pgq.get("w1")
        assert got.shape == arr.shape

    def test_put_many_total_bytes_correct(self):
        pgq = PGQ("tbytes")
        data = self._make_data(3, 40)
        result = pgq.put_many(data, num_workers=0)
        expected = sum(arr.nbytes for arr in data.values())
        assert result["total_bytes"] == expected

    def test_get_many_empty_list(self):
        pgq = PGQ("get-empty")
        result = pgq.get_many([], num_workers=0)
        assert result == {}

    def test_exists_many_all_missing(self):
        pgq = PGQ("all-missing")
        result = pgq.exists_many(["a", "b", "c"])
        assert all(not v for v in result.values())

    def test_put_many_single_item(self):
        pgq = PGQ("single-put")
        data = {"only": np.random.randn(10, 10).astype(np.float32)}
        result = pgq.put_many(data, num_workers=0)
        assert result["count"] == 1

    def test_remove_many_partial(self):
        pgq = PGQ("partial")
        data = self._make_data(3, 30)
        pgq.put_many(data, num_workers=0)
        removed = pgq.remove_many(["arr_0", "nonexistent"])
        assert removed == 1

    def test_stats_after_put_many(self):
        pgq = PGQ("stats")
        data = self._make_data(5, 60)
        pgq.put_many(data, num_workers=0)
        s = pgq.stats()
        assert s["tree"]["num_weights"] == 5

    def test_put_many_no_compress(self):
        pgq = PGQ("nocomp")
        data = self._make_data(2, 30)
        result = pgq.put_many(data, compress=False, num_workers=0)
        assert result["count"] == 2

    def test_put_many_parallel_total_bytes(self):
        pgq = PGQ("par-tbytes")
        data = self._make_data(3, 40)
        result = pgq.put_many(data, num_workers=2)
        expected = sum(arr.nbytes for arr in data.values())
        assert result["total_bytes"] == expected

    def test_get_many_parallel_missing(self):
        pgq = PGQ("par-missing")
        result = pgq.get_many(["x", "y"], num_workers=2)
        assert result["x"] is None
        assert result["y"] is None

    def test_remove_many_empty(self):
        pgq = PGQ("rem-empty")
        removed = pgq.remove_many([])
        assert removed == 0

    def test_exists_many_partial(self):
        pgq = PGQ("exists-partial")
        arr = np.random.randn(10, 10).astype(np.float32)
        pgq.put("a", arr)
        result = pgq.exists_many(["a", "b"])
        assert result["a"] is True
        assert result["b"] is False

    def test_put_many_returns_zero_total_bytes_empty(self):
        pgq = PGQ("zero-bytes")
        result = pgq.put_many({}, num_workers=0)
        assert result["total_bytes"] == 0
