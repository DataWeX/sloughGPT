"""Tests for PointLibrary, ModelTree, and Point serialization."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from domains.infrastructure.point_compressor import (
    Point, PointCompressor, PointLibrary, ModelTree,
)
from domains.infrastructure.safetensors_loader import _find_safetensors, _get_model_dir

QWEN2_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def _is_cached(model_id: str) -> bool:
    return _find_safetensors(_get_model_dir(model_id)) is not None


# ── Fixtures ──

@pytest.fixture
def compressor():
    return PointCompressor()


@pytest.fixture
def library():
    return PointLibrary(name="test_lib")


@pytest.fixture
def sample_weights():
    rng = np.random.default_rng(42)
    return rng.standard_normal(1024).astype(np.float32)


@pytest.fixture
def structured_weights():
    """Linear weights — should compress well with function fitting."""
    i = np.arange(512, dtype=np.float32)
    return (0.01 * i + 0.5).astype(np.float32)


# ── Point tests ──

class TestPoint:
    def test_cluster_generate(self):
        centroids = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        assignments = np.array([0, 1, 2, 0, 1], dtype=np.uint8)
        p = Point(identity="test", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments})
        result = p.generate(5)
        expected = centroids[assignments]
        np.testing.assert_array_almost_equal(result, expected)

    def test_periodic_generate(self):
        p = Point(identity="test", function_type="periodic",
                  params={"a": 1.0, "b": 0.5, "w": 0.0})
        result = p.generate(3)
        i = np.arange(3, dtype=np.float32)
        expected = 1.0 * np.cos(i) + 0.5 * np.sin(i) + 0.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_linear_generate(self):
        p = Point(identity="test", function_type="linear",
                  params={"a": 2.0, "b": 1.0})
        result = p.generate(4)
        i = np.arange(4, dtype=np.float32)
        expected = 2.0 * i + 1.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_polynomial_generate(self):
        p = Point(identity="test", function_type="polynomial",
                  params={"a": 0.1, "b": 0.5, "c": 1.0})
        result = p.generate(3)
        i = np.arange(3, dtype=np.float32)
        expected = 0.1 * i**2 + 0.5 * i + 1.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_raw_generate(self):
        raw_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        import base64
        p = Point(identity="test", function_type="raw",
                  params={"data_b64": base64.b64encode(raw_data.tobytes()).decode(),
                          "shape": [3], "dtype": "float32"})
        result = p.generate(3)
        np.testing.assert_array_almost_equal(result, raw_data)

    def test_roundtrip_dict(self, sample_weights):
        compressor = PointCompressor()
        point = compressor.compress_cluster(sample_weights, "test_cluster")
        d = point.to_dict()
        restored = Point.from_dict(d)
        assert restored.identity == point.identity
        assert restored.function_type == point.function_type
        assert restored.accuracy == pytest.approx(point.accuracy)
        np.testing.assert_array_equal(
            restored.params["centroids"], point.params["centroids"])
        np.testing.assert_array_equal(
            restored.params["assignments"], point.params["assignments"])

    def test_roundtrip_dict_periodic(self):
        p = Point(identity="test", function_type="periodic",
                  params={"a": 1.5, "b": -0.3, "w": 0.7}, accuracy=0.95)
        d = p.to_dict()
        restored = Point.from_dict(d)
        assert restored.function_type == "periodic"
        assert restored.params["a"] == pytest.approx(1.5)
        assert restored.params["b"] == pytest.approx(-0.3)

    def test_roundtrip_dict_with_residual(self):
        residual = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        p = Point(identity="test", function_type="linear",
                  params={"a": 1.0, "b": 0.0}, residual=residual, accuracy=0.9)
        d = p.to_dict()
        restored = Point.from_dict(d)
        assert restored.residual is not None
        np.testing.assert_array_almost_equal(restored.residual, residual)


# ── PointCompressor tests ──

class TestPointCompressor:
    def test_compress_cluster(self, compressor, sample_weights):
        point = compressor.compress_cluster(sample_weights, "w0")
        assert point.function_type == "cluster"
        assert point.accuracy > 0.8
        assert "centroids" in point.params
        assert "assignments" in point.params

    def test_compress_function(self, compressor, structured_weights):
        point = compressor.compress_function(structured_weights, "w1")
        assert point.function_type in ("periodic", "linear", "polynomial")
        assert point.accuracy > 0.5

    def test_compress_method_dispatch(self, compressor, sample_weights):
        p1 = compressor.compress(sample_weights, "c1", method="cluster")
        p2 = compressor.compress(sample_weights, "f1", method="function")
        assert p1.function_type == "cluster"

    def test_decompress(self, compressor, sample_weights):
        point = compressor.compress_cluster(sample_weights, "w0")
        n = len(sample_weights.flatten())
        decompressed = compressor.decompress(point, n)
        assert decompressed.shape == (n,)
        # Should be close to original
        mse = np.mean((sample_weights.flatten() - decompressed) ** 2)
        var = np.var(sample_weights)
        accuracy = 1.0 - mse / (var + 1e-8)
        assert accuracy > 0.8

    def test_measure_compression(self, compressor, sample_weights):
        point = compressor.compress_cluster(sample_weights, "w0")
        m = compressor.measure_compression(sample_weights, point)
        assert m["raw_bytes"] > 0
        assert m["compressed_bytes"] > 0
        assert m["ratio"] > 1.0
        assert 0.0 <= m["accuracy"] <= 1.0


# ── PointLibrary tests ──

class TestPointLibrary:
    def test_add_and_get(self, library):
        p = Point(identity="p1", function_type="linear",
                  params={"a": 1.0, "b": 0.0}, accuracy=0.9)
        library.add(p)
        assert library.get("p1") is p

    def test_get_nonexistent(self, library):
        assert library.get("nope") is None

    def test_has(self, library):
        p = Point(identity="p1", function_type="linear",
                  params={"a": 1.0, "b": 0.0}, accuracy=0.9)
        library.add(p)
        assert library.has("p1")
        assert not library.has("nope")

    def test_remove(self, library):
        p = Point(identity="p1", function_type="linear",
                  params={"a": 1.0, "b": 0.0}, accuracy=0.9)
        library.add(p)
        assert library.remove("p1")
        assert library.get("p1") is None
        assert not library.remove("p1")

    def test_list_all(self, library):
        for i in range(5):
            library.add(Point(identity=f"p{i}", function_type="linear",
                              params={"a": float(i), "b": 0.0}))
        assert len(library.list_all()) == 5

    def test_list_by_type(self, library):
        library.add(Point(identity="lin1", function_type="linear",
                          params={"a": 1.0, "b": 0.0}))
        library.add(Point(identity="clu1", function_type="cluster",
                          params={"centroids": np.zeros(5), "assignments": np.zeros(10, dtype=np.uint8)}))
        library.add(Point(identity="lin2", function_type="linear",
                          params={"a": 2.0, "b": 0.0}))
        linear = library.list_by_type("linear")
        assert len(linear) == 2
        cluster = library.list_by_type("cluster")
        assert len(cluster) == 1

    def test_search(self, library):
        library.add(Point(identity="attn.qkv.w0", function_type="linear",
                          params={"a": 1.0, "b": 0.0}))
        library.add(Point(identity="attn.qkv.w1", function_type="linear",
                          params={"a": 2.0, "b": 0.0}))
        library.add(Point(identity="ffn.up.w0", function_type="linear",
                          params={"a": 3.0, "b": 0.0}))
        results = library.search("attn.qkv")
        assert len(results) == 2

    def test_best_points(self, library):
        for i in range(10):
            library.add(Point(identity=f"p{i}", function_type="linear",
                              params={"a": 1.0, "b": 0.0}, accuracy=i / 10.0))
        best = library.best_points(3)
        assert len(best) == 3
        assert best[0].accuracy >= best[1].accuracy >= best[2].accuracy

    def test_compress_and_store(self, library, sample_weights):
        point = library.compress_and_store(sample_weights, "w0")
        assert library.has("w0")
        assert point.function_type == "cluster"
        assert point.accuracy > 0.8

    def test_clear(self, library):
        for i in range(5):
            library.add(Point(identity=f"p{i}", function_type="linear",
                              params={"a": 1.0, "b": 0.0}))
        library.clear()
        assert len(library.list_all()) == 0

    def test_stats(self, library):
        assert library.stats()["total_points"] == 0
        library.add(Point(identity="p1", function_type="linear",
                          params={"a": 1.0, "b": 0.0}, accuracy=0.9))
        s = library.stats()
        assert s["total_points"] == 1
        assert s["avg_accuracy"] == pytest.approx(0.9)
        assert "linear" in s["types"]

    def test_save_and_load(self, library):
        for i in range(3):
            library.add(Point(identity=f"p{i}", function_type="linear",
                              params={"a": float(i), "b": 0.0}, accuracy=0.8 + i * 0.05))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = library.save(Path(tmpdir) / "test.points.json")
            assert path.exists()

            loaded = PointLibrary.load(path)
            assert loaded.name == "test_lib"
            assert len(loaded.list_all()) == 3
            assert loaded.get("p0").params["a"] == pytest.approx(0.0)
            assert loaded.get("p2").accuracy == pytest.approx(0.9)

    def test_save_load_cluster_points(self, compressor, sample_weights):
        point = compressor.compress_cluster(sample_weights, "cluster_w")
        library = PointLibrary(name="cluster_test")
        library.add(point)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = library.save(Path(tmpdir) / "cluster.points.json")
            loaded = PointLibrary.load(path)
            restored = loaded.get("cluster_w")
            assert restored is not None
            np.testing.assert_array_equal(
                restored.params["centroids"], point.params["centroids"])
            np.testing.assert_array_equal(
                restored.params["assignments"], point.params["assignments"])

    def test_replace_on_duplicate_identity(self, library):
        library.add(Point(identity="p1", function_type="linear",
                          params={"a": 1.0, "b": 0.0}))
        library.add(Point(identity="p1", function_type="polynomial",
                          params={"a": 2.0, "b": 0.0, "c": 0.0}))
        assert len(library.list_all()) == 1
        assert library.get("p1").function_type == "polynomial"


# ── ModelTree tests ──

class TestModelTree:
    def test_load_weights(self, library):
        tree = ModelTree("test_model", library, n_clusters=8)
        weights = {
            "w0": np.random.default_rng(42).standard_normal(512).astype(np.float32),
            "w1": np.random.default_rng(43).standard_normal(256).astype(np.float32),
        }
        stats = tree.load_weights(weights)
        assert stats["num_weights"] == 2
        assert stats["ratio"] > 1.0
        assert tree.is_loaded

    def test_get_weight(self, library):
        tree = ModelTree("test_model", library, n_clusters=8)
        original = np.random.default_rng(42).standard_normal(512).astype(np.float32)
        tree.load_weights({"w0": original})
        recovered = tree.get_weight("w0")
        assert recovered is not None
        assert recovered.shape == original.shape
        # Should be close (VQ compression is lossy)
        mse = np.mean((original - recovered) ** 2)
        var = np.var(original)
        accuracy = 1.0 - mse / (var + 1e-8)
        assert accuracy > 0.8

    def test_get_weight_nonexistent(self, library):
        tree = ModelTree("test_model", library)
        assert tree.get_weight("nope") is None

    def test_small_weights_stored_raw(self, library):
        tree = ModelTree("test_model", library, n_clusters=16)
        tiny = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tree.load_weights({"tiny": tiny})
        recovered = tree.get_weight("tiny")
        np.testing.assert_array_almost_equal(recovered, tiny)

    def test_stats(self, library):
        tree = ModelTree("test_model", library)
        assert not tree.is_loaded
        tree.load_weights({"w0": np.zeros(100, dtype=np.float32)})
        s = tree.stats()
        assert s["model"] == "test_model"
        assert s["loaded"]
        assert s["num_weights"] == 1

    def test_library_shared(self):
        lib = PointLibrary(name="shared")
        t1 = ModelTree("model_a", lib)
        t2 = ModelTree("model_b", lib)
        t1.load_weights({"w0": np.zeros(100, dtype=np.float32)})
        t2.load_weights({"w1": np.ones(100, dtype=np.float32)})
        # Both trees share the same library
        assert lib.stats()["total_points"] == 2


# ── Integration: full round-trip ──

class TestIntegration:
    def test_compress_decompress_roundtrip(self, compressor):
        """Compress → decompress → verify accuracy."""
        original = np.random.default_rng(42).standard_normal(2048).astype(np.float32)
        point = compressor.compress_cluster(original, "test_weight", n_clusters=16)
        decompressed = compressor.decompress(point, len(original))
        mse = np.mean((original - decompressed) ** 2)
        var = np.var(original)
        accuracy = 1.0 - mse / (var + 1e-8)
        assert accuracy > 0.85

    def test_library_persistence_roundtrip(self):
        """Compress → save → load → verify points match."""
        lib = PointLibrary(name="persist_test")
        compressor = PointCompressor()
        weights = {
            f"w{i}": np.random.default_rng(i).standard_normal(512).astype(np.float32)
            for i in range(5)
        }
        for name, w in weights.items():
            lib.compress_and_store(w, name)

        with tempfile.TemporaryDirectory() as tmpdir:
            lib.save(Path(tmpdir) / "lib.points.json")
            loaded = PointLibrary.load(Path(tmpdir) / "lib.points.json")
            assert loaded.stats()["total_points"] == 5
            for name in weights:
                assert loaded.has(name)

    def test_model_tree_full_pipeline(self):
        """Full pipeline: create tree → load weights → compress → verify."""
        lib = PointLibrary(name="full_test")
        tree = ModelTree("gpt2_test", lib, n_clusters=16)

        rng = np.random.default_rng(42)
        weights = {
            "h.0.ln_1.weight": rng.standard_normal(768).astype(np.float32),
            "h.0.attn.c_attn.weight": rng.standard_normal((768, 2304)).astype(np.float32),
            "wte.weight": rng.standard_normal((50257, 768)).astype(np.float32),
        }

        stats = tree.load_weights(weights)
        assert stats["ratio"] > 1.0

        for name, original in weights.items():
            recovered = tree.get_weight(name)
            assert recovered is not None
            assert recovered.shape == original.shape
            mse = np.mean((original.flatten() - recovered.flatten()) ** 2)
            var = np.var(original)
            accuracy = 1.0 - mse / (var + 1e-8)
            assert accuracy > 0.7, f"{name} accuracy too low: {accuracy}"

    def test_multi_model_sharing(self):
        """Two models share the same point library."""
        lib = PointLibrary(name="shared_lib")
        t1 = ModelTree("model_a", lib)
        t2 = ModelTree("model_b", lib)

        rng = np.random.default_rng(42)
        t1.load_weights({"shared_w": rng.standard_normal(256).astype(np.float32)})
        t2.load_weights({"my_w": rng.standard_normal(256).astype(np.float32)})

        # Both exist in the shared library
        assert lib.has("model_a.shared_w")
        assert lib.has("model_b.my_w")
        assert lib.stats()["total_points"] == 2


# ── NumpyEngine + ModelTree integration ──

class TestNumpyEngineModelTree:
    """Integration tests: NumpyEngine with ModelTree (Point-based storage)."""

    def test_numpy_engine_with_model_tree(self):
        """NumpyEngine stores weights as Points via ModelTree."""
        from domains.infrastructure.numpy_engine import NumpyEngine

        config = {
            "architectures": ["GPT2LMHeadModel"],
            "vocab_size": 100,
            "n_positions": 128,
            "n_embd": 64,
            "n_head": 4,
            "n_layer": 2,
            "n_ctx": 128,
        }
        rng = np.random.default_rng(42)
        weights = {
            "wte.weight": rng.standard_normal((100, 64)).astype(np.float32),
            "wpe.weight": rng.standard_normal((128, 64)).astype(np.float32),
            "h.0.ln_1.weight": rng.standard_normal(64).astype(np.float32),
            "h.0.ln_1.bias": rng.standard_normal(64).astype(np.float32),
        }

        lib = PointLibrary(name="test_engine")
        tree = ModelTree("gpt2_test", lib)
        engine = NumpyEngine(
            config=config, weights=weights, compress=True,
            model_tree=tree,
        )

        # Weights stored as Points, not _CompressedWeight
        assert engine._model_tree is not None
        assert len(engine._compressed_weights) == 0

        # Can still get weights
        w = engine._get_weight("wte.weight")
        assert w.shape == (100, 64)

        # Stats
        info = engine.info()
        assert info["compressed"] is True

    @pytest.mark.skipif(not _is_cached(QWEN2_ID), reason=f"{QWEN2_ID} not cached locally")
    def test_numpy_engine_from_pretrained_with_points(self):
        """NumpyEngine.from_pretrained with use_points=True."""
        pytest.importorskip("safetensors")
        from domains.infrastructure.numpy_engine import NumpyEngine

        engine = NumpyEngine.from_pretrained(QWEN2_ID, use_points=True, n_clusters=8)

        # ModelTree created
        assert engine._model_tree is not None
        assert engine._model_tree.is_loaded

        # Library has points
        lib = engine._model_tree.library
        assert lib.stats()["total_points"] > 0

        # Can get weights
        w = engine._get_weight("model.embed_tokens.weight")
        assert w.shape[0] == 151936  # Qwen2.5 vocab

    @pytest.mark.skipif(not _is_cached(QWEN2_ID), reason=f"{QWEN2_ID} not cached locally")
    def test_numpy_engine_from_pretrained_with_shared_library(self):
        """Two engines share the same PointLibrary via ModelTree."""
        pytest.importorskip("safetensors")
        from domains.infrastructure.numpy_engine import NumpyEngine

        lib = PointLibrary(name="shared")
        e1 = NumpyEngine.from_pretrained(QWEN2_ID, use_points=True, library=lib)
        e2 = NumpyEngine.from_pretrained(QWEN2_ID, use_points=True, library=lib)

        # Both use the same library
        assert e1._model_tree.library is e2._model_tree.library

        # Library has points from both engines (with different model prefixes)
        stats = lib.stats()
        assert stats["total_points"] > 0

    @pytest.mark.skipif(not _is_cached(QWEN2_ID), reason=f"{QWEN2_ID} not cached locally")
    def test_points_persistence(self):
        """Save PointLibrary from NumpyEngine, load and verify."""
        pytest.importorskip("safetensors")
        from domains.infrastructure.numpy_engine import NumpyEngine

        engine = NumpyEngine.from_pretrained(QWEN2_ID, use_points=True, n_clusters=8)
        lib = engine._model_tree.library

        with tempfile.TemporaryDirectory() as tmpdir:
            lib.save(Path(tmpdir) / "qwen2.points.json")

            loaded_lib = PointLibrary.load(Path(tmpdir) / "qwen2.points.json")
            assert loaded_lib.stats()["total_points"] == lib.stats()["total_points"]

            # Verify a weight can be reconstructed
            original = engine._get_weight("model.layers.0.input_layernorm.weight")
            tree2 = ModelTree(QWEN2_ID, loaded_lib)
            recovered = tree2.get_weight("model.layers.0.input_layernorm.weight")
            assert recovered is not None
            assert recovered.shape == original.shape


# ── PointDeduplicator tests ──

class TestPointDeduplicator:
    def test_find_duplicates(self):
        from domains.infrastructure.point_compressor import PointDeduplicator

        centroids = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        assignments = np.array([0, 1, 2, 0, 1], dtype=np.uint8)

        lib1 = PointLibrary("lib1")
        lib2 = PointLibrary("lib2")
        lib1.add(Point("model_a.weight_0", "cluster",
                       {"centroids": centroids.copy(), "assignments": assignments.copy()}))
        lib2.add(Point("model_b.weight_0", "cluster",
                       {"centroids": centroids.copy(), "assignments": assignments.copy()}))

        dedup = PointDeduplicator()
        dedup.add_library(lib1)
        dedup.add_library(lib2)
        groups = dedup.find_duplicates()
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_deduplicate_merges(self):
        from domains.infrastructure.point_compressor import PointDeduplicator

        centroids = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        assignments = np.array([0, 1, 2, 0, 1], dtype=np.uint8)

        lib1 = PointLibrary("lib1")
        lib2 = PointLibrary("lib2")
        lib1.add(Point("a.w", "cluster",
                       {"centroids": centroids.copy(), "assignments": assignments.copy()}))
        lib2.add(Point("b.w", "cluster",
                       {"centroids": centroids.copy(), "assignments": assignments.copy()}))

        dedup = PointDeduplicator()
        dedup.add_library(lib1)
        dedup.add_library(lib2)
        stats = dedup.deduplicate()
        assert stats["merged"] >= 1
        assert stats["bytes_saved"] > 0
        assert stats["groups"] == 1

    def test_no_duplicates(self):
        from domains.infrastructure.point_compressor import PointDeduplicator

        lib1 = PointLibrary("lib1")
        lib2 = PointLibrary("lib2")
        lib1.add(Point("a.w", "cluster",
                       {"centroids": np.zeros(16), "assignments": np.zeros(100, dtype=np.uint8)}))
        lib2.add(Point("b.w", "cluster",
                       {"centroids": np.ones(16), "assignments": np.ones(100, dtype=np.uint8)}))

        dedup = PointDeduplicator()
        dedup.add_library(lib1)
        dedup.add_library(lib2)
        groups = dedup.find_duplicates()
        assert len(groups) == 0

    def test_keep_first_occurrence(self):
        from domains.infrastructure.point_compressor import PointDeduplicator

        cents = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        assns = np.array([0, 1, 2], dtype=np.uint8)

        lib1 = PointLibrary("lib1")
        lib2 = PointLibrary("lib2")
        lib1.add(Point("keep_this", "cluster",
                       {"centroids": cents.copy(), "assignments": assns.copy()}))
        lib2.add(Point("remove_this", "cluster",
                       {"centroids": cents.copy(), "assignments": assns.copy()}))

        dedup = PointDeduplicator()
        dedup.add_library(lib1)
        dedup.add_library(lib2)
        dedup.deduplicate()
        assert lib1.has("keep_this")
        assert not lib2.has("remove_this")

    def test_different_types_not_duplicates(self):
        from domains.infrastructure.point_compressor import PointDeduplicator

        lib1 = PointLibrary("lib1")
        lib2 = PointLibrary("lib2")
        lib1.add(Point("a.w", "linear", {"a": 1.0, "b": 0.0}))
        lib2.add(Point("b.w", "periodic", {"a": 1.0, "b": 0.0, "w": 0.0}))

        dedup = PointDeduplicator()
        dedup.add_library(lib1)
        dedup.add_library(lib2)
        groups = dedup.find_duplicates()
        assert len(groups) == 0

    def test_multi_model_sharing(self):
        """Two models sharing a library — dedup should find cross-model duplicates."""
        from domains.infrastructure.point_compressor import PointDeduplicator

        shared_cents = np.linspace(-1, 1, 16).astype(np.float32)
        shared_assns = np.random.randint(0, 16, size=512).astype(np.uint8)

        lib = PointLibrary("shared")
        tree_a = ModelTree("model_a", lib)
        tree_b = ModelTree("model_b", lib)

        # Both models have identical weights for one layer
        tree_a.load_weights({"layer_0": shared_cents[shared_assns].reshape(512).astype(np.float32)})
        tree_b.load_weights({"layer_0": shared_cents[shared_assns].reshape(512).astype(np.float32)})

        dedup = PointDeduplicator()
        dedup.add_library(lib)
        groups = dedup.find_duplicates()
        # Both trees store as model_a.layer_0 and model_b.layer_0
        # They should be duplicates
        assert len(groups) >= 1


# ── PointLibrarySync tests ──

class TestPointLibrarySync:
    def test_export_import_bytes(self):
        from domains.infrastructure.point_compressor import PointLibrarySync

        lib = PointLibrary("sync_test")
        lib.add(Point("p1", "linear", {"a": 1.0, "b": 0.0}, accuracy=0.9))
        lib.add(Point("p2", "cluster", {
            "centroids": np.array([0.1, 0.5, 0.9], dtype=np.float32),
            "assignments": np.array([0, 1, 2], dtype=np.uint8),
        }, accuracy=0.95))

        sync = PointLibrarySync()
        data = sync.export_bytes(lib)
        assert isinstance(data, bytes)

        restored = sync.import_bytes(data)
        assert restored.name == "sync_test"
        assert len(restored.list_all()) == 2
        assert restored.get("p1").params["a"] == pytest.approx(1.0)

    def test_sync_to_directory(self):
        from domains.infrastructure.point_compressor import PointLibrarySync

        lib = PointLibrary("dir_sync_test")
        lib.add(Point("p1", "linear", {"a": 1.0, "b": 0.0}))

        sync = PointLibrarySync()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = sync.sync_to_directory(lib, Path(tmpdir))
            assert path.exists()
            assert path.name == "dir_sync_test.points.json"

    def test_sync_from_directory(self):
        from domains.infrastructure.point_compressor import PointLibrarySync

        lib = PointLibrary("dir_load_test")
        lib.add(Point("p1", "linear", {"a": 1.0, "b": 0.0}))

        sync = PointLibrarySync()
        with tempfile.TemporaryDirectory() as tmpdir:
            sync.sync_to_directory(lib, Path(tmpdir))
            loaded = sync.sync_from_directory(Path(tmpdir), name="dir_load_test")
            assert loaded is not None
            assert loaded.stats()["total_points"] == 1

    def test_sync_from_directory_not_found(self):
        from domains.infrastructure.point_compressor import PointLibrarySync

        sync = PointLibrarySync()
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = sync.sync_from_directory(Path(tmpdir), name="nonexistent")
            assert loaded is None

    def test_merge_libraries(self):
        from domains.infrastructure.point_compressor import PointLibrarySync

        lib1 = PointLibrary("merge_1")
        lib2 = PointLibrary("merge_2")
        lib1.add(Point("shared", "linear", {"a": 1.0, "b": 0.0}))
        lib1.add(Point("unique_1", "linear", {"a": 2.0, "b": 0.0}))
        lib2.add(Point("shared", "linear", {"a": 1.0, "b": 0.0}))
        lib2.add(Point("unique_2", "linear", {"a": 3.0, "b": 0.0}))

        sync = PointLibrarySync()
        merged = sync.merge([lib1, lib2])
        # Should have 3 unique points (shared deduplicated)
        assert merged.stats()["total_points"] == 3

    def test_roundtrip_persistence(self):
        from domains.infrastructure.point_compressor import PointLibrarySync

        lib = PointLibrary("roundtrip")
        for i in range(5):
            lib.add(Point(f"p{i}", "linear", {"a": float(i), "b": 0.0}, accuracy=0.8 + i * 0.04))

        sync = PointLibrarySync()
        with tempfile.TemporaryDirectory() as tmpdir:
            sync.sync_to_directory(lib, Path(tmpdir))
            loaded = sync.sync_from_directory(Path(tmpdir), name="roundtrip")
            assert loaded.stats()["total_points"] == 5
            for i in range(5):
                assert loaded.has(f"p{i}")


# ── Point binary serialization coverage ──

class TestPointBytes:
    """to_bytes / from_bytes round-trips with residuals and error paths."""

    def test_to_bytes_periodic_with_residual(self):
        residual = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        p = Point(identity="t", function_type="periodic",
                  params={"a": 1.0, "b": 0.5, "w": 0.0}, residual=residual)
        data = p.to_bytes()
        assert data[:4] == b"PER "
        p2 = Point.from_bytes(data, identity="t")
        assert p2.function_type == "periodic"
        assert p2.params["a"] == pytest.approx(1.0)
        assert p2.residual is not None
        np.testing.assert_array_almost_equal(p2.residual, residual)

    def test_to_bytes_linear_with_residual(self):
        residual = np.array([0.01, -0.01], dtype=np.float32)
        p = Point(identity="t", function_type="linear",
                  params={"a": 2.0, "b": 1.0}, residual=residual)
        data = p.to_bytes()
        assert data[:4] == b"LIN "
        p2 = Point.from_bytes(data, identity="t")
        assert p2.function_type == "linear"
        assert p2.residual is not None
        np.testing.assert_array_almost_equal(p2.residual, residual)

    def test_to_bytes_polynomial_with_residual(self):
        residual = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        p = Point(identity="t", function_type="polynomial",
                  params={"a": 0.1, "b": 0.5, "c": 1.0}, residual=residual)
        data = p.to_bytes()
        assert data[:4] == b"POLY"
        p2 = Point.from_bytes(data, identity="t")
        assert p2.function_type == "polynomial"
        assert p2.residual is not None
        np.testing.assert_array_almost_equal(p2.residual, residual)

    def test_to_bytes_unknown_type_returns_empty_params(self):
        p = Point(identity="t", function_type="mystery", params={})
        assert p.to_bytes() == b"\x00\x00\x00\x00"

    def test_from_bytes_unknown_type_code_raises(self):
        with pytest.raises(ValueError):
            Point.from_bytes(b"\x00\x00\x00\x00")

    def test_from_bytes_unknown_function_type_raises(self, monkeypatch):
        import domains.infrastructure.pugqeep.point as point_module
        decode = dict(point_module.Point._TYPE_DECODE)
        decode[b"ZZZZ"] = "mystery"
        monkeypatch.setattr(point_module.Point, "_TYPE_DECODE", decode)
        with pytest.raises(ValueError):
            point_module.Point.from_bytes(b"ZZZZ")

    def test_generate_with_residual(self):
        residual = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        p = Point(identity="t", function_type="linear",
                  params={"a": 0.0, "b": 0.0}, residual=residual)
        np.testing.assert_array_almost_equal(p.generate(3), residual)

    def test_generate_unknown_type_raises(self):
        p = Point(identity="t", function_type="mystery", params={})
        with pytest.raises(ValueError):
            p.generate(3)

    def test_nbytes_raw(self):
        import base64
        raw = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        p = Point(identity="t", function_type="raw",
                  params={"data_b64": base64.b64encode(raw.tobytes()).decode()})
        assert p.nbytes() == raw.nbytes

    def test_nbytes_function_with_residual(self):
        residual = np.array([0.1, 0.2], dtype=np.float32)
        p = Point(identity="t", function_type="linear",
                  params={"a": 1.0, "b": 0.0}, residual=residual)
        assert p.nbytes() == 4 + len(p.params) * 4 + residual.nbytes

    def test_nbytes_cluster_with_residual(self):
        centroids = np.zeros(4, dtype=np.float32)
        assignments = np.zeros(8, dtype=np.uint8)
        residual = np.zeros(8, dtype=np.float32)
        p = Point(identity="t", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments},
                  residual=residual)
        assert p.nbytes() == centroids.nbytes + assignments.nbytes + residual.nbytes


# ── PointLibrary edge-case coverage ──

class TestPointLibraryCoverage:
    def test_remove_auto_save(self, tmp_path):
        from domains.infrastructure.pugqeep.config import LibraryConfig
        cfg = LibraryConfig(name="autosave", auto_save=True, storage_dir=tmp_path)
        lib = PointLibrary(config=cfg)
        lib.add(Point(identity="p1", function_type="linear", params={"a": 1.0, "b": 0.0}))
        assert lib.remove("p1")
        assert not lib.has("p1")
        assert (tmp_path / "autosave.points.json").exists()

    def test_compress_and_store_function(self, library, structured_weights):
        point = library.compress_and_store(structured_weights, "w_f", method="function")
        assert library.has("w_f")
        assert point.function_type in ("periodic", "linear", "polynomial")

    def test_decompress_to_nonexistent(self, library):
        assert library.decompress_to("nope") is None

    def test_decompress_to_cluster(self, library):
        centroids = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        assignments = np.array([0, 1, 2, 0], dtype=np.uint8)
        library.add(Point(identity="clu", function_type="cluster",
                          params={"centroids": centroids, "assignments": assignments}))
        np.testing.assert_array_equal(library.decompress_to("clu"), centroids[assignments])

    def test_decompress_to_function(self, library):
        library.add(Point(identity="lin", function_type="linear",
                          params={"a": 1.0, "b": 0.0}))
        result = library.decompress_to("lin")
        assert result.shape == (2 * 100,)

    def test_decompress_to_with_shape(self, library):
        library.add(Point(identity="lin", function_type="linear",
                          params={"a": 1.0, "b": 0.0}))
        result = library.decompress_to("lin", shape=(10, 20))
        assert result.shape == (10, 20)

    def test_stats_with_residual(self, library):
        residual = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        library.add(Point(identity="lin", function_type="linear",
                          params={"a": 1.0, "b": 0.0}, residual=residual))
        s = library.stats()
        expected = 4 + len({"a": 1.0, "b": 0.0}) * 4 + residual.nbytes
        assert s["total_compressed_bytes"] == expected

    def test_save_without_storage_dir_raises(self, library):
        with pytest.raises(ValueError):
            library.save()
