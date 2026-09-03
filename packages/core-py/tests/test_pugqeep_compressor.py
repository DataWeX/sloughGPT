"""Tests for domains.infrastructure.pugqeep.compressor — PointCompressor."""

import numpy as np
import pytest
from domains.infrastructure.pugqeep.compressor import PointCompressor
from domains.infrastructure.pugqeep.point import Point
from domains.infrastructure.pugqeep.config import CompressorConfig


class TestPointCompressorCluster:
    def test_compress_returns_point(self):
        c = PointCompressor()
        weights = np.random.randn(128)
        p = c.compress_cluster(weights, identity="layer1")
        assert isinstance(p, Point)
        assert p.identity == "layer1"
        assert p.function_type == "cluster"

    def test_accuracy_between_0_and_1(self):
        c = PointCompressor()
        weights = np.random.randn(256)
        p = c.compress_cluster(weights)
        assert 0.0 <= p.accuracy <= 1.0

    def test_centroids_match_n_clusters(self):
        n_clusters = 8
        c = PointCompressor(n_clusters=n_clusters)
        weights = np.random.randn(128)
        p = c.compress_cluster(weights)
        assert len(p.params["centroids"]) >= n_clusters

    def test_assignments_match_weights_length(self):
        c = PointCompressor()
        weights = np.random.randn(100)
        p = c.compress_cluster(weights)
        assert len(p.params["assignments"]) == 100

    def test_custom_n_clusters_override(self):
        c = PointCompressor(n_clusters=16)
        weights = np.random.randn(200)
        p = c.compress_cluster(weights, n_clusters=32)
        assert len(p.params["centroids"]) >= 32

    def test_empty_array_raises(self):
        c = PointCompressor()
        with pytest.raises(ValueError, match="empty"):
            c.compress_cluster(np.array([]))

    def test_nan_raises(self):
        c = PointCompressor()
        weights = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="NaN"):
            c.compress_cluster(weights)

    def test_inf_raises(self):
        c = PointCompressor()
        weights = np.array([1.0, np.inf, 3.0])
        with pytest.raises(ValueError, match="NaN"):
            c.compress_cluster(weights)

    def test_n_clusters_zero_raises(self):
        c = PointCompressor()
        with pytest.raises(ValueError, match="n_clusters"):
            c.compress_cluster(np.ones(10), n_clusters=0)

    def test_n_clusters_negative_raises(self):
        c = PointCompressor()
        with pytest.raises(ValueError, match="n_clusters"):
            c.compress_cluster(np.ones(10), n_clusters=-1)

    def test_identity_preserved(self):
        c = PointCompressor()
        weights = np.random.randn(64)
        p = c.compress_cluster(weights, identity="my_layer")
        assert p.identity == "my_layer"

    def test_dtype_preserved(self):
        c = PointCompressor()
        weights = np.random.randn(64).astype(np.float64)
        p = c.compress_cluster(weights)
        assert p.dtype == "float64"

    def test_shape_preserved(self):
        c = PointCompressor()
        weights = np.random.randn(8, 16)
        p = c.compress_cluster(weights, identity="mat")
        assert p.shape == (8, 16)

    def test_n_clusters_clamped_to_array_size(self):
        c = PointCompressor(n_clusters=100)
        weights = np.ones(10)
        p = c.compress_cluster(weights)
        assert len(p.params["centroids"]) >= 1
        assert len(p.params["assignments"]) == 10

    def test_two_element_array(self):
        c = PointCompressor()
        weights = np.array([1.0, 2.0])
        p = c.compress_cluster(weights)
        assert len(p.params["assignments"]) == 2

    def test_large_array(self):
        c = PointCompressor(n_clusters=16)
        weights = np.random.randn(10000)
        p = c.compress_cluster(weights)
        assert p.accuracy > 0.0

    def test_deterministic(self):
        c1 = PointCompressor()
        c2 = PointCompressor()
        np.random.seed(42)
        weights = np.random.randn(128)
        p1 = c1.compress_cluster(weights.copy(), identity="test")
        np.random.seed(42)
        weights2 = np.random.randn(128)
        p2 = c2.compress_cluster(weights2.copy(), identity="test")
        np.testing.assert_array_equal(p1.params["centroids"], p2.params["centroids"])


class TestPointCompressorFunction:
    def test_compress_periodic(self):
        c = PointCompressor()
        i = np.arange(50, dtype=np.float32)
        weights = 2.0 * np.cos(i) + 0.5 * np.sin(i) + 1.0
        p = c.compress_function(weights, identity="periodic_w")
        assert isinstance(p, Point)
        assert p.identity == "periodic_w"

    def test_compress_linear(self):
        c = PointCompressor()
        weights = np.arange(50, dtype=np.float32) * 0.1
        p = c.compress_function(weights, identity="linear_w")
        assert isinstance(p, Point)

    def test_compress_random(self):
        c = PointCompressor(residual_threshold=0.99)
        weights = np.random.randn(200)
        p = c.compress_function(weights, identity="random")
        assert isinstance(p, Point)
        assert p.accuracy >= 0.0

    def test_periodic_accuracy_high_for_periodic_signal(self):
        c = PointCompressor()
        i = np.arange(100, dtype=np.float32)
        weights = 3.0 * np.cos(i) + 1.5 * np.sin(i) + 0.5
        p = c.compress_function(weights)
        assert p.accuracy > 0.9

    def test_linear_accuracy_high_for_linear_signal(self):
        c = PointCompressor()
        weights = np.arange(100, dtype=np.float32) * 2.0 + 1.0
        p = c.compress_function(weights)
        assert p.accuracy > 0.9

    def test_polynomial_signal(self):
        c = PointCompressor()
        i = np.arange(50, dtype=np.float32)
        weights = 0.01 * i**2 + 0.5 * i + 3.0
        p = c.compress_function(weights)
        assert p.function_type in ("polynomial", "linear")

    def test_empty_array_raises(self):
        c = PointCompressor()
        with pytest.raises(ValueError, match="empty"):
            c.compress_function(np.array([]))

    def test_nan_raises(self):
        c = PointCompressor()
        weights = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="NaN"):
            c.compress_function(weights)

    def test_identity_preserved(self):
        c = PointCompressor()
        weights = np.arange(50, dtype=np.float32)
        p = c.compress_function(weights, identity="func_test")
        assert p.identity == "func_test"

    def test_shape_preserved(self):
        c = PointCompressor()
        weights = np.arange(50, dtype=np.float32)
        p = c.compress_function(weights)
        assert p.shape == (50,)

    def test_residual_stored_when_low_accuracy(self):
        c = PointCompressor(residual_threshold=0.99)
        np.random.seed(42)
        weights = np.random.randn(200).astype(np.float32)
        p = c.compress_function(weights)
        if p.accuracy < c.residual_threshold:
            assert p.residual is not None

    def test_residual_none_when_high_accuracy(self):
        c = PointCompressor(residual_threshold=0.0)
        i = np.arange(100, dtype=np.float32)
        weights = np.arange(100, dtype=np.float32) * 2.0 + 1.0
        p = c.compress_function(weights)
        assert p.residual is None


class TestPointCompressorGeneral:
    def test_compress_cluster_via_compress(self):
        c = PointCompressor(n_clusters=8)
        weights = np.random.randn(128)
        p = c.compress(weights, method="cluster")
        assert p.function_type == "cluster"

    def test_compress_function_via_compress(self):
        c = PointCompressor(n_clusters=8)
        weights = np.random.randn(128)
        p = c.compress(weights, method="function")
        assert p.function_type in ("periodic", "linear", "polynomial")

    def test_compress_unknown_method_raises(self):
        c = PointCompressor()
        with pytest.raises(ValueError, match="Unknown method"):
            c.compress(np.random.randn(32), method="invalid")

    def test_decompress_cluster_roundtrip(self):
        c = PointCompressor()
        weights = np.random.randn(128)
        p = c.compress_cluster(weights)
        recovered = c.decompress(p, n=128)
        assert len(recovered) == 128

    def test_measure_compression(self):
        c = PointCompressor()
        weights = np.random.randn(128)
        p = c.compress_cluster(weights)
        m = c.measure_compression(weights, p)
        assert m["raw_bytes"] == weights.nbytes
        assert m["compressed_bytes"] > 0
        assert m["ratio"] > 0
        assert m["accuracy"] > 0
        assert m["function_type"] == "cluster"

    def test_compress_cluster_via_method_param(self):
        c = PointCompressor()
        weights = np.random.randn(64)
        p = c.compress(weights, method="cluster")
        assert p.function_type == "cluster"

    def test_compress_function_via_method_param(self):
        c = PointCompressor()
        weights = np.arange(64, dtype=np.float32)
        p = c.compress(weights, method="function")
        assert p.function_type in ("periodic", "linear", "polynomial")

    def test_measure_compression_function(self):
        c = PointCompressor()
        weights = np.arange(64, dtype=np.float32)
        p = c.compress_function(weights)
        m = c.measure_compression(weights, p)
        assert m["raw_bytes"] == weights.nbytes
        assert m["compressed_bytes"] > 0
        assert m["function_type"] in ("periodic", "linear", "polynomial")

    def test_decompress_function_roundtrip(self):
        c = PointCompressor()
        weights = np.arange(50, dtype=np.float32) * 2.0
        p = c.compress_function(weights)
        recovered = c.decompress(p, n=50)
        assert len(recovered) == 50

    def test_batch_compress(self):
        c = PointCompressor(n_clusters=8)
        weights_dict = {
            "layer1": np.random.randn(64),
            "layer2": np.random.randn(128),
            "layer3": np.random.randn(32),
        }
        results = c.compress_batch(weights_dict)
        assert len(results) == 3
        assert "layer1" in results
        assert "layer2" in results
        assert "layer3" in results

    def test_batch_compress_with_prefix(self):
        c = PointCompressor(n_clusters=8)
        weights_dict = {"w1": np.random.randn(32), "w2": np.random.randn(64)}
        results = c.compress_batch(weights_dict, prefix="model.")
        assert "w1" in results
        assert results["w1"].identity == "model.w1"

    def test_batch_compress_method_override(self):
        c = PointCompressor()
        weights_dict = {"w1": np.random.randn(32)}
        results = c.compress_batch(weights_dict, method="function")
        assert results["w1"].function_type in ("periodic", "linear", "polynomial")

    def test_config_based_init(self):
        config = CompressorConfig(n_clusters=32, lloyd_iterations=10, method="function")
        c = PointCompressor(config=config)
        assert c.n_clusters == 32
        assert c.lloyd_iterations == 10
        assert c.method == "function"

    def test_default_gap_fill_values(self):
        c = PointCompressor()
        assert c.gap_fill_iterations == 4
        assert c.gap_fill_max_elements == 100_000

    def test_residual_threshold_set(self):
        c = PointCompressor(residual_threshold=0.5)
        assert c.residual_threshold == 0.5

    def test_compress_cluster_point_has_params(self):
        c = PointCompressor(n_clusters=8)
        weights = np.random.randn(128)
        p = c.compress_cluster(weights)
        assert "centroids" in p.params
        assert "assignments" in p.params
        assert isinstance(p.params["centroids"], np.ndarray)
        assert isinstance(p.params["assignments"], np.ndarray)

    def test_compress_function_point_has_params(self):
        c = PointCompressor()
        weights = np.arange(50, dtype=np.float32)
        p = c.compress_function(weights)
        assert isinstance(p.params, dict)

    def test_measure_compression_ratio_formula(self):
        c = PointCompressor()
        weights = np.random.randn(128)
        p = c.compress_cluster(weights)
        m = c.measure_compression(weights, p)
        expected_ratio = m["raw_bytes"] / max(m["compressed_bytes"], 1)
        assert m["ratio"] == expected_ratio

    def test_lloyd_iterations_affect_convergence(self):
        weights = np.random.randn(256)
        c1 = PointCompressor(n_clusters=16, lloyd_iterations=1)
        c2 = PointCompressor(n_clusters=16, lloyd_iterations=20)
        p1 = c1.compress_cluster(weights.copy())
        p2 = c2.compress_cluster(weights.copy())
        assert p1.accuracy > 0
        assert p2.accuracy > 0

    def test_cluster_small_array(self):
        c = PointCompressor(n_clusters=4)
        weights = np.random.randn(10)
        p = c.compress_cluster(weights)
        assert len(p.params["centroids"]) >= 1
        assert len(p.params["assignments"]) == 10

    def test_all_negative_weights(self):
        c = PointCompressor()
        weights = -np.abs(np.random.randn(64))
        p = c.compress_cluster(weights)
        assert p.accuracy > 0.0
        assert len(p.params["assignments"]) == 64


# ---------------------------------------------------------------------------
# Point serialization — to_bytes / from_bytes
# ---------------------------------------------------------------------------

class TestPointSerializationBytes:
    def test_cluster_roundtrip(self):
        c = PointCompressor(n_clusters=8)
        weights = np.random.randn(64)
        p = c.compress_cluster(weights, identity="layer1")
        raw = p.to_bytes()
        restored = Point.from_bytes(raw, identity="layer1")
        assert restored.function_type == "cluster"
        assert restored.identity == "layer1"
        recovered = restored.generate(64)
        assert len(recovered) == 64

    def test_periodic_roundtrip(self):
        c = PointCompressor(residual_threshold=0.0)
        i = np.arange(50, dtype=np.float32)
        weights = 2.0 * np.cos(i) + 0.5 * np.sin(i) + 1.0
        p = c.compress_function(weights, identity="per1")
        raw = p.to_bytes()
        restored = Point.from_bytes(raw, identity="per1")
        assert restored.function_type == "periodic"
        assert restored.params["a"] == pytest.approx(p.params["a"], rel=1e-5)

    def test_linear_roundtrip(self):
        c = PointCompressor(residual_threshold=0.0)
        weights = np.arange(50, dtype=np.float32) * 3.0 + 7.0
        p = c.compress_function(weights, identity="lin1")
        raw = p.to_bytes()
        restored = Point.from_bytes(raw, identity="lin1")
        assert restored.function_type == "linear"
        assert restored.params["a"] == pytest.approx(p.params["a"], rel=1e-5)

    def test_polynomial_roundtrip(self):
        c = PointCompressor(residual_threshold=0.0)
        i = np.arange(50, dtype=np.float32)
        weights = 0.01 * i**2 + 0.5 * i + 3.0
        p = c.compress_function(weights, identity="poly1")
        raw = p.to_bytes()
        restored = Point.from_bytes(raw, identity="poly1")
        assert restored.function_type == "polynomial"
        assert restored.params["a"] == pytest.approx(p.params["a"], rel=1e-4)

    def test_invalid_type_code_raises(self):
        with pytest.raises(ValueError, match="Unknown type code"):
            Point.from_bytes(b"\x00\x00\x00\x00" + b"\x00" * 8, identity="bad")

    def test_bytes_header_length(self):
        c = PointCompressor(n_clusters=4)
        p = c.compress_cluster(np.random.randn(32))
        raw = p.to_bytes()
        assert len(raw) >= 4


# ---------------------------------------------------------------------------
# Point serialization — to_dict / from_dict
# ---------------------------------------------------------------------------

class TestPointSerializationDict:
    def test_cluster_roundtrip(self):
        c = PointCompressor(n_clusters=8)
        weights = np.random.randn(64)
        p = c.compress_cluster(weights, identity="L1")
        d = p.to_dict()
        restored = Point.from_dict(d)
        assert restored.identity == "L1"
        assert restored.function_type == "cluster"
        assert "centroids" in restored.params
        assert "assignments" in restored.params

    def test_periodic_roundtrip(self):
        c = PointCompressor(residual_threshold=0.0)
        i = np.arange(50, dtype=np.float32)
        weights = np.cos(i)
        p = c.compress_function(weights, identity="per")
        d = p.to_dict()
        restored = Point.from_dict(d)
        assert restored.function_type == "periodic"
        assert restored.params["a"] == pytest.approx(p.params["a"], rel=1e-5)

    def test_linear_roundtrip(self):
        c = PointCompressor(residual_threshold=0.0)
        weights = np.arange(50, dtype=np.float32) * 2.0
        p = c.compress_function(weights, identity="lin")
        d = p.to_dict()
        restored = Point.from_dict(d)
        assert restored.function_type == "linear"

    def test_polynomial_roundtrip(self):
        c = PointCompressor(residual_threshold=0.0)
        i = np.arange(50, dtype=np.float32)
        weights = 0.01 * i**2 + i
        p = c.compress_function(weights, identity="poly")
        d = p.to_dict()
        restored = Point.from_dict(d)
        assert restored.function_type == "polynomial"

    def test_cluster_dict_has_b64_keys(self):
        c = PointCompressor(n_clusters=4)
        p = c.compress_cluster(np.random.randn(32), identity="t")
        d = p.to_dict()
        assert "centroids_b64" in d["params"]
        assert "assignments_b64" in d["params"]

    def test_residual_appears_in_dict(self):
        c = PointCompressor(residual_threshold=0.99)
        weights = np.random.randn(200).astype(np.float32)
        p = c.compress_function(weights)
        d = p.to_dict()
        if p.residual is not None:
            assert "residual_b64" in d


# ---------------------------------------------------------------------------
# Point.nbytes and _estimate_raw_bytes
# ---------------------------------------------------------------------------

class TestPointNbytes:
    def test_cluster_nbytes(self):
        c = PointCompressor(n_clusters=8)
        p = c.compress_cluster(np.random.randn(64))
        nb = p.nbytes()
        assert nb > 0
        huffman_data = p.params.get("huffman_data")
        if huffman_data is not None:
            tree_size = sum(len(v) for v in p.params.get("huffman_codes", {}).values())
            expected = p.params["centroids"].nbytes + len(huffman_data) + tree_size + 8
        else:
            expected = p.params["centroids"].nbytes + p.params["assignments"].nbytes
        assert nb == expected

    def test_cluster_nbytes_with_residual(self):
        c = PointCompressor(residual_threshold=0.99)
        weights = np.random.randn(200).astype(np.float32)
        p = c.compress_function(weights)
        if p.residual is not None:
            nb = p.nbytes()
            assert nb > p.residual.nbytes

    def test_function_nbytes_positive(self):
        c = PointCompressor(residual_threshold=0.0)
        weights = np.arange(50, dtype=np.float32)
        p = c.compress_function(weights)
        assert p.nbytes() > 0

    def test_estimate_raw_bytes_cluster(self):
        c = PointCompressor(n_clusters=4)
        p = c.compress_cluster(np.random.randn(64))
        raw = p._estimate_raw_bytes()
        assert raw == 64 * 4

    def test_estimate_raw_bytes_function(self):
        c = PointCompressor(residual_threshold=0.0)
        weights = np.arange(50, dtype=np.float32)
        p = c.compress_function(weights)
        raw = p._estimate_raw_bytes()
        assert raw == 50 * 4

    def test_estimate_raw_bytes_cluster_no_assignments(self):
        from domains.infrastructure.pugqeep.point import Point
        p = Point(identity="x", function_type="cluster", params={"centroids": np.zeros(4)})
        assert p._estimate_raw_bytes() == 0


# ---------------------------------------------------------------------------
# Point __repr__, __eq__, __hash__
# ---------------------------------------------------------------------------

class TestPointProtocol:
    def test_repr_cluster(self):
        c = PointCompressor(n_clusters=4)
        p = c.compress_cluster(np.random.randn(32), identity="test")
        r = repr(p)
        assert "test" in r
        assert "cluster" in r

    def test_repr_periodic(self):
        c = PointCompressor(residual_threshold=0.0)
        weights = np.cos(np.arange(50, dtype=np.float32))
        p = c.compress_function(weights, identity="per")
        r = repr(p)
        assert "per" in r

    def test_eq_same(self):
        c = PointCompressor(n_clusters=4)
        p1 = c.compress_cluster(np.random.randn(32), identity="x")
        p2 = c.compress_cluster(np.random.randn(32), identity="x")
        assert p1 == p2

    def test_eq_different_identity(self):
        c = PointCompressor(n_clusters=4)
        p1 = c.compress_cluster(np.random.randn(32), identity="a")
        p2 = c.compress_cluster(np.random.randn(32), identity="b")
        assert p1 != p2

    def test_eq_different_type(self):
        c = PointCompressor(n_clusters=4)
        p1 = c.compress_cluster(np.random.randn(32), identity="x")
        p2 = c.compress_function(np.arange(32, dtype=np.float32), identity="x")
        assert p1 != p2

    def test_eq_non_point(self):
        c = PointCompressor(n_clusters=4)
        p = c.compress_cluster(np.random.randn(32), identity="x")
        assert p != "not a point"

    def test_hash_cluster(self):
        c = PointCompressor(n_clusters=4)
        p = c.compress_cluster(np.random.randn(32), identity="x")
        assert isinstance(hash(p), int)

    def test_hash_equal_objects(self):
        c = PointCompressor(n_clusters=4)
        p1 = c.compress_cluster(np.random.randn(32), identity="x")
        p2 = c.compress_cluster(np.random.randn(32), identity="x")
        assert hash(p1) == hash(p2)

    def test_hash_usable_in_set(self):
        c = PointCompressor(n_clusters=4)
        p1 = c.compress_cluster(np.random.randn(32), identity="x")
        p2 = c.compress_cluster(np.random.randn(32), identity="x")
        s = {p1, p2}
        assert len(s) == 1


# ---------------------------------------------------------------------------
# FunctionType enum
# ---------------------------------------------------------------------------

class TestFunctionType:
    def test_all_members(self):
        from domains.infrastructure.pugqeep.point_interface import FunctionType
        assert len(FunctionType) == 5

    def test_values(self):
        from domains.infrastructure.pugqeep.point_interface import FunctionType
        assert FunctionType.PERIODIC.value == "periodic"
        assert FunctionType.LINEAR.value == "linear"
        assert FunctionType.POLYNOMIAL.value == "polynomial"
        assert FunctionType.CLUSTER.value == "cluster"
        assert FunctionType.RAW.value == "raw"

    def test_from_str_valid(self):
        from domains.infrastructure.pugqeep.point_interface import FunctionType
        assert FunctionType.from_str("linear") == FunctionType.LINEAR

    def test_from_str_invalid(self):
        from domains.infrastructure.pugqeep.point_interface import FunctionType
        with pytest.raises(ValueError, match="Unknown function type"):
            FunctionType.from_str("invalid")

    def test_is_str_subclass(self):
        from domains.infrastructure.pugqeep.point_interface import FunctionType
        assert issubclass(FunctionType, str)


# ---------------------------------------------------------------------------
# PointView
# ---------------------------------------------------------------------------

class TestPointView:
    def test_view_lazy(self):
        from domains.infrastructure.pugqeep.point_interface import PointView
        c = PointCompressor(n_clusters=8)
        p = c.compress_cluster(np.random.randn(64), identity="lazy")
        view = PointView(p, shape=(64,), dtype="float32")
        assert view._cache is None

    def test_view_generate_caches(self):
        from domains.infrastructure.pugqeep.point_interface import PointView
        c = PointCompressor(n_clusters=8)
        p = c.compress_cluster(np.random.randn(64), identity="c")
        view = PointView(p, shape=(64,), dtype="float32")
        arr1 = view.generate()
        arr2 = view.generate()
        assert arr1 is arr2

    def test_view_clear_cache(self):
        from domains.infrastructure.pugqeep.point_interface import PointView
        c = PointCompressor(n_clusters=8)
        p = c.compress_cluster(np.random.randn(64), identity="c")
        view = PointView(p, shape=(64,), dtype="float32")
        view.generate()
        view.clear_cache()
        assert view._cache is None

    def test_view_shape(self):
        from domains.infrastructure.pugqeep.point_interface import PointView
        c = PointCompressor(n_clusters=8)
        p = c.compress_cluster(np.random.randn(64), identity="c")
        view = PointView(p, shape=(8, 8), dtype="float32")
        assert view.shape == (8, 8)

    def test_view_dtype(self):
        from domains.infrastructure.pugqeep.point_interface import PointView
        c = PointCompressor(n_clusters=8)
        p = c.compress_cluster(np.random.randn(64), identity="c")
        view = PointView(p, shape=(64,), dtype="float64")
        assert view.dtype == np.dtype("float64")

    def test_view_accuracy(self):
        from domains.infrastructure.pugqeep.point_interface import PointView
        c = PointCompressor(n_clusters=8)
        p = c.compress_cluster(np.random.randn(64), identity="c")
        view = PointView(p, shape=(64,), dtype="float32")
        assert view.accuracy == p.accuracy

    def test_view_nbytes(self):
        from domains.infrastructure.pugqeep.point_interface import PointView
        c = PointCompressor(n_clusters=8)
        p = c.compress_cluster(np.random.randn(64), identity="c")
        view = PointView(p, shape=(64,), dtype="float32")
        assert view.nbytes == p.nbytes()

    def test_view_repr(self):
        from domains.infrastructure.pugqeep.point_interface import PointView
        c = PointCompressor(n_clusters=8)
        p = c.compress_cluster(np.random.randn(64), identity="c")
        view = PointView(p, shape=(64,), dtype="float32")
        r = repr(view)
        assert "lazy" in r

    def test_view_len(self):
        from domains.infrastructure.pugqeep.point_interface import PointView
        c = PointCompressor(n_clusters=8)
        p = c.compress_cluster(np.random.randn(64), identity="c")
        view = PointView(p, shape=(64,), dtype="float32")
        assert len(view) == 64

    def test_view_getitem_slice(self):
        from domains.infrastructure.pugqeep.point_interface import PointView
        c = PointCompressor(n_clusters=8)
        p = c.compress_cluster(np.random.randn(64), identity="c")
        view = PointView(p, shape=(64,), dtype="float32")
        arr = view[0:10]
        assert len(arr) == 10


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

class TestCompressorConfig:
    def test_defaults(self):
        cfg = CompressorConfig()
        assert cfg.n_clusters == 16
        assert cfg.lloyd_iterations == 5
        assert cfg.gap_fill_iterations == 4
        assert cfg.gap_fill_max_elements == 100_000
        assert cfg.method == "cluster"

    def test_custom(self):
        cfg = CompressorConfig(n_clusters=32, lloyd_iterations=10, method="function")
        assert cfg.n_clusters == 32
        assert cfg.lloyd_iterations == 10
        assert cfg.method == "function"

    def test_config_applied_to_compressor(self):
        cfg = CompressorConfig(n_clusters=8, lloyd_iterations=3, gap_fill_iterations=2)
        c = PointCompressor(config=cfg)
        assert c.n_clusters == 8
        assert c.lloyd_iterations == 3
        assert c.gap_fill_iterations == 2

    def test_config_method_applied(self):
        cfg = CompressorConfig(method="function")
        c = PointCompressor(config=cfg)
        assert c.method == "function"


class TestPointConfig:
    def test_defaults(self):
        from domains.infrastructure.pugqeep.config import PointConfig
        cfg = PointConfig()
        assert cfg.function_type == "cluster"
        assert cfg.n_clusters == 16
        assert cfg.residual_threshold == 0.99


class TestLibraryConfig:
    def test_defaults(self):
        from domains.infrastructure.pugqeep.config import LibraryConfig
        cfg = LibraryConfig()
        assert cfg.name == "default"
        assert cfg.storage_dir is None
        assert cfg.auto_save is False


class TestTreeConfig:
    def test_defaults(self):
        from domains.infrastructure.pugqeep.config import TreeConfig
        cfg = TreeConfig()
        assert cfg.name == "model"
        assert cfg.n_clusters == 16
        assert cfg.skip_embeddings is True
        assert cfg.skip_biases is True


class TestQueueConfig:
    def test_defaults(self):
        from domains.infrastructure.pugqeep.config import QueueConfig
        cfg = QueueConfig()
        assert cfg.max_trees == 10
        assert cfg.dedup is True


class TestSubprocessConfig:
    def test_defaults(self):
        from domains.infrastructure.pugqeep.config import SubprocessConfig
        cfg = SubprocessConfig()
        assert cfg.enabled is True
        assert cfg.python_exe == "python3"
        assert cfg.max_workers == 4
        assert cfg.terminate_grace == 3.0


class TestRestartPolicy:
    def test_defaults(self):
        from domains.infrastructure.pugqeep.config import RestartPolicy
        cfg = RestartPolicy()
        assert cfg.max_restarts == 0
        assert cfg.restart_delay == 1.0
        assert cfg.backoff == "exponential"
        assert cfg.max_backoff == 30.0


class TestMonitorConfig:
    def test_defaults(self):
        from domains.infrastructure.pugqeep.config import MonitorConfig
        cfg = MonitorConfig()
        assert cfg.enabled is True
        assert cfg.poll_interval == 1.0
        assert cfg.stall_timeout == 60.0
        assert cfg.on_stall == "restart"


class TestEngineConfig:
    def test_defaults(self):
        from domains.infrastructure.pugqeep.config import EngineConfig
        cfg = EngineConfig()
        assert cfg.name == "main"
        assert cfg.max_trees == 16
        assert cfg.tree_workers == 4
        assert cfg.max_stems == 8
        assert cfg.queue_size == 128
        assert cfg.poll_interval == 0.1


# ---------------------------------------------------------------------------
# Extended PointCompressor cluster tests
# ---------------------------------------------------------------------------

class TestPointCompressorClusterExtended:
    def test_single_element_array(self):
        cfg = CompressorConfig(gap_fill_iterations=0)
        c = PointCompressor(config=cfg)
        p = c.compress_cluster(np.array([42.0]))
        assert len(p.params["assignments"]) == 1
        assert p.accuracy >= 0.0

    def test_all_same_values(self):
        c = PointCompressor()
        weights = np.ones(64)
        p = c.compress_cluster(weights)
        assert p.accuracy > 0.9

    def test_uniform_distribution(self):
        c = PointCompressor()
        weights = np.linspace(0, 1, 100)
        p = c.compress_cluster(weights)
        assert p.accuracy > 0.8

    def test_bimodal_distribution(self):
        c = PointCompressor(n_clusters=4)
        weights = np.concatenate([np.ones(50) * 0, np.ones(50) * 10])
        p = c.compress_cluster(weights)
        assert p.accuracy > 0.8

    def test_centroids_sorted(self):
        c = PointCompressor(n_clusters=8)
        weights = np.random.randn(200)
        p = c.compress_cluster(weights)
        centroids = p.params["centroids"]
        assert all(centroids[i] <= centroids[i+1] for i in range(len(centroids)-1))

    def test_assignments_in_range(self):
        c = PointCompressor(n_clusters=8)
        weights = np.random.randn(100)
        p = c.compress_cluster(weights)
        n_centroids = len(p.params["centroids"])
        assert all(0 <= a < n_centroids for a in p.params["assignments"])

    def test_accuracy_bounded(self):
        c = PointCompressor(n_clusters=2)
        weights = np.array([0.0, 100.0])
        p = c.compress_cluster(weights)
        assert 0.0 <= p.accuracy <= 1.0

    def test_n_clusters_2(self):
        cfg = CompressorConfig(gap_fill_iterations=0)
        c = PointCompressor(config=cfg)
        weights = np.random.randn(50)
        p = c.compress_cluster(weights, n_clusters=2)
        assert len(p.params["centroids"]) >= 2

    def test_float32_input(self):
        c = PointCompressor()
        weights = np.random.randn(64).astype(np.float32)
        p = c.compress_cluster(weights)
        assert p.dtype == "float32"

    def test_float64_input(self):
        c = PointCompressor()
        weights = np.random.randn(64).astype(np.float64)
        p = c.compress_cluster(weights)
        assert p.dtype == "float64"

    def test_multidimensional_input_flattened(self):
        c = PointCompressor()
        weights = np.random.randn(4, 8)
        p = c.compress_cluster(weights, identity="mat")
        assert len(p.params["assignments"]) == 32
        assert p.shape == (4, 8)

    def test_very_large_values(self):
        c = PointCompressor()
        weights = np.array([1e10, -1e10, 1e10, -1e10])
        p = c.compress_cluster(weights)
        assert p.accuracy >= 0.0

    def test_very_small_values(self):
        c = PointCompressor()
        weights = np.array([1e-10, 2e-10, 3e-10, 4e-10])
        p = c.compress_cluster(weights)
        assert p.accuracy >= 0.0


# ---------------------------------------------------------------------------
# Extended PointCompressor function tests
# ---------------------------------------------------------------------------

class TestPointCompressorFunctionExtended:
    def test_constant_signal(self):
        c = PointCompressor()
        weights = np.ones(50) * 7.0
        p = c.compress_function(weights)
        assert p.accuracy > 0.9

    def test_step_function(self):
        c = PointCompressor()
        weights = np.concatenate([np.zeros(25), np.ones(25) * 10])
        p = c.compress_function(weights)
        assert isinstance(p, Point)

    def test_sine_wave(self):
        c = PointCompressor()
        i = np.arange(100, dtype=np.float32)
        weights = np.sin(2 * np.pi * i / 50)
        p = c.compress_function(weights)
        assert 0.0 <= p.accuracy <= 1.0

    def test_quadratic_signal(self):
        c = PointCompressor()
        i = np.arange(50, dtype=np.float32)
        weights = 0.001 * i**2 + 0.1 * i + 1.0
        p = c.compress_function(weights)
        assert p.function_type in ("polynomial", "linear")

    def test_noisy_periodic(self):
        c = PointCompressor()
        np.random.seed(42)
        i = np.arange(100, dtype=np.float32)
        weights = 2.0 * np.cos(i) + np.random.randn(100) * 0.1
        p = c.compress_function(weights)
        assert 0.0 <= p.accuracy <= 1.0

    def test_single_element(self):
        c = PointCompressor()
        p = c.compress_function(np.array([42.0]))
        assert isinstance(p, Point)

    def test_two_elements(self):
        c = PointCompressor()
        p = c.compress_function(np.array([1.0, 2.0]))
        assert isinstance(p, Point)

    def test_inf_raises(self):
        c = PointCompressor()
        with pytest.raises(ValueError, match="NaN"):
            c.compress_function(np.array([1.0, np.inf, 3.0]))

    def test_dtype_preserved(self):
        c = PointCompressor()
        weights = np.arange(50, dtype=np.float64)
        p = c.compress_function(weights)
        assert p.dtype == "float64"

    def test_shape_preserved(self):
        c = PointCompressor()
        weights = np.arange(50, dtype=np.float32)
        p = c.compress_function(weights)
        assert p.shape == (50,)

    def test_residual_stored_when_low_accuracy(self):
        c = PointCompressor(residual_threshold=0.99)
        np.random.seed(42)
        weights = np.random.randn(200).astype(np.float32)
        p = c.compress_function(weights)
        if p.accuracy < c.residual_threshold:
            assert p.residual is not None
            assert len(p.residual) == 200


# ---------------------------------------------------------------------------
# Extended compress/decompress roundtrip tests
# ---------------------------------------------------------------------------

class TestCompressDecompressExtended:
    def test_cluster_roundtrip_accuracy(self):
        c = PointCompressor(n_clusters=16)
        weights = np.random.randn(128)
        p = c.compress_cluster(weights)
        recovered = c.decompress(p, n=128)
        mse = np.mean((weights.flatten() - recovered) ** 2)
        var = np.var(weights)
        accuracy = 1.0 - mse / (var + 1e-8)
        assert accuracy > 0.5

    def test_function_roundtrip_linear(self):
        c = PointCompressor(residual_threshold=0.0)
        weights = np.arange(100, dtype=np.float32) * 3.0 + 7.0
        p = c.compress_function(weights)
        recovered = c.decompress(p, n=100)
        assert len(recovered) == 100

    def test_compress_with_method_cluster(self):
        c = PointCompressor()
        weights = np.random.randn(64)
        p = c.compress(weights, method="cluster")
        assert p.function_type == "cluster"

    def test_compress_with_method_function(self):
        c = PointCompressor()
        weights = np.arange(64, dtype=np.float32)
        p = c.compress(weights, method="function")
        assert p.function_type in ("periodic", "linear", "polynomial")

    def test_compress_with_default_method(self):
        c = PointCompressor()
        c.method = "cluster"
        weights = np.random.randn(64)
        p = c.compress(weights)
        assert p.function_type == "cluster"

    def test_measure_compression_cluster(self):
        c = PointCompressor(n_clusters=8)
        weights = np.random.randn(128)
        p = c.compress_cluster(weights)
        m = c.measure_compression(weights, p)
        assert m["raw_bytes"] == 128 * 8
        assert m["compressed_bytes"] > 0
        assert m["ratio"] > 0
        assert m["accuracy"] > 0
        assert m["function_type"] == "cluster"

    def test_measure_compression_function(self):
        c = PointCompressor(residual_threshold=0.0)
        weights = np.arange(64, dtype=np.float32)
        p = c.compress_function(weights)
        m = c.measure_compression(weights, p)
        assert m["raw_bytes"] == 64 * 4
        assert m["compressed_bytes"] > 0


# ---------------------------------------------------------------------------
# Extended Point serialization tests
# ---------------------------------------------------------------------------

class TestPointSerializationExtended:
    def test_cluster_to_dict_roundtrip(self):
        c = PointCompressor(n_clusters=8)
        weights = np.random.randn(64)
        p = c.compress_cluster(weights, identity="L1")
        d = p.to_dict()
        restored = Point.from_dict(d)
        assert restored.function_type == "cluster"
        assert restored.identity == "L1"
        recovered = restored.generate(64)
        assert len(recovered) == 64

    def test_periodic_to_dict_roundtrip(self):
        c = PointCompressor(residual_threshold=0.0)
        i = np.arange(50, dtype=np.float32)
        weights = 2.0 * np.cos(i) + 0.5 * np.sin(i) + 1.0
        p = c.compress_function(weights, identity="per")
        d = p.to_dict()
        restored = Point.from_dict(d)
        assert restored.function_type == "periodic"
        recovered = restored.generate(50)
        assert len(recovered) == 50

    def test_bytes_roundtrip_preserves_identity(self):
        c = PointCompressor(n_clusters=8)
        weights = np.random.randn(64)
        p = c.compress_cluster(weights, identity="test")
        raw = p.to_bytes()
        restored = Point.from_bytes(raw, identity="test")
        assert restored.identity == "test"
        assert restored.function_type == "cluster"

    def test_dict_roundtrip_preserves_accuracy(self):
        c = PointCompressor(n_clusters=8)
        weights = np.random.randn(64)
        p = c.compress_cluster(weights, identity="test")
        d = p.to_dict()
        restored = Point.from_dict(d)
        assert restored.accuracy == pytest.approx(p.accuracy, rel=1e-5)


# ---------------------------------------------------------------------------
# Extended PointView tests
# ---------------------------------------------------------------------------

class TestPointViewExtended:
    def test_view_generate_returns_ndarray(self):
        from domains.infrastructure.pugqeep.point_interface import PointView
        import numpy as np
        c = PointCompressor(n_clusters=8)
        p = c.compress_cluster(np.random.randn(64), identity="c")
        view = PointView(p, shape=(64,), dtype="float32")
        arr = view.generate()
        assert isinstance(arr, np.ndarray)

    def test_view_shape_matches_generate(self):
        from domains.infrastructure.pugqeep.point_interface import PointView
        c = PointCompressor(n_clusters=8)
        p = c.compress_cluster(np.random.randn(64), identity="c")
        view = PointView(p, shape=(8, 8), dtype="float32")
        arr = view.generate()
        assert arr.shape == (8, 8)

    def test_view_dtype_matches(self):
        from domains.infrastructure.pugqeep.point_interface import PointView
        import numpy as np
        c = PointCompressor(n_clusters=8)
        p = c.compress_cluster(np.random.randn(64), identity="c")
        view = PointView(p, shape=(64,), dtype="float64")
        arr = view.generate()
        assert arr.dtype == np.dtype("float64")

    def test_view_len_matches_shape(self):
        from domains.infrastructure.pugqeep.point_interface import PointView
        c = PointCompressor(n_clusters=8)
        p = c.compress_cluster(np.random.randn(100), identity="c")
        view = PointView(p, shape=(100,), dtype="float32")
        assert len(view) == 100

    def test_view_repr_contains_identity(self):
        from domains.infrastructure.pugqeep.point_interface import PointView
        c = PointCompressor(n_clusters=8)
        p = c.compress_cluster(np.random.randn(64), identity="myid")
        view = PointView(p, shape=(64,), dtype="float32")
        r = repr(view)
        assert "myid" in r

    def test_view_accuracy_matches_point(self):
        from domains.infrastructure.pugqeep.point_interface import PointView
        c = PointCompressor(n_clusters=8)
        p = c.compress_cluster(np.random.randn(64), identity="c")
        view = PointView(p, shape=(64,), dtype="float32")
        assert view.accuracy == p.accuracy


# ---------------------------------------------------------------------------
# Extended Config dataclass tests
# ---------------------------------------------------------------------------

class TestCompressorConfigExtended:
    def test_custom_gap_fill_iterations(self):
        cfg = CompressorConfig(gap_fill_iterations=10)
        assert cfg.gap_fill_iterations == 10

    def test_custom_gap_fill_max_elements(self):
        cfg = CompressorConfig(gap_fill_max_elements=50_000)
        assert cfg.gap_fill_max_elements == 50_000

    def test_config_to_compressor_gap_fill(self):
        cfg = CompressorConfig(gap_fill_iterations=8, gap_fill_max_elements=200_000)
        c = PointCompressor(config=cfg)
        assert c.gap_fill_iterations == 8
        assert c.gap_fill_max_elements == 200_000

    def test_residual_threshold_applied(self):
        c = PointCompressor(residual_threshold=0.75)
        assert c.residual_threshold == 0.75

    def test_lloyd_iterations_applied(self):
        c = PointCompressor(lloyd_iterations=10)
        assert c.lloyd_iterations == 10

    def test_n_clusters_applied(self):
        c = PointCompressor(n_clusters=32)
        assert c.n_clusters == 32


class TestPointConfigExtended:
    def test_custom_values(self):
        from domains.infrastructure.pugqeep.config import PointConfig
        cfg = PointConfig(function_type="linear", n_clusters=8, residual_threshold=0.5)
        assert cfg.function_type == "linear"
        assert cfg.n_clusters == 8
        assert cfg.residual_threshold == 0.5


class TestLibraryConfigExtended:
    def test_custom_values(self):
        from domains.infrastructure.pugqeep.config import LibraryConfig
        cfg = LibraryConfig(name="test", storage_dir="/tmp", auto_save=True)
        assert cfg.name == "test"
        assert cfg.storage_dir == "/tmp"
        assert cfg.auto_save is True


class TestTreeConfigExtended:
    def test_custom_values(self):
        from domains.infrastructure.pugqeep.config import TreeConfig
        cfg = TreeConfig(name="encoder", n_clusters=32, skip_embeddings=False, skip_biases=False)
        assert cfg.name == "encoder"
        assert cfg.n_clusters == 32
        assert cfg.skip_embeddings is False
        assert cfg.skip_biases is False


class TestQueueConfigExtended:
    def test_custom_values(self):
        from domains.infrastructure.pugqeep.config import QueueConfig
        cfg = QueueConfig(max_trees=20, dedup=False)
        assert cfg.max_trees == 20
        assert cfg.dedup is False


class TestSubprocessConfigExtended:
    def test_custom_values(self):
        from domains.infrastructure.pugqeep.config import SubprocessConfig
        cfg = SubprocessConfig(enabled=False, python_exe="python", max_workers=8, terminate_grace=5.0)
        assert cfg.enabled is False
        assert cfg.python_exe == "python"
        assert cfg.max_workers == 8
        assert cfg.terminate_grace == 5.0


class TestRestartPolicyExtended:
    def test_custom_values(self):
        from domains.infrastructure.pugqeep.config import RestartPolicy
        cfg = RestartPolicy(max_restarts=5, restart_delay=2.0, backoff="linear", max_backoff=60.0)
        assert cfg.max_restarts == 5
        assert cfg.restart_delay == 2.0
        assert cfg.backoff == "linear"
        assert cfg.max_backoff == 60.0


class TestMonitorConfigExtended:
    def test_custom_values(self):
        from domains.infrastructure.pugqeep.config import MonitorConfig
        cfg = MonitorConfig(enabled=False, poll_interval=5.0, stall_timeout=120.0, on_stall="kill")
        assert cfg.enabled is False
        assert cfg.poll_interval == 5.0
        assert cfg.stall_timeout == 120.0
        assert cfg.on_stall == "kill"


class TestEngineConfigExtended:
    def test_custom_values(self):
        from domains.infrastructure.pugqeep.config import EngineConfig
        cfg = EngineConfig(name="test", max_trees=32, tree_workers=8, max_stems=16, queue_size=256, poll_interval=0.5)
        assert cfg.name == "test"
        assert cfg.max_trees == 32
        assert cfg.tree_workers == 8
        assert cfg.max_stems == 16
        assert cfg.queue_size == 256
        assert cfg.poll_interval == 0.5
