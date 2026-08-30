"""Tests for PointWeight — Point-based weight representation in SloNet.

Verifies:
  - PointWeight compression and generation
  - SloLinear.set_point_weight() integration
  - SloLinear.compress_to_point() end-to-end
  - Round-trip: numpy -> Point -> generate -> numpy
  - Compression ratio and accuracy
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from domains.infrastructure.pugqeep.point import Point
from domains.infrastructure.pugqeep.point_weight import PointWeight, compress_slonet_to_points
from domains.infrastructure.pugqeep.compressor import PointCompressor


class TestPointWeight:
    def test_from_array_cluster(self):
        w = np.random.randn(64, 128).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test", method="cluster")
        assert pw.shape == (64, 128)
        assert pw.accuracy() > 0.5
        out = pw.generate()
        assert out.shape == (64, 128)
        assert out.dtype == np.float32

    def test_from_array_function(self):
        w = np.linspace(0, 1, 128).reshape(32, 4).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test", method="function")
        assert pw.shape == (32, 4)
        out = pw.generate()
        assert out.shape == (32, 4)

    def test_from_array_auto_picks_best(self):
        w = np.random.randn(64, 128).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test", method="auto")
        assert pw.accuracy() > 0.5

    def test_caching(self):
        w = np.random.randn(32, 32).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test")
        out1 = pw.generate()
        out2 = pw.generate()
        np.testing.assert_array_equal(out1, out2)

    def test_invalidate_cache(self):
        w = np.random.randn(32, 32).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test")
        out1 = pw.generate()
        pw.invalidate_cache()
        out2 = pw.generate()
        np.testing.assert_array_equal(out1, out2)

    def test_from_point(self):
        p = Point(
            identity="test",
            function_type="linear",
            params={"a": 0.01, "b": 0.5},
            dtype="float32",
            shape=(16, 16),
        )
        pw = PointWeight.from_point(p, shape=(16, 16))
        out = pw.generate()
        assert out.shape == (16, 16)

    def test_small_weights_stored_raw(self):
        w = np.array([1.0, 2.0, 3.0])
        pw = PointWeight.from_array(w, identity="tiny")
        out = pw.generate()
        np.testing.assert_allclose(out, w, atol=1e-6)

    def test_nbytes(self):
        w = np.random.randn(64, 128).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test")
        assert pw.nbytes() < w.nbytes

    def test_repr(self):
        w = np.random.randn(32, 32).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test")
        r = repr(pw)
        assert "PointWeight" in r
        assert "shape" in r

    def test_data_property(self):
        w = np.random.randn(16, 16).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test")
        data = pw.data
        assert data.shape == (16, 16)

    def test_data_is_same_as_generate(self):
        w = np.random.randn(16, 16).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test")
        np.testing.assert_array_equal(pw.data, pw.generate())

    def test_nbytes_nonzero(self):
        w = np.random.randn(32, 32).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test")
        assert pw.nbytes() > 0

    def test_accuracy_range(self):
        w = np.random.randn(64, 64).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test")
        assert 0.0 <= pw.accuracy() <= 1.0

    def test_shape_preserved(self):
        w = np.random.randn(8, 16, 32).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test")
        assert pw.shape == (8, 16, 32)

    def test_dtype_preserved(self):
        w = np.random.randn(32, 32).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test")
        assert pw.dtype == "float32"

    def test_invalidate_then_regenerate(self):
        w = np.random.randn(16, 16).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test")
        first = pw.generate()
        pw.invalidate_cache()
        pw._cached = None
        second = pw.generate()
        np.testing.assert_array_equal(first, second)

    def test_from_point_with_residual(self):
        residual = np.random.randn(64).astype(np.float32)
        p = Point(
            identity="test",
            function_type="linear",
            params={"a": 0.01, "b": 0.5},
            residual=residual,
            dtype="float32",
            shape=(64,),
        )
        pw = PointWeight.from_point(p, shape=(64,))
        out = pw.generate()
        assert out.shape == (64,)

    def test_from_point_periodic(self):
        p = Point(
            identity="test",
            function_type="periodic",
            params={"a": 1.0, "b": 0.5, "w": 0.0},
            dtype="float32",
            shape=(32,),
        )
        pw = PointWeight.from_point(p, shape=(32,))
        out = pw.generate()
        assert out.shape == (32,)

    def test_from_point_polynomial(self):
        p = Point(
            identity="test",
            function_type="polynomial",
            params={"a": 0.001, "b": 0.01, "c": 0.5},
            dtype="float32",
            shape=(32,),
        )
        pw = PointWeight.from_point(p, shape=(32,))
        out = pw.generate()
        assert out.shape == (32,)

    def test_from_point_cluster(self):
        centroids = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        assignments = np.array([0, 1, 2, 0, 1, 2], dtype=np.uint8)
        p = Point(
            identity="test",
            function_type="cluster",
            params={"centroids": centroids, "assignments": assignments},
            dtype="float32",
            shape=(6,),
        )
        pw = PointWeight.from_point(p, shape=(6,))
        out = pw.generate()
        assert out.shape == (6,)

    def test_from_array_two_elements(self):
        w = np.array([42.0, 43.0], dtype=np.float32)
        pw = PointWeight.from_array(w, identity="tiny_two")
        out = pw.generate()
        np.testing.assert_allclose(out, w, atol=1e-6)

    def test_from_array_1d(self):
        w = np.random.randn(128).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test_1d")
        assert pw.shape == (128,)
        out = pw.generate()
        assert out.shape == (128,)

    def test_from_array_large(self):
        w = np.random.randn(256, 512).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test_large")
        assert pw.shape == (256, 512)
        out = pw.generate()
        assert out.shape == (256, 512)

    def test_from_array_all_zeros(self):
        w = np.zeros((32, 32), dtype=np.float32)
        pw = PointWeight.from_array(w, identity="test_zeros")
        out = pw.generate()
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_from_array_all_ones(self):
        w = np.ones((32, 32), dtype=np.float32)
        pw = PointWeight.from_array(w, identity="test_ones")
        out = pw.generate()
        np.testing.assert_allclose(out, 1.0, atol=0.5)

    def test_from_array_constant(self):
        w = np.full((16, 16), 42.0, dtype=np.float32)
        pw = PointWeight.from_array(w, identity="test_const")
        out = pw.generate()
        assert out.shape == (16, 16)

    def test_from_array_monotonic(self):
        w = np.arange(100, dtype=np.float32).reshape(10, 10)
        pw = PointWeight.from_array(w, identity="test_mono", method="function")
        out = pw.generate()
        assert out.shape == (10, 10)

    def test_from_array_random(self):
        w = np.random.randn(64, 64).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test_random")
        out = pw.generate()
        assert out.shape == (64, 64)

    def test_nbytes_smaller_than_raw(self):
        w = np.random.randn(128, 128).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test_compress")
        assert pw.nbytes() < w.nbytes

    def test_accuracy_above_threshold(self):
        w = np.random.randn(64, 64).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test_acc")
        assert pw.accuracy() > 0.3

    def test_dtype_float64(self):
        w = np.random.randn(32, 32).astype(np.float64)
        pw = PointWeight.from_array(w, identity="test_f64")
        assert pw.dtype == "float64"
        assert pw.shape == (32, 32)

    def test_multiple_invalidates(self):
        w = np.random.randn(16, 16).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test_multi_inval")
        for _ in range(5):
            pw.invalidate_cache()
        out = pw.generate()
        assert out.shape == (16, 16)

    def test_generate_preserves_values_close(self):
        w = np.random.randn(32, 32).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test_close")
        out = pw.generate()
        mse = np.mean((w - out) ** 2)
        var = np.var(w)
        accuracy = 1.0 - mse / (var + 1e-8)
        assert accuracy > 0.5

    def test_point_property(self):
        w = np.random.randn(16, 16).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test_point")
        assert isinstance(pw.point, Point)

    def test_from_point_raw(self):
        raw_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        import base64
        p = Point(
            identity="test",
            function_type="raw",
            params={"data_b64": base64.b64encode(raw_data.tobytes()).decode(),
                    "dtype": "float32"},
            accuracy=1.0,
            dtype="float32",
            shape=(3,),
        )
        pw = PointWeight.from_point(p, shape=(3,))
        out = pw.generate()
        np.testing.assert_allclose(out, raw_data, atol=1e-6)

    def test_from_array_method_cluster_explicit(self):
        w = np.random.randn(64, 64).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test_cluster", method="cluster", n_clusters=8)
        assert pw.accuracy() > 0.3

    def test_from_array_method_function_explicit(self):
        w = np.random.randn(64, 64).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test_func", method="function")
        assert pw.accuracy() > 0.0


class TestSloLinearPointWeight:
    def test_set_point_weight(self):
        from domains.training.slonet import SloLinear
        layer = SloLinear(64, 32, name="test")
        w = np.random.randn(32, 64).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test.weight")
        layer.set_point_weight(pw)
        assert layer.get_point_weight() is pw
        assert layer.weight.data.shape == (32, 64)

    def test_compress_to_point(self):
        from domains.training.slonet import SloLinear
        layer = SloLinear(64, 32, name="test")
        pw = layer.compress_to_point(method="cluster")
        assert isinstance(pw, PointWeight)
        assert pw.accuracy() > 0.5
        assert layer.get_point_weight() is pw

    def test_forward_with_point_weight(self):
        from domains.training.slonet import SloLinear, Tensor
        layer = SloLinear(64, 32, name="test")
        w = layer.weight.data.copy()
        pw = PointWeight.from_array(w, identity="test.weight")
        layer.set_point_weight(pw)
        x = Tensor(np.random.randn(1, 64).astype(np.float32))
        out = layer.forward(x)
        assert out.data.shape == (1, 32)

    def test_forward_numpy_with_point_weight(self):
        from domains.training.slonet import SloLinear
        layer = SloLinear(64, 32, name="test")
        w = layer.weight.data.copy()
        pw = PointWeight.from_array(w, identity="test.weight")
        layer.set_point_weight(pw)
        x = np.random.randn(1, 64).astype(np.float32)
        out = layer.forward_numpy(x)
        assert out.shape == (1, 32)

    def test_compress_to_point_function(self):
        from domains.training.slonet import SloLinear
        layer = SloLinear(64, 32, name="test")
        pw = layer.compress_to_point(method="function")
        assert isinstance(pw, PointWeight)
        assert pw.accuracy() > 0.0

    def test_set_then_get_point_weight(self):
        from domains.training.slonet import SloLinear
        layer = SloLinear(32, 16, name="test")
        assert layer.get_point_weight() is None
        w = np.random.randn(16, 32).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test.weight")
        layer.set_point_weight(pw)
        assert layer.get_point_weight() is pw

    def test_compress_to_point_auto(self):
        from domains.training.slonet import SloLinear
        layer = SloLinear(64, 32, name="test")
        pw = layer.compress_to_point(method="auto")
        assert isinstance(pw, PointWeight)
        assert pw.accuracy() > 0.0

    def test_set_point_weight_replaces(self):
        from domains.training.slonet import SloLinear
        layer = SloLinear(64, 32, name="test")
        w1 = np.random.randn(32, 64).astype(np.float32)
        w2 = np.random.randn(32, 64).astype(np.float32)
        pw1 = PointWeight.from_array(w1, identity="test.weight1")
        pw2 = PointWeight.from_array(w2, identity="test.weight2")
        layer.set_point_weight(pw1)
        layer.set_point_weight(pw2)
        assert layer.get_point_weight() is pw2

    def test_forward_consistency(self):
        from domains.training.slonet import SloLinear, Tensor
        layer = SloLinear(32, 16, name="test")
        w = layer.weight.data.copy()
        pw = PointWeight.from_array(w, identity="test.weight")
        layer.set_point_weight(pw)
        x = Tensor(np.random.randn(1, 32).astype(np.float32))
        out1 = layer.forward(x)
        out2 = layer.forward(x)
        np.testing.assert_array_almost_equal(out1.data, out2.data)

    def test_numpy_forward_consistency(self):
        from domains.training.slonet import SloLinear
        layer = SloLinear(32, 16, name="test")
        w = layer.weight.data.copy()
        pw = PointWeight.from_array(w, identity="test.weight")
        layer.set_point_weight(pw)
        x = np.random.randn(1, 32).astype(np.float32)
        out1 = layer.forward_numpy(x)
        out2 = layer.forward_numpy(x)
        np.testing.assert_array_almost_equal(out1, out2)

    def test_get_point_weight_before_set(self):
        from domains.training.slonet import SloLinear
        layer = SloLinear(32, 16, name="test")
        assert layer.get_point_weight() is None

    def test_compress_to_point_accuracy_range(self):
        from domains.training.slonet import SloLinear
        layer = SloLinear(64, 32, name="test")
        pw = layer.compress_to_point(method="cluster")
        assert 0.0 <= pw.accuracy() <= 1.0


class TestCompressSloNetToPoints:
    def test_compress_slo_transformer(self):
        from domains.training.slonet import SloTransformer
        model = SloTransformer(
            vocab_size=256, n_embed=32, n_layer=2, n_head=2,
            block_size=32, max_seq_len=64, use_rope=False,
        )
        points = compress_slonet_to_points(model, method="cluster")
        assert len(points) > 0
        for name, pw in points.items():
            assert isinstance(pw, PointWeight)
            assert pw.accuracy() > 0.3

    def test_compress_method_function(self):
        from domains.training.slonet import SloTransformer
        model = SloTransformer(
            vocab_size=256, n_embed=32, n_layer=2, n_head=2,
            block_size=32, max_seq_len=64, use_rope=False,
        )
        points = compress_slonet_to_points(model, method="function")
        assert len(points) > 0
        for name, pw in points.items():
            assert isinstance(pw, PointWeight)

    def test_compress_returns_dict(self):
        from domains.training.slonet import SloTransformer
        model = SloTransformer(
            vocab_size=256, n_embed=32, n_layer=2, n_head=2,
            block_size=32, max_seq_len=64, use_rope=False,
        )
        points = compress_slonet_to_points(model)
        assert isinstance(points, dict)

    def test_compress_keys_are_strings(self):
        from domains.training.slonet import SloTransformer
        model = SloTransformer(
            vocab_size=256, n_embed=32, n_layer=2, n_head=2,
            block_size=32, max_seq_len=64, use_rope=False,
        )
        points = compress_slonet_to_points(model)
        for key in points:
            assert isinstance(key, str)

    def test_compress_preserves_shapes(self):
        from domains.training.slonet import SloTransformer
        model = SloTransformer(
            vocab_size=256, n_embed=32, n_layer=2, n_head=2,
            block_size=32, max_seq_len=64, use_rope=False,
        )
        points = compress_slonet_to_points(model, method="cluster")
        for name, pw in points.items():
            assert len(pw.shape) > 0

    def test_compress_nclusters_parameter(self):
        from domains.training.slonet import SloTransformer
        model = SloTransformer(
            vocab_size=256, n_embed=32, n_layer=2, n_head=2,
            block_size=32, max_seq_len=64, use_rope=False,
        )
        points = compress_slonet_to_points(model, method="cluster", n_clusters=8)
        assert len(points) > 0

    def test_compress_all_weights_have_points(self):
        from domains.training.slonet import SloTransformer
        model = SloTransformer(
            vocab_size=256, n_embed=32, n_layer=2, n_head=2,
            block_size=32, max_seq_len=64, use_rope=False,
        )
        points = compress_slonet_to_points(model)
        for name, pw in points.items():
            assert pw.point is not None
            assert pw.point.identity == name

    def test_compress_small_model(self):
        from domains.training.slonet import SloTransformer
        model = SloTransformer(
            vocab_size=32, n_embed=16, n_layer=1, n_head=2,
            block_size=16, max_seq_len=32, use_rope=False,
        )
        points = compress_slonet_to_points(model)
        assert len(points) > 0

    def test_compress_auto_method(self):
        from domains.training.slonet import SloTransformer
        model = SloTransformer(
            vocab_size=256, n_embed=32, n_layer=2, n_head=2,
            block_size=32, max_seq_len=64, use_rope=False,
        )
        points = compress_slonet_to_points(model, method="auto")
        assert len(points) > 0
        for name, pw in points.items():
            assert isinstance(pw, PointWeight)

    def test_compress_generate_all(self):
        from domains.training.slonet import SloTransformer
        model = SloTransformer(
            vocab_size=256, n_embed=32, n_layer=2, n_head=2,
            block_size=32, max_seq_len=64, use_rope=False,
        )
        points = compress_slonet_to_points(model, method="cluster")
        for name, pw in points.items():
            out = pw.generate()
            assert out.shape == pw.shape
