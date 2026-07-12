"""Tests for PointWeight — Point-based weight representation in SloNet.

Verifies:
  - PointWeight compression and generation
  - SloLinear.set_point_weight() integration
  - SloLinear.compress_to_point() end-to-end
  - Round-trip: numpy → Point → generate → numpy
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
        # Linear weights should fit well
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
        np.testing.assert_array_equal(out1, out2)  # same values, different object

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


class TestSloLinearPointWeight:
    def test_set_point_weight(self):
        from domains.training.slonet import SloLinear
        layer = SloLinear(64, 32, name="test")
        w = np.random.randn(32, 64).astype(np.float32)
        pw = PointWeight.from_array(w, identity="test.weight")
        layer.set_point_weight(pw)
        assert layer.get_point_weight() is pw
        # weight data should be synced
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
        # Forward should work normally
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
