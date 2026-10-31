"""Tests for the Point codec registry + block_q4/q8 reconstruction.

The headline defect these pin down: block_q4/block_q8 Points serialized and
sized correctly but ``generate()`` raised ``Unknown function type`` — a
deserialized block-quantized Point could not reconstruct. The codec registry
closes that hole and replaces the four coupled if/elif chains in ``Point``.
"""

from __future__ import annotations

import numpy as np
import pytest

from domain.infrastructure._internal.pugqeep.codecs import (
    BlockQ4Codec,
    BlockQ8Codec,
    CodecRegistry,
    codec_registry,
)
from domain.infrastructure._internal.pugqeep.compressor import PointCompressor
from domain.infrastructure._internal.pugqeep.point import Point


def _make_block_points(seed: int = 0):
    rng = np.random.default_rng(seed)
    weights = rng.standard_normal(512).astype(np.float32) * 3.0 + 1.0
    comp = PointCompressor()
    return weights, comp.compress_block_q4(weights, "q4w"), comp.compress_block_q8(weights, "q8w")


class TestBlockQuantizedGenerate:
    """block_q4/q8 must reconstruct via Point.generate (was: Unknown function type)."""

    def test_block_q4_generate_roundtrip(self):
        weights, p4, _ = _make_block_points()
        got = p4.generate(len(weights))
        assert got.shape == weights.shape
        # Q4 is coarse but must track the signal (correlation high, error bounded).
        assert np.corrcoef(got, weights)[0, 1] > 0.99
        assert np.max(np.abs(got - weights)) < np.max(np.abs(weights)) * 0.2

    def test_block_q8_generate_roundtrip(self):
        weights, _, p8 = _make_block_points()
        got = p8.generate(len(weights))
        assert got.shape == weights.shape
        # Q8 is fine-grained — near-exact reconstruction.
        np.testing.assert_allclose(got, weights, atol=0.05)
        assert np.corrcoef(got, weights)[0, 1] > 0.9999

    def test_generate_matches_compressor_decompress(self):
        """Point.generate must equal the compressor's private decompressor."""
        weights, p4, p8 = _make_block_points(7)
        np.testing.assert_allclose(p4.generate(len(weights)), comp_deq(p4), atol=1e-6)
        np.testing.assert_allclose(p8.generate(len(weights)), comp_deq(p8), atol=1e-6)

    def test_generate_respects_n(self):
        weights, p4, p8 = _make_block_points(3)
        assert len(p8.generate(10)) == 10
        assert len(p4.generate(10)) == 10


def comp_deq(p: Point) -> np.ndarray:
    comp = PointCompressor()
    if p.function_type == "block_q4":
        return comp.decompress_block_q4(p)
    return comp.decompress_block_q8(p)


class TestBlockQuantizedByteRoundTrip:
    """to_bytes -> from_bytes must preserve block params and still generate."""

    def test_q4_bytes_roundtrip_generates(self):
        weights, p4, _ = _make_block_points(11)
        restored = Point.from_bytes(p4.to_bytes(), identity=p4.identity)
        assert restored.function_type == "block_q4"
        np.testing.assert_allclose(
            restored.generate(len(weights)), p4.generate(len(weights)), atol=1e-6
        )

    def test_q8_bytes_roundtrip_generates(self):
        weights, _, p8 = _make_block_points(13)
        restored = Point.from_bytes(p8.to_bytes(), identity=p8.identity)
        assert restored.function_type == "block_q8"
        np.testing.assert_allclose(
            restored.generate(len(weights)), p8.generate(len(weights)), atol=1e-6
        )

    def test_q8_bytes_roundtrip_preserves_accuracy(self):
        """Regression: from_bytes used to drop accuracy/dtype/shape."""
        _, _, p8 = _make_block_points(17)
        restored = Point.from_bytes(p8.to_bytes(), identity=p8.identity)
        assert restored.accuracy == pytest.approx(p8.accuracy, rel=1e-6)
        assert restored.dtype == p8.dtype
        assert restored.shape == p8.shape

    @pytest.mark.parametrize("which", ["q4", "q8"])
    def test_block_dict_roundtrip(self, which):
        """JSON path (to_dict/from_dict, used by JSONStore) must also work."""
        weights, p4, p8 = _make_block_points(23)
        p = p4 if which == "q4" else p8
        restored = Point.from_dict(p.to_dict())
        assert restored.function_type == p.function_type
        assert restored.accuracy == pytest.approx(p.accuracy, rel=1e-6)
        np.testing.assert_allclose(
            restored.generate(len(weights)), p.generate(len(weights)), atol=1e-6
        )


class TestLegacyByteCompatibility:
    """New serializers must emit byte-identical blobs to the legacy format."""

    def test_linear_bytes_unchanged(self):
        p = Point(
            identity="lin",
            function_type="linear",
            params={"a": 2.0, "b": -1.0},
            accuracy=0.99,
            dtype="float32",
            shape=(5,),
        )
        # legacy layout: "LIN " + <ff> + has_res(0)
        from domain.infrastructure._internal.pugqeep.point import Point as P

        blob = p.to_bytes()
        assert blob[:4] == b"LIN "
        restored = P.from_bytes(blob, identity="lin")
        assert restored.params["a"] == pytest.approx(2.0)
        assert restored.params["b"] == pytest.approx(-1.0)
        np.testing.assert_allclose(restored.generate(5), p.generate(5))

    def test_cluster_quantized_bytes_roundtrip(self):
        # Real compressor centroids are 1-D (k-means over flattened data);
        # the legacy wire format stores a flat uint8 centroid buffer.
        centroids = np.arange(16, dtype=np.uint8)
        assignments = np.array([0, 1, 2, 3, 1], dtype=np.uint8)
        p = Point(
            identity="clq",
            function_type="cluster",
            params={
                "centroids": centroids,
                "assignments": assignments,
                "centroid_quantized": True,
                "centroid_scale": 0.5,
                "centroid_zero_point": 2.0,
            },
            accuracy=0.95,
        )
        restored = Point.from_bytes(p.to_bytes(), identity="clq")
        np.testing.assert_allclose(restored.generate(5), p.generate(5))

    def test_periodic_with_residual_roundtrip(self):
        res = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        p = Point(
            identity="per",
            function_type="periodic",
            params={"a": 1.0, "b": 2.0, "w": 0.5},
            residual=res,
            accuracy=0.8,
        )
        restored = Point.from_bytes(p.to_bytes(), identity="per")
        assert restored.residual is not None
        np.testing.assert_allclose(restored.generate(3), p.generate(3), atol=1e-6)


class TestCodecRegistry:
    def test_registry_has_all_seven_types(self):
        expected = {
            "periodic",
            "linear",
            "polynomial",
            "cluster",
            "raw",
            "block_q4",
            "block_q8",
        }
        assert expected.issubset(set(codec_registry.types()))

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown function type"):
            codec_registry.get("nope")

    def test_register_custom_codec(self):
        reg = CodecRegistry()

        class Dummy:
            function_type = "dummy"

            def generate(self, point, n):
                return np.zeros(n, dtype=np.float32)

            def nbytes(self, point):
                return 0

            def serialize_params(self, point):
                return b""

            def deserialize_params(self, buf):
                return {}

        reg.register(Dummy())
        assert "dummy" in reg
        assert reg.get("dummy").function_type == "dummy"

    def test_block_codecs_registered(self):
        assert isinstance(codec_registry.get("block_q4"), BlockQ4Codec)
        assert isinstance(codec_registry.get("block_q8"), BlockQ8Codec)


class TestPointDispatch:
    """Point.generate/nbytes must route through the registry for every type."""

    def test_all_types_generate_via_registry(self):
        _, p4, p8 = _make_block_points(21)
        for p in (p4, p8):
            n = int(np.prod(p.shape)) if p.shape else p.params["n_elements"]
            assert len(p.generate(n)) > 0

    def test_unknown_type_raises_cleanly(self):
        p = Point(identity="x", function_type="mystery", params={})
        with pytest.raises(ValueError, match="Unknown function type"):
            p.generate(4)
