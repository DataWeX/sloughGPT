"""
Tests for domains/infrastructure/pugqeep/point.py — Point data class.

Covers:
    - generate(): periodic, linear, polynomial, cluster, raw, unknown type
    - nbytes(): all function types, with/without residual
    - to_bytes()/from_bytes(): round-trip for all function types
    - to_dict()/from_dict(): round-trip for all function types
    - Edge cases: empty arrays, residual addition, unknown type
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_CORE_PY = Path(__file__).resolve().parents[1]
if str(_CORE_PY) not in sys.path:
    sys.path.insert(0, str(_CORE_PY))

from domains.infrastructure.pugqeep.point import Point


# ── generate() ────────────────────────────────────────────────────────


class TestPointGenerate:
    def test_periodic(self):
        p = Point(identity="t", function_type="periodic",
                  params={"a": 1.0, "b": 0.5, "w": 0.1})
        vals = p.generate(10)
        assert vals.shape == (10,)
        # a*cos(i) + b*sin(i) + w
        i = np.arange(10, dtype=np.float32)
        expected = 1.0 * np.cos(i) + 0.5 * np.sin(i) + 0.1
        np.testing.assert_allclose(vals, expected, atol=1e-6)

    def test_linear(self):
        p = Point(identity="l", function_type="linear",
                  params={"a": 2.0, "b": 1.0})
        vals = p.generate(5)
        i = np.arange(5, dtype=np.float32)
        np.testing.assert_allclose(vals, 2.0 * i + 1.0, atol=1e-6)

    def test_polynomial(self):
        p = Point(identity="p", function_type="polynomial",
                  params={"a": 0.5, "b": 1.0, "c": 2.0})
        vals = p.generate(5)
        i = np.arange(5, dtype=np.float32)
        np.testing.assert_allclose(vals, 0.5 * i**2 + 1.0 * i + 2.0, atol=1e-6)

    def test_cluster(self):
        centroids = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        assignments = np.array([0, 1, 2, 0, 1], dtype=np.uint8)
        p = Point(identity="c", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments})
        vals = p.generate(5)
        np.testing.assert_allclose(vals, [10.0, 20.0, 30.0, 10.0, 20.0])

    def test_raw(self):
        import base64
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        raw_b64 = base64.b64encode(data.tobytes()).decode()
        p = Point(identity="r", function_type="raw",
                  params={"data_b64": raw_b64, "dtype": "float32", "shape": [3]})
        vals = p.generate(3)
        np.testing.assert_allclose(vals, data)

    def test_unknown_type_raises(self):
        p = Point(identity="x", function_type="unknown", params={})
        with pytest.raises(ValueError, match="Unknown function type"):
            p.generate(5)

    def test_residual_added(self):
        p = Point(identity="l", function_type="linear",
                  params={"a": 1.0, "b": 0.0},
                  residual=np.array([0.1, 0.2, 0.3], dtype=np.float32))
        vals = p.generate(3)
        i = np.arange(3, dtype=np.float32)
        expected = 1.0 * i + np.array([0.1, 0.2, 0.3])
        np.testing.assert_allclose(vals, expected, atol=1e-6)


# ── nbytes() ──────────────────────────────────────────────────────────


class TestPointNbytes:
    def test_periodic_no_residual(self):
        p = Point(identity="t", function_type="periodic",
                  params={"a": 1.0, "b": 0.5, "w": 0.1})
        assert p.nbytes() > 0

    def test_linear_no_residual(self):
        p = Point(identity="l", function_type="linear",
                  params={"a": 1.0, "b": 0.0})
        assert p.nbytes() > 0

    def test_cluster(self):
        centroids = np.ones(10, dtype=np.float32)
        assignments = np.zeros(20, dtype=np.uint8)
        p = Point(identity="c", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments})
        assert p.nbytes() == centroids.nbytes + assignments.nbytes

    def test_raw(self):
        import base64
        data = b"hello world"
        p = Point(identity="r", function_type="raw",
                  params={"data_b64": base64.b64encode(data).decode()})
        assert p.nbytes() == len(data)

    def test_block_q4(self):
        n_blocks, bs = 4, 32
        mins = np.zeros(n_blocks, dtype=np.float32)
        scales = np.ones(n_blocks, dtype=np.float32)
        packed = np.zeros(n_blocks * bs // 2, dtype=np.uint8)
        p = Point(identity="q4", function_type="block_q4",
                  params={"mins": mins, "scales": scales, "packed": packed,
                          "n_elements": n_blocks * bs, "n_blocks": n_blocks, "block_size": bs})
        assert p.nbytes() == mins.nbytes + scales.nbytes + packed.nbytes

    def test_block_q8(self):
        n_blocks, bs = 4, 32
        mins = np.zeros(n_blocks, dtype=np.float32)
        scales = np.ones(n_blocks, dtype=np.float32)
        values = np.zeros(n_blocks * bs, dtype=np.uint8)
        p = Point(identity="q8", function_type="block_q8",
                  params={"mins": mins, "scales": scales, "values": values,
                          "n_elements": n_blocks * bs, "n_blocks": n_blocks, "block_size": bs})
        assert p.nbytes() == mins.nbytes + scales.nbytes + values.nbytes

    def test_block_q4_missing_params(self):
        p = Point(identity="q4", function_type="block_q4", params={})
        assert p.nbytes() == 0

    def test_block_q8_missing_params(self):
        p = Point(identity="q8", function_type="block_q8", params={})
        assert p.nbytes() == 0


# ── to_bytes / from_bytes round-trip ─────────────────────────────────


class TestPointBytesRoundTrip:
    def test_periodic(self):
        p = Point(identity="t", function_type="periodic",
                  params={"a": 1.0, "b": 0.5, "w": 0.1})
        data = p.to_bytes()
        p2 = Point.from_bytes(data, identity="t2")
        assert p2.function_type == "periodic"
        assert p2.params["a"] == pytest.approx(1.0)
        assert p2.params["b"] == pytest.approx(0.5)

    def test_periodic_with_residual(self):
        res = np.array([0.1, 0.2], dtype=np.float32)
        p = Point(identity="t", function_type="periodic",
                  params={"a": 1.0, "b": 0.0, "w": 0.0}, residual=res)
        data = p.to_bytes()
        p2 = Point.from_bytes(data)
        assert p2.residual is not None
        np.testing.assert_allclose(p2.residual, res)

    def test_linear(self):
        p = Point(identity="l", function_type="linear",
                  params={"a": 2.0, "b": 3.0})
        data = p.to_bytes()
        p2 = Point.from_bytes(data)
        assert p2.function_type == "linear"
        assert p2.params["a"] == pytest.approx(2.0)

    def test_polynomial(self):
        p = Point(identity="p", function_type="polynomial",
                  params={"a": 0.5, "b": 1.0, "c": 2.0})
        data = p.to_bytes()
        p2 = Point.from_bytes(data)
        assert p2.function_type == "polynomial"
        assert p2.params["c"] == pytest.approx(2.0)

    def test_cluster(self):
        centroids = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assignments = np.array([0, 1, 2], dtype=np.uint8)
        p = Point(identity="c", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments})
        data = p.to_bytes()
        p2 = Point.from_bytes(data)
        assert p2.function_type == "cluster"
        np.testing.assert_allclose(p2.params["centroids"], centroids)
        np.testing.assert_allclose(p2.params["assignments"], assignments)

    def test_raw(self):
        raw_data = b"hello world"
        import base64
        p = Point(identity="r", function_type="raw",
                  params={"data_b64": base64.b64encode(raw_data).decode(),
                          "dtype": "float32", "shape": []})
        data = p.to_bytes()
        p2 = Point.from_bytes(data)
        assert p2.function_type == "raw"

    def test_unknown_type_code_raises(self):
        with pytest.raises(ValueError, match="Unknown type code"):
            Point.from_bytes(b"\xff\xff\xff\xff" + b"\x00" * 8)


# ── to_dict / from_dict round-trip ───────────────────────────────────


class TestPointDictRoundTrip:
    def test_periodic(self):
        p = Point(identity="t", function_type="periodic",
                  params={"a": 1.0, "b": 0.5, "w": 0.1})
        d = p.to_dict()
        p2 = Point.from_dict(d)
        assert p2.function_type == "periodic"
        assert p2.params["a"] == pytest.approx(1.0)

    def test_cluster(self):
        centroids = np.array([1.0, 2.0], dtype=np.float32)
        assignments = np.array([0, 1], dtype=np.uint8)
        p = Point(identity="c", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments})
        d = p.to_dict()
        p2 = Point.from_dict(d)
        np.testing.assert_allclose(p2.params["centroids"], centroids)
        np.testing.assert_allclose(p2.params["assignments"], assignments)

    def test_with_residual(self):
        res = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        p = Point(identity="l", function_type="linear",
                  params={"a": 1.0, "b": 0.0}, residual=res)
        d = p.to_dict()
        assert "residual_b64" in d
        p2 = Point.from_dict(d)
        np.testing.assert_allclose(p2.residual, res)

    def test_accuracy_preserved(self):
        p = Point(identity="t", function_type="linear",
                  params={"a": 1.0, "b": 0.0}, accuracy=0.95)
        d = p.to_dict()
        p2 = Point.from_dict(d)
        assert p2.accuracy == 0.95

    def test_raw(self):
        import base64
        p = Point(identity="r", function_type="raw",
                  params={"data_b64": base64.b64encode(b"x").decode(),
                          "dtype": "float32", "shape": []})
        d = p.to_dict()
        p2 = Point.from_dict(d)
        assert p2.function_type == "raw"


# ── Edge cases ────────────────────────────────────────────────────────


class TestPointEdgeCases:
    def test_generate_zero_length(self):
        p = Point(identity="l", function_type="linear",
                  params={"a": 1.0, "b": 0.0})
        vals = p.generate(0)
        assert vals.shape == (0,)

    def test_cluster_short_request(self):
        centroids = np.array([10.0, 20.0], dtype=np.float32)
        assignments = np.array([0, 1], dtype=np.uint8)
        p = Point(identity="c", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments})
        vals = p.generate(1)
        assert vals.shape == (1,)
        assert vals[0] == 10.0

    def test_bytes_no_residual(self):
        p = Point(identity="l", function_type="linear",
                  params={"a": 1.0, "b": 0.0})
        data = p.to_bytes()
        p2 = Point.from_bytes(data)
        assert p2.residual is None
