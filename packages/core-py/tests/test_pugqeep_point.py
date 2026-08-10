"""Tests for pugqeep/point.py — Point serialization, generation, to_bytes/from_bytes round-trip."""

import numpy as np
import pytest
from domains.infrastructure.pugqeep.point import Point


class TestPointGenerate:
    def test_periodic(self):
        p = Point(identity="x", function_type="periodic",
                  params={"a": 1.0, "b": 0.5, "w": 0.0})
        vals = p.generate(10)
        assert vals.shape == (10,)
        i = np.arange(10, dtype=np.float32)
        expected = 1.0 * np.cos(i) + 0.5 * np.sin(i)
        np.testing.assert_allclose(vals, expected, atol=1e-6)

    def test_linear(self):
        p = Point(identity="x", function_type="linear",
                  params={"a": 2.0, "b": 1.0})
        vals = p.generate(5)
        expected = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        np.testing.assert_allclose(vals, expected, atol=1e-6)

    def test_polynomial(self):
        p = Point(identity="x", function_type="polynomial",
                  params={"a": 1.0, "b": 0.0, "c": 0.0})
        vals = p.generate(4)
        expected = np.array([0.0, 1.0, 4.0, 9.0])
        np.testing.assert_allclose(vals, expected, atol=1e-6)

    def test_cluster(self):
        centroids = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        assignments = np.array([0, 1, 2, 0, 1], dtype=np.uint8)
        p = Point(identity="x", function_type="cluster",
                  params={"centroids": centroids, "assignments": assignments})
        vals = p.generate(5)
        expected = np.array([10.0, 20.0, 30.0, 10.0, 20.0])
        np.testing.assert_array_equal(vals, expected)

    def test_raw(self):
        import base64
        raw = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b64 = base64.b64encode(raw.tobytes()).decode()
        p = Point(identity="x", function_type="raw",
                  params={"data_b64": b64, "dtype": "float32", "shape": []})
        vals = p.generate(3)
        np.testing.assert_array_equal(vals, raw)

    def test_with_residual(self):
        p = Point(identity="x", function_type="linear",
                  params={"a": 1.0, "b": 0.0},
                  residual=np.array([0.1, 0.1, 0.1], dtype=np.float32))
        vals = p.generate(3)
        expected = np.array([0.1, 1.1, 2.1])
        np.testing.assert_allclose(vals, expected, atol=1e-6)

    def test_unknown_type_raises(self):
        p = Point(identity="x", function_type="unknown", params={})
        with pytest.raises(ValueError, match="Unknown function type"):
            p.generate(5)


class TestPointNbytes:
    def test_cluster(self):
        c = np.zeros(10, dtype=np.float32)
        a = np.zeros(20, dtype=np.uint8)
        p = Point("x", "cluster", {"centroids": c, "assignments": a})
        assert p.nbytes() == 40 + 20

    def test_raw(self):
        import base64
        raw = np.zeros(5, dtype=np.float32)
        b64 = base64.b64encode(raw.tobytes()).decode()
        p = Point("x", "raw", {"data_b64": b64, "dtype": "float32", "shape": []})
        assert p.nbytes() == 20


class TestPointBytesRoundTrip:
    def test_periodic(self):
        p = Point("test", "periodic", {"a": 1.5, "b": -0.3, "w": 0.7})
        data = p.to_bytes()
        p2 = Point.from_bytes(data, identity="test")
        assert p2.function_type == "periodic"
        assert p2.params["a"] == pytest.approx(1.5)
        vals1 = p.generate(10)
        vals2 = p2.generate(10)
        np.testing.assert_allclose(vals1, vals2, atol=1e-6)

    def test_linear(self):
        p = Point("test", "linear", {"a": 2.0, "b": 1.0})
        data = p.to_bytes()
        p2 = Point.from_bytes(data, identity="test")
        np.testing.assert_allclose(p.generate(5), p2.generate(5))

    def test_polynomial(self):
        p = Point("test", "polynomial", {"a": 1.0, "b": -2.0, "c": 3.0})
        data = p.to_bytes()
        p2 = Point.from_bytes(data, identity="test")
        np.testing.assert_allclose(p.generate(5), p2.generate(5))

    def test_cluster(self):
        c = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a = np.array([0, 1, 2, 1, 0], dtype=np.uint8)
        p = Point("test", "cluster", {"centroids": c, "assignments": a})
        data = p.to_bytes()
        p2 = Point.from_bytes(data, identity="test")
        np.testing.assert_array_equal(p.generate(5), p2.generate(5))

    def test_raw(self):
        import base64
        raw = np.array([10.0, 20.0], dtype=np.float32)
        b64 = base64.b64encode(raw.tobytes()).decode()
        p = Point("test", "raw", {"data_b64": b64, "dtype": "float32", "shape": []})
        data = p.to_bytes()
        p2 = Point.from_bytes(data, identity="test")
        np.testing.assert_array_equal(p.generate(2), p2.generate(2))

    def test_periodic_with_residual(self):
        res = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        p = Point("test", "periodic", {"a": 1.0, "b": 0.0, "w": 0.0}, residual=res)
        data = p.to_bytes()
        p2 = Point.from_bytes(data, identity="test")
        np.testing.assert_allclose(p.generate(3), p2.generate(3), atol=1e-6)

    def test_unknown_type_code_raises(self):
        with pytest.raises(ValueError, match="Unknown type code"):
            Point.from_bytes(b"XXXX")


class TestPointDictRoundTrip:
    def test_linear(self):
        p = Point("test", "linear", {"a": 2.5, "b": -1.0}, accuracy=0.95)
        d = p.to_dict()
        p2 = Point.from_dict(d)
        assert p2.identity == "test"
        assert p2.accuracy == pytest.approx(0.95)
        np.testing.assert_allclose(p.generate(5), p2.generate(5))

    def test_cluster(self):
        c = np.array([1.0, 2.0], dtype=np.float32)
        a = np.array([0, 1, 0], dtype=np.uint8)
        p = Point("test", "cluster", {"centroids": c, "assignments": a})
        d = p.to_dict()
        p2 = Point.from_dict(d)
        np.testing.assert_array_equal(p.generate(3), p2.generate(3))

    def test_raw(self):
        import base64
        raw = np.array([5.0], dtype=np.float32)
        b64 = base64.b64encode(raw.tobytes()).decode()
        p = Point("test", "raw", {"data_b64": b64, "dtype": "float32", "shape": []})
        d = p.to_dict()
        p2 = Point.from_dict(d)
        np.testing.assert_array_equal(p.generate(1), p2.generate(1))
