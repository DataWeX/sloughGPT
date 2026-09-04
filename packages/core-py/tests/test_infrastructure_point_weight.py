"""Tests for PointWeight — weight representation using Points."""
from __future__ import annotations

import numpy as np

from domains.infrastructure.pugqeep.point import Point
from domains.infrastructure.pugqeep.point_weight import PointWeight


class TestPointWeight:
    def test_generate(self):
        point = Point(
            identity="test",
            function_type="raw",
            params={"data_b64": "AAAA", "shape": [2], "dtype": "float32"},
            accuracy=1.0,
        )
        # Manually create a point that can generate
        arr = np.array([1.0, 2.0], dtype=np.float32)
        import base64
        point = Point(
            identity="test",
            function_type="raw",
            params={"data_b64": base64.b64encode(arr.tobytes()).decode(), "shape": [2], "dtype": "float32"},
            accuracy=1.0,
        )
        pw = PointWeight(point, shape=(2,), dtype="float32")
        data = pw.generate()
        assert data.shape == (2,)
        np.testing.assert_array_almost_equal(data, arr)

    def test_caching(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        import base64
        point = Point(
            identity="test",
            function_type="raw",
            params={"data_b64": base64.b64encode(arr.tobytes()).decode(), "shape": [3], "dtype": "float32"},
            accuracy=1.0,
        )
        pw = PointWeight(point, shape=(3,), dtype="float32")
        data1 = pw.generate()
        data2 = pw.generate()
        assert data1 is data2  # same cached object

    def test_invalidate_cache(self):
        arr = np.array([1.0], dtype=np.float32)
        import base64
        point = Point(
            identity="test",
            function_type="raw",
            params={"data_b64": base64.b64encode(arr.tobytes()).decode(), "shape": [1], "dtype": "float32"},
            accuracy=1.0,
        )
        pw = PointWeight(point, shape=(1,), dtype="float32")
        data1 = pw.generate()
        pw.invalidate_cache()
        data2 = pw.generate()
        assert data1 is not data2  # new object after invalidation

    def test_data_property(self):
        arr = np.array([4.0, 5.0], dtype=np.float32)
        import base64
        point = Point(
            identity="test",
            function_type="raw",
            params={"data_b64": base64.b64encode(arr.tobytes()).decode(), "shape": [2], "dtype": "float32"},
            accuracy=1.0,
        )
        pw = PointWeight(point, shape=(2,), dtype="float32")
        np.testing.assert_array_almost_equal(pw.data, arr)

    def test_from_point(self):
        point = Point(identity="w", function_type="linear", params={"slope": 1.0, "intercept": 0.0}, accuracy=0.9, dtype="float32")
        pw = PointWeight.from_point(point, shape=(4,))
        assert pw.shape == (4,)
        assert pw.accuracy() == 0.9

    def test_repr(self):
        point = Point(identity="w", function_type="cluster", params={}, accuracy=0.85)
        pw = PointWeight(point, shape=(10,), dtype="float32")
        r = repr(pw)
        assert "cluster" in r
        assert "shape=(10,)" in r
