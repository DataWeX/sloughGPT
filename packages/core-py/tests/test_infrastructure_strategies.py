"""Tests for compression strategies — RawStrategy."""
from __future__ import annotations

import base64

import numpy as np

from domains.infrastructure.pugqeep.point import Point
from domains.infrastructure.pugqeep.strategies import RawStrategy


class TestRawStrategy:
    def test_compress(self):
        strategy = RawStrategy()
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        point, nbytes = strategy.compress("weight", data, "p1", n_clusters=10)
        assert isinstance(point, Point)
        assert point.function_type == "raw"
        assert nbytes == data.nbytes

    def test_preserves_data(self):
        strategy = RawStrategy()
        data = np.array([10.0, 20.0], dtype=np.float32)
        point, _ = strategy.compress("w", data, "p1", n_clusters=10)
        decoded = np.frombuffer(base64.b64decode(point.params["data_b64"]), dtype=np.float32)
        np.testing.assert_array_equal(decoded, data)

    def test_metadata(self):
        strategy = RawStrategy()
        data = np.zeros((2, 3), dtype=np.float64)
        point, _ = strategy.compress("w", data, "p1", n_clusters=10)
        assert point.params["shape"] == [2, 3]
        assert point.params["dtype"] == "float64"
        assert point.accuracy == 1.0
