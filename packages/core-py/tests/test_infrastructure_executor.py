"""Tests for ParallelExecutor — batch parallel processing."""
from __future__ import annotations

import threading

from domains.infrastructure.pugqeep.executor import ParallelExecutor


class TestParallelExecutor:
    def test_run(self):
        results = []
        lock = threading.Lock()
        items = [("a", 1), ("b", 2), ("c", 3)]

        def handler(item):
            with lock:
                results.append(item)

        ex = ParallelExecutor(num_workers=2)
        ex.run(items, handler)
        assert len(results) == 3

    def test_map(self):
        items = [("a", 1), ("b", 2), ("c", 3)]
        ex = ParallelExecutor(num_workers=2)
        result = ex.map(items, lambda x: x * 10)
        assert result["a"] == 10
        assert result["b"] == 20
        assert result["c"] == 30

    def test_num_workers(self):
        ex = ParallelExecutor(num_workers=4)
        assert ex.num_workers == 4

    def test_map_empty(self):
        ex = ParallelExecutor(num_workers=2)
        result = ex.map([], lambda x: x)
        assert result == {}
