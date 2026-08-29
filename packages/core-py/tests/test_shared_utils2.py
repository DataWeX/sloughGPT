"""Tests for domains.shared.utils — utility functions and Timer."""

import time
from domains.shared.utils import (
    generate_id, hash_string, format_size, format_time,
    merge_dicts, clamp, Timer,
)


class TestGenerateId:
    def test_returns_string(self):
        assert isinstance(generate_id(), str)

    def test_prefix(self):
        assert generate_id("run").startswith("run")

    def test_unique(self):
        ids = {generate_id() for _ in range(50)}
        assert len(ids) == 50


class TestHashString:
    def test_sha256(self):
        h = hash_string("hello")
        assert len(h) == 64

    def test_deterministic(self):
        assert hash_string("test") == hash_string("test")

    def test_different(self):
        assert hash_string("a") != hash_string("b")


class TestFormatSize:
    def test_bytes(self):
        assert format_size(0) == "0.0 B"

    def test_kb(self):
        assert "KB" in format_size(1024)

    def test_mb(self):
        assert "MB" in format_size(1024 * 1024)

    def test_gb(self):
        assert "GB" in format_size(1024 ** 3)


class TestFormatTime:
    def test_seconds(self):
        result = format_time(30)
        assert "30" in result

    def test_minutes(self):
        result = format_time(120)
        assert "2" in result

    def test_hours(self):
        result = format_time(7200)
        assert "2" in result


class TestMergeDicts:
    def test_basic(self):
        result = merge_dicts({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_overlap_last_wins(self):
        result = merge_dicts({"a": 1}, {"a": 2})
        assert result["a"] == 2

    def test_empty(self):
        assert merge_dicts() == {}


class TestClamp:
    def test_in_range(self):
        assert clamp(5, 0, 10) == 5

    def test_below_min(self):
        assert clamp(-1, 0, 10) == 0

    def test_above_max(self):
        assert clamp(20, 0, 10) == 10


class TestTimer:
    def test_context_manager(self):
        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed >= 0.01

    def test_elapsed_is_float(self):
        with Timer() as t:
            pass
        assert isinstance(t.elapsed, float)
