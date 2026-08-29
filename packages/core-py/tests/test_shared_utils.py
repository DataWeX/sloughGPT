"""Tests for domains.shared.utils — pure utility functions."""

import tempfile
import time
from pathlib import Path

import pytest


class TestGenerateId:
    def test_returns_string(self):
        from domains.shared.utils import generate_id
        result = generate_id()
        assert isinstance(result, str)
        assert len(result) == 8

    def test_with_prefix(self):
        from domains.shared.utils import generate_id
        result = generate_id("run_")
        assert result.startswith("run_")
        assert len(result) == 12

    def test_unique(self):
        from domains.shared.utils import generate_id
        ids = {generate_id() for _ in range(100)}
        assert len(ids) == 100


class TestHashString:
    def test_sha256(self):
        from domains.shared.utils import hash_string
        result = hash_string("hello")
        assert len(result) == 64  # SHA-256 hex digest length

    def test_md5(self):
        from domains.shared.utils import hash_string
        result = hash_string("hello", algorithm="md5")
        assert len(result) == 32  # MD5 hex digest length

    def test_sha1(self):
        from domains.shared.utils import hash_string
        result = hash_string("hello", algorithm="sha1")
        assert len(result) == 40

    def test_unknown_returns_input(self):
        from domains.shared.utils import hash_string
        assert hash_string("hello", algorithm="unknown") == "hello"

    def test_deterministic(self):
        from domains.shared.utils import hash_string
        assert hash_string("test") == hash_string("test")


class TestFormatSize:
    def test_bytes(self):
        from domains.shared.utils import format_size
        assert format_size(0) == "0.0 B"
        assert format_size(512) == "512.0 B"

    def test_kb(self):
        from domains.shared.utils import format_size
        assert format_size(1024) == "1.0 KB"

    def test_mb(self):
        from domains.shared.utils import format_size
        assert format_size(1024 * 1024) == "1.0 MB"

    def test_gb(self):
        from domains.shared.utils import format_size
        assert format_size(1024 ** 3) == "1.0 GB"

    def test_tb(self):
        from domains.shared.utils import format_size
        assert format_size(1024 ** 4) == "1.0 TB"

    def test_pb(self):
        from domains.shared.utils import format_size
        assert format_size(1024 ** 5) == "1.0 PB"


class TestFormatTime:
    def test_seconds(self):
        from domains.shared.utils import format_time
        assert format_time(5.0) == "5.0s"

    def test_minutes(self):
        from domains.shared.utils import format_time
        assert format_time(120.0) == "2.0m"

    def test_hours(self):
        from domains.shared.utils import format_time
        assert format_time(7200.0) == "2.0h"

    def test_days(self):
        from domains.shared.utils import format_time
        assert format_time(172800.0) == "2.0d"


class TestMergeDicts:
    def test_single_dict(self):
        from domains.shared.utils import merge_dicts
        assert merge_dicts({"a": 1}) == {"a": 1}

    def test_multiple_dicts(self):
        from domains.shared.utils import merge_dicts
        result = merge_dicts({"a": 1}, {"b": 2}, {"c": 3})
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_overlap_last_wins(self):
        from domains.shared.utils import merge_dicts
        result = merge_dicts({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_empty(self):
        from domains.shared.utils import merge_dicts
        assert merge_dicts() == {}


class TestClamp:
    def test_within_range(self):
        from domains.shared.utils import clamp
        assert clamp(5, 0, 10) == 5

    def test_below_min(self):
        from domains.shared.utils import clamp
        assert clamp(-5, 0, 10) == 0

    def test_above_max(self):
        from domains.shared.utils import clamp
        assert clamp(15, 0, 10) == 10

    def test_at_boundaries(self):
        from domains.shared.utils import clamp
        assert clamp(0, 0, 10) == 0
        assert clamp(10, 0, 10) == 10


class TestTimer:
    def test_context_manager(self):
        from domains.shared.utils import Timer
        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed is not None
        assert t.elapsed >= 0.01

    def test_start_end_set(self):
        from domains.shared.utils import Timer
        with Timer() as t:
            time.sleep(0.005)
        assert t.start is not None
        assert t.end is not None
        assert t.end >= t.start


class TestCache:
    def test_get_set(self):
        from domains.shared.utils import Cache
        c = Cache()
        c.set("a", 1)
        assert c.get("a") == 1

    def test_get_miss(self):
        from domains.shared.utils import Cache
        c = Cache()
        assert c.get("missing") is None

    def test_max_size_eviction(self):
        from domains.shared.utils import Cache
        c = Cache(max_size=3)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        c.set("d", 4)  # should evict "a"
        assert c.get("a") is None
        assert c.get("d") == 4

    def test_clear(self):
        from domains.shared.utils import Cache
        c = Cache()
        c.set("a", 1)
        c.clear()
        assert c.get("a") is None
        assert len(c) == 0

    def test_len(self):
        from domains.shared.utils import Cache
        c = Cache()
        assert len(c) == 0
        c.set("a", 1)
        assert len(c) == 1


class TestRateLimiter:
    def test_within_limit(self):
        from domains.shared.utils import RateLimiter
        limiter = RateLimiter(max_calls=3, period=1.0)

        @limiter
        def f():
            return "ok"

        assert f() == "ok"
        assert f() == "ok"
        assert f() == "ok"

    def test_exceeds_limit(self):
        from domains.shared.utils import RateLimiter
        limiter = RateLimiter(max_calls=2, period=10.0)

        @limiter
        def f():
            return "ok"

        f()
        f()
        with pytest.raises(Exception, match="Rate limit exceeded"):
            f()


class TestValidateConfig:
    def test_valid(self):
        from domains.shared.utils import validate_config
        assert validate_config({"a": 1, "b": 2}, ["a", "b"]) is True

    def test_missing_key(self):
        from domains.shared.utils import validate_config
        assert validate_config({"a": 1}, ["a", "b"]) is False

    def test_empty_required(self):
        from domains.shared.utils import validate_config
        assert validate_config({"a": 1}, []) is True


class TestFindAvailablePort:
    def test_finds_port(self):
        from domains.shared.utils import find_available_port
        port = find_available_port(start_port=10000, max_attempts=10)
        assert 10000 <= port < 10010

    def test_skips_occupied(self):
        import socket
        from domains.shared.utils import find_available_port
        # Bind a port to occupy it
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 10100))
        s.listen(1)
        try:
            port = find_available_port(start_port=10100, max_attempts=5)
            assert port != 10100
        finally:
            s.close()


class TestFindRepoRoot:
    def test_finds_root(self):
        from domains.shared.utils import find_repo_root
        root = find_repo_root()
        assert (root / "apps").is_dir() or (root / "packages").is_dir()
