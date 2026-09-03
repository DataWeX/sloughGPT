"""Tests for domains.shared.utils — utility functions, Cache, Timer, RateLimiter."""

from __future__ import annotations

import json
import time
import pytest
from pathlib import Path

from domains.shared.utils import (
    generate_id,
    hash_string,
    format_size,
    format_time,
    load_json,
    save_json,
    merge_dicts,
    clamp,
    retry,
    Timer,
    Cache,
    RateLimiter,
    validate_config,
    get_timestamp,
    find_available_port,
    find_repo_root,
    find_server_python,
)


# ── generate_id ───────────────────────────────────────────────────────────────

class TestGenerateId:
    def test_default(self):
        id1 = generate_id()
        id2 = generate_id()
        assert len(id1) == 8
        assert id1 != id2

    def test_with_prefix(self):
        id_ = generate_id("user_")
        assert id_.startswith("user_")
        assert len(id_) == 13  # 5 + 8


# ── hash_string ───────────────────────────────────────────────────────────────

class TestHashString:
    def test_sha256(self):
        h = hash_string("hello")
        assert len(h) == 64
        assert h == hashlib.sha256(b"hello").hexdigest()

    def test_md5(self):
        h = hash_string("hello", "md5")
        assert len(h) == 32

    def test_sha1(self):
        h = hash_string("hello", "sha1")
        assert len(h) == 40

    def test_unknown_algo(self):
        assert hash_string("hello", "unknown") == "hello"

import hashlib


# ── format_size ───────────────────────────────────────────────────────────────

class TestFormatSize:
    def test_bytes(self):
        assert format_size(500) == "500.0 B"

    def test_kb(self):
        assert format_size(2048) == "2.0 KB"

    def test_mb(self):
        assert format_size(1048576) == "1.0 MB"

    def test_gb(self):
        assert format_size(1073741824) == "1.0 GB"

    def test_tb(self):
        assert format_size(1099511627776) == "1.0 TB"


# ── format_time ───────────────────────────────────────────────────────────────

class TestFormatTime:
    def test_seconds(self):
        assert format_time(30.5) == "30.5s"

    def test_minutes(self):
        assert format_time(120.0) == "2.0m"

    def test_hours(self):
        assert format_time(7200.0) == "2.0h"

    def test_days(self):
        assert format_time(172800.0) == "2.0d"


# ── load_json / save_json ────────────────────────────────────────────────────

class TestJsonIO:
    def test_roundtrip(self, tmp_path):
        data = {"key": "value", "num": 42}
        path = str(tmp_path / "test.json")
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_indent(self, tmp_path):
        path = str(tmp_path / "test.json")
        save_json({"a": 1}, path, indent=4)
        with open(path) as f:
            content = f.read()
        assert "    " in content  # 4-space indent


# ── merge_dicts ───────────────────────────────────────────────────────────────

class TestMergeDicts:
    def test_basic(self):
        result = merge_dicts({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_overlap_last_wins(self):
        result = merge_dicts({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_empty(self):
        assert merge_dicts() == {}


# ── clamp ─────────────────────────────────────────────────────────────────────

class TestClamp:
    def test_in_range(self):
        assert clamp(5, 0, 10) == 5

    def test_below_min(self):
        assert clamp(-5, 0, 10) == 0

    def test_above_max(self):
        assert clamp(15, 0, 10) == 10


# ── retry ─────────────────────────────────────────────────────────────────────

class TestRetry:
    def test_success_first_try(self):
        call_count = 0
        @retry(max_attempts=3, delay=0.01)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"
        assert succeed() == "ok"
        assert call_count == 1

    def test_retry_then_success(self):
        call_count = 0
        @retry(max_attempts=3, delay=0.01)
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "ok"
        assert fail_twice() == "ok"
        assert call_count == 3

    def test_all_attempts_fail(self):
        @retry(max_attempts=2, delay=0.01)
        def always_fail():
            raise ValueError("always")
        with pytest.raises(ValueError):
            always_fail()


# ── Timer ─────────────────────────────────────────────────────────────────────

class TestTimer:
    def test_context_manager(self):
        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed >= 0.01

    def test_attributes(self):
        t = Timer()
        assert t.start is None
        assert t.end is None
        assert t.elapsed is None


# ── Cache ─────────────────────────────────────────────────────────────────────

class TestCache:
    def test_set_get(self):
        c = Cache()
        c.set("k", "v")
        assert c.get("k") == "v"

    def test_get_missing(self):
        c = Cache()
        assert c.get("missing") is None

    def test_max_size_eviction(self):
        c = Cache(max_size=2)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)  # evicts "a"
        assert c.get("a") is None
        assert c.get("b") == 2
        assert c.get("c") == 3

    def test_clear(self):
        c = Cache()
        c.set("k", "v")
        c.clear()
        assert len(c) == 0

    def test_len(self):
        c = Cache()
        c.set("a", 1)
        c.set("b", 2)
        assert len(c) == 2


# ── RateLimiter ───────────────────────────────────────────────────────────────

class TestRateLimiter:
    def test_allows_within_limit(self):
        limiter = RateLimiter(max_calls=3, period=1.0)
        @limiter
        def fn():
            return "ok"
        assert fn() == "ok"
        assert fn() == "ok"
        assert fn() == "ok"

    def test_raises_on_exceed(self):
        limiter = RateLimiter(max_calls=2, period=10.0)
        @limiter
        def fn():
            return "ok"
        fn()
        fn()
        with pytest.raises(Exception, match="Rate limit"):
            fn()


# ── validate_config ───────────────────────────────────────────────────────────

class TestValidateConfig:
    def test_valid(self):
        assert validate_config({"a": 1, "b": 2}, ["a", "b"]) is True

    def test_missing_key(self):
        assert validate_config({"a": 1}, ["a", "b"]) is False

    def test_empty_required(self):
        assert validate_config({"a": 1}, []) is True


# ── get_timestamp ─────────────────────────────────────────────────────────────

class TestGetTimestamp:
    def test_returns_iso(self):
        ts = get_timestamp()
        assert "T" in ts
        assert "+" in ts or "Z" in ts


# ── find_available_port ──────────────────────────────────────────────────────

class TestFindAvailablePort:
    def test_finds_port(self):
        port = find_available_port(start_port=9000, max_attempts=100)
        assert 9000 <= port < 9100


# ── find_repo_root ────────────────────────────────────────────────────────────

class TestFindRepoRoot:
    def test_finds_root(self):
        root = find_repo_root()
        assert (root / "apps").is_dir() or (root / "packages").is_dir()


# ── find_server_python ────────────────────────────────────────────────────────

class TestFindServerPython:
    def test_returns_string(self):
        result = find_server_python()
        assert isinstance(result, str)
        assert "python" in result.lower()
