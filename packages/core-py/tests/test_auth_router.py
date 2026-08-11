"""Tests for the auth API router (routers/auth.py).

Covers: AuthRouter password hashing (static methods, no HTTP mocking needed).
HTTP-level auth tests deferred — instance method patching requires careful setup.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from routers.auth import AuthRouter  # noqa: E402


class TestPasswordHashing:
    def test_hash_and_verify(self):
        hashed = AuthRouter._hash_password("mypassword")
        assert hashed.startswith("v1:")
        assert AuthRouter._verify_password("mypassword", hashed) is True

    def test_wrong_password_fails(self):
        hashed = AuthRouter._hash_password("mypassword")
        assert AuthRouter._verify_password("wrongpassword", hashed) is False

    def test_different_hashes_for_same_password(self):
        h1 = AuthRouter._hash_password("test")
        h2 = AuthRouter._hash_password("test")
        assert h1 != h2  # salt-based

    def test_legacy_hash_verify(self):
        legacy = hashlib.sha256("testpass".encode()).hexdigest()
        assert AuthRouter._verify_password("testpass", legacy) is True

    def test_legacy_hash_wrong_password(self):
        legacy = hashlib.sha256("testpass".encode()).hexdigest()
        assert AuthRouter._verify_password("wrong", legacy) is False

    def test_non_v1_non_hex_rejects(self):
        assert AuthRouter._verify_password("x", "not-a-hash") is False

    def test_v1_bad_hex_rejects(self):
        assert AuthRouter._verify_password("x", "v1:zzzz") is False

    def test_empty_password_hashes(self):
        hashed = AuthRouter._hash_password("")
        assert hashed.startswith("v1:")
        assert AuthRouter._verify_password("", hashed) is True

    def test_empty_password_wrong(self):
        hashed = AuthRouter._hash_password("nonempty")
        assert AuthRouter._verify_password("", hashed) is False

    def test_long_password(self):
        long_pw = "a" * 10000
        hashed = AuthRouter._hash_password(long_pw)
        assert AuthRouter._verify_password(long_pw, hashed) is True
        assert AuthRouter._verify_password("a" * 9999, hashed) is False

    def test_unicode_password(self):
        hashed = AuthRouter._hash_password("p@$$w0rd 🔐")
        assert AuthRouter._verify_password("p@$$w0rd 🔐", hashed) is True
        assert AuthRouter._verify_password("p@$$w0rd 🔑", hashed) is False

    def test_hash_format_contains_colon(self):
        hashed = AuthRouter._hash_password("test")
        parts = hashed.split(":")
        assert len(parts) == 3  # v1:salt:hash
        assert parts[0] == "v1"
        assert len(parts[1]) > 0  # non-empty salt
        assert len(parts[2]) > 0  # non-empty hash

    def test_verify_empty_string_hash(self):
        assert AuthRouter._verify_password("x", "") is False

    def test_verify_none_like_hash(self):
        assert AuthRouter._verify_password("x", "v1:") is False
