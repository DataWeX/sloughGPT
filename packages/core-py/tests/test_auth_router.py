"""Tests for the auth API router (routers/auth.py).

Covers: AuthRouter password hashing (static methods, no HTTP mocking needed).
HTTP-level auth tests deferred — instance method patching requires careful setup.
"""
from __future__ import annotations

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
        import hashlib
        legacy = hashlib.sha256("testpass".encode()).hexdigest()
        assert AuthRouter._verify_password("testpass", legacy) is True

    def test_legacy_hash_wrong_password(self):
        import hashlib
        legacy = hashlib.sha256("testpass".encode()).hexdigest()
        assert AuthRouter._verify_password("wrong", legacy) is False

    def test_non_v1_non_hex_rejects(self):
        assert AuthRouter._verify_password("x", "not-a-hash") is False

    def test_v1_bad_hex_rejects(self):
        assert AuthRouter._verify_password("x", "v1:zzzz") is False
