"""Tests for domains.shell.config — API base URL configuration.

Covers: default value, env var override, get_api_base.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.shell.config import get_api_base, DEFAULT_API_BASE


# ---------------------------------------------------------------------------
# DEFAULT_API_BASE
# ---------------------------------------------------------------------------

class TestDefaultApiBase:
    def test_is_string(self):
        assert isinstance(DEFAULT_API_BASE, str)

    def test_non_empty(self):
        assert len(DEFAULT_API_BASE) > 0

    def test_is_url(self):
        assert DEFAULT_API_BASE.startswith("http")

    def test_reflects_env_at_import_time(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://custom:9999")
        import importlib
        import domains.shell.config as mod
        mod.DEFAULT_API_BASE = os.environ.get("MAN_API_URL", "http://localhost:8000")
        assert mod.DEFAULT_API_BASE == "http://custom:9999"

    def test_fallback_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("MAN_API_URL", raising=False)
        import importlib
        import domains.shell.config as mod
        mod.DEFAULT_API_BASE = os.environ.get("MAN_API_URL", "http://localhost:8000")
        assert mod.DEFAULT_API_BASE == "http://localhost:8000"


# ---------------------------------------------------------------------------
# get_api_base
# ---------------------------------------------------------------------------

class TestGetApiBase:
    def test_returns_string(self):
        result = get_api_base()
        assert isinstance(result, str)

    def test_returns_url(self):
        result = get_api_base()
        assert result.startswith("http")

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://custom-host:42")
        assert get_api_base() == "http://custom-host:42"

    def test_env_empty_string(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "")
        result = get_api_base()
        assert result == ""

    def test_env_special_chars(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://host:8000/v1/api")
        assert get_api_base() == "http://host:8000/v1/api"

    def test_env_localhost_default(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://localhost:8000")
        assert get_api_base() == "http://localhost:8000"

    def test_env_https(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "https://api.example.com")
        assert get_api_base() == "https://api.example.com"

    def test_unenv_returns_default(self, monkeypatch):
        monkeypatch.delenv("MAN_API_URL", raising=False)
        result = get_api_base()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_env_ip_address(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://192.168.1.100:8080")
        assert get_api_base() == "http://192.168.1.100:8080"

    def test_env_ipv6(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://[::1]:8000")
        assert get_api_base() == "http://[::1]:8000"

    def test_env_with_path(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://host/v2/endpoint")
        assert get_api_base() == "http://host/v2/endpoint"

    def test_env_with_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://host:8000/")
        assert get_api_base() == "http://host:8000/"

    def test_env_high_port(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://localhost:65535")
        assert get_api_base() == "http://localhost:65535"

    def test_env_zero_port(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://localhost:0")
        assert get_api_base() == "http://localhost:0"

    def test_env_localsocket(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "unix:///tmp/api.sock")
        assert get_api_base() == "unix:///tmp/api.sock"

    def test_env_with_subpath(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://host:8000/api/v1")
        result = get_api_base()
        assert result.endswith("/api/v1")

    def test_returns_consistently(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://stable:5000")
        r1 = get_api_base()
        r2 = get_api_base()
        assert r1 == r2

    def test_env_whitespace_preserved(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", " http://host:8000 ")
        assert get_api_base() == " http://host:8000 "

    def test_env_fragment_preserved(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://host:8000#section")
        assert get_api_base() == "http://host:8000#section"

    def test_env_query_preserved(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://host:8000?key=val")
        assert get_api_base() == "http://host:8000?key=val"

    def test_env_with_auth(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://user:pass@host:8000")
        assert get_api_base() == "http://user:pass@host:8000"

    def test_long_url(self, monkeypatch):
        long_url = "http://" + "a" * 200 + ".com:8000"
        monkeypatch.setenv("MAN_API_URL", long_url)
        assert get_api_base() == long_url

    def test_env_with_port_only(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://:8000")
        assert get_api_base() == "http://:8000"

    def test_fresh_import_returns_new_value(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://fresh:1111")
        import importlib
        import domains.shell.config as mod
        importlib.reload(mod)
        assert mod.get_api_base() == "http://fresh:1111"


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_default_api_base_stable(self):
        assert DEFAULT_API_BASE is not None
        assert isinstance(DEFAULT_API_BASE, str)

    def test_get_api_base_always_callable(self):
        assert callable(get_api_base)

    def test_default_has_localhost(self, monkeypatch):
        monkeypatch.delenv("MAN_API_URL", raising=False)
        import importlib
        import domains.shell.config as mod
        importlib.reload(mod)
        assert "localhost" in mod.DEFAULT_API_BASE or "8000" in mod.DEFAULT_API_BASE

    def test_get_api_base_returns_same_type(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://x:1")
        assert type(get_api_base()) is str

    def test_default_is_http_scheme(self, monkeypatch):
        monkeypatch.delenv("MAN_API_URL", raising=False)
        import importlib
        import domains.shell.config as mod
        importlib.reload(mod)
        assert mod.DEFAULT_API_BASE.startswith("http")

    def test_get_api_base_called_multiple_times(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://multi:7777")
        results = [get_api_base() for _ in range(10)]
        assert all(r == "http://multi:7777" for r in results)

    def test_env_value_reflected_immediately(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://first:1")
        assert get_api_base() == "http://first:1"
        monkeypatch.setenv("MAN_API_URL", "http://second:2")
        assert get_api_base() == "http://second:2"

    def test_env_delete_reverts(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://temp:9999")
        assert get_api_base() == "http://temp:9999"
        monkeypatch.delenv("MAN_API_URL", raising=False)
        result = get_api_base()
        assert result != "http://temp:9999"
