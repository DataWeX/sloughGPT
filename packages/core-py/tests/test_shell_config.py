"""Tests for shell/config.py."""

from unittest.mock import patch

from domains.shell import config


class TestShellConfig:
    def test_default_api_base(self, monkeypatch):
        monkeypatch.delenv("MAN_API_URL", raising=False)
        assert config.get_api_base() == "http://localhost:8000"

    def test_uses_env_var(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://example.com:9000")
        assert config.get_api_base() == "http://example.com:9000"

    def test_env_takes_precedence_over_module_default(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "https://api.local")
        assert config.get_api_base() == "https://api.local"
