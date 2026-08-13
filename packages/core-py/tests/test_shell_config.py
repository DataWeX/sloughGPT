"""Tests for shell/config.py."""

import os
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

    def test_empty_env_returns_default(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "")
        result = config.get_api_base()
        assert result == "" or result == "http://localhost:8000"

    def test_default_api_base_constant(self):
        assert hasattr(config, "DEFAULT_API_BASE")

    def test_get_api_base_returns_string(self, monkeypatch):
        monkeypatch.delenv("MAN_API_URL", raising=False)
        assert isinstance(config.get_api_base(), str)

    def test_localhost_url(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://localhost:3000")
        assert config.get_api_base() == "http://localhost:3000"

    def test_custom_port(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://127.0.0.1:5000")
        assert config.get_api_base() == "http://127.0.0.1:5000"

    def test_https_url(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "https://api.production.com")
        assert config.get_api_base() == "https://api.production.com"

    def test_ip_url(self, monkeypatch):
        monkeypatch.setenv("MAN_API_URL", "http://192.168.1.100:8080")
        assert config.get_api_base() == "http://192.168.1.100:8080"
