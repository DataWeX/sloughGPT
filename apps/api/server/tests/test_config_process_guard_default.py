"""
Tests for the ProcessGuard production-default invariant: the runtime
toggle must agree with ``ServerConfig.enable_process_guard`` (single source
of truth), and both must default to enabled per docs/ENVIRONMENT.md.
"""

import importlib

import config


def _reload_config(monkeypatch, env_value):
    if env_value is None:
        monkeypatch.delenv("SLO_ENABLE_PROCESS_GUARD", raising=False)
    else:
        monkeypatch.setenv("SLO_ENABLE_PROCESS_GUARD", env_value)
    reloaded = importlib.reload(config)
    return reloaded


def test_enable_process_guard_defaults_enabled():
    assert config.ServerConfig().enable_process_guard is True


def test_from_env_defaults_enabled():
    assert config.ServerConfig.from_env().enable_process_guard is True


def test_from_env_respects_explicit_false(monkeypatch):
    monkeypatch.setenv("SLO_ENABLE_PROCESS_GUARD", "false")
    assert config.ServerConfig.from_env().enable_process_guard is False


def test_runtime_toggle_derives_from_config_default(monkeypatch):
    reloaded = _reload_config(monkeypatch, None)
    assert reloaded.ServerConfig.from_env().enable_process_guard is True
    assert reloaded.get_process_guard_enabled() is True


def test_runtime_toggle_derives_from_config_false(monkeypatch):
    reloaded = _reload_config(monkeypatch, "false")
    assert reloaded.ServerConfig.from_env().enable_process_guard is False
    assert reloaded.get_process_guard_enabled() is False


def test_runtime_toggle_still_overridable(monkeypatch):
    reloaded = _reload_config(monkeypatch, None)
    reloaded.set_process_guard_enabled(False)
    assert reloaded.get_process_guard_enabled() is False
