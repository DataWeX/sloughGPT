"""Tests for the Plugins router — 4 endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.exception_handlers import register_app_error_handler

_AUTH_USER = {"sub": "user1", "tenant_id": "t1"}


def _build_app(auth_user_dict=_AUTH_USER):
    from routers.plugins import PluginsRouter

    router_obj = PluginsRouter()
    _app = FastAPI()
    register_app_error_handler(_app)
    _app.include_router(router_obj.router)
    _app.dependency_overrides[require_auth_if_enabled] = lambda: auth_user_dict
    return _app


# ── List plugins ─────────────────────────────────────────────────────────────


class TestListPlugins:
    def test_list_plugins(self):
        _app = _build_app()
        mock_pm = MagicMock()
        mock_pm.list_plugins.return_value = [
            {"name": "test-plugin", "enabled": True, "version": "1.0"},
        ]
        with patch("domains.plugins.get_plugin_manager", return_value=mock_pm):
            resp = TestClient(_app).get("/plugins")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "plugins" in data
        assert len(data["plugins"]) == 1


# ── Enable plugin ────────────────────────────────────────────────────────────


class TestEnablePlugin:
    def test_enable_plugin(self):
        _app = _build_app()
        mock_pm = MagicMock()
        mock_meta = MagicMock()
        mock_pm._metadata = {"test-plugin": mock_meta}
        with patch("domains.plugins.get_plugin_manager", return_value=mock_pm):
            resp = TestClient(_app).post("/plugins/test-plugin/enable")
        assert resp.status_code == 200
        assert resp.json()["data"]["enabled"] is True


# ── Disable plugin ───────────────────────────────────────────────────────────


class TestDisablePlugin:
    def test_disable_plugin(self):
        _app = _build_app()
        mock_pm = MagicMock()
        mock_meta = MagicMock()
        mock_pm._metadata = {"test-plugin": mock_meta}
        with patch("domains.plugins.get_plugin_manager", return_value=mock_pm):
            resp = TestClient(_app).post("/plugins/test-plugin/disable")
        assert resp.status_code == 200
        assert resp.json()["data"]["enabled"] is False


# ── Reload plugins ───────────────────────────────────────────────────────────


class TestReloadPlugins:
    def test_reload_plugins(self):
        _app = _build_app()
        mock_pm = MagicMock()
        mock_pm.load_from_directory.return_value = 3
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        with (
            patch("domains.plugins.get_plugin_manager", return_value=mock_pm),
            patch("routers.plugins.find_repo_root", return_value=MagicMock(__truediv__=lambda self, x: mock_path)),
        ):
            resp = TestClient(_app).post("/plugins/reload")
        assert resp.status_code == 200
