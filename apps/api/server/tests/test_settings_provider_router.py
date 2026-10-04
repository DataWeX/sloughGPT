"""Tests for /settings/providers/api endpoints."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.exception_handlers import register_app_error_handler

_AUTH_USER = {"sub": "user1", "tenant_id": "t1"}


def _build_app():
    from routers.settings import SettingsRouter

    _app = FastAPI()
    register_app_error_handler(_app)
    _app.include_router(SettingsRouter().router)
    _app.dependency_overrides[require_auth_if_enabled] = lambda: _AUTH_USER
    return _app


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """Point the persistent settings singleton at a temp file."""
    from domain.settings._internal import persistent

    ps = persistent.PersistentSettings(settings_path=tmp_path / "settings.json")
    monkeypatch.setattr(persistent, "_default_settings", ps)
    return ps


@pytest.fixture
def clean_registry():
    from domain.models._internal.provider import registry

    saved = dict(registry._providers)
    registry._providers.clear()
    try:
        yield
    finally:
        registry._providers.clear()
        registry._providers.update(saved)


class TestGetProviderSettings:
    def test_get_defaults(self, isolated_settings, clean_registry):
        resp = TestClient(_build_app()).get("/settings/providers/api")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["enabled"] is False
        assert data["api_url"] == ""
        assert isinstance(data["model"], str)
        assert data["api_key_set"] is False
        assert "api_key" not in data

    def test_get_reflects_persisted_values(self, isolated_settings, clean_registry):
        isolated_settings.update(
            "providers", enabled=True, api_url="https://api.openai.com/v1", model="gpt-4o"
        )
        resp = TestClient(_build_app()).get("/settings/providers/api")
        data = resp.json()["data"]
        assert data["enabled"] is True
        assert data["api_url"] == "https://api.openai.com/v1"
        assert data["model"] == "gpt-4o"


class TestPatchProviderSettings:
    def test_patch_no_changes(self, isolated_settings, clean_registry):
        resp = TestClient(_build_app()).patch("/settings/providers/api", json={})
        assert resp.status_code == 200
        assert "message" in resp.json()["data"]

    def test_patch_invalid_url_rejected(self, isolated_settings, clean_registry):
        resp = TestClient(_build_app()).patch(
            "/settings/providers/api",
            json={"enabled": True, "api_url": "not-a-url", "api_key": "sk-test"},
        )
        assert resp.status_code == 422

    def test_patch_enable_applies_runtime_and_persists(self, isolated_settings, clean_registry):
        resp = TestClient(_build_app()).patch(
            "/settings/providers/api",
            json={
                "enabled": True,
                "api_url": "https://api.openai.com/v1",
                "api_key": "sk-test",
                "model": "gpt-4o-mini",
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["enabled"] is True
        assert data["api_key_set"] is True
        assert isolated_settings.settings.providers.api_url == "https://api.openai.com/v1"
        assert isolated_settings.settings.providers.model == "gpt-4o-mini"

    def test_patch_enable_without_key_not_applied(self, isolated_settings, clean_registry):
        from domain.models import get_provider

        resp = TestClient(_build_app()).patch(
            "/settings/providers/api",
            json={"enabled": True, "api_url": "https://api.openai.com/v1"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["registered"] is False
        assert get_provider("api") is None

    def test_patch_enable_registers_runtime_provider(self, isolated_settings, clean_registry):
        from domain.inference._internal.api_provider import ApiProvider
        from domain.models import get_provider

        TestClient(_build_app()).patch(
            "/settings/providers/api",
            json={
                "enabled": True,
                "api_url": "https://api.openai.com/v1",
                "api_key": "sk-test",
                "model": "gpt-4o-mini",
            },
        )
        assert isinstance(get_provider("api"), ApiProvider)
