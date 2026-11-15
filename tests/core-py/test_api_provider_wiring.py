"""Tests for ApiProvider runtime wiring (configure_api_provider)."""

from __future__ import annotations

import pytest

from domain.inference._internal.api_provider import ApiProvider, configure_api_provider
from domain.models._internal.provider.registry import _providers, get_provider, register_provider
from domain.models._internal.provider.router import ProviderRouter


@pytest.fixture
def native_router_with_registry():
    """Registry with a slonet-native provider + default router on it."""
    saved = dict(_providers)
    _providers.clear()
    register_provider("slonet-native", object())
    router = ProviderRouter()
    router.set_text_provider("slonet-native")
    register_provider("default", router)
    try:
        yield router
    finally:
        _providers.clear()
        _providers.update(saved)


class TestConfigureApiProvider:
    def test_enabled_registers_api_provider_as_text(self, native_router_with_registry):
        result = configure_api_provider(
            True,
            api_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4o-mini",
        )
        provider = get_provider("api")
        assert isinstance(provider, ApiProvider)
        assert provider.model_name == "gpt-4o-mini"
        assert native_router_with_registry._text_name == "api"
        assert result["registered"] is True
        assert result["text_provider"] == "api"

    def test_disabled_restores_native(self, native_router_with_registry):
        configure_api_provider(
            True,
            api_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4o-mini",
        )
        result = configure_api_provider(False)
        assert get_provider("api") is None
        assert native_router_with_registry._text_name == "slonet-native"
        assert result["registered"] is False

    def test_missing_credentials_not_registered(self, native_router_with_registry):
        result = configure_api_provider(True, api_url="", api_key="")
        assert result["registered"] is False
        assert "required" in result["error"]
        assert get_provider("api") is None
        assert native_router_with_registry._text_name == "slonet-native"

    def test_no_router_still_registers(self):
        try:
            _providers.clear()
            result = configure_api_provider(
                True,
                api_url="https://api.openai.com/v1",
                api_key="sk-test",
            )
            assert isinstance(get_provider("api"), ApiProvider)
            assert result["registered"] is True
        finally:
            _providers.clear()
