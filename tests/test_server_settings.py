"""Tests for ``apps/api/server/settings.py`` (canonical vs legacy env names)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SERVER_DIR = Path(__file__).resolve().parent.parent / "apps" / "api" / "server"
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))


@pytest.fixture
def clear_settings_cache() -> None:
    from settings import get_security_settings

    get_security_settings.cache_clear()
    yield
    get_security_settings.cache_clear()


def test_prefers_sloughgpt_over_legacy_typo(monkeypatch: pytest.MonkeyPatch, clear_settings_cache: None) -> None:
    from settings import get_security_settings

    monkeypatch.setenv("SLOUGHGPT_API_KEY", "canonical-key")
    monkeypatch.setenv("SLAUGHGPT_API_KEY", "should-not-win")
    s = get_security_settings()
    assert s.primary_api_key == "canonical-key"


def test_legacy_slaughgpt_api_key_still_works(monkeypatch: pytest.MonkeyPatch, clear_settings_cache: None) -> None:
    from settings import get_security_settings

    monkeypatch.delenv("SLOUGHGPT_API_KEY", raising=False)
    monkeypatch.setenv("SLAUGHGPT_API_KEY", "legacy-only-key")
    s = get_security_settings()
    assert s.primary_api_key == "legacy-only-key"
