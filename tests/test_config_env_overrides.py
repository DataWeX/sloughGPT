"""Env-override loader behaviour for the infrastructure AppConfig
(domain/infrastructure/_internal/config.py `_apply_env_overrides`).

Guard for the fail-closed auth change (card 20261029_005644): every
deployment now sets SLO_AUTH_REQUIRED (default true; false only for local
dev), and the flag is consumed directly via os.environ by the auth code —
the typed-config loader must neither warn about it nor let it leak into
the schema (AppConfig is extra="forbid").
"""

from __future__ import annotations

import logging

import pytest

_CONFIG_LOGGER = "slo.config"


class TestSkipEnvKeys:
    def test_auth_required_is_skipped_without_warning(self, monkeypatch, caplog):
        from domain.infrastructure._internal.config import AppConfig, _apply_env_overrides

        monkeypatch.setenv("SLO_AUTH_REQUIRED", "false")
        with caplog.at_level(logging.WARNING, logger=_CONFIG_LOGGER):
            cfg = _apply_env_overrides(AppConfig())
        assert not [r for r in caplog.records if "Unknown config key" in r.getMessage()]
        # Validate the same way load_config does — extra="forbid" must not
        # trip on the flag having been folded into the dump.
        AppConfig.model_validate(cfg.model_dump())

    def test_unknown_key_still_warns(self, monkeypatch, caplog):
        # The skip-list is precise: an actually-unknown key must keep
        # warning, or the loader's typo feedback disappears silently.
        from domain.infrastructure._internal.config import AppConfig, _apply_env_overrides

        monkeypatch.setenv("SLO_TOTALLY_FAKE_KEY", "x")
        with caplog.at_level(logging.WARNING, logger=_CONFIG_LOGGER):
            _apply_env_overrides(AppConfig())
        assert any("Unknown config key" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("value", ["false", "true", "1", "yes"])
    def test_auth_required_any_value_is_quiet(self, monkeypatch, caplog, value):
        from domain.infrastructure._internal.config import AppConfig, _apply_env_overrides

        monkeypatch.setenv("SLO_AUTH_REQUIRED", value)
        with caplog.at_level(logging.WARNING, logger=_CONFIG_LOGGER):
            _apply_env_overrides(AppConfig())
        assert not [r for r in caplog.records if "Unknown config key" in r.getMessage()]

    def test_cors_origins_directly_consumed_key_is_quiet(self, monkeypatch, caplog):
        # SLO_CORS_ORIGINS is read by main.py via os.environ.get — same
        # defect class as SLO_AUTH_REQUIRED (warned at every boot since the
        # local dev drop-in sets it).
        from domain.infrastructure._internal.config import AppConfig, _apply_env_overrides

        monkeypatch.setenv("SLO_CORS_ORIGINS", "http://localhost:5173")
        with caplog.at_level(logging.WARNING, logger=_CONFIG_LOGGER):
            _apply_env_overrides(AppConfig())
        assert not [r for r in caplog.records if "Unknown config key" in r.getMessage()]
