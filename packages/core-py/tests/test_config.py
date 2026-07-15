"""
Tests for Config System (config.py).
"""

import os
import pytest
from pydantic import ValidationError
from domains.infrastructure.config import (
    AppConfig, ModelConfig, ServerConfig, FeaturesConfig,
    AuthConfig, StorageConfig,
    ConfigManager, get_config, get_config_manager,
    set_config_manager, reload_config,
)


class TestConfigModels:
    def test_app_config_defaults(self):
        cfg = AppConfig()
        assert cfg.model.name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert cfg.server.port == 8000
        assert cfg.features.auto_workflow is True
        assert cfg.auth.jwt_algorithm == "HS256"
        assert cfg.storage.data_dir == "data"

    def test_model_config_override(self):
        cfg = AppConfig(model=ModelConfig(name="gpt2", device="cpu"))
        assert cfg.model.name == "gpt2"
        assert cfg.model.device == "cpu"

    def test_server_config_defaults(self):
        cfg = ServerConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000
        assert cfg.log_level == "INFO"
        assert cfg.request_timeout == 60.0

    def test_features_config_defaults(self):
        cfg = FeaturesConfig()
        assert cfg.watchdog is True
        assert cfg.health_monitor is True

    def test_unknown_fields_rejected(self):
        with pytest.raises(ValidationError):
            AppConfig(**{"unknown_field": "value"})

    def test_nested_override(self):
        cfg = AppConfig(**{
            "model": {"name": "gpt2-large", "temperature": 0.5},
        })
        assert cfg.model.name == "gpt2-large"
        assert cfg.model.temperature == 0.5

    def test_model_serialize(self):
        cfg = AppConfig()
        d = cfg.model_dump()
        assert d["model"]["name"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert d["server"]["port"] == 8000


class TestEnvOverrides:
    def test_env_override_model_name(self, monkeypatch):
        monkeypatch.setenv("SLO_MODEL__NAME", "gpt2")
        cfg = AppConfig()
        from domains.infrastructure.config import _apply_env_overrides
        cfg = _apply_env_overrides(cfg)
        assert cfg.model.name == "gpt2"

    def test_env_override_int(self, monkeypatch):
        monkeypatch.setenv("SLO_SERVER__PORT", "9000")
        cfg = AppConfig()
        from domains.infrastructure.config import _apply_env_overrides
        cfg = _apply_env_overrides(cfg)
        assert cfg.server.port == 9000

    def test_env_override_float(self, monkeypatch):
        monkeypatch.setenv("SLO_MODEL__TEMPERATURE", "0.5")
        cfg = AppConfig()
        from domains.infrastructure.config import _apply_env_overrides
        cfg = _apply_env_overrides(cfg)
        assert cfg.model.temperature == 0.5

    def test_env_override_bool(self, monkeypatch):
        monkeypatch.setenv("SLO_MODEL__AUTOLOAD", "false")
        cfg = AppConfig()
        from domains.infrastructure.config import _apply_env_overrides
        cfg = _apply_env_overrides(cfg)
        assert cfg.model.autoload is False

    def test_env_override_bool_true(self, monkeypatch):
        monkeypatch.setenv("SLO_FEATURES__WATCHDOG", "1")
        cfg = AppConfig()
        from domains.infrastructure.config import _apply_env_overrides
        cfg = _apply_env_overrides(cfg)
        assert cfg.features.watchdog is True

    def test_env_override_unknown_key_warns(self, monkeypatch):
        monkeypatch.setenv("SLO_UNKNOWN__KEY", "value")
        cfg = AppConfig()
        from domains.infrastructure.config import _apply_env_overrides
        # Should not raise
        _apply_env_overrides(cfg)

    def test_non_man_env_ignored(self, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("HOME", "/root")
        cfg = AppConfig()
        from domains.infrastructure.config import _apply_env_overrides
        result = _apply_env_overrides(cfg)
        assert result == cfg  # no changes


class TestConfigManager:
    def test_manager_creates_config(self, tmp_path):
        mgr = ConfigManager(config_dir=str(tmp_path))
        cfg = mgr.config
        assert cfg.model.name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert cfg.server.port == 8000

    def test_reload_returns_updated(self, tmp_path):
        mgr = ConfigManager(config_dir=str(tmp_path))
        old = mgr.config
        os.environ["SLO_MODEL__NAME"] = "gpt2-test"
        try:
            new = mgr.reload()
            assert new.model.name == "gpt2-test"
        finally:
            del os.environ["SLO_MODEL__NAME"]

    def test_reload_callbacks_fired(self, tmp_path):
        mgr = ConfigManager(config_dir=str(tmp_path))
        results = []

        def cb(new_cfg, old_cfg):
            results.append((new_cfg.model.name, old_cfg.model.name))

        mgr.on_reload(cb)
        os.environ["SLO_MODEL__NAME"] = "gpt2-cb"
        try:
            mgr.reload()
            assert len(results) == 1
            assert results[0][0] == "gpt2-cb"
        finally:
            del os.environ["SLO_MODEL__NAME"]

    def test_yaml_override(self, tmp_path):
        yaml_file = tmp_path / "defaults.yaml"
        yaml_file.write_text("model:\n  name: yaml-model\n  temperature: 0.3\n")
        mgr = ConfigManager(config_dir=str(tmp_path))
        assert mgr.config.model.name == "yaml-model"
        assert mgr.config.model.temperature == 0.3

    def test_env_overrides_yaml(self, tmp_path, monkeypatch):
        yaml_file = tmp_path / "defaults.yaml"
        yaml_file.write_text("model:\n  name: yaml-model\n  temperature: 0.3\n")
        monkeypatch.setenv("SLO_MODEL__NAME", "env-beats-yaml")
        mgr = ConfigManager(config_dir=str(tmp_path))
        assert mgr.config.model.name == "env-beats-yaml"
        assert mgr.config.model.temperature == 0.3

    def test_profile_override(self, tmp_path, monkeypatch):
        defaults = tmp_path / "defaults.yaml"
        defaults.write_text("server:\n  port: 8000\n  log_level: INFO\n")
        profile = tmp_path / "prod.yaml"
        profile.write_text("server:\n  port: 443\n")
        monkeypatch.setenv("SLO_ENV", "prod")
        mgr = ConfigManager(config_dir=str(tmp_path))
        assert mgr.config.server.port == 443
        assert mgr.config.server.log_level == "INFO"

    def test_dump(self, tmp_path):
        mgr = ConfigManager(config_dir=str(tmp_path))
        d = mgr.dump()
        assert isinstance(d, dict)
        assert "model" in d
        assert "server" in d


class TestSingleton:
    def test_get_config(self):
        cfg = get_config()
        assert isinstance(cfg, AppConfig)

    def test_get_config_manager(self):
        mgr = get_config_manager()
        assert isinstance(mgr, ConfigManager)

    def test_set_config_manager(self, tmp_path):
        custom = ConfigManager(config_dir=str(tmp_path))
        set_config_manager(custom)
        assert get_config_manager() is custom

    def test_reload_config_function(self):
        cfg = reload_config()
        assert isinstance(cfg, AppConfig)


class TestEdgeCases:
    def test_empty_env_no_crash(self, monkeypatch):
        monkeypatch.setenv("SLO_", "")
        cfg = AppConfig()
        from domains.infrastructure.config import _apply_env_overrides
        cfg = _apply_env_overrides(cfg)  # Should not crash
        assert cfg.model.name == "Qwen/Qwen2.5-0.5B-Instruct"

    def test_type_coercion_failure_falls_back(self, monkeypatch):
        monkeypatch.setenv("SLO_SERVER__PORT", "not-a-number")
        cfg = AppConfig()
        from domains.infrastructure.config import _apply_env_overrides
        cfg = _apply_env_overrides(cfg)
        assert cfg.server.port == 8000  # unchanged
