"""Tests for ConfigController."""
import pytest
from unittest.mock import patch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'api', 'server'))

from controllers.config import ConfigController, get_config_controller


@pytest.fixture
def ctrl():
    return ConfigController()


class TestGetGenerationConfig:
    def test_returns_dict(self, ctrl):
        result = ctrl.get_generation_config()
        assert isinstance(result, dict)

    def test_has_default_keys(self, ctrl):
        result = ctrl.get_generation_config()
        for key in ["temperature", "top_p", "top_k", "max_new_tokens"]:
            assert key in result

    def test_returns_copy(self, ctrl):
        r1 = ctrl.get_generation_config()
        r1["temperature"] = 999
        r2 = ctrl.get_generation_config()
        assert r2["temperature"] != 999

    def test_default_temperature(self, ctrl):
        assert ctrl.get_generation_config()["temperature"] == 0.8

    def test_has_full_known_key_set(self, ctrl):
        result = ctrl.get_generation_config()
        for key in [
            "temperature", "top_p", "top_k", "repetition_penalty",
            "max_new_tokens", "max_context_length",
        ]:
            assert key in result

    def test_defaults_values(self, ctrl):
        result = ctrl.get_generation_config()
        assert result["top_p"] == 0.9
        assert result["top_k"] == 50
        assert result["repetition_penalty"] == 1.2
        assert result["max_new_tokens"] == 200
        assert result["max_context_length"] == 1024


class TestUpdateGenerationConfig:
    def test_update_temperature(self, ctrl):
        result = ctrl.update_generation_config(temperature=0.5)
        assert result["temperature"] == 0.5

    def test_update_multiple(self, ctrl):
        result = ctrl.update_generation_config(temperature=0.3, top_p=0.95)
        assert result["temperature"] == 0.3
        assert result["top_p"] == 0.95

    def test_ignores_none(self, ctrl):
        original = ctrl.get_generation_config()["temperature"]
        result = ctrl.update_generation_config(temperature=None)
        assert result["temperature"] == original

    def test_ignores_unknown_keys(self, ctrl):
        result = ctrl.update_generation_config(unknown_key=42)
        assert "unknown_key" not in result

    def test_persists_across_calls(self, ctrl):
        ctrl.update_generation_config(temperature=0.1)
        assert ctrl.get_generation_config()["temperature"] == 0.1

    def test_updates_every_known_key(self, ctrl):
        updates = {
            "temperature": 0.1,
            "top_p": 0.5,
            "top_k": 10,
            "repetition_penalty": 0.8,
            "max_new_tokens": 512,
            "max_context_length": 2048,
        }
        result = ctrl.update_generation_config(**updates)
        for key, value in updates.items():
            assert result[key] == value
        assert ctrl.get_generation_config()["max_new_tokens"] == 512

    def test_mixed_valid_and_unknown(self, ctrl):
        result = ctrl.update_generation_config(temperature=0.7, bogus=1)
        assert result["temperature"] == 0.7
        assert "bogus" not in result

    def test_returns_copy_after_update(self, ctrl):
        result = ctrl.update_generation_config(top_k=77)
        result["top_k"] = 999
        assert ctrl.get_generation_config()["top_k"] == 77

    def test_none_value_leaves_other_updated_keys(self, ctrl):
        result = ctrl.update_generation_config(temperature=None, top_p=0.75)
        assert result["top_p"] == 0.75


class TestGetConfigControllerSingleton:
    def test_returns_same_instance(self):
        c1 = get_config_controller()
        c2 = get_config_controller()
        assert c1 is c2
