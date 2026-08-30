"""Tests for domains.infrastructure.model_registry — ModelRegistry, get_model_registry.

Covers: register, register_engine, unregister, get, list_models, default_id,
generate, health_summary, reset_metrics, singleton pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.infrastructure.model_registry import ModelRegistry, get_model_registry, DEFAULT_MODEL_ID
from domains.infrastructure.model_server import ModelStatus


def _mock_server(model_id="m1", status=ModelStatus.READY):
    s = MagicMock()
    s.model_id = model_id
    s._device = "cpu"
    s.status = status
    s.get_metrics_snapshot.return_value = {
        "status": status.value if hasattr(status, "value") else status,
        "requests_total": 0,
        "errors_total": 0,
    }
    s.generate = AsyncMock(return_value={"text": "hello", "tokens_generated": 1})
    return s


class TestRegister:
    def test_register_creates_server(self):
        reg = ModelRegistry()
        model, tok = MagicMock(), MagicMock()
        server = reg.register("m1", model, tok)
        assert server is not None
        assert reg.get("m1") is server

    def test_register_sets_default_when_first(self):
        reg = ModelRegistry()
        reg.register("m1", MagicMock(), MagicMock())
        assert reg.default_id == "m1"

    def test_register_make_default(self):
        reg = ModelRegistry()
        reg.register("m1", MagicMock(), MagicMock())
        reg.register("m2", MagicMock(), MagicMock(), make_default=True)
        assert reg.default_id == "m2"

    def test_register_replaces_existing(self):
        reg = ModelRegistry()
        old = _mock_server("m1")
        reg._servers["m1"] = old
        reg.register("m1", MagicMock(), MagicMock())
        old.set_status.assert_called_with(ModelStatus.UNLOADED)
        assert reg.get("m1") is not old

    def test_register_passes_params(self):
        reg = ModelRegistry()
        server = reg.register("m1", MagicMock(), MagicMock(),
                              max_concurrent=4, generate_timeout=60.0,
                              enable_circuit_breaker=False, idle_timeout_s=300)
        assert server is not None


class TestRegisterEngine:
    def test_register_engine(self):
        reg = ModelRegistry()
        provider = MagicMock()
        reg.register_engine("eng1", provider)
        assert reg.get("eng1") is provider

    def test_register_engine_make_default(self):
        reg = ModelRegistry()
        reg.register_engine("eng1", MagicMock(), make_default=True)
        assert reg.default_id == "eng1"


class TestUnregister:
    def test_unregister_existing(self):
        reg = ModelRegistry()
        reg.register("m1", MagicMock(), MagicMock())
        assert reg.unregister("m1") is True
        assert reg.get("m1") is None

    def test_unregister_nonexistent(self):
        reg = ModelRegistry()
        assert reg.unregister("nope") is False

    def test_unregister_default_falls_back(self):
        reg = ModelRegistry()
        reg.register("m1", MagicMock(), MagicMock())
        reg.register("m2", MagicMock(), MagicMock())
        reg.unregister("m1")
        assert reg.default_id == "m2"

    def test_unregister_last_model(self):
        reg = ModelRegistry()
        reg.register("m1", MagicMock(), MagicMock())
        reg.unregister("m1")
        assert reg.default_id is None


class TestGet:
    def test_get_by_id(self):
        reg = ModelRegistry()
        srv = _mock_server("m1")
        reg._servers["m1"] = srv
        assert reg.get("m1") is srv

    def test_get_default(self):
        reg = ModelRegistry()
        srv = _mock_server("m1")
        reg._servers["m1"] = srv
        reg._default_id = "m1"
        assert reg.get() is srv

    def test_get_empty(self):
        reg = ModelRegistry()
        assert reg.get() is None
        assert reg.get("missing") is None


class TestDefaultId:
    def test_default_id_getter(self):
        reg = ModelRegistry()
        reg.register("m1", MagicMock(), MagicMock())
        assert reg.default_id == "m1"

    def test_default_id_setter(self):
        reg = ModelRegistry()
        reg.register("m1", MagicMock(), MagicMock())
        reg.register("m2", MagicMock(), MagicMock())
        reg.default_id = "m2"
        assert reg.default_id == "m2"

    def test_default_id_setter_ignores_unknown(self):
        reg = ModelRegistry()
        reg.register("m1", MagicMock(), MagicMock())
        reg.default_id = "nope"
        assert reg.default_id == "m1"


class TestListModels:
    def test_list_empty(self):
        reg = ModelRegistry()
        assert reg.list_models() == []

    def test_list_with_models(self):
        reg = ModelRegistry()
        reg.register("m1", MagicMock(), MagicMock())
        reg.register("m2", MagicMock(), MagicMock(), make_default=True)
        models = reg.list_models()
        assert len(models) == 2
        ids = {m["model_id"] for m in models}
        assert ids == {"m1", "m2"}
        defaults = [m for m in models if m["is_default"]]
        assert len(defaults) == 1
        assert defaults[0]["model_id"] == "m2"


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_default(self):
        reg = ModelRegistry()
        srv = _mock_server("m1")
        reg._servers["m1"] = srv
        reg._default_id = "m1"
        result = await reg.generate("hello")
        assert result["text"] == "hello"
        srv.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_by_id(self):
        reg = ModelRegistry()
        srv = _mock_server("m2")
        reg._servers["m2"] = srv
        result = await reg.generate("hello", model_id="m2")
        assert result["text"] == "hello"

    @pytest.mark.asyncio
    async def test_generate_no_model(self):
        reg = ModelRegistry()
        with pytest.raises(RuntimeError, match="No model registered"):
            await reg.generate("hello")

    @pytest.mark.asyncio
    async def test_generate_unknown_model(self):
        reg = ModelRegistry()
        reg.register("m1", MagicMock(), MagicMock())
        with pytest.raises(RuntimeError, match="requested 'nope'"):
            await reg.generate("hello", model_id="nope")


class TestHealthSummary:
    def test_health_empty(self):
        reg = ModelRegistry()
        h = reg.health_summary()
        assert h["models_loaded"] == 0
        assert h["models_registered"] == 0
        assert h["healthy"] is False
        assert h["has_errors"] is False

    def test_health_with_ready(self):
        reg = ModelRegistry()
        srv = _mock_server("m1", ModelStatus.READY)
        reg._servers["m1"] = srv
        reg._default_id = "m1"
        h = reg.health_summary()
        assert h["models_loaded"] == 1
        assert h["healthy"] is True

    def test_health_with_error(self):
        reg = ModelRegistry()
        srv = _mock_server("m1", ModelStatus.ERROR)
        reg._servers["m1"] = srv
        reg._default_id = "m1"
        h = reg.health_summary()
        assert h["has_errors"] is True
        assert h["healthy"] is False

    def test_health_degraded(self):
        reg = ModelRegistry()
        srv1 = _mock_server("m1", ModelStatus.READY)
        srv2 = _mock_server("m2", ModelStatus.DEGRADED)
        reg._servers["m1"] = srv1
        reg._servers["m2"] = srv2
        h = reg.health_summary()
        assert h["degraded"] is True
        assert h["healthy"] is False


class TestResetMetrics:
    def test_reset_metrics(self):
        reg = ModelRegistry()
        srv = _mock_server("m1")
        reg._servers["m1"] = srv
        reg._default_id = "m1"
        reg.reset_metrics()
        srv.set_status.assert_called_with(ModelStatus.READY)


class TestSingleton:
    def test_get_model_registry_returns_same(self):
        r1 = get_model_registry()
        r2 = get_model_registry()
        assert r1 is r2


class TestIterServers:
    def test_iter_servers_returns_all(self):
        reg = ModelRegistry()
        srv1 = _mock_server("m1")
        srv2 = _mock_server("m2")
        reg._servers["m1"] = srv1
        reg._servers["m2"] = srv2
        result = reg.iter_servers()
        assert len(result) == 2
        ids = {mid for mid, _ in result}
        assert ids == {"m1", "m2"}

    def test_iter_servers_empty(self):
        reg = ModelRegistry()
        result = reg.iter_servers()
        assert result == []

    def test_iter_servers_returns_snapshot(self):
        reg = ModelRegistry()
        srv1 = _mock_server("m1")
        reg._servers["m1"] = srv1
        result1 = reg.iter_servers()
        reg._servers["m2"] = _mock_server("m2")
        result2 = reg.iter_servers()
        assert len(result1) == 1
        assert len(result2) == 2
