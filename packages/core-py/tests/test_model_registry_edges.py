"""Edge-case tests for ModelRegistry (fast, no server)."""

from unittest.mock import patch

import pytest

from domains.infrastructure.model_registry import ModelRegistry


class _FakeProvider:
    def __init__(self):
        self.metadata = {"name": "fake"}
        self.status = "ready"

    def get_metrics_snapshot(self):
        return {"status": self.status, "metrics": {}}

    def set_status(self, status):
        self.status = status


class TestRegistryEdges:
    def test_get_empty_returns_none(self):
        reg = ModelRegistry()
        assert reg.get() is None
        assert reg.get("missing") is None

    def test_generate_without_model_raises(self):
        reg = ModelRegistry()
        with pytest.raises(RuntimeError, match="No model registered"):
            import asyncio

            asyncio.run(reg.generate("hello"))

    def test_generate_with_unknown_model_raises(self):
        reg = ModelRegistry()
        with pytest.raises(RuntimeError, match="requested 'nope'"):
            import asyncio

            asyncio.run(reg.generate("hello", model_id="nope"))

    def test_register_engine_makes_first_default(self):
        reg = ModelRegistry()
        reg.register_engine("eng1", _FakeProvider())
        assert reg.default_id == "eng1"
        assert reg.get("eng1") is reg.get()

    def test_register_engine_make_default_flag(self):
        reg = ModelRegistry()
        reg.register_engine("eng1", _FakeProvider())
        reg.register_engine("eng2", _FakeProvider(), make_default=True)
        assert reg.default_id == "eng2"

    def test_register_engine_list_models(self):
        reg = ModelRegistry()
        reg.register_engine("eng1", _FakeProvider())
        models = reg.list_models()
        assert len(models) == 1
        assert models[0]["model_id"] == "eng1"
        assert models[0]["is_default"] is True

    def test_emit_event_exception_swallowed(self):
        reg = ModelRegistry()
        with patch(
            "domains.infrastructure.event_bus.get_event_bus",
            side_effect=RuntimeError("bus down"),
        ):
            reg._emit_event("model.registered", "m")
            reg._emit_event("model.unregistered", "m")

    def test_emit_event_emit_failure_swallowed(self):
        reg = ModelRegistry()

        class BadBus:
            def emit_sync(self, *args, **kwargs):
                raise RuntimeError("emit failed")

        with patch(
            "domains.infrastructure.event_bus.get_event_bus",
            return_value=BadBus(),
        ):
            reg._emit_event("model.registered", "m")

    def test_unregister_only_model_clears_default(self):
        reg = ModelRegistry()
        reg.register_engine("eng1", _FakeProvider())
        assert reg.unregister("eng1") is True
        assert reg.default_id is None
        assert reg.get() is None
        assert reg.unregister("eng1") is False

    def test_default_id_setter_ignores_unknown(self):
        reg = ModelRegistry()
        reg.register_engine("eng1", _FakeProvider())
        reg.default_id = "missing"
        assert reg.default_id == "eng1"
