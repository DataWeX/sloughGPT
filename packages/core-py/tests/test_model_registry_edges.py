"""Edge-case tests for ModelRegistry (fast, no server)."""

import asyncio
from unittest.mock import patch

import pytest

from domains.infrastructure.model_registry import ModelRegistry, get_model_registry


class _FakeProvider:
    def __init__(self, name="fake"):
        self.metadata = {"name": name}
        self.status = "ready"

    def get_metrics_snapshot(self):
        return {"status": self.status, "metrics": {}}

    def set_status(self, status):
        self.status = status


# ---------------------------------------------------------------------------
# Basic access
# ---------------------------------------------------------------------------

class TestRegistryEdges:
    def test_get_empty_returns_none(self):
        reg = ModelRegistry()
        assert reg.get() is None
        assert reg.get("missing") is None

    def test_generate_without_model_raises(self):
        reg = ModelRegistry()
        with pytest.raises(RuntimeError, match="No model registered"):
            asyncio.run(reg.generate("hello"))

    def test_generate_with_unknown_model_raises(self):
        reg = ModelRegistry()
        with pytest.raises(RuntimeError, match="requested 'nope'"):
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


# ---------------------------------------------------------------------------
# Multiple engines
# ---------------------------------------------------------------------------

class TestRegistryMultipleEngines:
    def test_register_multiple(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider("a"))
        reg.register_engine("b", _FakeProvider("b"))
        reg.register_engine("c", _FakeProvider("c"))
        assert len(reg.list_models()) == 3

    def test_unregister_leaves_others(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider("a"))
        reg.register_engine("b", _FakeProvider("b"))
        reg.unregister("a")
        models = reg.list_models()
        assert len(models) == 1
        assert models[0]["model_id"] == "b"

    def test_default_shifts_on_unregister(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider())
        reg.register_engine("b", _FakeProvider())
        assert reg.default_id == "a"
        reg.unregister("a")
        assert reg.default_id == "b"

    def test_list_models_default_flag(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider())
        reg.register_engine("b", _FakeProvider(), make_default=True)
        models = reg.list_models()
        for m in models:
            if m["model_id"] == "a":
                assert m["is_default"] is False
            elif m["model_id"] == "b":
                assert m["is_default"] is True

    def test_get_specific_engine(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider("a"))
        reg.register_engine("b", _FakeProvider("b"))
        assert reg.get("b") is not None
        assert reg.get("a") is not None
        assert reg.get("a") is not reg.get("b")

    def test_default_id_setter_valid(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider())
        reg.register_engine("b", _FakeProvider())
        reg.default_id = "b"
        assert reg.default_id == "b"
        assert reg.get() is reg.get("b")

    def test_unregister_nonexistent_returns_false(self):
        reg = ModelRegistry()
        assert reg.unregister("ghost") is False

    def test_register_same_id_twice_replaces(self):
        reg = ModelRegistry()
        p1 = _FakeProvider("first")
        p2 = _FakeProvider("second")
        reg.register_engine("x", p1)
        reg.register_engine("x", p2)
        models = reg.list_models()
        assert len(models) == 1

    def test_empty_registry_health_summary(self):
        reg = ModelRegistry()
        h = reg.health_summary()
        assert h["models_loaded"] == 0
        assert h["models_registered"] == 0
        assert h["healthy"] is False

    def test_health_summary_after_register(self):
        reg = ModelRegistry()
        reg.register_engine("eng", _FakeProvider())
        h = reg.health_summary()
        assert h["models_registered"] == 1
        assert h["default_model"] == "eng"


# ---------------------------------------------------------------------------
# Default ID management
# ---------------------------------------------------------------------------

class TestDefaultIdManagement:
    def test_default_id_none_initially(self):
        reg = ModelRegistry()
        assert reg.default_id is None

    def test_first_register_becomes_default(self):
        reg = ModelRegistry()
        reg.register_engine("first", _FakeProvider())
        assert reg.default_id == "first"

    def test_make_default_overrides(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider())
        reg.register_engine("b", _FakeProvider(), make_default=True)
        assert reg.default_id == "b"

    def test_make_default_false_keeps_current(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider())
        reg.register_engine("b", _FakeProvider(), make_default=False)
        assert reg.default_id == "a"

    def test_setter_to_existing(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider())
        reg.register_engine("b", _FakeProvider())
        reg.default_id = "b"
        assert reg.default_id == "b"

    def test_setter_to_none_is_noop(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider())
        reg.default_id = None
        assert reg.default_id == "a"


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------

class TestEventEmission:
    def test_emit_registered_event(self):
        reg = ModelRegistry()
        emitted = []

        class FakeBus:
            def emit_sync(self, event, data, **kw):
                emitted.append((event, data))

        with patch("domains.infrastructure.event_bus.get_event_bus", return_value=FakeBus()):
            reg.register_engine("eng1", _FakeProvider())
        assert any(e[0] == "model.registered" for e in emitted)

    def test_emit_unregistered_event(self):
        reg = ModelRegistry()
        reg.register_engine("eng1", _FakeProvider())
        emitted = []

        class FakeBus:
            def emit_sync(self, event, data, **kw):
                emitted.append((event, data))

        with patch("domains.infrastructure.event_bus.get_event_bus", return_value=FakeBus()):
            reg.unregister("eng1")
        assert any(e[0] == "model.unregistered" for e in emitted)


# ---------------------------------------------------------------------------
# Health summary
# ---------------------------------------------------------------------------

class TestHealthSummary:
    def test_health_no_models(self):
        reg = ModelRegistry()
        h = reg.health_summary()
        assert h["healthy"] is False
        assert h["degraded"] is False
        assert h["has_errors"] is False
        assert h["models"] == []

    def test_health_one_model(self):
        reg = ModelRegistry()
        reg.register_engine("eng", _FakeProvider())
        h = reg.health_summary()
        assert h["models_registered"] == 1
        assert h["default_model"] == "eng"

    def test_health_degraded_model(self):
        reg = ModelRegistry()
        p = _FakeProvider()
        p.status = "degraded"
        reg.register_engine("eng", p)
        h = reg.health_summary()
        assert h["degraded"] is True

    def test_health_error_model(self):
        reg = ModelRegistry()
        p = _FakeProvider()
        p.status = "error"
        reg.register_engine("eng", p)
        h = reg.health_summary()
        assert h["has_errors"] is True

    def test_health_multiple_models(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider())
        reg.register_engine("b", _FakeProvider())
        h = reg.health_summary()
        assert h["models_registered"] == 2

    def test_health_default_model(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider())
        reg.register_engine("b", _FakeProvider(), make_default=True)
        h = reg.health_summary()
        assert h["default_model"] == "b"


# ---------------------------------------------------------------------------
# get_model_registry singleton
# ---------------------------------------------------------------------------

class TestGetModelRegistry:
    def test_singleton(self):
        r1 = get_model_registry()
        r2 = get_model_registry()
        assert r1 is r2

    def test_singleton_has_models(self):
        r = get_model_registry()
        assert isinstance(r, ModelRegistry)


# ---------------------------------------------------------------------------
# register() — full ModelServer path (not register_engine)
# ---------------------------------------------------------------------------

class TestRegisterMethod:
    def test_register_creates_server(self):
        reg = ModelRegistry()
        server = reg.register("m1", model="fake_model", tokenizer="fake_tok")
        assert server is not None
        assert reg.get("m1") is server

    def test_register_first_becomes_default(self):
        reg = ModelRegistry()
        reg.register("m1", model="m", tokenizer="t")
        assert reg.default_id == "m1"

    def test_register_make_default(self):
        reg = ModelRegistry()
        reg.register("m1", model="m", tokenizer="t")
        reg.register("m2", model="m2", tokenizer="t2", make_default=True)
        assert reg.default_id == "m2"

    def test_register_replaces_same_id(self):
        reg = ModelRegistry()
        reg.register("m1", model="first", tokenizer="t1")
        reg.register("m1", model="second", tokenizer="t2")
        models = reg.list_models()
        assert len(models) == 1

    def test_register_with_timeout(self):
        reg = ModelRegistry()
        server = reg.register("m1", model="m", tokenizer="t", generate_timeout=30.0)
        assert server is not None

    def test_register_with_max_concurrent(self):
        reg = ModelRegistry()
        server = reg.register("m1", model="m", tokenizer="t", max_concurrent=4)
        assert server is not None

    def test_register_with_circuit_breaker_disabled(self):
        reg = ModelRegistry()
        server = reg.register("m1", model="m", tokenizer="t", enable_circuit_breaker=False)
        assert server is not None

    def test_register_with_idle_timeout(self):
        reg = ModelRegistry()
        server = reg.register("m1", model="m", tokenizer="t", idle_timeout_s=60.0)
        assert server is not None

    def test_register_multiple_models(self):
        reg = ModelRegistry()
        reg.register("a", model="ma", tokenizer="ta")
        reg.register("b", model="mb", tokenizer="tb")
        reg.register("c", model="mc", tokenizer="tc")
        assert len(reg.list_models()) == 3

    def test_register_with_process_guard(self):
        reg = ModelRegistry()
        server = reg.register("m1", model="m", tokenizer="t", process_guard=None)
        assert server is not None


# ---------------------------------------------------------------------------
# reset_metrics
# ---------------------------------------------------------------------------

class _FakeProviderWithMetrics:
    def __init__(self, name="fake"):
        self.metadata = {"name": name}
        self.status = "ready"
        self.metrics = type("Metrics", (), {"__init__": lambda s: None})()

    def get_metrics_snapshot(self):
        return {"status": self.status, "metrics": {}}

    def set_status(self, status):
        self.status = status


class TestResetMetrics:
    def test_reset_metrics_no_models(self):
        reg = ModelRegistry()
        reg.reset_metrics()  # should not raise

    def test_reset_metrics_with_model(self):
        reg = ModelRegistry()
        reg.register_engine("eng", _FakeProviderWithMetrics())
        reg.reset_metrics()  # should not raise

    def test_reset_metrics_restores_status(self):
        from domains.infrastructure.model_server import ModelStatus
        reg = ModelRegistry()
        p = _FakeProviderWithMetrics()
        p.status = "degraded"
        reg.register_engine("eng", p)
        reg.reset_metrics()
        assert p.status == ModelStatus.READY


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_register_unregister(self):
        import threading
        reg = ModelRegistry()
        errors = []

        def worker(tid):
            try:
                reg.register_engine(f"e{tid}", _FakeProvider(f"p{tid}"))
                reg.unregister(f"e{tid}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_concurrent_get_default(self):
        import threading
        reg = ModelRegistry()
        reg.register_engine("e1", _FakeProvider())
        results = []

        def reader():
            for _ in range(50):
                results.append(reg.default_id)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r == "e1" for r in results)

    def test_concurrent_health_summary(self):
        import threading
        reg = ModelRegistry()
        reg.register_engine("e1", _FakeProvider())
        errors = []

        def reader():
            try:
                for _ in range(50):
                    reg.health_summary()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# list_models — additional edge cases
# ---------------------------------------------------------------------------

class TestListModelsExtra:
    def test_list_models_after_unregister_all(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider())
        reg.unregister("a")
        assert reg.list_models() == []

    def test_list_models_replace_preserves_count(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider())
        reg.register_engine("a", _FakeProvider())
        models = reg.list_models()
        assert len(models) == 1

    def test_list_models_default_flag_after_shift(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider())
        reg.register_engine("b", _FakeProvider())
        reg.unregister("a")
        models = reg.list_models()
        assert any(m["model_id"] == "b" and m["is_default"] for m in models)


# ---------------------------------------------------------------------------
# health_summary — additional edge cases
# ---------------------------------------------------------------------------

class TestHealthSummaryExtra:
    def test_health_mixed_statuses(self):
        reg = ModelRegistry()
        p1 = _FakeProvider()
        p1.status = "ready"
        p2 = _FakeProvider()
        p2.status = "error"
        reg.register_engine("good", p1)
        reg.register_engine("bad", p2)
        h = reg.health_summary()
        assert h["models_registered"] == 2
        assert h["has_errors"] is True

    def test_health_all_healthy(self):
        reg = ModelRegistry()
        p1 = _FakeProvider()
        p1.status = "ready"
        p2 = _FakeProvider()
        p2.status = "ready"
        reg.register_engine("a", p1)
        reg.register_engine("b", p2)
        h = reg.health_summary()
        assert h["healthy"] is True

    def test_health_after_unregister(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider())
        reg.unregister("a")
        h = reg.health_summary()
        assert h["models_registered"] == 0
        assert h["healthy"] is False


# ---------------------------------------------------------------------------
# generate — additional cases
# ---------------------------------------------------------------------------

class TestGenerateExtra:
    def test_generate_no_default_no_model_id(self):
        reg = ModelRegistry()
        with pytest.raises(RuntimeError, match="No model registered"):
            asyncio.run(reg.generate("hello"))

    def test_generate_error_includes_model_id(self):
        reg = ModelRegistry()
        with pytest.raises(RuntimeError, match="requested 'x'"):
            asyncio.run(reg.generate("hello", model_id="x"))


# ---------------------------------------------------------------------------
# unregister — additional edge cases
# ---------------------------------------------------------------------------

class TestUnregisterExtra:
    def test_unregister_then_re_register(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider())
        reg.unregister("a")
        reg.register_engine("a", _FakeProvider())
        assert reg.get("a") is not None
        assert reg.default_id == "a"

    def test_unregister_middle_of_three(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider())
        reg.register_engine("b", _FakeProvider())
        reg.register_engine("c", _FakeProvider())
        reg.unregister("b")
        models = reg.list_models()
        ids = [m["model_id"] for m in models]
        assert "a" in ids
        assert "c" in ids
        assert "b" not in ids

    def test_unregister_does_not_affect_others_default(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider())
        reg.register_engine("b", _FakeProvider(), make_default=True)
        reg.unregister("a")
        assert reg.default_id == "b"


# ---------------------------------------------------------------------------
# default_id — additional edge cases
# ---------------------------------------------------------------------------

class TestDefaultIdExtra:
    def test_setter_to_self_is_noop(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider())
        reg.default_id = "a"
        assert reg.default_id == "a"

    def test_setter_after_all_unregister(self):
        reg = ModelRegistry()
        reg.register_engine("a", _FakeProvider())
        reg.unregister("a")
        reg.default_id = "b"  # should be ignored
        assert reg.default_id is None

    def test_first_register_default(self):
        reg = ModelRegistry()
        reg.register_engine("first", _FakeProvider())
        assert reg.default_id == "first"
        reg.register_engine("second", _FakeProvider())
        assert reg.default_id == "first"  # first stays default
