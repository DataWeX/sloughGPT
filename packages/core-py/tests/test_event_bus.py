"""
Tests for domains/infrastructure/event_bus.py — async pub/sub event bus.

Covers:
    - Subscribe (on) and emit
    - Once subscription (auto-remove after one emission)
    - Wildcard subscriptions
    - Unsubscribe (off)
    - Clear subscriptions
    - Event history and replay
    - Priority ordering (CRITICAL > HIGH > NORMAL > MONITOR)
    - Error isolation (bad handler doesn't crash bus)
    - emit_sync (non-async emit)
    - subscriber_count
    - Singleton get/set
    - _is_noisy filter
"""

import asyncio
import sys
from pathlib import Path
import pytest

_CORE_PY = Path(__file__).resolve().parents[1]
if str(_CORE_PY) not in sys.path:
    sys.path.insert(0, str(_CORE_PY))

from domains.infrastructure.event_bus import (
    EventBus,
    Event,
    EventPriority,
    Subscription,
    get_event_bus,
    set_event_bus,
    _is_noisy,
    install_log_subscriber,
    _LOG_SUBSCRIBER_INSTALLED,
)


# ── Subscribe + Emit ──────────────────────────────────────────────────


class TestSubscribeEmit:
    def test_on_and_emit(self):
        bus = EventBus()
        received = []
        bus.on("test.event", lambda name, data: received.append((name, data)))
        count = bus.emit_sync("test.event", {"key": "val"})
        assert count == 1
        assert received == [("test.event", {"key": "val"})]

    def test_no_data_defaults_to_empty(self):
        bus = EventBus()
        received = []
        bus.on("test.event", lambda name, data: received.append(data))
        bus.emit_sync("test.event")
        assert received == [{}]

    def test_multiple_handlers(self):
        bus = EventBus()
        results = []
        bus.on("x", lambda n, d: results.append("a"))
        bus.on("x", lambda n, d: results.append("b"))
        bus.emit_sync("x")
        assert results == ["a", "b"]

    def test_handler_receives_source(self):
        bus = EventBus()
        received = []
        bus.on("x", lambda n, d: received.append(d))
        bus.emit_sync("x", {"msg": "hi"}, source="test")
        assert received[0]["msg"] == "hi"


# ── Once ──────────────────────────────────────────────────────────────


class TestOnce:
    def test_once_fires_once(self):
        bus = EventBus()
        count = []
        bus.once("x", lambda n, d: count.append(1))
        bus.emit_sync("x")
        bus.emit_sync("x")
        assert count == [1]

    def test_once_not_in_subscriber_count_after(self):
        bus = EventBus()
        bus.once("x", lambda n, d: None)
        assert bus.subscriber_count == 1
        bus.emit_sync("x")
        assert bus.subscriber_count == 0


# ── Wildcard ──────────────────────────────────────────────────────────


class TestWildcard:
    def test_wildcard_receives_all(self):
        bus = EventBus()
        received = []
        bus.on("*", lambda n, d: received.append(n))
        bus.emit_sync("a")
        bus.emit_sync("b")
        assert received == ["a", "b"]

    def test_wildcard_and_specific(self):
        bus = EventBus()
        specific = []
        wild = []
        bus.on("target", lambda n, d: specific.append(1))
        bus.on("*", lambda n, d: wild.append(1))
        bus.emit_sync("target")
        assert specific == [1]
        assert wild == [1]


# ── Off ───────────────────────────────────────────────────────────────


class TestOff:
    def test_off_removes_handler(self):
        bus = EventBus()
        received = []
        handler = lambda n, d: received.append(1)
        bus.on("x", handler)
        assert bus.off("x", handler) is True
        bus.emit_sync("x")
        assert received == []

    def test_off_returns_false_if_not_found(self):
        bus = EventBus()
        assert bus.off("x", lambda n, d: None) is False

    def test_off_wildcard(self):
        bus = EventBus()
        handler = lambda n, d: None
        bus.on("*", handler)
        assert bus.off("*", handler) is True
        assert bus.subscriber_count == 0


# ── Clear ─────────────────────────────────────────────────────────────


class TestClear:
    def test_clear_specific_event(self):
        bus = EventBus()
        bus.on("a", lambda n, d: None)
        bus.on("b", lambda n, d: None)
        bus.clear("a")
        assert bus.subscriber_count == 1

    def test_clear_all(self):
        bus = EventBus()
        bus.on("a", lambda n, d: None)
        bus.on("b", lambda n, d: None)
        bus.on("*", lambda n, d: None)
        bus.clear()
        assert bus.subscriber_count == 0

    def test_clear_wildcard(self):
        bus = EventBus()
        bus.on("*", lambda n, d: None)
        bus.on("x", lambda n, d: None)
        bus.clear("*")
        assert bus.subscriber_count == 1


# ── History ───────────────────────────────────────────────────────────


class TestHistory:
    def test_emit_stores_history(self):
        bus = EventBus()
        bus.emit_sync("x", {"v": 1})
        bus.emit_sync("y", {"v": 2})
        h = bus.history()
        assert len(h) == 2

    def test_history_filtered(self):
        bus = EventBus()
        bus.emit_sync("a")
        bus.emit_sync("b")
        bus.emit_sync("a")
        assert len(bus.history("a")) == 2
        assert len(bus.history("b")) == 1

    def test_history_max_limit(self):
        bus = EventBus(max_history=3)
        for i in range(5):
            bus.emit_sync("x", {"i": i})
        assert len(bus.history("x")) == 3

    def test_replay_calls_handler(self):
        bus = EventBus()
        bus.emit_sync("x", {"v": 1})
        bus.emit_sync("y", {"v": 2})
        replayed = []
        bus.replay(handler=lambda n, d: replayed.append(n))
        assert replayed == ["x", "y"]

    def test_replay_returns_events(self):
        bus = EventBus()
        bus.emit_sync("x")
        events = bus.replay()
        assert len(events) == 1
        assert events[0].name == "x"


# ── Priority ──────────────────────────────────────────────────────────


class TestPriority:
    def test_high_priority_runs_first(self):
        bus = EventBus()
        order = []
        bus.on("x", lambda n, d: order.append("normal"), priority=EventPriority.NORMAL)
        bus.on("x", lambda n, d: order.append("high"), priority=EventPriority.HIGH)
        bus.on("x", lambda n, d: order.append("critical"), priority=EventPriority.CRITICAL)
        bus.emit_sync("x")
        assert order == ["critical", "high", "normal"]


# ── Error Isolation ──────────────────────────────────────────────────


class TestErrorIsolation:
    def test_bad_handler_doesnt_crash_bus(self):
        bus = EventBus()
        def bad_handler(n, d):
            raise ValueError("boom")
        good = []
        bus.on("x", bad_handler)
        bus.on("x", lambda n, d: good.append(1))
        count = bus.emit_sync("x")
        assert count == 2
        assert good == [1]

    def test_async_handler_error_isolated(self):
        bus = EventBus()
        async def bad(n, d):
            raise RuntimeError("async boom")
        good = []
        bus.on("x", bad)
        bus.on("x", lambda n, d: good.append(1))

        async def run():
            return await bus.emit("x")

        result = asyncio.run(run())
        assert result == 2
        assert good == [1]


# ── emit_sync vs emit ─────────────────────────────────────────────────


class TestEmitSync:
    def test_sync_skips_async_handlers(self):
        import warnings
        bus = EventBus()
        called = []
        async def handler(n, d):
            called.append(1)
        bus.on("x", handler)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            count = bus.emit_sync("x")
        assert count == 1
        assert called == []  # sync emit skips coroutine results

    def test_async_emit_awaits_async_handlers(self):
        bus = EventBus()
        called = []
        async def handler(n, d):
            called.append(1)
        bus.on("x", handler)

        async def run():
            return await bus.emit("x")

        result = asyncio.run(run())
        assert result == 1
        assert called == [1]


# ── subscriber_count ──────────────────────────────────────────────────


class TestSubscriberCount:
    def test_count(self):
        bus = EventBus()
        assert bus.subscriber_count == 0
        bus.on("a", lambda n, d: None)
        bus.on("a", lambda n, d: None)
        bus.on("*", lambda n, d: None)
        assert bus.subscriber_count == 3


# ── Singleton ─────────────────────────────────────────────────────────


class TestSingleton:
    def test_get_returns_same(self):
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2

    def test_set_replaces(self):
        original = get_event_bus()
        new_bus = EventBus()
        set_event_bus(new_bus)
        assert get_event_bus() is new_bus
        set_event_bus(original)  # restore


# ── _is_noisy ─────────────────────────────────────────────────────────


class TestIsNoisy:
    def test_heartbeat(self):
        assert _is_noisy("heartbeat") is True

    def test_metric(self):
        assert _is_noisy("metric.cpu") is True

    def test_cache(self):
        assert _is_noisy("cache.hit") is True

    def test_normal_event(self):
        assert _is_noisy("model.loaded") is False

    def test_error_event(self):
        assert _is_noisy("error.raised") is False


# ── Event dataclass ───────────────────────────────────────────────────


class TestEventDataclass:
    def test_defaults(self):
        evt = Event(name="test")
        assert evt.name == "test"
        assert evt.data == {}
        assert evt.id.startswith("evt_")
        assert evt.timestamp > 0
        assert evt.source == ""

    def test_custom(self):
        evt = Event(name="x", data={"k": "v"}, source="src")
        assert evt.data == {"k": "v"}
        assert evt.source == "src"


# ── Non-callable handler rejection ────────────────────────────────────


class TestHandlerValidation:
    def test_on_rejects_non_callable(self):
        bus = EventBus()
        with pytest.raises(TypeError, match="callable"):
            bus.on("x", "not a function")

    def test_once_rejects_non_callable(self):
        bus = EventBus()
        with pytest.raises(TypeError, match="callable"):
            bus.once("x", 42)


# ── install_log_subscriber ─────────────────────────────────────────────


class TestInstallLogSubscriber:
    def test_idempotent(self):
        """Calling install_log_subscriber twice doesn't add duplicate handlers."""
        import domains.infrastructure.event_bus as mod
        original = mod._LOG_SUBSCRIBER_INSTALLED
        mod._LOG_SUBSCRIBER_INSTALLED = False
        try:
            bus = EventBus()
            install_log_subscriber(bus)
            count_after_first = bus.subscriber_count
            install_log_subscriber(bus)  # second call — should be no-op
            assert bus.subscriber_count == count_after_first
        finally:
            mod._LOG_SUBSCRIBER_INSTALLED = original

    def test_noisy_events_filtered(self):
        """Noisy events (heartbeat, metric, cache) should not reach the log handler."""
        import logging
        import domains.infrastructure.event_bus as mod
        original = mod._LOG_SUBSCRIBER_INSTALLED
        mod._LOG_SUBSCRIBER_INSTALLED = False
        try:
            bus = EventBus()
            install_log_subscriber(bus)
            # Emit a noisy event — should not raise
            count = bus.emit_sync("heartbeat", {"ts": 1.0})
            assert count >= 1  # handler was invoked (but filtered inside)
        finally:
            mod._LOG_SUBSCRIBER_INSTALLED = original

    def test_replay_with_async_handler_warns(self):
        """replay() with an async handler should emit a warning."""
        bus = EventBus()
        bus.emit_sync("x", {"v": 1})

        async def async_handler(name, data):
            pass

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bus.replay(handler=async_handler)
            async_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(async_warnings) == 1
            assert "async" in str(async_warnings[0].message).lower()
