"""
Tests for Event Bus (event_bus.py).
"""

import asyncio
import logging
import pytest
from domains.infrastructure import event_bus as eb
from domains.infrastructure.event_bus import (
    EventBus, Event, EventPriority,
    get_event_bus, set_event_bus,
)


@pytest.fixture
def bus():
    return EventBus(max_history=10)


class TestEvent:
    def test_default_fields(self):
        e = Event(name="test.event")
        assert e.name == "test.event"
        assert e.data == {}
        assert e.id.startswith("evt_")
        assert isinstance(e.timestamp, float)
        assert e.source == ""


@pytest.mark.asyncio
class TestEventBus:
    async def test_on_and_emit(self, bus):
        received = []

        async def handler(event, data):
            received.append((event, data))

        bus.on("test.event", handler)
        n = await bus.emit("test.event", {"key": "val"})
        assert n == 1
        assert received == [("test.event", {"key": "val"})]

    async def test_multiple_handlers(self, bus):
        results = []

        async def h1(event, data):
            results.append("h1")

        async def h2(event, data):
            results.append("h2")

        bus.on("test", h1)
        bus.on("test", h2)
        n = await bus.emit("test")
        assert n == 2
        assert results == ["h1", "h2"]

    async def test_emit_with_no_subscribers(self, bus):
        n = await bus.emit("no_one_listens")
        assert n == 0

    async def test_once_auto_unsubscribes(self, bus):
        calls = []

        async def handler(event, data):
            calls.append(1)

        bus.once("test.once", handler)
        await bus.emit("test.once")
        await bus.emit("test.once")
        await bus.emit("test.once")
        assert calls == [1]

    async def test_off_removes_handler(self, bus):
        calls = []

        async def handler(event, data):
            calls.append(1)

        bus.on("test", handler)
        ok = bus.off("test", handler)
        assert ok is True
        await bus.emit("test")
        assert calls == []

    async def test_off_nonexistent_returns_false(self, bus):
        async def handler(event, data):
            pass

        ok = bus.off("test", handler)
        assert ok is False

    async def test_clear_event(self, bus):
        async def handler(event, data):
            pass

        bus.on("a", handler)
        bus.on("b", handler)
        bus.clear("a")
        assert bus.subscriber_count == 1

    async def test_clear_all(self, bus):
        async def handler(event, data):
            pass

        bus.on("a", handler)
        bus.on("b", handler)
        bus.clear()
        assert bus.subscriber_count == 0

    async def test_error_isolation(self, bus):
        """One handler that throws should not prevent other handlers from running."""
        results = []

        async def bad_handler(event, data):
            raise ValueError("oops")

        async def good_handler(event, data):
            results.append("ok")

        bus.on("test", bad_handler)
        bus.on("test", good_handler)
        n = await bus.emit("test")
        assert n == 2
        assert results == ["ok"]

    async def test_wildcard_subscription(self, bus):
        results = []

        async def wild(event, data):
            results.append((event, data.get("x")))

        bus.on("*", wild)
        await bus.emit("a", {"x": 1})
        await bus.emit("b", {"x": 2})
        assert results == [("a", 1), ("b", 2)]

    async def test_wildcard_off(self, bus):
        calls = []

        async def handler(event, data):
            calls.append(1)

        bus.on("*", handler)
        bus.off("*", handler)
        await bus.emit("test")
        assert calls == []

    async def test_priority_ordering(self, bus):
        order = []

        async def low(event, data):
            order.append("low")

        async def high(event, data):
            order.append("high")

        bus.on("test", low, priority=EventPriority.MONITOR)
        bus.on("test", high, priority=EventPriority.CRITICAL)
        await bus.emit("test")
        assert order == ["high", "low"]

    async def test_history_stored(self, bus):
        await bus.emit("e1", {"n": 1})
        await bus.emit("e2", {"n": 2})
        assert len(bus.history()) == 2
        assert len(bus.history("e1")) == 1
        assert len(bus.history("e2")) == 1

    async def test_history_max(self, bus):
        bus._max_history = 3
        for i in range(5):
            await bus.emit("e", {"i": i})
        assert len(bus.history("e")) == 3

    async def test_replay_to_handler(self, bus):
        await bus.emit("e", {"x": 1})
        await bus.emit("e", {"x": 2})
        results = []

        def handler(event, data):
            results.append(data["x"])

        bus.replay("e", handler)
        assert results == [1, 2]

    async def test_replay_no_handler_returns_events(self, bus):
        await bus.emit("e", {"x": 1})
        events = bus.replay("e")
        assert len(events) == 1
        assert events[0].data["x"] == 1

    async def test_event_id_unique(self, bus):
        e1 = Event(name="a")
        e2 = Event(name="a")
        assert e1.id != e2.id

    async def test_event_source_propagated(self, bus):
        received = []

        async def handler(event, data):
            received.append(data.get("_source"))

        bus.on("test", handler)
        await bus.emit("test", {"msg": "hi"}, source="test-suite")
        hist = bus.history("test")
        assert hist[0].source == "test-suite"

    async def test_sync_emit(self, bus):
        results = []

        def handler(event, data):
            results.append(data.get("x"))

        bus.on("test", handler)
        n = bus.emit_sync("test", {"x": 42})
        assert n == 1
        assert results == [42]

    async def test_singleton(self):
        b1 = get_event_bus()
        b2 = get_event_bus()
        assert b1 is b2

    async def test_set_singleton(self):
        b = EventBus()
        set_event_bus(b)
        assert get_event_bus() is b

    async def test_subscriber_count(self, bus):
        assert bus.subscriber_count == 0

        async def h(event, data):
            pass

        bus.on("a", h)
        bus.on("b", h)
        assert bus.subscriber_count == 2

    async def test_on_raises_on_non_callable(self, bus):
        with pytest.raises(TypeError, match="handler must be callable"):
            bus.on("test", "not_callable")

    async def test_once_raises_on_non_callable(self, bus):
        with pytest.raises(TypeError, match="handler must be callable"):
            bus.once("test", 42)

    async def test_clear_star_clears_wildcards(self, bus):
        async def h(event, data):
            pass
        bus.on("*", h)
        bus.clear("*")
        assert bus.subscriber_count == 0

    @pytest.mark.filterwarnings("ignore:coroutine .* was never awaited:RuntimeWarning")
    async def test_emit_sync_skips_async_handlers(self, bus):
        results = []
        async def async_handler(event, data):
            results.append("async")

        bus.on("test", async_handler)
        n = bus.emit_sync("test")
        assert n == 1
        assert results == []

    async def test_mixed_handler_types(self, bus):
        results = []
        def sync_h(event, data):
            results.append("sync")
        async def async_h(event, data):
            results.append("async")

        bus.on("test", sync_h)
        bus.on("test", async_h)
        n = await bus.emit("test")
        assert n == 2
        assert results == ["sync", "async"]

    async def test_wildcard_and_specific_both_receive(self, bus):
        results = []
        async def wild(event, data):
            results.append(f"wild:{event}")
        async def specific(event, data):
            results.append(f"specific:{event}")

        bus.on("*", wild)
        bus.on("foo", specific)
        await bus.emit("foo")
        assert "wild:foo" in results
        assert "specific:foo" in results

    async def test_replay_empty_history_returns_empty_list(self, bus):
        events = bus.replay("nonexistent")
        assert events == []

    async def test_off_twice_returns_false(self, bus):
        async def h(event, data):
            pass
        bus.on("test", h)
        bus.off("test", h)
        assert bus.off("test", h) is False

    async def test_emit_count_includes_errored_handlers(self, bus):
        async def bad(event, data):
            raise ValueError("fail")

        bus.on("test", bad)
        n = await bus.emit("test")
        assert n == 1


# ── Sync / once / replay / log-subscriber branch coverage ──


class TestEventBusCoverage:
    def test_once_wildcard_fires_once(self, bus):
        calls = []

        def h(event, data):
            calls.append(event)

        bus.once("*", h)
        n1 = bus.emit_sync("a")
        n2 = bus.emit_sync("b")
        assert calls == ["a"]
        assert n1 == 1
        assert n2 == 0

    def test_once_specific_fires_once_sync(self, bus):
        calls = []

        def h(event, data):
            calls.append(1)

        bus.once("test", h)
        bus.emit_sync("test")
        bus.emit_sync("test")
        assert calls == [1]

    def test_emit_sync_errored_once_handler(self, bus, caplog):
        def bad(event, data):
            raise ValueError("boom")

        bus.once("test", bad)
        with caplog.at_level(logging.ERROR, logger="slo.event_bus"):
            n = bus.emit_sync("test")
        assert n == 1
        assert "Sync handler failed for" in caplog.text
        assert bus.subscriber_count == 0

    @pytest.mark.filterwarnings("ignore:coroutine .* was never awaited:RuntimeWarning")
    def test_replay_async_handler_warns_filtered(self, bus):
        async def h(event, data):
            pass

        bus.emit_sync("e", {"x": 1})
        with pytest.warns(UserWarning, match="async handler passed to sync replay"):
            bus.replay("e", h)

    def test_replay_errored_handler_logs(self, bus, caplog):
        def bad(event, data):
            raise ValueError("nope")

        bus.emit_sync("e", {"x": 1})
        with caplog.at_level(logging.ERROR, logger="slo.event_bus"):
            bus.replay("e", bad)
        assert "Replay handler failed for" in caplog.text

    def test_is_noisy(self):
        assert eb._is_noisy("heartbeat")
        assert eb._is_noisy("metric.cpu")
        assert not eb._is_noisy("model.loaded")

    def test_install_log_subscriber(self, monkeypatch, caplog):
        monkeypatch.setattr(eb, "_LOG_SUBSCRIBER_INSTALLED", False)
        b = EventBus(max_history=5)
        eb.install_log_subscriber(b)
        assert b.subscriber_count == 1
        with caplog.at_level(logging.INFO, logger="slo.event_sensor"):
            b.emit_sync("model.loaded", {"name": "gpt2"})
            b.emit_sync("heartbeat")
        assert "EVENT model.loaded" in caplog.text
        assert "{'name': 'gpt2'}" in caplog.text
        assert "EVENT heartbeat" not in caplog.text

    def test_install_log_subscriber_idempotent(self, monkeypatch):
        monkeypatch.setattr(eb, "_LOG_SUBSCRIBER_INSTALLED", False)
        b1 = EventBus()
        b2 = EventBus()
        eb.install_log_subscriber(b1)
        eb.install_log_subscriber(b2)
        assert b1.subscriber_count == 1
        assert b2.subscriber_count == 0

    def test_install_log_subscriber_default_bus(self, monkeypatch):
        monkeypatch.setattr(eb, "_LOG_SUBSCRIBER_INSTALLED", False)
        b = EventBus()
        monkeypatch.setattr(eb, "get_event_bus", lambda: b)
        eb.install_log_subscriber()
        assert b.subscriber_count == 1
