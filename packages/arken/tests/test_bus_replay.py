"""Bus + replay tests — patterns, priority, interceptors, replay."""

import asyncio

from arken.core.element import ElementLocator
from arken.events import (
    EventBus,
    EventRecorder,
    EventReplay,
    EventType,
    ReplayConfig,
)
from arken.events.bus import Subscription
from arken.events.models import Event
from arken.events.replay import parse_locator
from test_autoclicker import FakeBackend


def run(coro):
    return asyncio.run(coro)


def evt(type_, name="e"):
    return Event(type=type_, name=name)


class TestBus:
    def test_exact_and_wildcard(self):
        bus = EventBus()
        seen = []
        bus.subscribe("click", lambda e: seen.append(("exact", e.name)))
        bus.subscribe("*", lambda e: seen.append(("wild", e.name)))
        assert bus.publish(evt(EventType.CLICK, "a")) == 2
        assert ("exact", "a") in seen and ("wild", "a") in seen

    def test_prefix_pattern(self):
        bus = EventBus()
        seen = []
        bus.subscribe("task_*", lambda e: seen.append(e.name))
        bus.publish(evt(EventType.TASK_STARTED, "t"))
        bus.publish(evt(EventType.CLICK, "c"))
        assert seen == ["t"]

    def test_priority_order(self):
        bus = EventBus()
        order = []
        bus.subscribe("click", lambda e: order.append("low"), priority=1)
        bus.subscribe("click", lambda e: order.append("high"), priority=10)
        bus.publish(evt(EventType.CLICK))
        assert order == ["high", "low"]

    def test_interceptor_drops(self):
        bus = EventBus()
        seen = []
        bus.subscribe("*", lambda e: seen.append(e.name))
        bus.add_interceptor(lambda e: None if e.name == "noise" else e)
        assert bus.publish(evt(EventType.CLICK, "noise")) == 0
        assert bus.publish(evt(EventType.CLICK, "real")) == 1
        assert seen == ["real"]

    def test_bad_subscriber_skipped(self):
        bus = EventBus()
        bus.subscribe("click", lambda e: 1 / 0)
        assert bus.publish(evt(EventType.CLICK)) == 0

    def test_unsubscribe(self):
        bus = EventBus()
        sub = bus.subscribe("click", lambda e: None)
        assert isinstance(sub, Subscription)
        assert bus.unsubscribe(sub.id) is True
        assert bus.unsubscribe(sub.id) is False


class TestParseLocator:
    def test_css(self):
        assert parse_locator("css=input").describe() == "css=input"

    def test_text(self):
        loc = parse_locator("text(contains)='Start'")
        assert loc == ElementLocator.text("Start")

    def test_role_testid_label(self):
        assert parse_locator("role=button").describe() == "role=button"
        assert parse_locator("testId='go'").describe() == "testId='go'"
        assert parse_locator("label='Name'").describe() == "label='Name'"

    def test_unknown_none(self):
        assert parse_locator("xpath=//x") is None


class TestReplay:
    def _session(self):
        from arken import Arken
        from arken.core.element import ElementFinder
        from arken.core.navigator import Navigator

        a = Arken(base_url="http://x")
        a._backend = FakeBackend()
        a._navigator = Navigator(a._backend, "http://x")
        a._finder = ElementFinder(a._backend)
        return a

    def _recorded(self):
        rec = EventRecorder()
        rec.record(EventType.NAVIGATE, name="/chat")
        rec.record(EventType.CLICK, name=ElementLocator.text("Start").describe())
        rec.record(
            EventType.FILL, name=ElementLocator.css("input").describe(), data={"value": "hi"}
        )
        rec.record(EventType.SESSION_STARTED, name="s")  # no action
        return rec.events

    def test_full_replay(self):
        a = self._session()
        result = run(EventReplay(a).replay(self._recorded()))
        assert result.success is True
        assert result.passed == 3
        assert a.current_url == "http://x/chat"
        assert a._backend.clicked == [ElementLocator.text("Start").describe()]

    def test_dry_run_resolves_only(self):
        a = self._session()
        result = run(EventReplay(a, ReplayConfig(dry_run=True)).replay(self._recorded()))
        assert result.success is True
        assert a._backend.clicked == []

    def test_error_stops_by_default(self):
        a = self._session()
        rec = EventRecorder()
        rec.record(EventType.CLICK, name="xpath=//nope")
        rec.record(EventType.NAVIGATE, name="/late")
        result = run(EventReplay(a).replay(rec.events))
        assert result.success is False
        assert len(result.steps) == 1

    def test_continue_on_error(self):
        a = self._session()
        rec = EventRecorder()
        rec.record(EventType.CLICK, name="xpath=//nope")
        rec.record(EventType.NAVIGATE, name="/late")
        result = run(EventReplay(a, ReplayConfig(stop_on_error=False)).replay(rec.events))
        assert result.passed == 1 and result.failed == 1
        assert a.current_url == "http://x/late"
