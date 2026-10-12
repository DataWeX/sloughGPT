"""Element-picker tests — headless, via a fake backend.

The picker is designed to run headless over CDP (JS injected via
``backend.evaluate``), so these tests drive it through a fake backend that
simulates the in-page overlay: ``evaluate`` returns the picker payload the real
overlay would produce. No browser, no human.
"""

from __future__ import annotations

from avion.core.element import ElementLocator, SelectorStrategy
from avion.events.models import EventType
from avion.events.recorder import EventRecorder
from avion.interact.picker import (
    _POLL_EXPR,
    ElementPicker,
    PickedElement,
    _locator_from_payload,
)


class FakeBackend:
    """Simulates the in-page overlay for the picker's evaluate() calls.

    Distinguishes the two evaluate() calls by the poll sentinel, NOT by
    substring: the overlay JS also contains "__avionPicked", so a loose
    substring match would misroute the activate call and never arm the picker.
    """

    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload or _SAMPLE_PAYLOAD
        self.evaluations: list[str] = []
        self.shots = 0
        self._picked_served = False

    async def evaluate(self, expression: str) -> object:
        self.evaluations.append(expression)
        if expression == _POLL_EXPR:
            if self._picked_served:
                return None
            self._picked_served = True
            return self.payload
        return "picker_active"

    async def find_element(self, locator: ElementLocator):
        return FakeElement()

    async def click(self, element) -> None:
        self.clicked = True

    async def screenshot(self) -> bytes:
        self.shots += 1
        return b"\x89PNG-fake"


class FakeElement:
    pass


# The payload the real in-page overlay produces for <button data-testid="save">.
_SAMPLE_PAYLOAD = {
    "strategy": "test_id",
    "value": "save",
    "tag": "button",
    "text": "Save document",
    "rect": {"x": 10, "y": 20, "width": 80, "height": 30},
    "candidates": [
        {"strategy": "test_id", "value": "save"},
        {"strategy": "role", "value": "button[Save document]"},
        {"strategy": "css", "value": "#save"},
    ],
}


# ── selector resolution (the correctness-critical part) ─────────────────────


def test_locator_keeps_candidate_priority_order():
    """The locator must try test_id first, then role, then css."""
    loc = _locator_from_payload(_SAMPLE_PAYLOAD)
    strategies = [s.strategy for s in loc.selectors]
    assert strategies == [
        SelectorStrategy.TEST_ID,
        SelectorStrategy.ROLE,
        SelectorStrategy.CSS,
    ]


def test_locator_values_match_payload():
    loc = _locator_from_payload(_SAMPLE_PAYLOAD)
    assert loc.selectors[0].value == "save"
    assert loc.selectors[1].value == "button[Save document]"
    assert loc.selectors[2].value == "#save"


def test_locator_description_includes_tag_and_text():
    loc = _locator_from_payload(_SAMPLE_PAYLOAD)
    assert "button" in loc.description
    assert "Save document" in loc.description


def test_locator_falls_back_to_css_when_all_others_empty():
    """If candidates are empty, the raw css value is still usable."""
    loc = _locator_from_payload({"strategy": "css", "value": "#only", "candidates": []})
    assert len(loc.selectors) == 1
    assert loc.selectors[0].strategy == SelectorStrategy.CSS
    assert loc.selectors[0].value == "#only"


def test_locator_skips_unknown_strategy_without_failing():
    payload = {
        "strategy": "css",
        "value": "#x",
        "candidates": [
            {"strategy": "bogus", "value": "nope"},  # unknown — must be skipped
            {"strategy": "css", "value": "#x"},
        ],
    }
    loc = _locator_from_payload(payload)
    assert [s.strategy for s in loc.selectors] == [SelectorStrategy.CSS]


# ── pick() flow: activate + record ──────────────────────────────────────────


async def test_pick_records_a_select_event():
    recorder = EventRecorder(session_name="test")
    backend = FakeBackend()
    picker = ElementPicker(backend, recorder=recorder)

    picked = await picker.pick(timeout=1)

    assert picked is not None
    assert picked.tag == "button"
    assert picked.locator.selectors[0].strategy == SelectorStrategy.TEST_ID
    selects = recorder.of_type(EventType.SELECT)
    assert len(selects) == 1
    assert selects[0].data["tag"] == "button"
    assert selects[0].data["best"] == picked.locator.describe()


async def test_pick_arms_the_overlay_first():
    backend = FakeBackend()
    picker = ElementPicker(backend)
    await picker.pick(timeout=1)
    # The overlay JS must be injected (not the poll) before we poll for a pick.
    non_poll = [e for e in backend.evaluations if e != _POLL_EXPR]
    assert non_poll, "overlay was never injected"
    assert any("__avionPicker.destroy" in e for e in non_poll)


async def test_pick_returns_none_on_timeout():
    class NoPickBackend(FakeBackend):
        async def evaluate(self, expression: str) -> object:
            self.evaluations.append(expression)
            if expression == _POLL_EXPR:
                return None  # never picked
            return "picker_active"

    picker = ElementPicker(NoPickBackend())
    assert await picker.pick(timeout=0.2) is None


# ── act() ────────────────────────────────────────────────────────────────────


async def test_act_clicks_and_records_a_click_event():
    recorder = EventRecorder(session_name="test")
    backend = FakeBackend()
    picker = ElementPicker(backend, recorder=recorder)
    picked = PickedElement(
        locator=_locator_from_payload(_SAMPLE_PAYLOAD),
        tag="button",
        text="Save document",
    )

    await picker.act(picked)

    assert getattr(backend, "clicked", False) is True
    clicks = recorder.of_type(EventType.CLICK)
    assert len(clicks) == 1
    assert clicks[0].success is True


async def test_act_records_failure_when_element_unresolvable():
    class GoneBackend(FakeBackend):
        async def find_element(self, locator):
            return None

    recorder = EventRecorder(session_name="test")
    picker = ElementPicker(GoneBackend(), recorder=recorder)
    picked = PickedElement(locator=_locator_from_payload(_SAMPLE_PAYLOAD), tag="button")

    await picker.act(picked)

    clicks = recorder.of_type(EventType.CLICK)
    assert len(clicks) == 1
    assert clicks[0].success is False
    assert "no longer resolvable" in clicks[0].error


# ── per-pick screenshots (the trace-viewer hallmark) ────────────────────────


async def test_pick_screenshot_reaches_the_recorder():
    backend = FakeBackend()
    recorder = EventRecorder(session_name="test")
    picker = ElementPicker(backend, recorder=recorder)

    await picker.pick(timeout=1)

    selects = recorder.of_type(EventType.SELECT)
    assert len(selects) == 1
    assert selects[0].data.get("screenshot") == b"\x89PNG-fake"
    assert backend.shots == 1


async def test_act_click_screenshot_reaches_the_recorder():
    backend = FakeBackend()
    recorder = EventRecorder(session_name="test")
    picker = ElementPicker(backend, recorder=recorder)
    picked = PickedElement(
        locator=_locator_from_payload(_SAMPLE_PAYLOAD), tag="button"
    )

    await picker.act(picked)

    clicks = recorder.of_type(EventType.CLICK)
    assert len(clicks) == 1 and clicks[0].success is True
    assert clicks[0].data.get("screenshot") == b"\x89PNG-fake"


async def test_failed_act_still_captures_the_broken_state():
    """The failure screenshot is the valuable one — it shows WHY the click broke."""

    class GoneBackend(FakeBackend):
        async def find_element(self, locator):
            return None

    recorder = EventRecorder(session_name="test")
    picker = ElementPicker(GoneBackend(), recorder=recorder)
    picked = PickedElement(
        locator=_locator_from_payload(_SAMPLE_PAYLOAD), tag="button"
    )

    await picker.act(picked)

    clicks = recorder.of_type(EventType.CLICK)
    assert len(clicks) == 1 and clicks[0].success is False
    assert clicks[0].data.get("screenshot") == b"\x89PNG-fake"


async def test_screenshot_failure_never_breaks_picking():
    """Best-effort contract: a blind backend loses the image, not the event."""

    class BlindBackend(FakeBackend):
        async def screenshot(self) -> bytes:
            raise RuntimeError("backend has no page")

    recorder = EventRecorder(session_name="test")
    picker = ElementPicker(BlindBackend(), recorder=recorder)

    picked = await picker.pick(timeout=1)

    assert picked is not None  # picking still works
    selects = recorder.of_type(EventType.SELECT)
    assert len(selects) == 1
    assert "screenshot" not in selects[0].data  # event kept, image dropped
