"""Arken v1 tests — autoclicker: find, click, type. No browser needed."""

from __future__ import annotations

import asyncio
from typing import Any

from avion import (
    Arken,
    Coordinate,
    Element,
    ElementLocator,
    InteractionChain,
    Keyboard,
    Mouse,
)
from avion.core.task import Task, TaskStatus, TaskStep


class FakeBackend:
    """In-memory backend: one page with named elements."""

    name = "fake"

    def __init__(self):
        self.started = False
        self.stopped = False
        self.url = ""
        self.clicked: list[str] = []
        self.filled: dict[str, str] = {}
        self.selected: dict[str, str] = {}
        self.pressed: list[str] = []
        self.typed: list[str] = []
        self.mouse_clicks: list[tuple[float, float]] = []
        self.page_text = ""
        self._elements = {
            ElementLocator.text("Start").describe(): Element(
                raw={"id": 1}, locator=ElementLocator.text("Start")
            ),
            ElementLocator.css("input").describe(): Element(
                raw={"id": 2}, locator=ElementLocator.css("input")
            ),
        }

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def navigate(self, url: str) -> None:
        self.url = url

    async def find_element(self, locator: ElementLocator) -> Element | None:
        return self._elements.get(locator.describe())

    async def find_elements(self, locator: ElementLocator) -> list[Element]:
        el = self._elements.get(locator.describe())
        return [el] if el else []

    async def click(self, element: Element) -> None:
        self.clicked.append(element.locator.describe())

    async def fill(self, element: Element, value: str) -> None:
        self.filled[element.locator.describe()] = value

    async def select_option(self, element: Element, value: str) -> None:
        self.selected[element.locator.describe()] = value

    async def check(self, element: Element) -> None:
        raise NotImplementedError

    async def uncheck(self, element: Element) -> None:
        raise NotImplementedError

    async def hover(self, element: Element) -> None:
        pass

    async def screenshot(self, path: str | None = None) -> bytes:
        return b"fake-png"

    async def get_url(self) -> str:
        return self.url

    async def get_title(self) -> str:
        return "fake"

    async def wait_for(self, locator: ElementLocator, timeout: float = 10.0) -> Element | None:
        return self._elements.get(locator.describe())

    async def wait_for_timeout(self, ms: float) -> None:
        pass

    async def evaluate(self, expression: str) -> Any:
        if expression.startswith("Array.from"):
            return 0
        return self.page_text

    async def get_accessibility_tree(self) -> dict[str, Any]:
        return {}

    # Coordinate-level (InteractionBackend)
    async def mouse_move(self, x: float, y: float) -> None:
        pass

    async def mouse_down(self, button=None) -> None:
        pass

    async def mouse_up(self, button=None) -> None:
        pass

    async def mouse_click(self, x: float, y: float, button=None) -> None:
        self.mouse_clicks.append((x, y))

    async def mouse_double_click(self, x: float, y: float) -> None:
        pass

    async def mouse_scroll(self, x: float, y: float, delta_x: int, delta_y: int) -> None:
        pass

    async def keyboard_down(self, key: str) -> None:
        pass

    async def keyboard_up(self, key: str) -> None:
        pass

    async def keyboard_press(self, key: str, hold_ms: float = 0) -> None:
        self.pressed.append(key)

    async def keyboard_type(self, text: str, delay_ms: float = 0) -> None:
        self.typed.append(text)

    async def get_viewport_size(self) -> tuple[int, int]:
        return (1280, 720)

    async def get_screenshot_as_bytes(self) -> bytes:
        return b"fake-png"


def run(coro):
    return asyncio.run(coro)


class TestLocators:
    def test_text_locator_describes(self):
        loc = ElementLocator.text("Start")
        assert loc.describe() == "text(contains)='Start'"

    def test_css_locator_describes(self):
        loc = ElementLocator.css("input")
        assert loc.describe() == "css=input"


class TestSession:
    def _session(self):
        from avion.interact.primitives import Keyboard as Kb
        from avion.interact.primitives import Mouse as Ms

        a = Arken(base_url="http://x")
        a._backend = FakeBackend()
        from avion.core.element import ElementFinder
        from avion.core.navigator import Navigator

        a._navigator = Navigator(a._backend, "http://x")  # type: ignore[arg-type]
        a._finder = ElementFinder(a._backend)  # type: ignore[arg-type]
        a._mouse = Ms(a._backend)  # type: ignore[arg-type]
        a._keyboard = Kb(a._backend)  # type: ignore[arg-type]
        return a

    def test_goto_returns_true_and_sets_url(self):
        a = self._session()
        assert run(a.goto("/chat")) is True
        assert a.current_url == "http://x/chat"

    def test_click_text_clicks_found_element(self):
        a = self._session()
        run(a.click_text("Start"))
        assert a._backend.clicked == [ElementLocator.text("Start").describe()]

    def test_fill_writes_value(self):
        a = self._session()
        run(a.fill(ElementLocator.css("input"), "hello"))
        assert a._backend.filled == {ElementLocator.css("input").describe(): "hello"}

    def test_find_missing_raises(self):
        from avion import ElementNotFoundError

        a = self._session()
        try:
            run(a.find(ElementLocator.text("Nope"), timeout=0.1))
        except ElementNotFoundError:
            return
        raise AssertionError("should have raised ElementNotFoundError")

    def test_screenshot_returns_bytes(self):
        a = self._session()
        assert run(a.screenshot()) == b"fake-png"

    def test_select_records_value(self):
        a = self._session()
        run(a.select(ElementLocator.css("input"), "opt1"))
        assert a._backend.selected == {"css=input": "opt1"}

    def test_press_records_key(self):
        from avion.events import EventType

        a = self._session()
        run(a.press("Enter"))
        assert a._backend.pressed == ["Enter"]
        assert a.recorder.of_type(EventType.KEY_PRESS)

    def test_wait_for_text(self):
        a = self._session()
        a._backend.page_text = "welcome home"
        assert run(a.wait_for_text("home", timeout=1.0)) is True
        assert run(a.wait_for_text("missing", timeout=0.1)) is False

    def test_wait_settled(self):
        a = self._session()
        assert run(a.wait_settled(timeout=1.0)) is True


class TestRunTask:
    def test_run_task_records_and_reports(self):
        a = TestSession()._session()

        async def go(ctx):
            await ctx["arken"].goto("/chat")

        task = Task("open-chat", steps=[TaskStep("goto", go)])
        result = run(a.run_task(task))
        assert result.status == TaskStatus.PASSED
        assert a.report().count("[PASS]") == 1
        types = [e.type.value for e in a.recorder.events]
        assert "task_started" in types and "task_passed" in types

    def test_failed_task_screenshots(self):
        a = TestSession()._session()

        async def boom(ctx):
            raise RuntimeError("nope")

        result = run(a.run_task(Task("bad", steps=[TaskStep("x", boom)])))
        assert result.status == TaskStatus.FAILED
        assert any(e.type.value == "screenshot" for e in a.recorder.events)


class TestCoordinates:
    def test_mouse_click_records_coords(self):
        b = FakeBackend()
        run(Mouse(b).click(Coordinate(10, 20)))
        assert b.mouse_clicks == [(10, 20)]

    def test_keyboard_type_records_text(self):
        b = FakeBackend()
        run(Keyboard(b).type_text("hi"))
        assert b.typed == ["hi"]

    def test_chain_replays_click_then_type(self):
        b = FakeBackend()
        chain = InteractionChain()
        chain.add_click(Coordinate(5, 5)).add_type("go").add_wait(1)
        results = run(chain.replay(Mouse(b), Keyboard(b)))
        assert all(r["success"] for r in results)
        assert b.mouse_clicks == [(5, 5)]
        assert b.typed == ["go"]
