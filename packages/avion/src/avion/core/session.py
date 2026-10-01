"""Arken session — minimal autoclicker entry point.

Goto a page, find elements, click them, fill text. Nothing else.

Usage::

    async with Arken(base_url="http://localhost:3000") as a:
        await a.goto("/training")
        el = await a.find(ElementLocator.text("Start Training"))
        await a.click(el)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from avion.core.element import Backend, Element, ElementFinder, ElementLocator
from avion.core.navigator import Navigator
from avion.core.reporter import Reporter
from avion.core.task import Task, TaskResult, TaskStatus
from avion.events.models import EventType
from avion.events.recorder import EventRecorder
from avion.interact.primitives import Keyboard, Mouse
from avion.logging.structured import StructuredLogger
from avion.rules.engine import Rule, RuleEngine
from avion.waits.smart import SmartWaiter, WaitConfig


@dataclass
class ArkenConfig:
    """Configuration for an Arken session."""

    base_url: str = "http://localhost:3000"
    backend: str = "playwright"
    headless: bool = True
    screenshot_on_error: bool = True


class Arken:
    """Minimal autoclicker session: goto, find, click, fill."""

    def __init__(self, **config_kwargs: Any):
        self.config = ArkenConfig(**config_kwargs)
        self._backend: Backend | None = None
        self._navigator: Navigator | None = None
        self._finder: ElementFinder | None = None
        self._mouse: Mouse | None = None
        self._keyboard: Keyboard | None = None
        self._recorder = EventRecorder(session_name="avion_session")
        self._logger = StructuredLogger("avion")
        self._reporter = Reporter("Arken Report")
        self._rule_engine = RuleEngine()

    async def __aenter__(self) -> Arken:
        await self.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.stop()

    async def start(self, backend: Backend | None = None) -> None:
        """Initialize the backend and start the session.

        Args:
            backend: optional pre-built backend (fakes in tests).
                     Defaults to a headless PlaywrightBackend.
        """
        if backend is None:
            if self.config.backend == "playwright":
                from avion.backends.playwright import PlaywrightBackend

                backend = PlaywrightBackend(headless=self.config.headless)
            else:
                raise ValueError(f"Unknown backend: {self.config.backend}")
        self._backend = backend
        await self._backend.start()
        self._navigator = Navigator(self._backend, self.config.base_url)
        self._finder = ElementFinder(self._backend)
        self._mouse = Mouse(self._backend)  # type: ignore[arg-type]
        self._keyboard = Keyboard(self._backend)  # type: ignore[arg-type]
        self._recorder.record(EventType.SESSION_STARTED, name="session_start")
        self._logger.info("Arken session started (backend=%s)" % self._backend.name)

    async def stop(self) -> None:
        """Shut down the backend."""
        self._recorder.record(EventType.SESSION_ENDED, name="session_end")
        self._logger.info("Arken session ended")
        if self._backend:
            await self._backend.stop()
            self._backend = None

    @property
    def mouse(self) -> Mouse:
        return self._mouse

    @property
    def keyboard(self) -> Keyboard:
        return self._keyboard

    @property
    def recorder(self) -> EventRecorder:
        return self._recorder

    @property
    def logger(self) -> StructuredLogger:
        return self._logger

    def set_context(self, **kwargs: str) -> None:
        """Attach logging context (e.g. run id)."""
        self._logger.set_context(**kwargs)

    @property
    def current_url(self) -> str:
        return self._navigator.current_url if self._navigator else ""

    async def goto(self, path: str) -> bool:
        """Navigate to a path. Returns True on success."""
        start = time.perf_counter()
        entry = await self._navigator.goto(path)
        self._recorder.record(
            EventType.NAVIGATE,
            name=path,
            success=entry.success,
            error=entry.error,
            duration_ms=(time.perf_counter() - start) * 1000,
        )
        self._logger.navigation(path, entry.success)
        return entry.success

    async def go_back(self) -> None:
        await self._navigator.go_back()

    async def reload(self) -> None:
        await self._navigator.reload()

    async def find(self, locator: ElementLocator, timeout: float | None = None) -> Element:
        """Find an element. Raises ElementNotFoundError if missing."""
        el = await self._finder.find(locator, timeout=timeout)
        self._recorder.record(
            EventType.ELEMENT_FOUND,
            name=locator.describe(),
            data={"text": el.text, "tag": el.tag},
        )
        self._logger.element("found", locator.describe())
        return el

    async def find_all(self, locator: ElementLocator) -> list[Element]:
        return await self._finder.find_all(locator)

    async def find_optional(self, locator: ElementLocator, timeout: float = 3.0) -> Element | None:
        el = await self._finder.find_optional(locator, timeout=timeout)
        self._recorder.record(
            EventType.ELEMENT_FOUND if el else EventType.ELEMENT_NOT_FOUND,
            name=locator.describe(),
            success=el is not None,
        )
        return el

    async def click(self, element: Element) -> None:
        """Click an element."""
        start = time.perf_counter()
        await self._backend.click(element)
        self._recorder.record(
            EventType.CLICK,
            name=element.locator.describe(),
            data={"text": element.text, "tag": element.tag},
            duration_ms=(time.perf_counter() - start) * 1000,
        )
        self._logger.element("clicked", element.locator.describe())

    async def click_text(self, text: str) -> None:
        """Find element by text and click it."""
        el = await self.find(ElementLocator.text(text))
        await self.click(el)

    async def fill(self, locator: ElementLocator, value: str) -> None:
        """Find a form field and fill it with text."""
        el = await self.find(locator)
        await self._backend.fill(el, value)
        self._recorder.record(EventType.FILL, name=locator.describe(), data={"value": value})

    async def select(self, locator: ElementLocator, value: str) -> None:
        """Select a value from a dropdown."""
        el = await self.find(locator)
        await self._backend.select_option(el, value)
        self._recorder.record(EventType.SELECT, name=locator.describe(), data={"value": value})

    async def press(self, key: str) -> None:
        """Press a key (e.g. "Enter", "Control+a")."""
        await self._keyboard.press(key)
        self._recorder.record(EventType.KEY_PRESS, name=key)
        self._logger.element("pressed", key)

    async def wait_for_text(self, text: str, timeout: float = 10.0) -> bool:
        """Wait until text appears on the page."""
        result = await SmartWaiter(self._backend).wait_for_text(text, timeout)
        return result.success

    async def wait_settled(self, timeout: float = 10.0) -> bool:
        """Wait until spinners/overlays settle."""
        result = await SmartWaiter(self._backend).wait(WaitConfig(timeout=timeout))
        return result.success

    async def screenshot(self) -> bytes:
        """Take a screenshot of the current page."""
        data = await self._backend.screenshot()
        self._recorder.record(EventType.SCREENSHOT, name="screenshot", screenshot=data)
        return data

    @property
    def reporter(self) -> Reporter:
        return self._reporter

    def report(self, format: str = "terminal") -> str:
        return self._reporter.report(format=format)

    @property
    def rule_engine(self) -> RuleEngine:
        return self._rule_engine

    def add_rule(self, rule: Rule) -> None:
        self._rule_engine.add_rule(rule)

    async def evaluate_rules(self, context: dict[str, Any] | None = None):
        """Evaluate rules against current page state."""
        ctx = context or {}
        ctx["current_url"] = self.current_url
        return await self._rule_engine.evaluate(ctx)

    async def run_task(self, task: Task) -> TaskResult:
        """Execute a task, record results, screenshot on error."""
        self._logger.task_start(task.name)
        self._recorder.record(EventType.TASK_STARTED, name=task.name)

        context = {
            "avion": self,
            "navigator": self._navigator,
            "finder": self._finder,
            "backend": self._backend,
            "recorder": self._recorder,
            "logger": self._logger,
            "current_url": self.current_url,
        }
        result = await task.execute(context)

        self._reporter.add(result)
        success = result.status == TaskStatus.PASSED
        self._recorder.record(
            EventType.TASK_PASSED if success else EventType.TASK_FAILED,
            name=task.name,
            success=success,
            error=result.error,
            duration_ms=result.duration_ms,
        )
        self._logger.task_end(task.name, success)

        if not success and self.config.screenshot_on_error and self._backend:
            await self.screenshot()
        return result

    def save_results(self, output_dir: str = "avion_output") -> None:
        """Save recorded events and logs to the output directory."""
        import os

        os.makedirs(output_dir, exist_ok=True)
        self._recorder.save(os.path.join(output_dir, "events.json"))
        self._logger.save(os.path.join(output_dir, "logs.json"))
        self._reporter.save(os.path.join(output_dir, "report.json"), format="json")
        self._reporter.save(os.path.join(output_dir, "report.md"), format="markdown")
