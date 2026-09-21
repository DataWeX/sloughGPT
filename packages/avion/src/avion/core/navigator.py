"""Page and flow navigation.

Provides a Navigator that manages page state, URL history, and
navigation actions with automatic waiting and verification.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from arken.core.element import Backend, ElementFinder, ElementLocator


@dataclass
class NavigationEntry:
    """Record of a single navigation event."""

    url: str
    title: str = ""
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    success: bool = True
    error: str = ""


class Navigator:
    """Manages page navigation with history tracking and verification.

    Usage::

        nav = Navigator(backend, base_url="http://localhost:3000")
        await nav.goto("/training")
        await nav.click_link("Start Training")
        assert nav.current_url.endswith("/training/job/123")
    """

    def __init__(self, backend: Backend, base_url: str = ""):
        self._backend = backend
        self._base_url = base_url.rstrip("/")
        self._finder = ElementFinder(backend)
        self._history: list[NavigationEntry] = []
        self._current_url = ""

    @property
    def current_url(self) -> str:
        return self._current_url

    @property
    def history(self) -> list[NavigationEntry]:
        return list(self._history)

    @property
    def finder(self) -> ElementFinder:
        return self._finder

    async def goto(self, path: str, verify: str | None = None) -> NavigationEntry:
        """Navigate to a URL path relative to base_url.

        Args:
            path: URL path (e.g. "/training" or full URL).
            verify: Optional string that must appear in the page content
                    after navigation to confirm success.
        """
        url = path
        if not (
            path.startswith("http")
            or path.startswith("data:")
            or path.startswith("about:")
            or path.startswith("file:")
        ):
            url = f"{self._base_url}{path}"
        start = time.perf_counter()

        entry = NavigationEntry(url=url)
        try:
            await self._backend.navigate(url)
            self._current_url = url
            entry.title = await self._backend.get_title()

            if verify:
                await self._wait_for_text(verify, timeout=10.0)

            entry.duration_ms = (time.perf_counter() - start) * 1000
            entry.success = True
        except Exception as e:
            entry.duration_ms = (time.perf_counter() - start) * 1000
            entry.success = False
            entry.error = str(e)

        self._history.append(entry)
        return entry

    async def go_back(self) -> NavigationEntry:
        """Navigate back in history."""
        await self._backend.evaluate("window.history.back()")
        await self._backend.wait_for_timeout(500)
        self._current_url = await self._backend.get_url()
        entry = NavigationEntry(url=self._current_url, title=await self._backend.get_title())
        self._history.append(entry)
        return entry

    async def reload(self) -> NavigationEntry:
        """Reload current page."""
        url = self._current_url
        start = time.perf_counter()
        entry = NavigationEntry(url=url)
        try:
            await self._backend.navigate(url)
            entry.title = await self._backend.get_title()
            entry.duration_ms = (time.perf_counter() - start) * 1000
            entry.success = True
        except Exception as e:
            entry.duration_ms = (time.perf_counter() - start) * 1000
            entry.success = False
            entry.error = str(e)
        self._history.append(entry)
        return entry

    async def click_link(self, text: str) -> None:
        """Click a link by its text content."""
        locator = ElementLocator.text(text)
        el = await self._finder.find(locator)
        await self._backend.click(el)
        await self._backend.wait_for_timeout(500)
        self._current_url = await self._backend.get_url()

    async def click_element(self, locator: ElementLocator) -> None:
        """Click an element identified by locator."""
        el = await self._finder.find(locator)
        await self._backend.click(el)

    async def fill_form_field(self, locator: ElementLocator, value: str) -> None:
        """Fill a form field."""
        el = await self._finder.find(locator)
        await self._backend.fill(el, value)

    async def select_dropdown(self, locator: ElementLocator, value: str) -> None:
        """Select an option from a dropdown."""
        el = await self._finder.find(locator)
        await self._backend.select_option(el, value)

    async def wait_for_page(self, text: str, timeout: float = 10.0) -> None:
        """Wait for text to appear on the page."""
        await self._wait_for_text(text, timeout=timeout)

    async def wait_for_url(self, pattern: str, timeout: float = 10.0) -> None:
        """Wait for URL to match a pattern (substring or regex)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            url = await self._backend.get_url()
            if pattern in url or (re.search(pattern, url) if "/" in pattern else False):
                self._current_url = url
                return
            await self._backend.wait_for_timeout(200)
        raise TimeoutError(f"URL did not match {pattern!r} within {timeout}s")

    async def screenshot(self, name: str = "page") -> bytes:
        """Take a screenshot of the current page."""
        return await self._backend.screenshot()

    async def get_page_snapshot(self) -> dict[str, Any]:
        """Get accessibility tree snapshot of the current page."""
        return await self._backend.get_accessibility_tree()

    async def _wait_for_text(self, text: str, timeout: float = 10.0) -> None:
        """Wait for text to appear anywhere on the page."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                content = await self._backend.evaluate("document.body.innerText")
                if text in content:
                    return
            except Exception:
                pass
            await self._backend.wait_for_timeout(200)
        raise TimeoutError(f"Text {text!r} not found within {timeout}s")
