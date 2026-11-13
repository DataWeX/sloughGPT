"""Page controls (PageControls protocol) — live Playwright tests.

Covers the capability additions that let ad-hoc Playwright callers port onto
avion (card e26dc68c): wait_for_function, set_viewport_size, focus,
press_element, force fill, and navigate's wait_until/timeout.

Needs Chromium (playwright install); skipped when playwright is absent.
"""

from __future__ import annotations

import asyncio
import importlib.util

import pytest
from avion import PageControls
from avion.backends.playwright import PlaywrightBackend
from avion.core.element import ElementLocator

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("playwright") is None, reason="playwright not installed"
)

PAGE = (
    "data:text/html,<html><head><title>PC</title></head>"
    "<body><input id='i'><button id='b'>Go</button></body></html>"
)


def run(coro):
    return asyncio.run(coro)


class TestWaitForFunction:
    def test_true_predicate_returns_true(self):
        async def main():
            b = PlaywrightBackend(headless=True)
            await b.start()
            try:
                await b.navigate(PAGE)
                assert await b.wait_for_function("() => true", timeout=5.0) is True
            finally:
                await b.stop()

        run(main())

    def test_unmet_predicate_times_out_false(self):
        async def main():
            b = PlaywrightBackend(headless=True)
            await b.start()
            try:
                await b.navigate(PAGE)
                assert await b.wait_for_function("() => false", timeout=0.5) is False
            finally:
                await b.stop()

        run(main())

    def test_predicates_against_dom(self):
        async def main():
            b = PlaywrightBackend(headless=True)
            await b.start()
            try:
                await b.navigate(PAGE)
                expr = "() => document.body.innerText.includes('Go')"
                assert await b.wait_for_function(expr, timeout=5.0) is True
            finally:
                await b.stop()

        run(main())


class TestViewport:
    def test_set_viewport_size(self):
        async def main():
            b = PlaywrightBackend(headless=True)
            await b.start()
            try:
                await b.set_viewport_size(800, 600)
                assert await b.get_viewport_size() == (800, 600)
                await b.set_viewport_size(375, 667)
                assert await b.get_viewport_size() == (375, 667)
            finally:
                await b.stop()

        run(main())


class TestFocusAndPress:
    def test_focus_moves_active_element(self):
        async def main():
            b = PlaywrightBackend(headless=True)
            await b.start()
            try:
                await b.navigate(PAGE)
                el = await b.wait_for(ElementLocator.css("#i"), timeout=5.0)
                await b.focus(el)
                active = await b.evaluate("() => document.activeElement.id")
                assert active == "i"
            finally:
                await b.stop()

        run(main())

    def test_press_element_types_into_field(self):
        async def main():
            b = PlaywrightBackend(headless=True)
            await b.start()
            try:
                await b.navigate(PAGE)
                el = await b.wait_for(ElementLocator.css("#i"), timeout=5.0)
                await b.press_element(el, "a")
                # Element caches its value at query time — re-read fresh.
                fresh = await b.find_element(ElementLocator.css("#i"))
                assert fresh is not None and fresh.value == "a"
            finally:
                await b.stop()

        run(main())


class TestForceFill:
    def test_force_fill_sets_value(self):
        async def main():
            b = PlaywrightBackend(headless=True)
            await b.start()
            try:
                await b.navigate(PAGE)
                el = await b.wait_for(ElementLocator.css("#i"), timeout=5.0)
                await b.fill(el, "forced", force=True)
                fresh = await b.find_element(ElementLocator.css("#i"))
                assert fresh is not None and fresh.value == "forced"
            finally:
                await b.stop()

        run(main())


class TestNavigateOptions:
    def test_wait_until_load_accepted(self):
        async def main():
            b = PlaywrightBackend(headless=True)
            await b.start()
            try:
                await b.navigate(PAGE, wait_until="load", timeout=15.0)
                assert (await b.get_title()) == "PC"
            finally:
                await b.stop()

        run(main())

    def test_default_still_works(self):
        async def main():
            b = PlaywrightBackend(headless=True)
            await b.start()
            try:
                await b.navigate(PAGE)
                assert "data:text/html" in await b.get_url()
            finally:
                await b.stop()

        run(main())


class TestProtocolShape:
    def test_playwright_satisfies_page_controls(self):
        b = PlaywrightBackend(headless=True)
        assert isinstance(b, PageControls)

    def test_api_backend_does_not(self):
        """Non-browser backends have no page — callers must probe first."""
        from avion.backends.api import ApiBackend

        assert not isinstance(ApiBackend(base_url="http://localhost:8000"), PageControls)
