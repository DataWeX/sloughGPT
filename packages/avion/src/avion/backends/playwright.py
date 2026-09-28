"""Playwright backend adapter.

Implements the Backend protocol using Playwright's async API.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from avion.core.element import (
    Element,
    ElementLocator,
    SelectorStrategy,
)


class PlaywrightBackend:
    """Playwright-based browser backend.

    Usage::

        backend = PlaywrightBackend()
        await backend.start()
        await backend.navigate("http://localhost:3000")
        el = await backend.find_element(ElementLocator.css(".my-button"))
        await backend.click(el)
        await backend.stop()
    """

    def __init__(self, headless: bool = True, browser_type: str = "chromium"):
        self._headless = headless
        self._browser_type = browser_type
        self._playwright = None
        self._browser = None
        self._page = None
        self._pages: dict[str, Any] = {}
        self._tab_seq = 0
        self._mocker = None

    @property
    def name(self) -> str:
        return f"playwright-{self._browser_type}"

    @property
    def page(self):
        return self._page

    async def start(self) -> None:
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        browser_cls = getattr(self._playwright, self._browser_type)
        self._browser = await browser_cls.launch(headless=self._headless)
        self._page = await self._browser.new_page()
        self._register_page(self._page)

    async def stop(self) -> None:
        if self._page:
            await self._page.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._page = None
        self._pages = {}

    def _register_page(self, page) -> str:
        self._tab_seq += 1
        tab_id = f"tab-{self._tab_seq}"
        self._pages[tab_id] = page
        return tab_id

    # ── Tabs (TabProvider) ──────────────────────────────────────────────

    async def tab_open(self, url: str) -> str:
        """Open a new tab (optionally navigating), return its id."""
        page = await self._browser.new_page()
        tab_id = self._register_page(page)
        if self._mocker is not None:
            await self._route_page(page)
        if url:
            await page.goto(url, wait_until="domcontentloaded")
        return tab_id

    async def tab_list(self) -> list[dict[str, Any]]:
        out = []
        for tab_id, page in self._pages.items():
            try:
                title = await page.title()
            except Exception:
                title = ""
            out.append({"id": tab_id, "url": page.url, "title": title})
        return out

    async def tab_activate(self, tab_id: str) -> None:
        if tab_id not in self._pages:
            raise KeyError(f"unknown tab: {tab_id}")
        self._page = self._pages[tab_id]
        try:
            await self._page.bring_to_front()
        except Exception:
            pass

    async def tab_close(self, tab_id: str) -> None:
        page = self._pages.pop(tab_id, None)
        if page is None:
            return
        try:
            await page.close()
        except Exception:
            pass
        if self._page not in self._pages.values():
            self._page = next(iter(self._pages.values()), None)

    # ── Network mocking ─────────────────────────────────────────────────

    async def enable_network_mock(self, mocker) -> None:
        """Route every page through a NetworkMocker. None = passthrough."""
        self._mocker = mocker
        for page in self._pages.values():
            await self._route_page(page)

    async def disable_network_mock(self) -> None:
        self._mocker = None
        for page in self._pages.values():
            try:
                await page.unroute("**/*")
            except Exception:
                pass

    async def _route_page(self, page) -> None:
        import json as _json

        async def handler(route) -> None:
            from avion.network.mocker import NetworkRequest

            req = route.request
            resp = await self._mocker.intercept(
                NetworkRequest(method=req.method, url=req.url, headers=dict(req.headers))
            )
            if resp is None:
                await route.continue_()
                return
            if resp.error:
                await route.abort()
                return
            body = resp.body
            headers = dict(resp.headers)
            if isinstance(body, (dict, list)):
                body = _json.dumps(body)
                headers.setdefault("content-type", "application/json")
            elif body is None:
                body = ""
            elif not isinstance(body, (str, bytes)):
                body = str(body)
            await route.fulfill(status=resp.status, body=body, headers=headers)

        await page.route("**/*", handler)

    async def navigate(self, url: str) -> None:
        await self._page.goto(url, wait_until="domcontentloaded")

    async def find_element(self, locator: ElementLocator) -> Element | None:
        for selector in locator.selectors:
            el = await self._query_selector(selector)
            if el is not None:
                return el
        return None

    async def find_elements(self, locator: ElementLocator) -> list[Element]:
        results = []
        for selector in locator.selectors:
            els = await self._query_selector_all(selector)
            results.extend(els)
            if results:
                break
        return results

    async def click(self, element: Element) -> None:
        await element.raw.click()

    async def fill(self, element: Element, value: str) -> None:
        await element.raw.fill(value)

    async def select_option(self, element: Element, value: str) -> None:
        await element.raw.select_option(value)

    async def check(self, element: Element) -> None:
        await element.raw.check()

    async def uncheck(self, element: Element) -> None:
        await element.raw.uncheck()

    async def hover(self, element: Element) -> None:
        await element.raw.hover()

    async def screenshot(self, path: str | None = None) -> bytes:
        return await self._page.screenshot(path=path, full_page=True)

    async def get_url(self) -> str:
        return self._page.url

    async def get_title(self) -> str:
        return await self._page.title()

    async def wait_for(self, locator: ElementLocator, timeout: float = 10.0) -> Element:
        for selector in locator.selectors:
            try:
                el = await self._wait_for_selector(selector, timeout)
                if el is not None:
                    return el
            except Exception:
                continue
        raise TimeoutError(f"Element not found: {locator.describe()}")

    async def wait_for_timeout(self, ms: float) -> None:
        await asyncio.sleep(ms / 1000)

    async def evaluate(self, expression: str) -> Any:
        return await self._page.evaluate(expression)

    async def get_accessibility_tree(self) -> dict[str, Any]:
        return await self._page.accessibility.snapshot()

    # ── Low-level interaction primitives ────────────────────────────────

    async def mouse_move(self, x: float, y: float) -> None:
        await self._page.mouse.move(x, y)

    async def mouse_down(self, button: str = "left") -> None:
        await self._page.mouse.down(button=button)

    async def mouse_up(self, button: str = "left") -> None:
        await self._page.mouse.up(button=button)

    async def mouse_click(self, x: float, y: float, button: str = "left") -> None:
        await self._page.mouse.click(x, y, button=button)

    async def mouse_double_click(self, x: float, y: float) -> None:
        await self._page.mouse.dblclick(x, y)

    async def mouse_scroll(self, x: float, y: float, delta_x: int = 0, delta_y: int = 0) -> None:
        await self._page.mouse.move(x, y)
        await self._page.mouse.wheel(delta_x=delta_x, delta_y=delta_y)

    async def keyboard_down(self, key: str) -> None:
        await self._page.keyboard.down(key)

    async def keyboard_up(self, key: str) -> None:
        await self._page.keyboard.up(key)

    async def keyboard_press(self, key: str, hold_ms: float = 0) -> None:
        if hold_ms > 0:
            await self._page.keyboard.down(key)
            await asyncio.sleep(hold_ms / 1000)
            await self._page.keyboard.up(key)
        else:
            await self._page.keyboard.press(key)

    async def keyboard_type(self, text: str, delay_ms: float = 0) -> None:
        delay = delay_ms / 1000 if delay_ms > 0 else 0
        await self._page.keyboard.type(text, delay=delay)

    async def get_viewport_size(self) -> tuple[int, int]:
        viewport = self._page.viewport_size
        return (viewport["width"], viewport["height"]) if viewport else (1280, 720)

    async def get_screenshot_as_bytes(self) -> bytes:
        return await self._page.screenshot()

    # ── Internal helpers ────────────────────────────────────────────────

    async def _query_selector(self, selector) -> Element | None:
        pw_selector = self._to_playwright_selector(selector)
        if pw_selector is None:
            return None
        raw = await self._page.query_selector(pw_selector)
        if raw is None:
            return None
        return await self._wrap_element(raw, selector)

    async def _query_selector_all(self, selector) -> list[Element]:
        pw_selector = self._to_playwright_selector(selector)
        if pw_selector is None:
            return []
        raw_list = await self._page.query_selector_all(pw_selector)
        return [await self._wrap_element(r, selector) for r in raw_list]

    async def _wait_for_selector(self, selector, timeout: float):
        pw_selector = self._to_playwright_selector(selector)
        if pw_selector is None:
            return None
        loc = self._page.locator(pw_selector).first
        await loc.wait_for(state="visible", timeout=timeout * 1000)
        raw = await loc.element_handle()
        if raw is None:
            return None
        return await self._wrap_element(raw, selector)

    def _to_playwright_selector(self, selector) -> str | None:
        strategy = selector.strategy
        value = selector.value

        if strategy == SelectorStrategy.CSS:
            return value
        if strategy == SelectorStrategy.TEXT:
            if selector.exact:
                return f"text={json.dumps(value, ensure_ascii=False)}"
            return f"text={value}"
        if strategy == SelectorStrategy.TEST_ID:
            return f"[data-testid='{value}']"
        if strategy == SelectorStrategy.LABEL:
            return f"label={value}"
        if strategy == SelectorStrategy.PLACEHOLDER:
            return f"[placeholder='{value}']"
        if strategy == SelectorStrategy.XPATH:
            return f"xpath={value}"
        if strategy == SelectorStrategy.ROLE:
            # Parse "role[name]" format
            if "[" in value:
                role, name = value.split("[", 1)
                name = name.rstrip("]")
                return f"role={role}[name='{name}']"
            return f"role={value}"
        return None

    async def _wrap_element(self, raw, selector) -> Element:
        cache = {}
        try:
            cache["tag"] = await raw.evaluate("el => el.tagName.toLowerCase()")
        except Exception:
            pass
        try:
            cache["text"] = await raw.inner_text()
        except Exception:
            pass
        try:
            cache["is_visible"] = await raw.is_visible()
        except Exception:
            pass
        try:
            cache["is_enabled"] = await raw.is_enabled()
        except Exception:
            pass
        try:
            bbox = await raw.bounding_box()
            cache["bounding_box"] = bbox
        except Exception:
            pass
        try:
            cache["attributes"] = await raw.evaluate(
                "el => Object.fromEntries([...el.attributes].map(a => [a.name, a.value]))"
            )
        except Exception:
            pass
        try:
            cache["value"] = await raw.evaluate("el => el.value || null")
        except Exception:
            pass

        return Element(
            raw=raw,
            locator=ElementLocator(selectors=(selector,)),
            backend=self.name,
            _cache=cache,
        )
