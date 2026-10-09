"""Appium backend — iOS/Android automation via WebDriver protocol.

Requires: pip install Appium-Python-Client + a running Appium server.
All third-party imports are lazy.
"""

from __future__ import annotations

from typing import Any

from avion.core.element import Element, ElementLocator, SelectorStrategy


def _appium():
    try:
        from appium import webdriver
        from appium.options.android import UiAutomator2Options
        from appium.options.ios import XCUITestOptions

        return webdriver, UiAutomator2Options, XCUITestOptions
    except ImportError as e:
        raise ImportError(
            "Appium-Python-Client is not installed. "
            "Install it with: pip install Appium-Python-Client"
        ) from e


class AppiumBackend:
    """Mobile automation. platform: android | ios."""

    name = "appium"

    def __init__(
        self,
        platform: str = "android",
        server_url: str = "http://localhost:4723",
        capabilities: dict[str, Any] | None = None,
    ):
        self._platform = platform
        self._server_url = server_url
        self._capabilities = dict(capabilities or {})
        self._driver = None

    async def start(self) -> None:
        webdriver, UiAutomator2Options, XCUITestOptions = _appium()
        import asyncio

        if self._platform == "ios":
            options = XCUITestOptions().load_capabilities(self._capabilities)
        else:
            options = UiAutomator2Options().load_capabilities(self._capabilities)
        self._driver = await asyncio.to_thread(webdriver.Remote, self._server_url, options=options)

    async def stop(self) -> None:
        if self._driver is not None:
            import asyncio

            await asyncio.to_thread(self._driver.quit)
            self._driver = None

    def _strategy(self, locator: ElementLocator) -> tuple[str, str]:
        for selector in locator.selectors:
            s = selector.strategy
            if s == SelectorStrategy.XPATH:
                return ("xpath", selector.value)
            if s == SelectorStrategy.CSS:
                return ("css selector", selector.value)
            if s in (SelectorStrategy.TEST_ID, SelectorStrategy.ARIA):
                return ("accessibility id", selector.value)
            if s == SelectorStrategy.TEXT:
                return ("xpath", f"//*[@text='{selector.value}']")
        first = locator.selectors[0]
        return ("xpath", first.value)

    async def navigate(self, url: str) -> None:
        import asyncio

        await asyncio.to_thread(self._driver.get, url)

    async def find_element(self, locator: ElementLocator) -> Element | None:
        elements = await self.find_elements(locator)
        return elements[0] if elements else None

    async def find_elements(self, locator: ElementLocator) -> list[Element]:
        import asyncio

        by, value = self._strategy(locator)
        try:
            raw = await asyncio.to_thread(self._driver.find_elements, by, value)
        except Exception:
            return []
        return [Element(raw=el, locator=locator, backend=self.name) for el in raw]

    async def click(self, element: Element, *, force: bool = False) -> None:
        """force is accepted for Backend parity but is a no-op on mobile:
        the Appium driver has no separate actionability gate to bypass."""
        import asyncio

        await asyncio.to_thread(element.raw.click)

    async def fill(self, element: Element, value: str, *, force: bool = False) -> None:
        """force accepted for Backend parity; no-op (see click)."""
        import asyncio

        await asyncio.to_thread(element.raw.clear)
        await asyncio.to_thread(element.raw.send_keys, value)

    async def select_option(self, element: Element, value: str) -> None:
        await self.click(element)

    async def check(self, element: Element) -> None:
        await self.click(element)

    async def uncheck(self, element: Element) -> None:
        await self.click(element)

    async def hover(self, element: Element) -> None:
        await self.click(element)

    async def screenshot(self, path: str | None = None) -> bytes:
        import asyncio

        return await asyncio.to_thread(self._driver.get_screenshot_as_png)

    async def get_url(self) -> str:
        import asyncio

        return await asyncio.to_thread(lambda: self._driver.current_url)

    async def get_title(self) -> str:
        return ""

    async def wait_for(self, locator: ElementLocator, timeout: float = 10.0) -> Element:
        import asyncio

        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            found = await self.find_elements(locator)
            if found:
                return found[0]
            await asyncio.sleep(0.5)
        raise TimeoutError(f"Element not found: {locator.describe()}")

    async def wait_for_timeout(self, ms: float) -> None:
        import asyncio

        await asyncio.sleep(ms / 1000)

    async def evaluate(self, expression: str) -> Any:
        import asyncio

        return await asyncio.to_thread(self._driver.execute_script, expression)

    async def get_accessibility_tree(self) -> dict[str, Any]:
        import asyncio

        src = await asyncio.to_thread(lambda: self._driver.page_source)
        return {"page_source": src}
