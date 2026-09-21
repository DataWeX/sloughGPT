"""Selenium/WebDriver backend — Chrome, Firefox, Edge.

Requires: pip install selenium (+ matching browser driver).
All selenium imports are lazy so the package installs without it.
"""

from __future__ import annotations

from typing import Any

from arken.core.element import Element, ElementLocator, SelectorStrategy


def _selenium():
    try:
        import selenium  # noqa: F401
        from selenium import webdriver
        from selenium.webdriver.common.by import By

        return webdriver, By
    except ImportError as e:
        raise ImportError("selenium is not installed. Install it with: pip install selenium") from e


def to_by(selector) -> tuple[Any, str]:
    """Map an Arken selector to a (By, value) pair. Pure, no driver needed."""
    _, By = _selenium()
    s = selector
    if s.strategy == SelectorStrategy.CSS:
        return (By.CSS_SELECTOR, s.value)
    if s.strategy == SelectorStrategy.XPATH:
        return (By.XPATH, s.value)
    if s.strategy == SelectorStrategy.TEST_ID:
        return (By.CSS_SELECTOR, f"[data-testid='{s.value}']")
    if s.strategy == SelectorStrategy.ROLE:
        return (By.CSS_SELECTOR, f"[role='{s.value}']")
    if s.strategy == SelectorStrategy.LABEL:
        return (By.XPATH, f"//*[text()='{s.value}']")
    if s.strategy == SelectorStrategy.PLACEHOLDER:
        return (By.CSS_SELECTOR, f"[placeholder='{s.value}']")
    if s.strategy == SelectorStrategy.ARIA:
        return (By.CSS_SELECTOR, f"[aria-label='{s.value}']")
    if s.strategy == SelectorStrategy.TEXT:
        if s.exact:
            return (By.XPATH, f"//*[text()='{s.value}']")
        return (By.XPATH, f"//*[contains(text(), '{s.value}')]")
    return (By.CSS_SELECTOR, s.value)


class SeleniumBackend:
    """WebDriver backend. browser_type: chromium | firefox | edge."""

    name = "selenium"

    def __init__(self, browser_type: str = "chromium", headless: bool = True):
        self._browser_type = browser_type
        self._headless = headless
        self._driver = None

    @property
    def driver(self):
        return self._driver

    async def start(self) -> None:
        webdriver, _ = _selenium()
        if self._browser_type == "firefox":
            options = webdriver.FirefoxOptions()
            if self._headless:
                options.add_argument("--headless")
            self._driver = webdriver.Firefox(options=options)
        elif self._browser_type == "edge":
            options = webdriver.EdgeOptions()
            if self._headless:
                options.add_argument("--headless")
            self._driver = webdriver.Edge(options=options)
        else:
            options = webdriver.ChromeOptions()
            if self._headless:
                options.add_argument("--headless=new")
            self._driver = webdriver.Chrome(options=options)

    async def stop(self) -> None:
        if self._driver is not None:
            self._driver.quit()
            self._driver = None

    async def navigate(self, url: str) -> None:
        import asyncio

        await asyncio.to_thread(self._driver.get, url)

    async def find_element(self, locator: ElementLocator) -> Element | None:
        elements = await self.find_elements(locator)
        return elements[0] if elements else None

    async def find_elements(self, locator: ElementLocator) -> list[Element]:
        import asyncio

        found = []
        for selector in locator.selectors:
            by, value = await asyncio.to_thread(to_by, selector)
            try:
                raw = await asyncio.to_thread(self._driver.find_elements, by, value)
            except Exception:
                continue
            for el in raw:
                found.append(await self._wrap(el, selector))
            if found:
                break
        return found

    async def _wrap(self, raw, selector) -> Element:
        import asyncio

        el = Element(raw=raw, locator=ElementLocator(selectors=(selector,)), backend=self.name)
        try:
            el._cache["tag"] = await asyncio.to_thread(lambda: raw.tag_name)
            el._cache["text"] = await asyncio.to_thread(lambda: raw.text)
            el._cache["is_visible"] = await asyncio.to_thread(lambda: raw.is_displayed())
            el._cache["is_enabled"] = await asyncio.to_thread(lambda: raw.is_enabled())
        except Exception:
            pass
        return el

    async def click(self, element: Element) -> None:
        import asyncio

        await asyncio.to_thread(element.raw.click)

    async def fill(self, element: Element, value: str) -> None:
        import asyncio

        await asyncio.to_thread(element.raw.clear)
        await asyncio.to_thread(element.raw.send_keys, value)

    async def select_option(self, element: Element, value: str) -> None:
        import asyncio

        from selenium.webdriver.support.ui import Select

        await asyncio.to_thread(lambda: Select(element.raw).select_by_visible_text(value))

    async def check(self, element: Element) -> None:
        import asyncio

        selected = await asyncio.to_thread(lambda: element.raw.is_selected())
        if not selected:
            await self.click(element)

    async def uncheck(self, element: Element) -> None:
        import asyncio

        selected = await asyncio.to_thread(lambda: element.raw.is_selected())
        if selected:
            await self.click(element)

    async def hover(self, element: Element) -> None:
        import asyncio

        from selenium.webdriver.common.action_chains import ActionChains

        await asyncio.to_thread(
            lambda: ActionChains(self._driver).move_to_element(element.raw).perform()
        )

    async def screenshot(self, path: str | None = None) -> bytes:
        import asyncio

        return await asyncio.to_thread(self._driver.get_screenshot_as_png)

    async def get_url(self) -> str:
        return self._driver.current_url

    async def get_title(self) -> str:
        return self._driver.title

    async def wait_for(self, locator: ElementLocator, timeout: float = 10.0) -> Element:
        import asyncio

        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        def _wait():
            for selector in locator.selectors:
                by, value = to_by(selector)
                try:
                    raw = WebDriverWait(self._driver, timeout).until(
                        EC.presence_of_element_located((by, value))
                    )
                    return raw, selector
                except Exception:
                    continue
            raise TimeoutError(f"Element not found: {locator.describe()}")

        raw, selector = await asyncio.to_thread(_wait)
        return await self._wrap(raw, selector)

    async def wait_for_timeout(self, ms: float) -> None:
        import asyncio

        await asyncio.sleep(ms / 1000)

    async def evaluate(self, expression: str) -> Any:
        import asyncio

        return await asyncio.to_thread(self._driver.execute_script, f"return {expression}")

    async def get_accessibility_tree(self) -> dict[str, Any]:
        return {}
