"""CDP backend — Chrome DevTools Protocol over websocket, no driver.

Requires: pip install websockets. Connects to a Chrome/Chromium instance
started with --remote-debugging-port. All third-party imports are lazy.
"""

from __future__ import annotations

import json
from typing import Any

from arken.core.element import Element, ElementLocator, SelectorStrategy


def _websockets():
    try:
        import websockets

        return websockets
    except ImportError as e:
        raise ImportError(
            "websockets is not installed. Install it with: pip install websockets"
        ) from e


def selector_to_js(selector) -> str | None:
    """Map an Arken selector to a document.querySelector expression."""
    s = selector
    if s.strategy == SelectorStrategy.CSS:
        return f"document.querySelector({s.value!r})"
    if s.strategy == SelectorStrategy.TEST_ID:
        return f"document.querySelector('[data-testid={s.value!r}]')"
    if s.strategy == SelectorStrategy.XPATH:
        return (
            "document.evaluate("
            f"{s.value!r}, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, "
            "null).singleNodeValue"
        )
    if s.strategy == SelectorStrategy.TEXT:
        needle = s.value.replace("'", "\\'")
        if s.exact:
            return (
                "[...document.querySelectorAll('body *')]"
                f".find(e => e.textContent?.trim() === '{needle}')"
            )
        return (
            "[...document.querySelectorAll('body *')]"
            f".find(e => e.textContent?.includes('{needle}')"
            " && e.children.length === 0)"
        )
    return None


class CdpBackend:
    """Direct CDP automation against a debuggable Chrome target."""

    name = "cdp"

    def __init__(self, cdp_url: str = "ws://localhost:9222", target_url: str = "about:blank"):
        self._cdp_url = cdp_url
        self._target_url = target_url
        self._ws = None
        self._msg_id = 0

    async def _connect(self):
        websockets = _websockets()
        import urllib.request

        with urllib.request.urlopen(
            self._cdp_url.replace("ws://", "http://").replace("ws:", "http:").rstrip("/") + "/json"
        ) as resp:
            targets = json.loads(resp.read().decode())
        page = next(
            (t for t in targets if t.get("type") == "page"), targets[0] if targets else None
        )
        if page is None:
            raise RuntimeError("no debuggable targets found")
        self._ws_url = page["webSocketDebuggerUrl"]
        self._ws = await websockets.connect(self._ws_url)

    async def start(self) -> None:
        _websockets()
        await self._connect()
        await self._send("Page.enable")
        await self._send("Runtime.enable")
        if self._target_url != "about:blank":
            await self.navigate(self._target_url)

    async def stop(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _send(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self._msg_id += 1
        msg_id = self._msg_id
        await self._ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        while True:
            raw = json.loads(await self._ws.recv())
            if raw.get("id") == msg_id:
                if "error" in raw:
                    raise RuntimeError(f"CDP error: {raw['error']}")
                return raw.get("result", {})

    async def navigate(self, url: str) -> None:
        await self._send("Page.navigate", {"url": url})

    async def evaluate(self, expression: str) -> Any:
        res = await self._send(
            "Runtime.evaluate", {"expression": expression, "returnByValue": True}
        )
        return (res.get("result") or {}).get("result", {}).get("value")

    async def find_element(self, locator: ElementLocator) -> Element | None:
        elements = await self.find_elements(locator)
        return elements[0] if elements else None

    async def find_elements(self, locator: ElementLocator) -> list[Element]:
        found = []
        for selector in locator.selectors:
            js = selector_to_js(selector)
            if js is None:
                continue
            try:
                exists = await self.evaluate(f"!!({js})")
            except Exception:
                continue
            if exists:
                found.append(
                    Element(
                        raw={"js": js},
                        locator=ElementLocator(selectors=(selector,)),
                        backend=self.name,
                    )
                )
                break
        return found

    async def click(self, element: Element) -> None:
        await self.evaluate(f"({element.raw['js']})?.click()")

    async def fill(self, element: Element, value: str) -> None:
        js = element.raw["js"]
        await self.evaluate(
            f"(() => {{ const el = ({js}); if (el) {{ el.focus(); "
            f"el.value = {value!r}; "
            "el.dispatchEvent(new Event('input', {bubbles: true})); } }})()"
        )

    async def select_option(self, element: Element, value: str) -> None:
        js = element.raw["js"]
        await self.evaluate(
            f"(() => {{ const el = ({js}); if (el) {{ el.value = {value!r}; "
            "el.dispatchEvent(new Event('change', {bubbles: true})); } }})()"
        )

    async def check(self, element: Element) -> None:
        await self.evaluate(
            f"(() => {{ const el = ({element.raw['js']}); "
            "if (el && !el.checked) el.click(); }})()"
        )

    async def uncheck(self, element: Element) -> None:
        await self.evaluate(
            f"(() => {{ const el = ({element.raw['js']}); if (el && el.checked) el.click(); }}}})()"
        )

    async def hover(self, element: Element) -> None:
        await self.evaluate(
            f"({element.raw['js']})?.dispatchEvent(new MouseEvent('mouseover', {{bubbles: true}}))"
        )

    async def screenshot(self, path: str | None = None) -> bytes:
        import base64

        res = await self._send("Page.captureScreenshot", {"format": "png"})
        return base64.b64decode(res.get("data", ""))

    async def get_url(self) -> str:
        return str(await self.evaluate("window.location.href") or "")

    async def get_title(self) -> str:
        return str(await self.evaluate("document.title") or "")

    async def wait_for(self, locator: ElementLocator, timeout: float = 10.0) -> Element:
        import asyncio

        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            found = await self.find_elements(locator)
            if found:
                return found[0]
            await asyncio.sleep(0.2)
        raise TimeoutError(f"Element not found: {locator.describe()}")

    async def wait_for_timeout(self, ms: float) -> None:
        import asyncio

        await asyncio.sleep(ms / 1000)

    async def get_accessibility_tree(self) -> dict[str, Any]:
        await self._send("Accessibility.enable")
        res = await self._send("Accessibility.getFullAXTree")
        return {"nodes": res.get("nodes", [])}
