"""CDP backend — Chrome DevTools Protocol over websocket, no driver.

Requires: pip install websockets. Connects to a Chrome/Chromium instance
started with --remote-debugging-port. All third-party imports are lazy.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from avion.core.element import Element, ElementLocator, SelectorStrategy


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


def _js_iife(body: str) -> str:
    """Wrap a statement body in an arrow IIFE.

    Built outside f-string literals so JS braces never need Python escaping —
    ``{{``/``}}`` are f-string escapes and silently corrupt hand-written JS.
    """
    return "(() => { " + body + " })()"


class CdpBackend:
    """Direct CDP automation against a debuggable Chrome target."""

    name = "cdp"

    def __init__(self, cdp_url: str = "ws://localhost:9222", target_url: str = "about:blank"):
        self._cdp_url = cdp_url
        self._target_url = target_url
        self._ws = None
        self._msg_id = 0
        self._console: list[dict[str, Any]] = []
        self._network: list[dict[str, Any]] = []
        self._inflight: dict[str, dict[str, Any]] = {}
        self._pending: dict[int, asyncio.Future] = {}
        self._reader: asyncio.Task | None = None

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
        self._reader = asyncio.create_task(self._read_loop())
        await self._send("Page.enable")
        await self._send("Runtime.enable")
        await self._send("Log.enable")
        await self._send("Network.enable")
        if self._target_url != "about:blank":
            await self.navigate(self._target_url)

    async def stop(self) -> None:
        task, self._reader = self._reader, None
        ws, self._ws = self._ws, None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    async def _read_loop(self) -> None:
        """Drain the socket: route events, resolve in-flight commands."""
        error: BaseException | None = None
        try:
            assert self._ws is not None
            async for raw in self._ws:
                msg = json.loads(raw)
                if "method" in msg:
                    self._handle_event(msg)
                    continue
                fut = self._pending.pop(msg.get("id"), None)
                if fut is not None and not fut.done():
                    fut.set_result(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # socket died — fail every waiter
            error = e
        if error is None:
            error = RuntimeError("CDP connection closed")
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(error)
        self._pending.clear()

    async def _send(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if self._ws is None or self._reader is None:
            raise RuntimeError("CDP backend is not started")
        self._msg_id += 1
        msg_id = self._msg_id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[msg_id] = fut
        try:
            await self._ws.send(
                json.dumps({"id": msg_id, "method": method, "params": params or {}})
            )
            raw = await fut
        finally:
            self._pending.pop(msg_id, None)
        if "error" in raw:
            raise RuntimeError(f"CDP error: {raw['error']}")
        return raw.get("result", {})

    def _handle_event(self, raw: dict[str, Any]) -> None:
        """Route async CDP events into the console/network logs."""
        method = raw.get("method", "")
        params = raw.get("params") or {}
        if method == "Runtime.consoleAPICalled":
            text = " ".join(
                str(a.get("value", a.get("description", "")))
                for a in params.get("args", [])
            )
            self._push_console({"level": str(params.get("type", "log")), "text": text})
        elif method == "Runtime.exceptionThrown":
            detail = (params.get("exceptionDetails") or {})
            desc = (detail.get("exception") or {}).get("description") or detail.get("text", "")
            self._push_console({"level": "error", "text": str(desc)})
        elif method == "Log.entryAdded":
            entry = params.get("entry") or {}
            # console API calls already arrive via Runtime.consoleAPICalled
            if entry.get("source") == "javascript-api":
                return
            self._push_console(
                {
                    "level": str(entry.get("level", "info")),
                    "text": str(entry.get("text", "")),
                }
            )
        elif method == "Network.requestWillBeSent":
            req = params.get("request") or {}
            self._inflight[str(params.get("requestId", ""))] = {
                "url": req.get("url", ""),
                "method": req.get("method", ""),
            }
        elif method == "Network.responseReceived":
            rec = self._inflight.pop(str(params.get("requestId", "")), {})
            response = params.get("response") or {}
            self._network.append(
                {
                    "url": response.get("url", rec.get("url", "")),
                    "method": rec.get("method", ""),
                    "status": response.get("status", 0),
                    "mime": response.get("mimeType", ""),
                }
            )
        elif method == "Network.loadingFailed":
            rec = self._inflight.pop(str(params.get("requestId", "")), {})
            self._network.append(
                {
                    "url": rec.get("url", ""),
                    "method": rec.get("method", ""),
                    "status": 0,
                    "error": params.get("errorText", "failed"),
                }
            )

    def _push_console(self, entry: dict[str, Any]) -> None:
        if self._console and self._console[-1] == entry:
            return
        self._console.append(entry)

    def console_entries(self, errors_only: bool = False) -> list[dict[str, Any]]:
        """Captured console output since start() (or last clear_events())."""
        if errors_only:
            return [e for e in self._console if e["level"] in ("error", "assert")]
        return list(self._console)

    def network_entries(self) -> list[dict[str, Any]]:
        """Network requests observed since start() (or last clear_events())."""
        return list(self._network)

    def clear_events(self) -> None:
        self._console.clear()
        self._network.clear()
        self._inflight.clear()

    async def navigate(self, url: str) -> None:
        await self._send("Page.navigate", {"url": url})

    async def evaluate(self, expression: str) -> Any:
        res = await self._send(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        if "exceptionDetails" in res:
            detail = res["exceptionDetails"] or {}
            desc = (detail.get("exception") or {}).get("description") or detail.get(
                "text", "evaluation failed"
            )
            # Chrome does not emit Runtime.exceptionThrown for evaluate-time
            # failures — record it here so `console --errors` sees everything.
            self._push_console({"level": "error", "text": str(desc)})
            raise RuntimeError(f"CDP evaluate error: {desc}")
        return (res.get("result") or {}).get("value")

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

    async def click(self, element: Element, *, force: bool = False) -> None:
        """JS-dispatched click already bypasses actionability — force is a
        no-op here (it documents intent for callers porting from Playwright)."""
        await self.evaluate(f"({element.raw['js']})?.click()")

    async def fill(self, element: Element, value: str, *, force: bool = False) -> None:
        """JS value write already bypasses actionability — force is a no-op."""
        js = element.raw["js"]
        await self.evaluate(
            _js_iife(
                "const el = (" + js + "); if (el) { el.focus(); "
                "el.value = " + repr(value) + "; "
                "el.dispatchEvent(new Event('input', {bubbles: true})); }"
            )
        )

    async def select_option(self, element: Element, value: str) -> None:
        js = element.raw["js"]
        await self.evaluate(
            _js_iife(
                "const el = (" + js + "); if (el) { el.value = " + repr(value) + "; "
                "el.dispatchEvent(new Event('change', {bubbles: true})); }"
            )
        )

    async def check(self, element: Element) -> None:
        await self.evaluate(
            _js_iife(
                "const el = (" + element.raw["js"] + "); "
                "if (el && !el.checked) el.click();"
            )
        )

    async def uncheck(self, element: Element) -> None:
        await self.evaluate(
            _js_iife(
                "const el = (" + element.raw["js"] + "); "
                "if (el && el.checked) el.click();"
            )
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

    async def wait_for_text(self, text: str, timeout: float = 10.0) -> bool:
        """Poll until the page body contains ``text``. True when found."""
        import asyncio

        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            body = await self.evaluate(
                "(document.body && document.body.innerText) || ''"
            )
            if text in str(body or ""):
                return True
            if asyncio.get_event_loop().time() >= deadline:
                return False
            await asyncio.sleep(0.2)

    async def get_accessibility_tree(self) -> dict[str, Any]:
        await self._send("Accessibility.enable")
        res = await self._send("Accessibility.getFullAXTree")
        return {"nodes": res.get("nodes", [])}
