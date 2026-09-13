"""
Computer-Use Agent — navigates the sloughGPT web UI via Playwright.

Provides a high-level API for browser automation with built-in DevTools
monitoring: captures console logs, network requests, performance metrics,
and error events during navigation.

Usage:
    from domains.agents.computer_use import ComputerUseAgent

    async with ComputerUseAgent() as agent:
        await agent.navigate("/training")
        await agent.click_button("Train")
        await agent.fill_input("epochs", "5")
        await agent.wait_for_text("Training complete")
        report = agent.devtools_report()

Requirements:
    playwright install chromium
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("slo.agents.computer_use")


@dataclass
class DevToolsEntry:
    kind: str  # console, network, performance, error
    timestamp: float
    data: dict[str, Any]


@dataclass
class NavigationResult:
    url: str
    status: int
    body_length: int
    duration_s: float
    errors: list[str] = field(default_factory=list)


@dataclass
class ClickResult:
    element: str
    found: bool
    duration_s: float
    error: str | None = None


@dataclass
class FillResult:
    element: str
    value: str
    success: bool
    error: str | None = None


@dataclass
class DevToolsReport:
    console_messages: list[dict[str, Any]]
    network_requests: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    performance: dict[str, Any]
    total_console: int
    total_requests: int
    total_errors: int
    summary: str


class ComputerUseAgent:
    """Browser automation agent with DevTools monitoring.

    Wraps Playwright to provide a clean API for navigating the sloughGPT
    web UI, interacting with elements, and collecting DevTools telemetry.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        api_url: str = "http://localhost:8000",
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
    ):
        self.base_url = base_url
        self.api_url = api_url
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._devtools_log: list[DevToolsEntry] = []
        self._console_messages: list[dict[str, Any]] = []
        self._network_requests: list[dict[str, Any]] = []
        self._errors: list[dict[str, Any]] = []
        self._performance: dict[str, Any] = {}
        self._step_counter = 0

    async def start(self) -> None:
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height}
        )
        self._page = await self._context.new_page()
        self._setup_listeners()
        logger.info("ComputerUseAgent started (base=%s)", self.base_url)

    async def stop(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()
        logger.info("ComputerUseAgent stopped")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()

    def _setup_listeners(self) -> None:
        page = self._page
        page.on("console", self._on_console)
        page.on("pageerror", self._on_page_error)
        page.on("request", self._on_request)
        page.on("response", self._on_response)

    def _on_console(self, msg) -> None:
        entry = {
            "type": msg.type,
            "text": msg.text,
            "timestamp": time.time(),
        }
        self._console_messages.append(entry)
        self._devtools_log.append(
            DevToolsEntry(kind="console", timestamp=time.time(), data=entry)
        )

    def _on_page_error(self, error) -> None:
        entry = {"message": str(error), "timestamp": time.time()}
        self._errors.append(entry)
        self._devtools_log.append(
            DevToolsEntry(kind="error", timestamp=time.time(), data=entry)
        )

    def _on_request(self, request) -> None:
        entry = {
            "url": request.url,
            "method": request.method,
            "timestamp": time.time(),
        }
        self._network_requests.append(entry)

    def _on_response(self, response) -> None:
        for req in self._network_requests:
            if req["url"] == response.url and "status" not in req:
                req["status"] = response.status
                req["duration_ms"] = (time.time() - req["timestamp"]) * 1000
                break

    # ── Navigation ─────────────────────────────────────────────
    async def navigate(self, path: str, timeout: int = 20000) -> NavigationResult:
        t0 = time.time()
        url = f"{self.base_url}{path}"
        try:
            resp = await self._page.goto(url, wait_until="load", timeout=timeout)
            status = resp.status if resp else 0

            try:
                await self._page.wait_for_function(
                    "() => !document.body.innerText.includes('Connecting...')",
                    timeout=15000,
                )
            except Exception:
                pass

            body = await self._page.inner_text("body")
            return NavigationResult(
                url=url,
                status=status,
                body_length=len(body),
                duration_s=time.time() - t0,
            )
        except Exception as exc:
            return NavigationResult(
                url=url, status=0, body_length=0, duration_s=time.time() - t0, errors=[str(exc)]
            )

    async def reload(self, ignore_cache: bool = False) -> NavigationResult:
        t0 = time.time()
        try:
            await self._page.reload(wait_until="load", timeout=20000, ignore_cache=ignore_cache)
            body = await self._page.inner_text("body")
            return NavigationResult(
                url=self._page.url, status=200, body_length=len(body), duration_s=time.time() - t0
            )
        except Exception as exc:
            return NavigationResult(
                url=self._page.url, status=0, body_length=0, duration_s=time.time() - t0, errors=[str(exc)]
            )

    # ── Interaction ────────────────────────────────────────────
    async def click_button(self, name: str, timeout: int = 5000) -> ClickResult:
        t0 = time.time()
        try:
            btn = self._page.get_by_role("button", name=name).first
            await btn.click(timeout=timeout, force=True)
            return ClickResult(element=name, found=True, duration_s=time.time() - t0)
        except Exception as exc:
            return ClickResult(element=name, found=False, duration_s=time.time() - t0, error=str(exc))

    async def click_link(self, name: str, timeout: int = 5000) -> ClickResult:
        t0 = time.time()
        try:
            link = self._page.get_by_role("link", name=name).first
            await link.click(timeout=timeout)
            return ClickResult(element=name, found=True, duration_s=time.time() - t0)
        except Exception as exc:
            return ClickResult(element=name, found=False, duration_s=time.time() - t0, error=str(exc))

    async def click_selector(self, selector: str, timeout: int = 5000) -> ClickResult:
        t0 = time.time()
        try:
            el = self._page.locator(selector).first
            await el.click(timeout=timeout, force=True)
            return ClickResult(element=selector, found=True, duration_s=time.time() - t0)
        except Exception as exc:
            return ClickResult(element=selector, found=False, duration_s=time.time() - t0, error=str(exc))

    async def fill_input(self, placeholder: str, value: str) -> FillResult:
        try:
            inp = self._page.locator(f"input[placeholder='{placeholder}']").first
            await inp.fill(value)
            return FillResult(element=placeholder, value=value, success=True)
        except Exception as exc:
            return FillResult(element=placeholder, value=value, success=False, error=str(exc))

    async def fill_by_label(self, label: str, value: str) -> FillResult:
        try:
            inp = self._page.get_by_label(label).first
            await inp.fill(value)
            return FillResult(element=label, value=value, success=True)
        except Exception as exc:
            return FillResult(element=label, value=value, success=False, error=str(exc))

    async def select_option(self, label: str, value: str) -> ClickResult:
        t0 = time.time()
        try:
            sel = self._page.get_by_label(label).first
            await sel.select_option(value)
            return ClickResult(element=label, found=True, duration_s=time.time() - t0)
        except Exception as exc:
            return ClickResult(element=label, found=False, duration_s=time.time() - t0, error=str(exc))

    async def press_key(self, key: str) -> None:
        await self._page.keyboard.press(key)

    # ── Queries ────────────────────────────────────────────────
    async def get_body_text(self) -> str:
        return await self._page.inner_text("body")

    async def get_page_title(self) -> str:
        return await self._page.title()

    async def get_url(self) -> str:
        return self._page.url

    async def element_exists(self, role: str, name: str = "") -> bool:
        try:
            if name:
                count = await self._page.get_by_role(role, name=name).count()
            else:
                count = await self._page.get_by_role(role).count()
            return count > 0
        except Exception:
            return False

    async def wait_for_text(self, text: str, timeout: int = 10000) -> bool:
        try:
            await self._page.wait_for_function(
                f"() => document.body.innerText.includes('{text}')",
                timeout=timeout,
            )
            return True
        except Exception:
            return False

    async def wait_for_selector(self, selector: str, timeout: int = 10000) -> bool:
        try:
            await self._page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception:
            return False

    async def take_screenshot(self, path: str) -> None:
        await self._page.screenshot(path=path, full_page=True)

    async def evaluate(self, expression: str) -> Any:
        return await self._page.evaluate(expression)

    # ── DevTools ───────────────────────────────────────────────
    async def collect_performance(self) -> dict[str, Any]:
        try:
            perf = await self._page.evaluate("""() => {
                const perf = performance;
                const entries = perf.getEntriesByType('navigation');
                const nav = entries.length > 0 ? entries[0] : null;
                return {
                    domContentLoaded: nav ? nav.domContentLoadedEventEnd : 0,
                    loadEvent: nav ? nav.loadEventEnd : 0,
                    domInteractive: nav ? nav.domInteractive : 0,
                    responseTime: nav ? nav.responseEnd - nav.requestStart : 0,
                    resourceCount: perf.getEntriesByType('resource').length,
                    memoryUsed: perf.memory ? perf.memory.usedJSHeapSize : 0,
                    memoryTotal: perf.memory ? perf.memory.totalJSHeapSize : 0,
                };
            }""")
            self._performance = perf
            return perf
        except Exception:
            return {}

    def devtools_report(self) -> DevToolsReport:
        return DevToolsReport(
            console_messages=list(self._console_messages),
            network_requests=list(self._network_requests),
            errors=list(self._errors),
            performance=dict(self._performance),
            total_console=len(self._console_messages),
            total_requests=len(self._network_requests),
            total_errors=len(self._errors),
            summary=self._build_summary(),
        )

    def _build_summary(self) -> str:
        parts = [
            f"Console: {len(self._console_messages)} messages",
            f"Network: {len(self._network_requests)} requests",
            f"Errors: {len(self._errors)} page errors",
        ]
        if self._performance:
            parts.append(f"Memory: {self._performance.get('memoryUsed', 0) / 1024 / 1024:.1f}MB")
        return " | ".join(parts)

    def clear_logs(self) -> None:
        self._devtools_log.clear()
        self._console_messages.clear()
        self._network_requests.clear()
        self._errors.clear()
        self._performance.clear()

    # ── Step recording ─────────────────────────────────────────
    def step(self, description: str) -> dict[str, Any]:
        self._step_counter += 1
        return {
            "step": self._step_counter,
            "description": description,
            "timestamp": time.time(),
            "url": self._page.url if self._page else "",
            "console_count": len(self._console_messages),
            "error_count": len(self._errors),
        }

    def get_full_log(self) -> list[dict[str, Any]]:
        return [
            {
                "kind": entry.kind,
                "timestamp": entry.timestamp,
                "data": entry.data,
            }
            for entry in self._devtools_log
        ]
