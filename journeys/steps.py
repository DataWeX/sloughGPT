"""Step builders for UX_FLOWS journeys (avion TaskStep factories).

Engine: avion (packages/avion) on Firefox — chromium never commits
navigations on this hardware.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any, Callable

_ROOT = os.path.abspath(__file__)
while _ROOT != os.path.dirname(_ROOT) and not os.path.isdir(
    os.path.join(_ROOT, "packages", "avion", "src")
):
    _ROOT = os.path.dirname(_ROOT)
for _sub in ("avion", "arken"):
    _src = os.path.join(_ROOT, "packages", _sub, "src")
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)

ASSISTANT_BUBBLE = '[aria-label="Message from Assistant"]'


def _step(name: str, fn: Callable[[dict[str, Any]], Any], optional: bool = False):
    from avion.core.task import TaskStep

    async def action(ctx: dict[str, Any]) -> None:
        res = fn(ctx)
        if hasattr(res, "__await__"):
            await res

    return TaskStep(name=name, action=action, optional=optional)


def s_goto(url: str, verify: str | None = None):
    async def action(ctx):
        ok = await ctx["arken"].goto(url)
        assert ok, f"goto failed: {url}"
        # Full reloads re-show the StartupOverlay (aria-label="Startup progress")
        # for every stage — wait until it fully unmounts, sustained, before asserting.
        page = ctx["backend"].page
        deadline = time.monotonic() + 60
        gone_since: float | None = None
        while time.monotonic() < deadline:
            n = await page.locator('[aria-label="Startup progress"]').count()
            if n == 0:
                if gone_since is None:
                    gone_since = time.monotonic()
                elif time.monotonic() - gone_since >= 1.0:
                    break
            else:
                gone_since = None
            await asyncio.sleep(0.25)
        else:
            raise AssertionError("boot overlay never dismissed")
        if verify:
            found = await ctx["arken"].wait_for_text(verify, timeout=25)
            assert found, f"after goto: missing text {verify!r}"

    return _step(f"goto {url}", action)


def s_click(text: str, optional: bool = False, timeout: float = 10):
    async def action(ctx):
        from avion.core.element import ElementLocator

        await ctx["arken"].find(ElementLocator.text(text), timeout=timeout)
        await ctx["arken"].click_text(text)

    return _step(f"click '{text}'", action, optional=optional)


def s_wait(text: str, timeout: float = 20):
    async def action(ctx):
        ok = await ctx["arken"].wait_for_text(text, timeout=timeout)
        assert ok, f"text not found: {text!r}"

    return _step(f"wait '{text}'", action)


def s_send(prompt: str, reply_timeout: float = 90):
    """Fill the composer, press Enter, wait for the assistant bubble."""

    async def action(ctx):
        page = ctx["backend"].page  # async Playwright page — every call awaits
        ta = page.locator("textarea").last
        await ta.wait_for(timeout=20000)
        await ta.fill(prompt)
        await ta.press("Enter")
        await page.locator(ASSISTANT_BUBBLE).first.wait_for(timeout=reply_timeout * 1000)

    return _step("send + assistant reply", action)


def s_exists(selector: str, description: str, timeout: float = 15):
    async def action(ctx):
        page = ctx["backend"].page
        deadline = time.monotonic() + timeout
        while True:
            if await page.locator(selector).count() > 0:
                return
            if time.monotonic() >= deadline:
                raise AssertionError(f"missing: {description}")
            await asyncio.sleep(0.25)

    return _step(f"exists: {description}", action)


def s_upload(path: str):
    async def action(ctx):
        page = ctx["backend"].page
        await page.locator("input[type=file]").first.set_input_files(path)
        await ctx["arken"].wait_for_text(os.path.basename(path), timeout=15)

    return _step(f"upload {os.path.basename(path)}", action)


def s_note(description: str, fn: Callable[[dict[str, Any]], Any]):
    """Record an observation (never fails the flow unless fn raises)."""
    return _step(description, fn)
