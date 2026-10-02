"""SyncRunner — drive avion's async API from synchronous callers.

avion's backends are async because Playwright, CDP and websockets are async.
Synchronous callers (pytest tests written against ``sync_playwright``,
standalone scripts) still need to drive them without rewriting themselves as
coroutines.

One runner owns one persistent event loop, and every coroutine runs on *that*
loop — Playwright objects bind to the loop that created them, so mixing
``asyncio.run()`` calls (a fresh loop each time) breaks a live browser. Run
everything through the same runner instead.

Usage::

    from avion.sync import SyncRunner

    with SyncRunner() as loop:
        loop.run(backend.start())
        loop.run(backend.navigate("http://localhost:5173"))
        el = loop.run(backend.find_element(ElementLocator.css("body")))
        loop.run(backend.stop())

This is an adapter (async → sync), not a duplicate API: it adds no
browser-facing methods of its own.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class SyncRunner:
    """Persistent event loop for calling avion coroutines synchronously."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """The runner's loop, created on first use."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run one coroutine to completion on the runner's loop."""
        return self.loop.run_until_complete(coro)

    def close(self) -> None:
        """Cancel stragglers and close the loop. Safe to call twice."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        finally:
            loop.close()
            self._loop = None

    def __enter__(self) -> SyncRunner:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


_default: SyncRunner | None = None


def get_default_runner() -> SyncRunner:
    """The process-wide runner, created on first use."""
    global _default
    if _default is None:
        _default = SyncRunner()
    return _default


# Type params stay as TypeVar: avion declares requires-python >=3.11 and
# PEP 695 syntax needs 3.12, while repo-wide ruff targets py312.
def run(coro: Coroutine[Any, Any, T]) -> T:  # noqa: UP047
    """Run a coroutine on the default runner."""
    return get_default_runner().run(coro)
