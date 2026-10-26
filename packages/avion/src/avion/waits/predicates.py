"""Predicate waits — poll a JS predicate the app can prove, not a heuristic.

Two layers, mirroring :mod:`avion.waits.smart`:

* ``wait_for`` — generic: any JS expression, any backend that satisfies the
  SmartWaiter contract (``evaluate`` + ``wait_for_timeout``; that is every
  avion browser backend). Polls until truthy or *timeout*, and the timeout
  path returns a :class:`WaitResult` naming **what** was being waited for,
  the expression itself, and the last backend error — never a bare "timeout".
* ``wait_app_ready`` — the app-declared readiness signal: sloughGPT's root
  element carries ``data-app-ready`` once React has committed the current
  route's content **and** startup reached ``ready``/``background``. Callers
  (``goto()`` wrappers) ask that single question instead of guessing with
  ``load``/``networkidle``/overlay-absence heuristics, each of which has a
  measured failure mode.
"""

from __future__ import annotations

import time
from typing import Any

from avion.waits.smart import WaitResult

#: Canonical predicate for the app-declared readiness signal. Kept as one
#: expression so every caller (suite ``goto()``, and later navigator/session
#: wrappers) waits on the exact same thing the app sets.
APP_READY_EXPRESSION = (
    'Boolean(document.getElementById("root") && '
    'document.getElementById("root").hasAttribute("data-app-ready"))'
)
APP_READY_DESCRIPTION = "app declared readiness (data-app-ready on #root)"


async def wait_for(
    backend: Any,
    expression: str,
    *,
    timeout: float = 15.0,
    poll_ms: float = 100.0,
    description: str = "predicate",
) -> WaitResult:
    """Poll *expression* until it is truthy or *timeout* seconds elapse.

    Args:
        backend: Anything with ``async evaluate(expr)`` and
            ``async wait_for_timeout(ms)`` (see ``avion.waits.smart``).
        expression: JS expression evaluated in the page.
        timeout: Seconds before giving up (the predicate is also evaluated
            once at the deadline, so a just-in-time flip is not lost).
        poll_ms: Delay between evaluations.
        description: Human name of the thing being waited for; it leads the
            timeout error so a failed run says which contract went unsatisfied.

    Returns:
        ``WaitResult(success=True)`` on the first truthy observation, else
        ``success=False`` with
        ``"timeout after <ms>ms waiting for <description>: <expression>"``.
    """
    start = time.monotonic()
    deadline = start + timeout
    last_error = ""
    while True:
        try:
            if await backend.evaluate(expression):
                return WaitResult(success=True, waited_ms=(time.monotonic() - start) * 1000)
        except Exception as e:  # backend hiccup: keep polling, report at timeout
            last_error = str(e)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        await backend.wait_for_timeout(min(poll_ms, remaining * 1000))
    waited = (time.monotonic() - start) * 1000
    suffix = f" (last backend error: {last_error})" if last_error else ""
    return WaitResult(
        success=False,
        waited_ms=waited,
        error=f"timeout after {waited:.0f}ms waiting for {description}: {expression}{suffix}",
    )


async def wait_app_ready(backend: Any, *, timeout: float = 15.0, poll_ms: float = 100.0):
    """Wait for the app's own readiness declaration (``data-app-ready``).

    Fires only after the app has committed the route content and reported a
    startup stage of ``ready``/``background``, so a query issued immediately
    after this returns sees an interactive page rather than a hydrating one.
    """
    return await wait_for(
        backend,
        APP_READY_EXPRESSION,
        timeout=timeout,
        poll_ms=poll_ms,
        description=APP_READY_DESCRIPTION,
    )
