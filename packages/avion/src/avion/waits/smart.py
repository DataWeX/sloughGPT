"""Smart waits — poll the page until loading settles, then proceed."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WaitCondition(Enum):
    """What to wait for before continuing."""

    NO_SPINNERS = "no_spinners"  # no visible loading spinners
    NO_OVERLAYS = "no_overlays"  # no modal/loading overlays
    TEXT_VISIBLE = "text_visible"  # a string appears in page text
    TEXT_GONE = "text_gone"  # a string disappears from page text
    ANIMATIONS_DONE = "animations_done"  # no running CSS animations


_SPINNER_JS = (
    "Array.from(document.querySelectorAll("
    "'[role=progressbar], .spinner, .loading, [aria-busy=true]'))"
    ".filter(e => e.offsetParent !== null).length"
)
_OVERLAY_JS = (
    "Array.from(document.querySelectorAll("
    "'.modal-backdrop, .overlay, [role=dialog]'))"
    ".filter(e => e.offsetParent !== null).length"
)
_ANIM_JS = (
    "Array.from(document.getAnimations ? document.getAnimations() : [])"
    ".filter(a => a.playState === 'running').length"
)
_TEXT_JS = "document.body ? document.body.innerText : ''"


@dataclass
class WaitConfig:
    """How to wait."""

    conditions: list[WaitCondition] = field(
        default_factory=lambda: [WaitCondition.NO_SPINNERS, WaitCondition.NO_OVERLAYS]
    )
    timeout: float = 10.0
    poll_ms: float = 200.0
    text: str = ""  # used by TEXT_VISIBLE / TEXT_GONE


@dataclass
class WaitResult:
    """Outcome of a wait."""

    success: bool
    waited_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "waited_ms": round(self.waited_ms, 1),
            "error": self.error,
        }


class SmartWaiter:
    """Polls page state until conditions hold or timeout hits.

    Only needs evaluate() + wait_for_timeout() on the backend,
    so it works with real browsers and fakes alike.
    """

    def __init__(self, backend):
        self._backend = backend

    async def wait(self, config: WaitConfig | None = None) -> WaitResult:
        cfg = config or WaitConfig()
        start = time.monotonic()
        deadline = start + cfg.timeout
        last_error = ""
        while time.monotonic() < deadline:
            try:
                if await self._settled(cfg):
                    waited = (time.monotonic() - start) * 1000
                    return WaitResult(success=True, waited_ms=waited)
            except Exception as e:
                last_error = str(e)
            await self._backend.wait_for_timeout(cfg.poll_ms)
        waited = (time.monotonic() - start) * 1000
        conds = ",".join(c.value for c in cfg.conditions)
        return WaitResult(
            success=False,
            waited_ms=waited,
            error=f"timeout waiting for [{conds}]: {last_error}",
        )

    async def wait_for_text(self, text: str, timeout: float = 10.0) -> WaitResult:
        return await self.wait(
            WaitConfig(conditions=[WaitCondition.TEXT_VISIBLE], timeout=timeout, text=text)
        )

    async def _settled(self, cfg: WaitConfig) -> bool:
        for cond in cfg.conditions:
            if cond == WaitCondition.NO_SPINNERS:
                if int(await self._backend.evaluate(_SPINNER_JS) or 0) > 0:
                    return False
            elif cond == WaitCondition.NO_OVERLAYS:
                if int(await self._backend.evaluate(_OVERLAY_JS) or 0) > 0:
                    return False
            elif cond == WaitCondition.ANIMATIONS_DONE:
                if int(await self._backend.evaluate(_ANIM_JS) or 0) > 0:
                    return False
            elif cond == WaitCondition.TEXT_VISIBLE:
                body = str(await self._backend.evaluate(_TEXT_JS) or "")
                if cfg.text not in body:
                    return False
            elif cond == WaitCondition.TEXT_GONE:
                body = str(await self._backend.evaluate(_TEXT_JS) or "")
                if cfg.text in body:
                    return False
        return True
