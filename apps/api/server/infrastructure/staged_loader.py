"""Staged startup loader — progressive server readiness.

The server has three stages:

    ┌─────────────────────────────────────────────────────────────┐
    │  STAGE 1: CRITICAL (blocks requests)                       │
    │    - DB connection pool                                     │
    │    - Core routers (health, status)                          │
    │    - Model load (background thread, ~45s)                   │
    │    Result: server accepts requests, model still loading     │
    ├─────────────────────────────────────────────────────────────┤
    │  STAGE 2: READY (health gates pass)                        │
    │    - Model loaded and warm                                  │
    │    - All feature routers registered                         │
    │    - Inference endpoints available                          │
    │    Result: full API available                               │
    ├─────────────────────────────────────────────────────────────┤
    │  STAGE 3: BACKGROUND (non-blocking)                        │
    │    - W&B / experiment tracking                              │
    │    - Prometheus metrics collector                           │
    │    - Auto-trainer daemon                                    │
    │    - RAG document ingestion                                 │
    │    - Model registry refresh                                 │
    │    Result: analytics + monitoring online                    │
    └─────────────────────────────────────────────────────────────┘

Usage:
    from infrastructure.staged_loader import get_staged_loader, Stage

    loader = get_staged_loader()
    if loader.stage >= Stage.READY:
        # model is loaded, inference available
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger("slo.startup")


class Stage(enum.IntEnum):
    """Server readiness stages (ordered)."""

    INIT = 0
    """Module imports complete, no subsystems started."""

    CRITICAL = 1
    """DB pool ready, core routers registered, model loading in background.
    Server accepts requests but inference returns 503."""

    READY = 2
    """Model loaded, all routers registered, full API available."""

    BACKGROUND = 3
    """Non-critical subsystems online (metrics, W&B, analytics)."""


class HookStatus(enum.Enum):
    """Status of a single hook."""

    PENDING = "pending"
    RUNNING = "running"
    OK = "ok"
    TIMEOUT = "timeout"
    ERROR = "error"


class HookInfo:
    """Per-hook timing and status."""

    __slots__ = ("name", "stage", "status", "start_time", "end_time", "error")

    def __init__(self, name: str, stage: Stage) -> None:
        self.name = name
        self.stage = stage
        self.status = HookStatus.PENDING
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.error: str | None = None

    @property
    def duration(self) -> float:
        if self.start_time == 0:
            return 0.0
        end = self.end_time if self.end_time > 0 else time.monotonic()
        return end - self.start_time

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "stage": self.stage.name.lower(),
            "status": self.status.value,
            "duration_seconds": round(self.duration, 2),
            "error": self.error,
        }


class StagedLoader:
    """Manages progressive server readiness.

    Each stage has a list of hooks. When the stage runs, all hooks
    execute (concurrently where possible). The stage completes when
    all hooks finish or their timeout expires.
    """

    def __init__(self) -> None:
        self._stage = Stage.INIT
        self._stage_time: dict[Stage, float] = {}
        self._hooks: dict[Stage, list[tuple[str, Callable[[], Coroutine[Any, Any, None]], float]]] = {
            Stage.CRITICAL: [],
            Stage.READY: [],
            Stage.BACKGROUND: [],
        }
        self._hook_infos: dict[str, HookInfo] = {}
        self._errors: dict[str, str] = {}
        self._model_progress: float = 0.0
        self._model_progress_message: str = ""

    @property
    def stage(self) -> Stage:
        return self._stage

    @property
    def stage_name(self) -> str:
        return self._stage.name.lower()

    @property
    def elapsed(self) -> float:
        """Seconds since INIT."""
        if Stage.INIT in self._stage_time:
            return time.monotonic() - self._stage_time[Stage.INIT]
        return 0.0

    def set_model_progress(self, progress: float, message: str = "") -> None:
        """Update model load progress (0.0 to 1.0)."""
        self._model_progress = min(1.0, max(0.0, progress))
        self._model_progress_message = message

    def on(self, stage: Stage, name: str, hook: Callable[[], Coroutine[Any, Any, None]], timeout: float = 30.0) -> None:
        """Register a hook for a given stage."""
        self._hooks[stage].append((name, hook, timeout))
        self._hook_infos[name] = HookInfo(name, stage)

    async def run_stage(self, stage: Stage) -> None:
        """Run all hooks for the given stage."""
        hooks = self._hooks.get(stage, [])
        if not hooks:
            self._stage = stage
            self._stage_time[stage] = time.monotonic()
            return

        logger.info(
            "Stage %s: running %d hooks",
            stage.name,
            len(hooks),
            extra={"tag": "START"},
        )

        tasks = []
        for name, hook, timeout in hooks:
            tasks.append(self._run_hook(stage, name, hook, timeout))

        await asyncio.gather(*tasks)

        self._stage = stage
        self._stage_time[stage] = time.monotonic()

        ok_count = sum(1 for name, _, _ in hooks if name not in self._errors)
        logger.info(
            "Stage %s: complete (%d/%d hooks OK, %.1fs elapsed)",
            stage.name,
            ok_count,
            len(hooks),
            self.elapsed,
            extra={"tag": "START"},
        )

    async def _run_hook(self, stage: Stage, name: str, hook: Callable, timeout: float) -> None:
        """Run a single hook with timeout and error isolation."""
        info = self._hook_infos.get(name)
        if info:
            info.status = HookStatus.RUNNING
            info.start_time = time.monotonic()

        try:
            await asyncio.wait_for(hook(), timeout=timeout)
            if info:
                info.status = HookStatus.OK
                info.end_time = time.monotonic()
        except TimeoutError:
            logger.warning(
                "Stage %s hook '%s' timed out after %.1fs",
                stage.name,
                name,
                timeout,
                extra={"tag": "START"},
            )
            self._errors[name] = f"timeout after {timeout}s"
            if info:
                info.status = HookStatus.TIMEOUT
                info.end_time = time.monotonic()
                info.error = f"timeout after {timeout}s"
        except Exception as exc:
            logger.warning(
                "Stage %s hook '%s' failed: %s",
                stage.name,
                name,
                exc,
                extra={"tag": "START"},
            )
            self._errors[name] = str(exc)
            if info:
                info.status = HookStatus.ERROR
                info.end_time = time.monotonic()
                info.error = str(exc)

    def get_status(self) -> dict:
        """Return current staged loader status for health endpoints."""
        return {
            "stage": self.stage_name,
            "stage_value": int(self._stage),
            "elapsed_seconds": round(self.elapsed, 1),
            "model_progress": round(self._model_progress, 2),
            "model_progress_message": self._model_progress_message,
            "errors": dict(self._errors),
            "hooks": {
                name: info.to_dict()
                for name, info in self._hook_infos.items()
            },
            "stages": {
                s.name.lower(): {
                    "hooks": [name for name, _, _ in self._hooks.get(s, [])],
                    "time": round(self._stage_time.get(s, 0) - self._stage_time.get(Stage.INIT, 0), 1) if s in self._stage_time else None,
                }
                for s in Stage
            },
        }


# ── Singleton ───────────────────────────────────────────────────────
_loader: StagedLoader | None = None


def get_staged_loader() -> StagedLoader:
    """Return the global staged loader instance."""
    global _loader
    if _loader is None:
        _loader = StagedLoader()
    return _loader
