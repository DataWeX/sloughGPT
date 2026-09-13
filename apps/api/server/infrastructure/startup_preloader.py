"""Startup preloader — imports commonly used modules in background to reduce cold start latency.

Usage:
    from infrastructure.startup_preloader import preload_common_modules, get_preload_status
    preload_common_modules()
    # Later...
    status = get_preload_status()
"""

import importlib
import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Commonly used modules that benefit from pre-importing
DEFAULT_PRELOAD_MODULES = [
    "json",
    "time",
    "datetime",
    "pathlib",
    "hashlib",
    "uuid",
    "re",
    "os",
    "sys",
    "logging",
    "asyncio",
    "functools",
    "collections",
    "typing",
    "dataclasses",
    "enum",
    "httpx",
    "fastapi",
    "pydantic",
    "sqlalchemy",
    "aiohttp",
    "starlette",
]


@dataclass
class PreloadResult:
    """Result of preloading a single module."""

    module: str
    duration_ms: float
    success: bool
    error: str | None = None


@dataclass
class PreloadStatus:
    """Status of the preloader."""

    running: bool = False
    finished: bool = False
    total_modules: int = 0
    completed: int = 0
    succeeded: int = 0
    failed: int = 0
    total_duration_ms: float = 0.0
    results: list[PreloadResult] = field(default_factory=list)


_status = PreloadStatus()
_lock = threading.Lock()


def _preload_module(module_name: str) -> PreloadResult:
    """Import a single module and return timing info."""
    start = time.perf_counter()
    try:
        from infrastructure.startup_cache import cached_import
        cached_import(module_name)
        duration_ms = (time.perf_counter() - start) * 1000
        return PreloadResult(
            module=module_name,
            duration_ms=round(duration_ms, 2),
            success=True,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        return PreloadResult(
            module=module_name,
            duration_ms=round(duration_ms, 2),
            success=False,
            error=str(e),
        )


def _run_preload(modules: list[str], delay: float) -> None:
    """Run preloading in a background thread."""
    global _status

    if delay > 0:
        time.sleep(delay)

    start = time.perf_counter()
    results: list[PreloadResult] = []

    for module in modules:
        result = _preload_module(module)
        results.append(result)
        with _lock:
            _status.completed += 1
            if result.success:
                _status.succeeded += 1
            else:
                _status.failed += 1
            logger.debug("Preloaded %s in %.1fms%s", module, result.duration_ms, " (failed)" if not result.success else "")

    total_ms = (time.perf_counter() - start) * 1000

    with _lock:
        _status.finished = True
        _status.running = False
        _status.total_duration_ms = round(total_ms, 2)
        _status.results = results

    logger.info(
        "Preloaded %d/%d modules in %.1fms",
        _status.succeeded,
        len(modules),
        total_ms,
    )


def preload_common_modules(
    modules: list[str] | None = None,
    delay: float = 0.0,
    daemon: bool = True,
) -> None:
    """Start preloading common modules in a background thread.

    Args:
        modules: List of module names to preload. Defaults to DEFAULT_PRELOAD_MODULES.
        delay: Seconds to wait before starting preload. Useful for avoiding
               contention with critical startup paths.
        daemon: Whether the preload thread should be a daemon thread.
    """
    global _status

    with _lock:
        if _status.running:
            logger.debug("Preload already running")
            return
        _status = PreloadStatus(
            running=True,
            finished=False,
            total_modules=len(modules or DEFAULT_PRELOAD_MODULES),
        )

    target_modules = modules or DEFAULT_PRELOAD_MODULES
    thread = threading.Thread(
        target=_run_preload,
        args=(target_modules, delay),
        daemon=daemon,
        name="startup-preloader",
    )
    thread.start()
    logger.info("Started preloading %d modules (delay=%.1fs)", len(target_modules), delay)


def get_preload_status() -> PreloadStatus:
    """Get current preload status."""
    with _lock:
        return _status


def wait_for_preload(timeout: float | None = None) -> PreloadStatus:
    """Block until preload finishes or timeout."""
    start = time.perf_counter()
    while True:
        with _lock:
            if _status.finished:
                return _status
        if timeout is not None and (time.perf_counter() - start) >= timeout:
            with _lock:
                return _status
        time.sleep(0.05)
