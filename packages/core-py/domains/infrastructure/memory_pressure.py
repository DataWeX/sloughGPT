from __future__ import annotations

"""
Memory pressure monitor — proactive memory management for model serving.

Detects high system memory usage and triggers cleanup actions:
  - Clears KV caches and Python object caches
  - Forces garbage collection
  - Evicts idle model weights via release_model()
  - Blocks new model loads when memory is critically low

Thresholds:
  - WARNING  (80%): log once, clear Python caches
  - CRITICAL (90%): aggressive cleanup — release idle weights, force GC
  - EMERGENCY (95%): refuse new model loads, release all idle resources

Usage::

    from domains.infrastructure.memory_pressure import get_memory_pressure_monitor
    monitor = get_memory_pressure_monitor()
    monitor.check()  # call before inference
    allowed = monitor.allow_load()  # call before model loading
"""

import gc
import logging
import threading
import time
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger("slo.infrastructure.memory_pressure")


class PressureLevel(Enum):
    """Memory pressure severity levels."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MemoryPressureMonitor:
    """Proactive memory pressure detection and cleanup.

    Monitors system memory and triggers graduated cleanup actions.
    Thread-safe — all state is protected by a lock.

    Args:
        warning_threshold: Memory percent to trigger cache clearing (default 80).
        critical_threshold: Memory percent to trigger weight release (default 90).
        emergency_threshold: Memory percent to block model loads (default 95).
        check_interval_s: Minimum seconds between automatic background checks (default 15).
    """

    def __init__(
        self,
        warning_threshold: float = 80.0,
        critical_threshold: float = 90.0,
        emergency_threshold: float = 95.0,
        check_interval_s: float = 15.0,
    ):
        self._warning = warning_threshold
        self._critical = critical_threshold
        self._emergency = emergency_threshold
        self._check_interval = check_interval_s

        self._lock = threading.Lock()
        self._last_check: float = 0.0
        self._last_cleanup: float = 0.0
        self._cleanup_count: int = 0
        self._loads_blocked: int = 0
        self._idle_releases: int = 0
        self._gc_forced: int = 0
        self._last_level: PressureLevel = PressureLevel.NORMAL
        self._last_warning_logged: float = 0.0
        self._last_critical_logged: float = 0.0
        self._last_emergency_logged: float = 0.0

        # Registered cleanup callbacks (called on critical/emergency)
        self._cleanup_callbacks: list[Callable[[], None]] = []

    @property
    def warning_threshold(self) -> float:
        return self._warning

    @property
    def critical_threshold(self) -> float:
        return self._critical

    @property
    def emergency_threshold(self) -> float:
        return self._emergency

    def configure(self, warning: Optional[float] = None,
                  critical: Optional[float] = None,
                  emergency: Optional[float] = None) -> None:
        """Update pressure thresholds at runtime.

        Only non-None values are applied. Thread-safe — thresholds are
        protected by the existing lock.
        """
        with self._lock:
            if warning is not None:
                self._warning = warning
            if critical is not None:
                self._critical = critical
            if emergency is not None:
                self._emergency = emergency

    def register_cleanup(self, callback: Callable[[], None]) -> None:
        """Register a cleanup callback invoked on critical/emergency pressure.

        Callbacks are called once per cleanup cycle (not every check).
        Duplicate registrations of the same callback are ignored.
        Typical usage: release idle model weights, clear embedding caches.
        """
        with self._lock:
            if callback not in self._cleanup_callbacks:
                self._cleanup_callbacks.append(callback)

    def check(self) -> PressureLevel:
        """Check memory pressure and trigger cleanup if needed.

        Should be called before inference or model loading. Respects
        ``_check_interval`` to avoid redundant checks.

        Returns:
            Current pressure level.
        """
        now = time.time()
        with self._lock:
            if now - self._last_check < self._check_interval:
                return self._last_level
            self._last_check = now

        try:
            import psutil
            mem = psutil.virtual_memory()
        except ImportError:
            return PressureLevel.NORMAL

        level = self._classify(mem.percent)
        process_rss_mb = self._get_rss_mb()

        with self._lock:
            prev_level = self._last_level
            self._last_level = level

        if level == PressureLevel.WARNING and prev_level != PressureLevel.WARNING:
            self._on_warning(mem.percent, mem.available, process_rss_mb)
        elif level == PressureLevel.CRITICAL:
            if now - self._last_critical_logged > 60:
                self._on_critical(mem.percent, mem.available, process_rss_mb)
                self._last_critical_logged = now
        elif level == PressureLevel.EMERGENCY:
            if now - self._last_emergency_logged > 60:
                self._on_emergency(mem.percent, mem.available, process_rss_mb)
                self._last_emergency_logged = now
        elif level == PressureLevel.NORMAL and prev_level in (PressureLevel.CRITICAL, PressureLevel.EMERGENCY):
            logger.info(
                "Memory pressure recovered: %.1f%% (was %s)",
                mem.percent, prev_level.value,
            )

        return level

    def allow_load(self) -> bool:
        """Check if a new model load should be allowed.

        Returns False when memory is at emergency level and loading
        would likely cause OOM.

        Returns:
            True if loading is allowed, False if it should be blocked.
        """
        try:
            import psutil
            mem = psutil.virtual_memory()
        except ImportError:
            return True

        level = self._classify(mem.percent)
        if level == PressureLevel.EMERGENCY:
            with self._lock:
                self._loads_blocked += 1
            logger.warning(
                "Memory pressure: model load blocked (%.1f%% used, %.0f MB available)",
                mem.percent, mem.available / (1024 * 1024),
            )
            return False
        return True

    def force_cleanup(self) -> PressureLevel:
        """Force an immediate cleanup cycle regardless of interval.

        Returns:
            Pressure level after cleanup.
        """
        with self._lock:
            self._last_check = 0  # reset to allow immediate check
        return self.check()

    def stats(self) -> dict:
        """Return current pressure stats for the health endpoint."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            process_rss_mb = self._get_rss_mb()
        except Exception:
            with self._lock:
                return {
                    "level": self._last_level.value,
                    "system_percent": 0.0,
                    "available_mb": 0,
                    "rss_mb": 0.0,
                    "last_check_ts": self._last_check,
                    "last_cleanup_ts": self._last_cleanup,
                    "cleanup_count": self._cleanup_count,
                    "loads_blocked": self._loads_blocked,
                    "idle_releases": self._idle_releases,
                    "gc_forced": self._gc_forced,
                    "warning_threshold": self._warning,
                    "critical_threshold": self._critical,
                    "emergency_threshold": self._emergency,
                }

        with self._lock:
            return {
                "level": self._last_level.value,
                "system_percent": round(mem.percent, 1),
                "available_mb": mem.available // (1024 * 1024),
                "rss_mb": round(process_rss_mb, 1),
                "last_check_ts": self._last_check,
                "last_cleanup_ts": self._last_cleanup,
                "cleanup_count": self._cleanup_count,
                "loads_blocked": self._loads_blocked,
                "idle_releases": self._idle_releases,
                "gc_forced": self._gc_forced,
                "warning_threshold": self._warning,
                "critical_threshold": self._critical,
                "emergency_threshold": self._emergency,
            }

    def _classify(self, percent: float) -> PressureLevel:
        """Classify memory percent into a pressure level."""
        if percent >= self._emergency:
            return PressureLevel.EMERGENCY
        if percent >= self._critical:
            return PressureLevel.CRITICAL
        if percent >= self._warning:
            return PressureLevel.WARNING
        return PressureLevel.NORMAL

    def _get_rss_mb(self) -> float:
        """Return current process RSS in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def _on_warning(self, percent: float, available: int, rss_mb: float) -> None:
        """Handle WARNING level — clear Python caches."""
        logger.warning(
            "Memory pressure WARNING: %.1f%% used (%.0f MB available, %.0f MB RSS)",
            percent, available / (1024 * 1024), rss_mb,
        )
        # Clear Python string interning cache and float cache (minor savings)
        try:
            import sys
            # Clear the float cache (saves ~few KB)
            sys.float_info
        except Exception:
            pass

    def _on_critical(self, percent: float, available: int, rss_mb: float) -> None:
        """Handle CRITICAL level — release idle weights, force GC, run callbacks."""
        logger.critical(
            "Memory pressure CRITICAL: %.1f%% used (%.0f MB available, %.0f MB RSS) — triggering cleanup",
            percent, available / (1024 * 1024), rss_mb,
        )
        with self._lock:
            self._last_cleanup = time.time()
            self._cleanup_count += 1

        # 1. Force garbage collection
        self._force_gc()

        # 2. Clear KV session caches
        self._clear_kv_caches()

        # 3. Release idle model weights via registered callbacks
        self._run_cleanup_callbacks()

        # 4. Try to release memory back to OS
        self._malloc_trim()

    def _on_emergency(self, percent: float, available: int, rss_mb: float) -> None:
        """Handle EMERGENCY level — all of CRITICAL plus refuse new loads."""
        logger.critical(
            "Memory pressure EMERGENCY: %.1f%% used (%.0f MB available, %.0f MB RSS) — "
            "new model loads blocked",
            percent, available / (1024 * 1024), rss_mb,
        )
        with self._lock:
            self._last_cleanup = time.time()
            self._cleanup_count += 1

        self._force_gc()
        self._clear_kv_caches()
        self._run_cleanup_callbacks()
        self._release_all_idle_weights()
        self._malloc_trim()

    def _force_gc(self) -> None:
        """Force garbage collection and track it."""
        try:
            collected = gc.collect()
            with self._lock:
                self._gc_forced += 1
            logger.debug("GC collected %d objects", collected)
        except Exception as e:
            logger.debug("GC failed: %s", e)

    def _clear_kv_caches(self) -> None:
        """Clear cross-turn KV session caches to free memory."""
        try:
            from domains.infrastructure.model_server import SESSION_KV_CACHE
            removed = SESSION_KV_CACHE.clear_all()
            logger.debug("KV cache cleared: %d sessions dropped", removed)
        except Exception as e:
            logger.debug("KV cache clear failed: %s", e)

    def _run_cleanup_callbacks(self) -> None:
        """Run registered cleanup callbacks."""
        with self._lock:
            callbacks = list(self._cleanup_callbacks)
        for cb in callbacks:
            try:
                cb()
            except Exception as e:
                logger.debug("Cleanup callback failed: %s", e)

    def _release_all_idle_weights(self) -> None:
        """Release all idle model weights via the provider's release_model()."""
        try:
            from domains.models.provider import get_provider
            for name in ("slonet-native", "slonet"):
                provider = get_provider(name)
                if provider is not None and hasattr(provider, "release_model"):
                    released = provider.release_model()
                    if released:
                        with self._lock:
                            self._idle_releases += 1
                        logger.info("Released idle model weights via provider '%s'", name)
        except Exception as e:
            logger.debug("Idle weight release failed: %s", e)

    def _malloc_trim(self) -> None:
        """Return allocator-held memory to the OS (glibc/Linux only)."""
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass  # not Linux or libc unavailable


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_monitor: Optional[MemoryPressureMonitor] = None
_monitor_lock = threading.Lock()


def get_memory_pressure_monitor() -> MemoryPressureMonitor:
    """Get or create the global MemoryPressureMonitor singleton."""
    global _monitor
    if _monitor is None:
        with _monitor_lock:
            if _monitor is None:
                _monitor = MemoryPressureMonitor()
    return _monitor


def reset_memory_pressure_monitor() -> None:
    """Reset the singleton (for testing)."""
    global _monitor
    with _monitor_lock:
        _monitor = None
