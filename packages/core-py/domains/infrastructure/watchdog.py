"""
Server Health Watchdog — monitors API health and auto-recovers from crashes.

Runs as a background thread that:
  - Polls /health endpoint every N seconds
  - Detects when server goes offline (consecutive failures)
  - Triggers auto-recovery (model reload, cache clear, provider re-init)
  - Logs recovery events
"""

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("slo.infrastructure.watchdog")


class HealthWatchdog:
    """
    Monitors server health and triggers auto-recovery on failure.

    Usage:
        watchdog = HealthWatchdog()
        watchdog.set_recovery_fn(reload_model_and_providers)
        watchdog.start(poll_interval=10, max_failures=3)
    """

    def __init__(self) -> None:
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._poll_interval = 10
        self._max_failures = 3
        self._consecutive_failures = 0
        self._recovery_fn: Optional[Callable] = None
        self._health_check_fn: Optional[Callable[[], bool]] = None
        self._on_recovery: Optional[Callable] = None
        self._lock = threading.Lock()

    def set_recovery_fn(self, fn: Callable):
        """Set the function to call when recovery is needed."""
        self._recovery_fn = fn

    def set_health_check_fn(self, fn: Callable[[], bool]):
        """Set the function to call for health checks."""
        self._health_check_fn = fn

    def set_on_recovery(self, fn: Callable):
        """Set callback after successful recovery."""
        self._on_recovery = fn

    def start(self, poll_interval: int = 10, max_failures: int = 3) -> None:
        """Start the watchdog background thread."""
        if self._running:
            return
        self._poll_interval = poll_interval
        self._max_failures = max_failures
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="health-watchdog")
        self._thread.start()
        logger.info("Health watchdog started (interval=%ds, max_failures=%d)", poll_interval, max_failures,
            extra={"tag": "INFRA"})

    def stop(self) -> None:
        """Stop the watchdog."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Health watchdog stopped",
            extra={"tag": "INFRA"})

    def _run(self) -> None:
        """Main watchdog loop."""
        while self._running:
            try:
                healthy = False
                if self._health_check_fn:
                    healthy = self._health_check_fn()
                else:
                    # Default: try importing and checking server state
                    try:
                        import state
                        healthy = state.model is not None
                    except Exception:
                        healthy = False

                with self._lock:
                    if healthy:
                        self._consecutive_failures = 0
                    else:
                        self._consecutive_failures += 1
                        logger.warning(
                            "Health check failed (%d/%d consecutive)",
                            self._consecutive_failures,
                            self._max_failures,
                            extra={"tag": "INFRA"},
                        )

                        if self._consecutive_failures >= self._max_failures:
                            logger.error("Server unhealthy — triggering recovery",
                                extra={"tag": "INFRA"})
                            self._consecutive_failures = 0
                            self._trigger_recovery()

            except Exception as e:
                logger.error("Watchdog error: %s", e,
                    extra={"tag": "INFRA"})

            # Sleep in small increments so we can stop quickly
            for _ in range(self._poll_interval * 10):
                if not self._running:
                    break
                time.sleep(0.1)

    def _trigger_recovery(self):
        """Attempt to recover the server."""
        if not self._recovery_fn:
            logger.warning("No recovery function set — cannot recover",
                extra={"tag": "INFRA"})
            return

        try:
            logger.info("Running recovery procedure...",
                extra={"tag": "INFRA"})
            result = self._recovery_fn()
            if result:
                logger.info("Recovery successful",
                    extra={"tag": "INFRA"})
                if self._on_recovery:
                    self._on_recovery()
            else:
                logger.error("Recovery failed",
                    extra={"tag": "INFRA"})
        except Exception as e:
            logger.error("Recovery procedure crashed: %s", e,
                extra={"tag": "INFRA"})


_watchdog: Optional[HealthWatchdog] = None
_watchdog_lock = threading.Lock()


def get_watchdog() -> HealthWatchdog:
    global _watchdog
    if _watchdog is None:
        with _watchdog_lock:
            if _watchdog is None:
                _watchdog = HealthWatchdog()
    return _watchdog


def reset_watchdog() -> None:
    """Reset the singleton (for testing)."""
    global _watchdog
    with _watchdog_lock:
        _watchdog = None
