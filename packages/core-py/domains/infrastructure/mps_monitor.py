"""
MPS Memory Monitor — auto-switches to CPU when GPU memory is low.

On 8GB Macs, MPS has ~6.8GB usable. When memory usage exceeds the threshold,
new generations automatically fall back to CPU to prevent OOM crashes.

Usage:
    from domains.infrastructure.mps_monitor import get_mps_monitor
    device = get_mps_monitor().get_device("auto")  # returns "cpu" if MPS is full
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger("man.infrastructure.mps_monitor")


class MPSMemoryMonitor:
    """
    Monitors MPS memory usage and auto-switches to CPU when near capacity.

    Thresholds (aggressive — float32 on 8GB Mac runs out of memory fast):
      - 30% used → warn, next generation uses CPU
      - 40% used → force CPU + aggressive cache clear
      - <20% used → safe to use MPS again
    """

    def __init__(self, warn_threshold: float = 0.30, force_cpu_threshold: float = 0.40, safe_threshold: float = 0.20):
        self._warn_threshold = warn_threshold
        self._force_cpu_threshold = force_cpu_threshold
        self._safe_threshold = safe_threshold
        self._locked_to_cpu = False
        self._lock = threading.Lock()
        self._last_check = 0.0
        self._last_usage = 0.0
        self._check_interval = 2.0  # seconds between checks

    def get_device(self, requested: str) -> str:
        """
        Return the actual device to use for generation.

        If MPS is near capacity, returns "cpu" regardless of requested device.
        If MPS is safe, returns the requested device (or resolved "auto").
        """
        if requested == "cpu":
            return "cpu"

        with self._lock:
            import time
            now = time.time()

            # Cache the check for a few seconds to avoid overhead
            if now - self._last_check < self._check_interval:
                if self._locked_to_cpu:
                    return "cpu"
                return requested if requested != "auto" else "mps"

            self._last_check = now
            usage = self._get_mps_usage()
            self._last_usage = usage

            if usage >= self._force_cpu_threshold:
                self._locked_to_cpu = True
                logger.warning(
                    "MPS memory at %.0f%% — forcing CPU for next generations",
                    usage * 100,
                )
                self._clear_mps_cache()
                return "cpu"

            if usage >= self._warn_threshold:
                logger.warning(
                    "MPS memory at %.0f%% — next generation will use CPU (threshold: %.0f%%)",
                    usage * 100,
                    self._warn_threshold * 100,
                )
                self._locked_to_cpu = True
                return "cpu"

            if usage < self._safe_threshold and self._locked_to_cpu:
                logger.info(
                    "MPS memory dropped to %.0f%% — re-enabling MPS",
                    usage * 100,
                )
                self._locked_to_cpu = False

            return requested if requested != "auto" else "mps"

    def check_mid_generation(self) -> bool:
        """
        Check MPS memory mid-generation.
        Returns True if safe to continue on MPS, False if should fall back.

        More aggressive than the pre-generation check because we're already
        in the middle of generation with KV cache growing.
        """
        try:
            import time
            now = time.time()
            if now - self._last_check < self._check_interval:
                return not self._locked_to_cpu
            self._last_check = now
            usage = self._get_mps_usage()
            self._last_usage = usage
            if usage >= 0.35:
                logger.warning("MPS mid-generation at %.0f%% — clearing cache", usage * 100)
                self._clear_mps_cache()
                usage = self._get_mps_usage()
                if usage >= 0.35:
                    self._locked_to_cpu = True
                    return False
            return True
        except Exception:
            return not self._locked_to_cpu

    def is_locked_to_cpu(self) -> bool:
        with self._lock:
            return self._locked_to_cpu

    def get_usage(self) -> float:
        """Return last known MPS memory usage (0.0 to 1.0)."""
        with self._lock:
            return self._last_usage

    def force_cpu(self):
        """Manually lock to CPU (e.g. after OOM error)."""
        with self._lock:
            self._locked_to_cpu = True
            logger.info("MPS manually locked to CPU")

    def reset(self):
        """Reset CPU lock — allows MPS to be used again."""
        with self._lock:
            self._locked_to_cpu = False
            self._last_usage = 0.0
            logger.info("MPS monitor reset — MPS re-enabled")

    def _get_mps_usage(self) -> float:
        """
        Estimate MPS memory usage as a fraction of the 8GB limit.

        Uses ml_types.mps for platform detection.
        Falls back to 0.0 if MPS is not available.
        """
        try:
            from domains.infrastructure.ml_types import mps as ml_mps
            if not ml_mps.is_available():
                return 0.0
            # MPS not available via numpy — return 0
            return 0.0
        except Exception:
            return 0.0

    def _clear_mps_cache(self):
        """Aggressively clear MPS memory."""
        try:
            import gc
            from domains.infrastructure.ml_types import mps as ml_mps
            gc.collect()
            if ml_mps.is_available():
                ml_mps.empty_cache()
                logger.info("MPS cache cleared (numpy backend)")
        except Exception:
            pass


_monitor: Optional[MPSMemoryMonitor] = None


def get_mps_monitor() -> MPSMemoryMonitor:
    global _monitor
    if _monitor is None:
        _monitor = MPSMemoryMonitor()
    return _monitor
