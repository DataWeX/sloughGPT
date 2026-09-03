"""Request coalescer: deduplicates identical concurrent in-flight requests.

First request runs; subsequent requests with the same hash await the shared result.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _hash_key(*parts: Any) -> str:
    """Build a deterministic SHA-256 hash from arbitrary parts."""
    canonical = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


class _InFlightRequest:
    """Tracks a single in-flight request and its waiters."""

    __slots__ = ("event", "result", "error", "created_at")

    def __init__(self) -> None:
        self.event = asyncio.Event()
        self.result: Optional[str] = None
        self.error: Optional[BaseException] = None
        self.created_at: float = time.monotonic()


class RequestCoalescer:
    """Deduplicates identical concurrent requests.

    Usage::

        coalescer = RequestCoalescer(ttl_seconds=30)

        key = coalescer.hash(messages, gen_params, model_type)
        existing = coalescer.start(key)
        if existing is not None:
            # Another request is already in-flight for this key.
            # await existing.wait() to get the result.
            result = await existing.wait()
        else:
            # We are the first request. Run generation, then complete.
            result = await do_generation()
            coalescer.complete(key, result)
    """

    def __init__(self, ttl_seconds: float = 30.0, max_entries: int = 512) -> None:
        self._in_flight: dict[str, _InFlightRequest] = {}
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    def hash(self, *parts: Any) -> str:
        """Compute a deterministic cache key from prompt components."""
        return _hash_key(*parts)

    async def start(self, key: str) -> Optional[_InFlightRequest]:
        """Register an in-flight request.

        Returns an existing ``_InFlightRequest`` if one is already in-flight
        for this key (caller should await it), or ``None`` if this caller
        is the first (caller should run the work and call ``complete``).
        """
        await self._maybe_cleanup()

        async with self._lock:
            existing = self._in_flight.get(key)
            if existing is not None:
                logger.debug("Coalescer hit: key=%s, age=%.1fs", key, time.monotonic() - existing.created_at)
                return existing

            entry = _InFlightRequest()
            self._in_flight[key] = entry
            logger.debug("Coalescer registered: key=%s, total_in_flight=%d", key, len(self._in_flight))
            return None

    async def complete(self, key: str, result: str) -> None:
        """Mark an in-flight request as complete and wake all waiters."""
        async with self._lock:
            entry = self._in_flight.get(key)
            if entry is None:
                return
            entry.result = result
            entry.event.set()

    async def complete_error(self, key: str, error: BaseException) -> None:
        """Mark an in-flight request as failed and wake all waiters."""
        async with self._lock:
            entry = self._in_flight.get(key)
            if entry is None:
                return
            entry.error = error
            entry.event.set()

    async def remove(self, key: str) -> None:
        """Remove an in-flight entry (e.g. on client disconnect)."""
        async with self._lock:
            self._in_flight.pop(key, None)

    @property
    def in_flight_count(self) -> int:
        return len(self._in_flight)

    async def _maybe_cleanup(self) -> None:
        """Spawn a periodic cleanup if not already running."""
        if self._cleanup_task is not None and not self._cleanup_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(self._cleanup_stale())
        except RuntimeError:
            pass

    async def _cleanup_stale(self) -> None:
        """Remove entries older than TTL or exceeding max_entries."""
        await asyncio.sleep(5.0)
        async with self._lock:
            now = time.monotonic()
            stale = [
                k for k, v in self._in_flight.items()
                if now - v.created_at > self._ttl
            ]
            for k in stale:
                entry = self._in_flight.pop(k)
                if not entry.event.is_set():
                    entry.error = TimeoutError(f"Coalesced request timed out after {self._ttl}s")
                    entry.event.set()
                logger.debug("Coalescer evicted stale key=%s, age=%.1fs", k, now - entry.created_at)

            if len(self._in_flight) > self._max_entries:
                sorted_keys = sorted(
                    self._in_flight.keys(),
                    key=lambda k: self._in_flight[k].created_at,
                )
                for k in sorted_keys[: len(self._in_flight) - self._max_entries]:
                    entry = self._in_flight.pop(k)
                    if not entry.event.is_set():
                        entry.error = TimeoutError("Coalescer evicted (max_entries exceeded)")
                        entry.event.set()
                logger.warning("Coalescer evicted %d entries (max_entries=%d)", len(sorted_keys[:len(sorted_keys) - self._max_entries + len(stale)]), self._max_entries)


_coalescer: Optional[RequestCoalescer] = None
_coalescer_lock = threading.Lock()


def get_coalescer() -> RequestCoalescer:
    """Get or create the global request coalescer singleton."""
    global _coalescer
    if _coalescer is None:
        with _coalescer_lock:
            if _coalescer is None:
                _coalescer = RequestCoalescer()
    return _coalescer


def reset_coalescer() -> None:
    """Reset the global coalescer (for testing)."""
    global _coalescer
    with _coalescer_lock:
        _coalescer = None
