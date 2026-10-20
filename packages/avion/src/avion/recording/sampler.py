"""Frame sampler — 2–5 fps capture on its OWN sync thread.

SYNC CORE: the sampler is a plain ``threading.Thread`` that owns cadence,
encoding and disk writes — no asyncio task, no event-loop coupling inside
avion's recording core. The capture callable is injected: the CLI bridges
the async backend screenshot onto its running loop
(``asyncio.run_coroutine_threadsafe`` at the seam), tests pass a plain
sync lambda.

RING BUFFER → DISK: captured frames land in a bounded deque that is
drained to disk every cycle. When disk or capture lags, the OLDEST pending
frame is dropped and counted (``dropped_frames`` in the manifest) — a
recording never grows an unbounded backlog and never stalls the page.

ONE CLOCK: every frame is stamped with the shared RecordingClock
(``time.monotonic_ns()`` origin), the same clock that stamps event
``_t_ns`` — that pairing is what the viewer aligns on.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable

from avion.recording.artifact import RecordingArtifact
from avion.recording.clock import RecordingClock

MIN_FPS = 2.0
MAX_FPS = 5.0


def clamp_fps(fps: float) -> float:
    """The sampler contract is 2–5 fps — anything else is clamped, not rejected."""
    try:
        value = float(fps)
    except (TypeError, ValueError):
        return 3.0
    return min(MAX_FPS, max(MIN_FPS, value))


class FrameSampler:
    """Samples frames at a fixed cadence into an artifact, on one thread.

    Usage::

        sampler = FrameSampler(
            capture=lambda: png_bytes,      # sync, blocking, thread-safe
            artifact=art,
            clock=clock,
            fps=3.0,
            crop=art.crop,                  # capture target (None = full)
        )
        sampler.start()
        ...
        stats = sampler.stop()              # joins; drains the ring first
    """

    def __init__(
        self,
        *,
        capture: Callable[[], bytes],
        artifact: RecordingArtifact,
        clock: RecordingClock,
        fps: float = 3.0,
        crop: dict[str, float] | None = None,
        ring: int = 16,
    ) -> None:
        self._capture = capture
        self._artifact = artifact
        self._clock = clock
        self._fps = clamp_fps(fps)
        # Cropping happens in the artifact at write time (it owns the target);
        # crop kept here only so callers can see what the sampler was told.
        self.crop = dict(crop) if crop else None
        self._ring: deque[tuple[int, float, bytes]] = deque(maxlen=max(1, int(ring)))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._written = 0
        self._dropped = 0
        self._errors = 0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def written(self) -> int:
        return self._written

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def errors(self) -> int:
        return self._errors

    def start(self) -> None:
        """Start sampling; the first frame is captured immediately (t≈0)."""
        if self._thread is not None:
            raise RuntimeError("FrameSampler already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="avion-frame-sampler", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 15.0) -> dict[str, Any]:
        """Signal, join, drain — returns the stats stamped into the manifest."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._drain()  # idempotent: the run loop drains before breaking too
        return {
            "written": self._written,
            "dropped": self._dropped,
            "errors": self._errors,
            "fps": self._fps,
        }

    # ── internals (thread side) ────────────────────────────────────────

    def _run(self) -> None:
        interval_ns = int(1_000_000_000 / self._fps)
        due_ns = self._clock.now_ns()  # first frame immediately
        while True:
            self._drain()
            if self._stop.is_set():
                return
            now_ns = self._clock.now_ns()
            if now_ns < due_ns:
                self._stop.wait((due_ns - now_ns) / 1_000_000_000)
                continue  # re-check stop, drain, then decide (no capture on shutdown)
            try:
                raw = self._capture()
            except Exception:
                raw = None
                with self._lock:
                    self._errors += 1
            if raw:
                if len(self._ring) == self._ring.maxlen:
                    self._ring.popleft()
                    with self._lock:
                        self._dropped += 1
                self._ring.append((self._clock.now_ns(), time.time(), raw))
            # Schedule from NOW: a slow capture skips missed ticks instead of
            # bursting to catch up (cadence is a ceiling, not a debt).
            now_ns = self._clock.now_ns()
            due_ns = now_ns + interval_ns if now_ns > due_ns else due_ns + interval_ns

    def _drain(self) -> None:
        while self._ring:
            t_ns, t_wall, raw = self._ring.popleft()
            try:
                self._artifact.write_frame(raw, t_ns=t_ns, t_wall=t_wall)
            except Exception:
                with self._lock:
                    self._errors += 1  # one bad frame never kills the sampler
                continue
            with self._lock:
                self._written += 1
