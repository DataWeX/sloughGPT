"""Recording session — the lifecycle that binds clock, artifact and sampler.

`avion record` drives a live CDP backend while a
session captures frames (own sync thread), stamps events on ONE clock and
closes into a viewer-ready artifact. `avion stop`/`view` re-derive boxes from
what this session persisted.

DO NOT DELETE. This is the only place that owns the record-time lifecycle:
the CLI stays a thin shell over begin()/finalize(), and every timing decision
(the shared clock origin, the element target resolved ONCE at record start,
the sampler's loop bridge) lives here so tests can drive it without a CLI.

The async/sync seam: begin/finalize run on the event loop, the FrameSampler
runs on its own thread. The bridge is explicit — the sampler's capture
callable schedules ``backend.screenshot()`` onto the running loop with
``asyncio.run_coroutine_threadsafe`` and waits (bounded). No asyncio inside
the sampler core; no thread-unsafe backend calls either.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from pathlib import Path
from typing import Any, Callable

from avion.core.element import ElementLocator
from avion.recording.artifact import (
    RecordingArtifact,
    frame_capabilities,
    new_recording_id,
)
from avion.recording.boxes import dom_sweep, rebuild_boxes, resolve_rect
from avion.recording.clock import RecordingClock
from avion.recording.sampler import FrameSampler, clamp_fps

_CAPTURE_TIMEOUT = 15.0  # one screenshot over the loop; bounded, never forever

# finalize() must never wait on a capture the loop can no longer service: the
# bridge polls in these slices and aborts as soon as _closing is set, so a
# screenshot stuck in flight at teardown costs one slice, not _CAPTURE_TIMEOUT.
_CAPTURE_SLICE_S = 0.05


def build_locator(value: str, by: str) -> ElementLocator | None:
    """Locator from ``--by`` semantics — None when the strategy is unknown
    (the session degrades to no target rect rather than SystemExit-ing:
    ``--by`` is validated at the parser layer, this is belt-and-braces)."""
    from avion.core.element import Selector, SelectorStrategy

    if by == "css":
        return ElementLocator.css(value)
    if by == "text":
        return ElementLocator.text(value)
    if by == "testid":
        return ElementLocator.test_id(value)
    if by == "label":
        return ElementLocator.label(value)
    if by == "role":
        return ElementLocator.role(value)
    if by == "xpath":
        return ElementLocator(selectors=(Selector(SelectorStrategy.XPATH, value),))
    return None


def parse_rect(text: str) -> dict[str, float] | None:
    """``--rect "x,y,width,height"`` → crop dict; anything unparseable → None
    (the caller degrades to a full-frame capture with the reason recorded —
    a typo'd rect never aborts a recording silently)."""
    parts = [p.strip() for p in str(text).replace(" ", ",").split(",") if p.strip()]
    if len(parts) != 4:
        return None
    try:
        x, y, w, h = (float(p) for p in parts)
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return {"x": x, "y": y, "width": w, "height": h}


class RecordingSession:
    """One recording run: begin() → (CLI drives, events stamped) → finalize().

    Usage::

        session = RecordingSession(out, fps=3.0, target="full")
        await session.begin(backend, recorder, url="http://…")
        … recorder.record(…, data=session.stamp()) …
        summary = session.finalize(recorder)
    """

    def __init__(
        self,
        out: Path | str,
        *,
        fps: float = 3.0,
        target: str = "full",
        rect: str | None = None,
        selector: str | None = None,
        by: str = "css",
        recording_id: str | None = None,
        clock: RecordingClock | None = None,
    ) -> None:
        if target not in ("full", "rect", "element"):
            raise ValueError(f"unknown capture target: {target!r} (full|rect|element)")
        if target == "rect" and not rect:
            raise ValueError("--target rect needs --rect 'x,y,width,height'")
        if target == "element" and not selector:
            raise ValueError("--target element needs --selector")
        self.out = Path(out)
        self.fps = clamp_fps(fps)
        self.target_kind = target
        self.rect_text = rect
        self.selector = selector
        self.by = by
        self.recording_id = recording_id or new_recording_id()
        self.clock = clock or RecordingClock()
        self.artifact: RecordingArtifact | None = None
        self._sampler: FrameSampler | None = None
        # Teardown gate: set the instant finalize() starts so an in-flight
        # bridge capture stops waiting on a loop that is about to join the
        # sampler thread (see _bridge_capture).
        self._closing = threading.Event()

    # ── lifecycle ──────────────────────────────────────────────────────

    async def begin(
        self,
        backend: Any,
        recorder: Any = None,
        *,
        url: str | None = None,
    ) -> RecordingArtifact:
        """Open the artifact, sweep the DOM, resolve the target, start sampling.

        The capture target is resolved ONCE here (element rect via bounding
        box / in-page rect — never re-probed per frame: a scrolled page would
        produce frames and boxes in different spaces mid-recording).
        """
        self._closing.clear()  # a reused session records again cleanly
        if url:
            await backend.navigate(url)
            open_t_ns = int(self.clock.now_ns())  # stamp BEFORE the sweep so
            try:  # the sweep segment isn't cut by its own page load
                await backend.wait_for_timeout(400)
            except Exception:
                pass
        else:
            open_t_ns = 0

        crop: dict[str, float] | None = None
        target: dict[str, Any] = {"kind": self.target_kind}
        if self.target_kind == "rect":
            crop = parse_rect(self.rect_text or "")
            if crop is None:
                target["degraded"] = f"unparseable --rect {self.rect_text!r}: full frame"
        elif self.target_kind == "element":
            crop = await self.resolve_target_rect(backend)
            if crop is None:
                target["degraded"] = "element rect unresolved: full frame"
            target["selector"] = self.selector
            target["by"] = self.by
        if crop is not None:
            target["crop"] = crop

        sweep = await dom_sweep(backend)
        sweep_t_ns = self.clock.now_ns()

        self.artifact = RecordingArtifact.create(
            self.out,
            recording_id=self.recording_id,
            fps=self.fps,
            target=target,
            clock=self.clock,
            sweep=sweep,
            sweep_t_ns=sweep_t_ns,
            capabilities=frame_capabilities(),
        )

        loop = asyncio.get_running_loop()
        self._sampler = FrameSampler(
            capture=self._bridge_capture(backend, loop),
            artifact=self.artifact,
            clock=self.clock,
            fps=self.fps,
            crop=self.artifact.crop,
        )
        self._sampler.start()

        if recorder is not None:
            self.attach(recorder)
            if url:
                # The open is part of the timeline: stamped at navigate time
                # (before the sweep) so the sweep segment starts on it rather
                # than being cut by its own page load (derive_boxes keeps
                # navigations strictly AFTER sweep_t_ns as segment ends).
                from avion.events.models import EventType

                recorder.record(EventType.NAVIGATE, name=url, data={"_t_ns": open_t_ns})
        return self.artifact

    def attach(self, recorder: Any) -> None:
        """Persist events incrementally — a killed record process still leaves
        an events.json a human (or ``avion stop``) can finish."""
        if self.artifact is None:
            raise RuntimeError("attach() before begin()")
        recorder.on_event(lambda _e: self.persist(recorder))

    def persist(self, recorder: Any) -> None:
        """Rewrite events.json from the recorder view (atomic, derived file)."""
        if self.artifact is None:
            return
        try:
            self.artifact.write_events(recorder.to_dict())
        except OSError:
            pass  # best-effort: a full disk must not kill the driving loop

    def stamp(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Event data carrying the shared-clock stamp ``_t_ns``.

        Everything the viewer aligns on goes through here: frames and events
        are then comparable by subtraction on ONE monotonic axis (see the
        clock module's desync note for why wall timestamps don't qualify).
        """
        payload = dict(data or {})
        payload["_t_ns"] = int(self.clock.now_ns())
        return payload

    async def resolve_target_rect(self, backend: Any) -> dict[str, float] | None:
        """Capture-target rect for ``--target element`` (structure tier only)."""
        locator = build_locator(self.selector or "", self.by)
        if locator is None:
            return None
        return await resolve_rect(backend, locator)

    async def event_rect(self, backend: Any, locator: Any) -> dict[str, float] | None:
        """Per-event rect for click/fill (recorded into event data as "rect")."""
        return await resolve_rect(backend, locator)

    def finalize(
        self,
        recorder: Any = None,
        *,
        completed: bool = True,
    ) -> dict[str, Any]:
        """Stop the sampler, persist events, derive boxes, close the manifest.

        Sync by design: it only joins a thread and writes files — safe to
        call from the CLI's finally block with no loop pumping needed.
        """
        if self.artifact is None:
            return {"ok": False, "error": "session never began"}
        # Open the gate FIRST: any capture waiting on the loop aborts within
        # one slice, so the sampler's stop().join() below is never held hostage
        # by a screenshot that can no longer be serviced (finalize runs on the
        # loop thread — it must not block on work that needs the loop).
        self._closing.set()
        stats: dict[str, Any] = {}
        if self._sampler is not None:
            stats = self._sampler.stop()
        if recorder is not None:
            self.persist(recorder)
        summary = rebuild_boxes(
            self.artifact,
            dropped_frames=int(stats.get("dropped", 0) or 0),
            completed=completed,
        )
        summary["ok"] = True
        summary["frames_written"] = int(stats.get("written", 0) or 0)
        summary["capture_errors"] = int(stats.get("errors", 0) or 0)
        summary["path"] = str(self.artifact.root)
        return summary

    # ── sampler bridge (thread side → loop) ────────────────────────────

    def _bridge_capture(self, backend: Any, loop: asyncio.AbstractEventLoop) -> Callable[[], bytes]:
        """Sync capture for the sampler thread: schedule on the loop, wait.

        Any failure raises into FrameSampler's handler (counted, retried next
        tick) — a page that stops answering screenshots degrades the frame
        count, it never wedges the recording.
        """

        def capture() -> bytes:
            try:
                coro = backend.screenshot()
            except Exception:
                raise  # sampler counts it; next tick retries
            try:
                fut = asyncio.run_coroutine_threadsafe(coro, loop)
            except RuntimeError:
                coro.close()  # loop gone (shutdown race): don't leak a coroutine
                raise
            data = self._await_capture(fut)
            if not data:
                raise RuntimeError("empty screenshot")
            return bytes(data)

        return capture

    def _await_capture(self, fut: concurrent.futures.Future[Any]) -> Any:
        """Wait for a bridge capture in slices — abortable by finalize().

        A plain ``fut.result(timeout=…)`` here would race the sampler's
        stop().join(): finalize() runs on the loop thread, so a screenshot
        still in flight when teardown starts has nobody left to complete it,
        and the join would stall for the full timeout. Polling in short
        slices lets the ``_closing`` gate cut the wait to one slice; the
        cancel propagates (task cancellation reaches the coroutine when the
        loop next runs it) instead of leaking a scheduled task.
        """
        deadline = time.monotonic() + _CAPTURE_TIMEOUT
        while True:
            if self._closing.is_set():
                fut.cancel()
                raise RuntimeError("capture aborted: session closing")
            try:
                return fut.result(timeout=_CAPTURE_SLICE_S)
            except concurrent.futures.TimeoutError:
                pass
            if time.monotonic() >= deadline:
                fut.cancel()
                raise TimeoutError(f"screenshot exceeded {_CAPTURE_TIMEOUT}s")
