"""run_on_engine — the async seam onto the pugqeep Engine (stage 4).

Boundary statement (``docs/plans/2026-09-20-execution-consolidation.md``):
the host is async, the core is sync, and long-running compute crosses
between them through the shared Engine — never a bare ``asyncio.to_thread``
and never inline on the loop.

Duration rule: long/indeterminate compute (training, inference,
captioning, model loads, shell) goes through :func:`run_on_engine`;
micro-I/O (``os.unlink``, config reads) stays on ``asyncio.to_thread``.

Design notes:
  - Lives OUTSIDE ``pugqeep/``: PGQ must stay event-loop-free (AGENTS.md
    execution philosophy) — this module is the async *host* of a sync
    core, so the ``asyncio`` import belongs here, not there.
  - Resolution is callback-based (``on_complete``/``on_fail``/``on_cancel``
    → ``loop.call_soon_threadsafe``), never ``proc._done_event.wait()``
    on the loop thread — blocking the loop is the starvation defect this
    seam exists to remove.
  - Exception fidelity: the engine stores only ``str(e)`` on failure, so
    the callable is wrapped to capture the original exception object and
    the seam re-raises *that*. Call sites keep their ``except
    InterruptedError`` (training cancel) exactly as written under
    ``asyncio.to_thread``.
  - Race: a process finishing between spawn and callback registration
    fires nothing — the ``is_done`` check right after registration
    settles it. Either ordering is safe (``complete()`` iterates the live
    callback list; a callback registered after completion is covered by
    the check).
  - Timeout gives up the await AND cancels the process: the compute must
    not keep burning a pool worker after the caller walked away.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from .pugqeep.engine import Process, ProcessStatus
from .pugqeep.singleton import get_engine

__all__ = ["EngineCancelled", "run_on_engine"]


class EngineCancelled(RuntimeError):
    """The awaited engine process was cancelled (engine stop, external
    ``cancel_process``, interpreter shutdown) before producing a result."""


async def run_on_engine[T](
    fn: Callable[..., T],
    *args: Any,
    name: str = "",
    tree: str = "default",
    timeout: float | None = None,
    **kwargs: Any,
) -> T:
    """Run *fn* on the shared Engine and await its result, loop-safely.

    Drop-in for ``await asyncio.to_thread(fn, ...)`` for long/indeterminate
    compute: same result/exception contract, but the work is a
    registry-visible Engine process (cancellable, pool-bounded, observable)
    instead of an anonymous executor thread.

    Args:
        fn: Sync callable. Runs on an Engine pool worker, never on the loop.
        *args, **kwargs: Passed through to *fn*.
        name: Process name (registry/observability); defaults to fn's name.
        tree: Engine tree — ``"default"`` for short/medium compute,
            ``"loops"`` for forever-style work (see singleton.get_engine).
        timeout: Seconds before the await raises ``TimeoutError``; the
            process is cancelled so the compute does not outlive the wait.

    Returns:
        Whatever *fn* returns.

    Raises:
        The original exception *fn* raised (type preserved).
        EngineCancelled: process cancelled before completing.
        TimeoutError: *timeout* elapsed; process cancelled.
    """
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[T] = loop.create_future()
    engine = get_engine()
    captured: list[BaseException] = []

    def _guarded(*a: Any, **k: Any) -> Any:
        # Capture the ORIGINAL exception: Engine.fail() keeps only str(e),
        # and call sites (training cancel!) match on exception type.
        try:
            return fn(*a, **k)
        except BaseException as e:  # noqa: BLE001 — re-raised below
            captured.append(e)
            raise

    proc = engine.spawn(
        _guarded,
        *args,
        name=name or getattr(fn, "__name__", ""),
        tree=tree,
        **kwargs,
    )

    def _settle(p: Process) -> None:
        if fut.done():
            return
        if p.status is ProcessStatus.COMPLETED:
            fut.set_result(p.result)
        elif p.status is ProcessStatus.FAILED:
            # Original exception wins; engine-text fallback covers
            # failures that never entered _guarded (engine-side).
            exc: BaseException = (
                captured[0]
                if captured
                else RuntimeError(p.error or f"engine process '{p.name}' failed")
            )
            fut.set_exception(exc)
        else:  # CANCELLED (or an engine status this seam doesn't expect)
            fut.set_exception(EngineCancelled(f"engine process '{p.name or p.id}' was cancelled"))

    def _resolve(p: Process) -> None:
        # Engine callbacks fire on pool/dispatch threads → hop to the loop.
        if not fut.done():
            loop.call_soon_threadsafe(_settle, p)

    proc.on_complete(_resolve)
    proc.on_fail(_resolve)
    proc.on_cancel(_resolve)
    if proc.is_done:
        # Finished before/while callbacks were registered — settle directly
        # (we are on the loop thread here, no threadsafe hop needed).
        _settle(proc)

    if timeout is None:
        return await fut
    try:
        return await asyncio.wait_for(fut, timeout)
    except TimeoutError:
        engine.cancel_process(proc.id)
        raise
