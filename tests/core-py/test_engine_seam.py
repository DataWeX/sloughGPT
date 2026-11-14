"""run_on_engine — the async seam onto the pugqeep Engine (stage 4).

The boundary from ``docs/plans/2026-09-20-execution-consolidation.md``:
long/indeterminate compute leaves the event loop through the shared
Engine (registry-visible, cancellable, pool-bounded) instead of a bare
``asyncio.to_thread``, while the await itself stays loop-safe.

Contracts proven here:
  - result / kwargs pass-through parity with ``asyncio.to_thread``
  - exception *type* fidelity (callers catch what they caught before —
    the engine only stores ``str(e)``, so the seam captures the original)
  - the loop keeps ticking while the await is in flight (the whole
    point of the seam: no starvation)
  - external cancel and timeout resolve the await instead of hanging it
  - the fast-path race: a process that finishes before the callbacks are
    registered still settles the future
"""

import asyncio
import threading
import time

import pytest

from domain.infrastructure._internal.engine_seam import (
    EngineCancelled,
    run_on_engine,
)
from domain.infrastructure._internal.pugqeep.engine import ProcessStatus
from domain.infrastructure._internal.pugqeep.singleton import get_engine


@pytest.fixture(autouse=True)
def _restore_event_loop_after_test():
    """Heal the MainThread loop after every test in this file.

    Each test drives the seam through a bare ``asyncio.run(...)``, which on
    exit leaves the process policy at ``set_event_loop(None)``. The core-py
    conftest heals that only at the *next core-py* test setup, so when this
    file is the last core-py file in a run, the unset loop leaks into later
    suites that read it (apps/server has no healer — proven by
    ``tests/server/test_chat_state_seam.py`` + consciousness failing on
    pristine origin/main, 2026-11-07). Same contract as the conftest
    healer, applied at teardown so change-scoped runs that select this file
    together with apps/ tests stay green.
    """
    yield
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def test_returns_result():
    async def _main():
        return await run_on_engine(lambda: 42, name="seam-result")

    assert asyncio.run(_main()) == 42


def test_args_kwargs_passthrough():
    async def _main():
        def _add(a, b, *, c=0):
            return a + b + c

        return await run_on_engine(_add, 1, 2, c=3, name="seam-args")

    assert asyncio.run(_main()) == 6


def test_work_runs_off_the_event_loop_thread():
    """The fn must not execute on the loop thread (to_thread parity)."""
    seen: dict[str, object] = {}

    async def _main():
        def _where():
            seen["thread"] = threading.current_thread()
            return 1

        await run_on_engine(_where, name="seam-offloop")
        return None

    asyncio.run(_main())
    assert seen["thread"] is not threading.main_thread()


def test_loop_keeps_ticking_while_awaiting():
    """Core invariant: awaiting engine work must not starve the loop.

    A naive blocking wait would stall the ticker; the seam resolves via
    ``call_soon_threadsafe``, so concurrent tasks keep running.
    """
    ticks = {"n": 0}
    started = threading.Event()

    async def _ticker():
        while True:
            await asyncio.sleep(0.02)
            ticks["n"] += 1

    async def _main():
        ticker = asyncio.ensure_future(_ticker())
        try:

            def _slow():
                started.set()
                time.sleep(0.3)
                return "done"

            result = await run_on_engine(_slow, name="seam-slow")
            assert result == "done"
            assert ticks["n"] >= 5  # loop advanced during the 0.3s await
        finally:
            ticker.cancel()

    asyncio.run(_main())
    assert started.is_set()


def test_exception_type_preserved():
    """``except InterruptedError`` at call sites must keep working: the
    engine stores only ``str(e)``, so the seam re-raises the original."""

    async def _main():
        def _boom():
            raise InterruptedError("Training cancelled by user")

        with pytest.raises(InterruptedError, match="cancelled by user"):
            await run_on_engine(_boom, name="seam-raise")

    asyncio.run(_main())


def test_generic_failure_raises_runtime_error_with_engine_error():
    """Failures not originating in fn (engine-side) surface the engine's
    stored error text."""

    async def _main():
        def _slow():
            time.sleep(0.2)
            return 1

        task = asyncio.ensure_future(run_on_engine(_slow, name="seam-cancelme"))
        await asyncio.sleep(0.05)  # let it dispatch and start running
        eng = get_engine()
        proc = next(p for p in eng.list_processes() if p.name == "seam-cancelme")
        assert eng.cancel_process(proc.id) >= 1
        with pytest.raises(EngineCancelled):
            await task

    asyncio.run(_main())


def test_timeout_cancels_the_process():
    async def _main():
        def _forever():
            time.sleep(5)
            return "never"

        with pytest.raises(TimeoutError):
            await run_on_engine(_forever, name="seam-timeout", timeout=0.1)
        # the compute must not keep running after the await gave up
        await asyncio.sleep(0.1)
        eng = get_engine()
        proc = next(p for p in eng.list_processes() if p.name == "seam-timeout")
        assert proc.status is ProcessStatus.CANCELLED

    asyncio.run(_main())


def test_fast_fn_settles_despite_callback_race():
    """A process completing between spawn and callback registration must
    still resolve the future (is_done fallback)."""

    async def _main():
        results = []
        for i in range(20):  # hammer the race window
            results.append(await run_on_engine(lambda i=i: i, name=f"seam-fast{i}"))
        return results

    assert asyncio.run(_main()) == list(range(20))
