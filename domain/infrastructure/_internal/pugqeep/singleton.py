"""Process-global pugqeep Engine — the shared spawn surface.

One lazily-created engine for ``process``-kind background work, per the
routing table in ``docs/plans/2026-09-20-execution-consolidation.md``.
Callers get :func:`get_engine` instead of hand-rolling
``threading.Thread(target=...)``.

Design notes:
  - A default tree is created up front: ``Engine.dispatch()`` silently
    drops ungrouped processes when no tree exists.
  - ``run_background()`` runs the dispatch loop (daemon thread);
    ``_running`` is set synchronously here so a later ``get_engine()``
    can't race the loop thread's own startup.
  - If the engine was stopped externally (``Engine.stop()``,
    ``stop_all_pugqeep()`` in test teardowns), the next ``get_engine()``
    recreates it — a stopped engine's run loop is dead and would never
    dispatch new spawns.
  - A ``threading._register_atexit`` hook (not plain ``atexit``) stops the
    engine so forever-loops running in tree pools cannot hang interpreter
    exit. Ordering is the load-bearing part: ``concurrent.futures``
    registers ``_python_exit`` — which *joins every pool worker* — the
    same way, and those hooks run (reversed) inside ``threading._shutdown``
    **before** plain ``atexit`` handlers. Registering with plain ``atexit``
    would deadlock exit on a worker blocked in a forever-loop. ``engine``
    imports ``concurrent.futures`` first, so our hook registers after
    ``_python_exit`` and reversed order runs it first: ``Engine.stop()``
    cancels every process, the ``on_cancel`` hooks release the workers,
    then ``_python_exit`` joins them cleanly.
"""

from __future__ import annotations

import atexit
import threading

from .engine import Engine

_engine: Engine | None = None
_atexit_registered = False
_lock = threading.Lock()


def get_engine() -> Engine:
    """Return the process-global Engine, creating it on first use."""
    global _engine, _atexit_registered
    with _lock:
        if _engine is not None and not _engine._running:
            # Stopped externally (stop() / stop_all_pugqeep()); its run loop
            # is dead — drop it so a fresh one is created below.
            _engine = None
        if _engine is None:
            engine = Engine("shared")
            engine.tree("default")
            # Long-lived loops get their own tree (stage 3B, routing table in
            # docs/plans/2026-09-20-execution-consolidation.md): a forever-loop
            # holds a pool worker for its whole life, so it must not crowd
            # short work off the default tree's 4 workers. A live loop's stem
            # never clears, so the stem cap carries restart-cycle headroom.
            engine.tree("loops", max_stems=32, pool_workers=16)
            engine._running = True  # close the race with run_background's thread
            engine.run_background()
            _engine = engine
            if not _atexit_registered:
                # threading._register_atexit, NOT atexit.register — see the
                # module docstring: plain atexit runs after _python_exit has
                # already joined the pool workers, which deadlocks exit on a
                # blocked forever-loop. Registered after cf's hook (engine
                # imports concurrent.futures first) → reversed order → ours
                # runs first. Fallback only for interpreters without it.
                register = getattr(threading, "_register_atexit", None) or atexit.register
                register(_shutdown)
                _atexit_registered = True
        return _engine


def _shutdown() -> None:
    """Stop the shared engine at interpreter exit (idempotent)."""
    global _engine
    with _lock:
        engine, _engine = _engine, None
    if engine is None:
        return
    engine.stop()
    if engine._signal_handlers_installed:
        engine.restore_signal_handlers()


class ThreadHandle:
    """Thread-shaped view of an engine Process (routing kind: process).

    Keeps the ``threading.Thread`` surface that loop consumers are written
    against — ``is_alive()``, ``join(timeout)``, ``daemon``, ``name`` — while
    the work itself runs on the shared engine instead of a raw thread at the
    call site (stage 3B of ``docs/plans/2026-09-20-execution-consolidation.md``).

    ``daemon`` is the lifecycle contract, not a thread flag: an engine-owned
    task never blocks interpreter exit, because the singleton's shutdown hook
    (``threading._register_atexit`` — it must beat ``_python_exit``'s worker
    join, see the module docstring) stops the engine, which cancels every
    non-done process and fires its cancel callbacks before
    ``concurrent.futures`` joins the pool worker.
    """

    __slots__ = ("_engine", "_proc")

    def __init__(self, engine: Engine, proc) -> None:
        self._engine = engine
        self._proc = proc

    @property
    def daemon(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return str(self._proc.name)

    def is_alive(self) -> bool:
        """True until the task completes, fails, or is cancelled."""
        return not self._proc.is_done

    def join(self, timeout: float | None = None) -> None:
        """Wait for the task to finish, bounded by *timeout*."""
        if timeout is None:
            self._engine.wait_for(self._proc.id)
        else:
            self._engine.wait_for(self._proc.id, timeout=timeout)


def spawn_loop(
    fn,
    *args,
    name: str = "",
    stop=None,
    **kwargs,
) -> ThreadHandle:
    """Spawn a background task on the shared engine's ``loops`` tree.

    The ``process`` kind for thread-shaped call sites (stage 3B): the caller
    keeps the Thread API it already consumes (``is_alive``/``join``); the work
    becomes a registry-visible engine process.

    ``stop`` releases the task when the process is cancelled (engine stop /
    interpreter exit — the daemon-equivalent guarantee): pass a
    ``threading.Event`` to set, or a zero-arg callable for flags that are not
    events. The loop fn must exit once its stop signal fires — the engine
    cancels the *process*, it cannot interrupt blocking work that ignores it.
    """
    engine = get_engine()
    proc = engine.spawn(fn, *args, name=name, tree="loops", **kwargs)
    if stop is not None:
        if isinstance(stop, threading.Event):
            proc.on_cancel(lambda _p: stop.set())
        else:
            proc.on_cancel(lambda _p: stop())
    return ThreadHandle(engine, proc)
