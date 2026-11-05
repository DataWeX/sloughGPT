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
  - An ``atexit`` hook stops the engine so forever-loops running in tree
    pools cannot hang interpreter exit — pool workers are non-daemon
    (verified: ``concurrent.futures`` joins them at exit), and
    ``Engine.stop()`` cancels every process, which fires the ``on_cancel``
    hooks each forever-loop registers to release its worker.
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
            engine._running = True  # close the race with run_background's thread
            engine.run_background()
            _engine = engine
            if not _atexit_registered:
                atexit.register(_shutdown)
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
