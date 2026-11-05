"""spawn_loop — thread-shaped process kind (stage 3B batch 2).

The handle contract loop consumers are written against
(``is_alive``/``join``/``daemon``/``name``), the dedicated loops tree
(forever-loops must not crowd short work off the default tree's pool),
stop-signal wiring, and the load-bearing interpreter-exit path: a
forever-loop on a pool worker must not block exit — the singleton's
atexit hook cancels it and fires its stop *before*
``concurrent.futures`` joins the worker.
"""

import subprocess
import sys
import threading
import time

from domain.infrastructure._internal.pugqeep.singleton import (
    get_engine,
    spawn_loop,
)

PY = sys.executable


def test_get_engine_creates_loops_tree():
    """Loop capacity is structural: default (short work) + loops (forever work)."""
    eng = get_engine()
    assert "default" in eng.list_trees()
    assert "loops" in eng.list_trees()


def test_handle_thread_surface():
    """The Thread API consumers call: name/daemon/is_alive/join."""
    ev = threading.Event()
    handle = spawn_loop(ev.wait, 60, name="surface-loop", stop=ev)
    try:
        assert handle.name == "surface-loop"
        assert handle.daemon is True  # lifecycle contract, see ThreadHandle
        assert handle.is_alive()  # pending or running — not done either way
        handle.join(timeout=0.05)  # bounded: returns while still alive
        assert handle.is_alive()
    finally:
        ev.set()
        handle.join(timeout=10)
    assert not handle.is_alive()


def test_callable_stop_signal_releases_loop():
    """stop= (zero-arg callable) for flags that aren't Events (continual shape)."""
    flag = {"running": True}
    eng = get_engine()

    def _spin():
        while flag["running"]:
            time.sleep(0.02)

    handle = spawn_loop(_spin, name="callable-stop-loop", stop=lambda: flag.__setitem__("running", False))
    try:
        proc = next(p for p in eng.list_processes() if p.name == "callable-stop-loop")
        assert eng.cancel_process(proc.id) >= 1  # engine cancel → on_cancel → flag flips
        handle.join(timeout=10)
        assert not handle.is_alive()
        assert flag["running"] is False
    finally:
        flag["running"] = False
        handle.join(timeout=10)


def test_forever_loop_does_not_block_interpreter_exit():
    """The atexit chain: engine stop cancels the proc, the wired stop event
    lets the fn return, and the pool worker is free before
    ``concurrent.futures`` joins it at exit."""
    code = (
        "import threading, time\n"
        "from domain.infrastructure._internal.pugqeep.singleton import spawn_loop\n"
        "ev = threading.Event()\n"
        "spawn_loop(ev.wait, 600, name='exit-loop', stop=ev)\n"
        "time.sleep(0.5)  # let the run loop dispatch it onto the pool\n"
    )
    proc = subprocess.run(  # noqa: S603
        [PY, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"interpreter hung or crashed: {proc.stderr[-2000:]}"
