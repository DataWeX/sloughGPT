"""External-command process kind — ``Engine.spawn_command`` supervision.

The external-command flavor of the ``process`` routing kind
(docs/plans/2026-09-20-execution-consolidation.md): eager start (no
dispatch latency, no pool worker), live streams, raw-Popen lifecycle
(``poll``/``returncode``/``terminate``), engine-side reaping (status,
metrics, ``timeout``) independent of the caller polling.
"""

import sys

import pytest

from domain.infrastructure._internal.pugqeep.engine import (
    CommandProcess,
    Engine,
    ProcessStatus,
)

PY = sys.executable


@pytest.fixture
def engine():
    eng = Engine("cmdtest")
    yield eng
    eng.stop()


def test_spawn_command_completes_without_run_loop(engine):
    """Eager start: the command runs and finalizes with no dispatch/run loop."""
    proc = engine.spawn_command([PY, "-c", "print('hi')"], name="echo")
    assert isinstance(proc, CommandProcess)
    assert engine._pending == []  # never queued for dispatch
    assert proc in engine.list_processes()
    rc = proc.wait(timeout=10)
    assert rc == 0
    assert proc.status == ProcessStatus.COMPLETED
    assert proc.result == 0
    assert proc.stdout.read() == "hi\n"


def test_nonzero_exit_fails_with_code(engine):
    proc = engine.spawn_command([PY, "-c", "import sys; sys.exit(3)"])
    rc = proc.wait(timeout=10)
    assert rc == 3
    assert proc.status == ProcessStatus.FAILED
    assert "exited with code 3" in (proc.error or "")
    # Second finalize is idempotent — first caller wins, status sticks.
    assert proc.poll() == 3
    assert proc.status == ProcessStatus.FAILED


def test_live_stream_and_lifecycle(engine):
    """Output is readable *while* the child runs; poll/terminate drive it."""
    proc = engine.spawn_command(
        [PY, "-u", "-c", "import time; print('ready', flush=True); time.sleep(30)"],
        name="sleeper",
    )
    try:
        assert proc.stdout.readline().strip() == "ready"
        assert proc.poll() is None  # still alive — not post-hoc capture
        assert proc.is_alive
        proc.terminate()
        proc.wait(timeout=10)
        assert not proc.is_alive
        assert proc.returncode is not None
    finally:
        if proc.is_alive:
            proc.kill()
            proc.wait(timeout=5)


def test_engine_stop_terminates_child(engine):
    """Engine.stop() (and atexit via the singleton) must not orphan the child."""
    proc = engine.spawn_command([PY, "-c", "import time; time.sleep(30)"], name="doomed")
    engine.stop()
    assert proc.status == ProcessStatus.CANCELLED
    assert proc.returncode is not None  # reaped, not a zombie
    assert not proc.is_alive


def test_timeout_enforced_by_run_loop(engine):
    """The run loop, not the caller, enforces spawn_command(timeout=...)."""
    engine.run_background()
    proc = engine.spawn_command(
        [PY, "-c", "import time; time.sleep(30)"], name="slow", timeout=0.5
    )
    done = engine.wait_for(proc.id, timeout=15)
    assert done is not None
    assert proc.status == ProcessStatus.FAILED
    assert "timed out" in (proc.error or "")
    assert not proc.is_alive


def test_spawn_command_rejects_bad_argv(engine):
    with pytest.raises(TypeError):
        engine.spawn_command("echo hi")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        engine.spawn_command([])


def test_missing_executable_raises_and_does_not_register(engine):
    with pytest.raises(OSError):
        engine.spawn_command(["definitely-not-a-real-binary-xyz"])
    assert engine.list_processes() == []


def test_singleton_engine_supervises_commands():
    """dev.py's actual path: get_engine().spawn_command(...)."""
    from domain.infrastructure._internal.pugqeep.singleton import get_engine

    eng = get_engine()
    proc = eng.spawn_command([PY, "-c", "print('ok')"], name="singleton-cmd")
    assert proc.wait(timeout=10) == 0
    assert proc in eng.list_processes()
    # Teardown: conftest's stop_all_pugqeep() stops the shared engine.
