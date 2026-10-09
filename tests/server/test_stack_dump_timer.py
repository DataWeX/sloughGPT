"""Regression tests for the ``SLO_DUMP_STACKS`` periodic stack-dump timer.

Root cause (2026-10-09): faulthandler's native ``dump_traceback_later``
watchdog dumps every thread's frame chain **without the GIL** (CPython's
``faulthandler_thread`` calls ``_Py_DumpTracebackThreads`` directly).  Since
CPython 3.11 interpreter frames live on each thread's *native C stack*, the
GIL-less walk races frame push/pop and can read a reused stack slot:
``f_executable`` comes back as garbage (0/1/2) and the dumper itself SEGVs at
``dump_traceback+0xbc``.  On slough-api this killed the service 10 times
(Oct 8-9 2026); each crash left the API unresponsive for ~4-6 minutes while
systemd-coredump compressed the ~1 GB core — which is why UI training jobs
"wouldn't start" roughly half the time.

The fix defaults to a **GIL-held** dumper thread (``faulthandler.dump_traceback``
runs under our GIL, so frame chains are frozen while we walk them) and keeps
the native gilless timer only behind ``SLO_DUMP_STACKS_GILLESS=1`` for when
you are actively debugging a total-GIL wedge and accept the crash risk.

These tests pin: default-off, mode selection, actual dump emission, and
cancellation.
"""

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = REPO_ROOT / "apps/api/server"
CORE_DIR = REPO_ROOT / "packages/core-py"
for _p in (SERVER_DIR, CORE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from apps.api.server import main as server_main  # noqa: E402

# faulthandler-style marker prefix (exact for integer intervals in prod:
# "Timeout (0:00:30)!"), kept so journal greps like `Timeout (0:00` still work.
_MARKER = "Timeout (0:00:00"


def _dumper_thread():
    """The gil-held dumper thread, or None (pre-implementation safe)."""
    return getattr(server_main, "_stack_dump_thread", None)


@pytest.fixture(autouse=True)
def _clean_stack_dump_state():
    """Make sure no timer (ours or faulthandler's) leaks across tests."""
    server_main._cancel_stack_dump_timer()
    yield
    server_main._cancel_stack_dump_timer()


def _wait_for_marker(capfd, marker: str = _MARKER, timeout: float = 5.0) -> str:
    """Drain fd-level capture until ``marker`` shows up; return all output."""
    seen = ""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        captured = capfd.readouterr()
        seen += captured.out + captured.err
        if marker in seen:
            return seen
        time.sleep(0.05)
    return seen


def test_disabled_by_default(monkeypatch):
    """Without ``SLO_DUMP_STACKS`` no thread and no native timer may start."""
    monkeypatch.delenv("SLO_DUMP_STACKS", raising=False)

    with patch.object(server_main.faulthandler, "dump_traceback_later") as native:
        server_main._install_stack_dump_timer()

    native.assert_not_called()
    assert _dumper_thread() is None


def test_gil_held_mode_starts_thread_and_dumps(monkeypatch, capfd):
    """Default mode must dump from OUR thread, never via the native timer.

    ``dump_traceback_later`` is the crashy GIL-less path — asserting it is
    not called is the regression pin for the SEGV bug this file documents.
    """
    monkeypatch.setenv("SLO_DUMP_STACKS", "1")
    monkeypatch.setenv("SLO_DUMP_STACKS_INTERVAL", "0.05")
    monkeypatch.delenv("SLO_DUMP_STACKS_GILLESS", raising=False)

    with patch.object(server_main.faulthandler, "dump_traceback_later") as native:
        server_main._install_stack_dump_timer()
        try:
            thread = _dumper_thread()
            assert thread is not None, "no gil-held dumper thread started"
            assert thread.is_alive()
            native.assert_not_called()

            # The dump must land on fd 2 with the faulthandler-style marker
            # and must show THIS thread's frame, proving our loop did the dump.
            seen = _wait_for_marker(capfd)
            assert _MARKER in seen, "no periodic dump emitted"
            assert "_stack_dump_loop" in seen, "dump not produced by our thread"
        finally:
            server_main._cancel_stack_dump_timer()


def test_cancel_stops_further_dumps(monkeypatch, capfd):
    """After cancellation the dumper thread must go quiet."""
    monkeypatch.setenv("SLO_DUMP_STACKS", "1")
    monkeypatch.setenv("SLO_DUMP_STACKS_INTERVAL", "0.05")

    server_main._install_stack_dump_timer()
    assert _dumper_thread() is not None, "gil-held dumper thread must be running"
    seen = _wait_for_marker(capfd)
    assert _MARKER in seen, "timer never dumped before cancel"

    server_main._cancel_stack_dump_timer()
    assert _dumper_thread() is None

    # Drain anything emitted during the cancel call itself, then prove silence.
    capfd.readouterr()
    time.sleep(0.3)
    capfd.readouterr()
    time.sleep(0.3)
    after = capfd.readouterr()
    assert "Timeout (" not in (after.out + after.err), "dumps continued after cancel"


def test_gilless_escape_hatch_arms_native_timer(monkeypatch):
    """``SLO_DUMP_STACKS_GILLESS=1`` keeps the old (unsafe) native behaviour."""
    monkeypatch.setenv("SLO_DUMP_STACKS", "1")
    monkeypatch.setenv("SLO_DUMP_STACKS_INTERVAL", "30")
    monkeypatch.setenv("SLO_DUMP_STACKS_GILLESS", "1")

    with patch.object(server_main.faulthandler, "dump_traceback_later") as native:
        server_main._install_stack_dump_timer()

    native.assert_called_once_with(30.0, repeat=True)
    assert _dumper_thread() is None, "gilless mode must not start our thread"
