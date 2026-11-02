"""Periodic stack dump: GIL-safe replacement contract (SLO_DUMP_STACKS).

The pre-fix implementation (``faulthandler.dump_traceback_later(...,
repeat=True)``) crash-looped slough-api SIGSEGV: 23 crashes over
2026-10-31..11-02, every 4-14 min under load and once during idle startup.
All 3 retained cores died at the identical instruction — ``dump_frame``,
``Python/traceback.c:1259``, reading ``code->co_filename`` off a foreign
thread's already-torn-down frame from a timer/signal context with no GIL
frame-lifetime guarantee.

These tests pin the replacement's contract:

- ``_dump_thread_stacks`` emits the faulthandler-shaped output (Thread
  headers + ``File "..."`` frames) to a caller-chosen fd;
- the env gate: off by default, ``SLO_DUMP_STACKS=1`` opts in, garbage
  fails safe (lesson of card 20261102_001616 — silent degradation costs
  an investigation);
- install is idempotent, the dumper is a daemon thread, cadence follows
  ``SLO_DUMP_STACKS_INTERVAL``, and cancel actually joins the thread.
"""

from __future__ import annotations

import os
import threading
import time

from apps.api.server import main as server_main

# Fresh env around every test: ambient SLO_DUMP_STACKS must not leak in.
_ENV_KEYS = ("SLO_DUMP_STACKS", "SLO_DUMP_STACKS_INTERVAL")


class _Recorder:
    """Stands in for _dump_thread_stacks when testing the loop's cadence."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> None:
        self.calls += 1


def _install(monkeypatch, dumps: str, interval: str | None = "0.05") -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLO_DUMP_STACKS", dumps)
    if interval is not None:
        monkeypatch.setenv("SLO_DUMP_STACKS_INTERVAL", interval)
    server_main._install_stack_dump_timer()


def _teardown() -> None:
    server_main._cancel_stack_dump_timer()


def _dump_via_pipe() -> bytes:
    """Call _dump_thread_stacks over a pipe, reading concurrently.

    A reader must exist before the write: a full all-threads dump can
    exceed the 64 KB pipe buffer, and ``os.write`` would then block
    forever with nobody draining it.
    """
    r, w = os.pipe()
    chunks: list[bytes] = []

    def reader() -> None:
        while True:
            block = os.read(r, 65536)
            if not block:
                return
            chunks.append(block)

    rt = threading.Thread(target=reader, daemon=True)
    rt.start()
    try:
        server_main._dump_thread_stacks(fd=w)
    finally:
        os.close(w)
    rt.join(5)
    os.close(r)
    assert not rt.is_alive(), "pipe reader never saw EOF"
    return b"".join(chunks).decode("utf-8", "replace")


class TestDumpOutput:
    def test_writes_thread_headers_and_frames(self) -> None:
        """One write, faulthandler-shaped: headers, our frame, File lines."""
        data = _dump_via_pipe()

        assert "Thread 0x" in data
        assert "(most recent call first):" in data
        assert "  File " in data
        # The dumping thread's own stack includes this test file.
        assert "test_stack_dump.py" in data

    def test_snapshot_covers_multiple_threads(self) -> None:
        """A live helper thread appears in the dump (foreign frames pin safely)."""
        ready = threading.Event()
        release = threading.Event()

        def helper() -> None:
            ready.set()
            release.wait(5)

        t = threading.Thread(target=helper, name="dump-helper", daemon=True)
        t.start()
        try:
            assert ready.wait(2)
            # Its frame is blocked in Event.wait inside this file's helper.
            assert "helper" in _dump_via_pipe()
        finally:
            release.set()
            t.join(5)


class TestEnvGate:
    def test_off_by_default(self, monkeypatch) -> None:
        for key in _ENV_KEYS:
            monkeypatch.delenv(key, raising=False)
        server_main._install_stack_dump_timer()
        try:
            assert server_main._STACK_DUMP_THREAD is None
        finally:
            _teardown()

    def test_garbage_value_fails_closed(self, monkeypatch) -> None:
        """SLO_DUMP_STACKS=garbage must NOT start the dumper."""
        _install(monkeypatch, dumps="garbage", interval=None)
        try:
            assert server_main._STACK_DUMP_THREAD is None
        finally:
            _teardown()

    def test_garbage_interval_warns_without_thread(self, monkeypatch) -> None:
        _install(monkeypatch, dumps="1", interval="not-a-number")
        try:
            assert server_main._STACK_DUMP_THREAD is None
        finally:
            _teardown()


class TestLifecycle:
    def test_install_starts_daemon_and_cancel_joins(self, monkeypatch) -> None:
        rec = _Recorder()
        monkeypatch.setattr(server_main, "_dump_thread_stacks", rec)
        _install(monkeypatch, dumps="1", interval="0.05")
        thread = server_main._STACK_DUMP_THREAD
        try:
            assert thread is not None and thread.daemon
            assert thread.name == "stack-dump"
            deadline = time.monotonic() + 3
            while rec.calls < 2 and time.monotonic() < deadline:
                time.sleep(0.02)
            assert rec.calls >= 2, "dumper did not tick within 3 intervals"
        finally:
            _teardown()

        assert server_main._STACK_DUMP_THREAD is None
        assert not thread.is_alive(), "cancel must join the dumper thread"
        frozen = rec.calls
        time.sleep(0.15)  # 3 more intervals: a cancelled loop must not tick
        assert rec.calls == frozen

    def test_install_is_idempotent(self, monkeypatch) -> None:
        rec = _Recorder()
        monkeypatch.setattr(server_main, "_dump_thread_stacks", rec)
        _install(monkeypatch, dumps="1", interval="0.05")
        first = server_main._STACK_DUMP_THREAD
        try:
            assert first is not None
            _install(monkeypatch, dumps="1", interval="0.05")
            second = server_main._STACK_DUMP_THREAD
            assert second is not None and second is not first
            assert not first.is_alive(), "reinstall must stop the old dumper"
        finally:
            _teardown()
            assert second is None or not second.is_alive()

    def test_cancel_without_install_is_safe(self) -> None:
        _teardown()  # must not raise when nothing is installed
