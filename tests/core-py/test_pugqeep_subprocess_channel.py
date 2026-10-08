"""The forked subprocess channel: no multiprocessing, framed protocol, real capture.

Complements ``test_pugqeep_process_mgmt`` (which covers process lifecycle) by
pinning *how* the child is created and how its output reaches the parent.
"""

from __future__ import annotations

import inspect
import logging
import os
import signal
import sys
import time

import pytest

from domain.infrastructure._internal.pugqeep import engine as engine_mod
from domain.infrastructure._internal.pugqeep.config import SubprocessConfig
from domain.infrastructure._internal.pugqeep.engine import (
    Process,
    ProcessStatus,
    SubprocessProcess,
)
from domain.infrastructure._internal.pugqeep.frame import PROTOCOL_VERSION


def _wait_done(proc: Process, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while not proc.is_done and time.time() < deadline:
        time.sleep(0.02)


def _getpid():
    return os.getpid()


def _report_sigterm_handler():
    return "SIG_DFL" if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL else "inherited"


def _print_both():
    print("hello stdout")
    print("hello stderr", file=sys.stderr, flush=True)
    return "done"


# ── Option A: our own fork, multiprocessing gone ─────────────────────────────


class TestNoMultiprocessing:
    def test_engine_module_does_not_bind_multiprocessing(self):
        assert not hasattr(engine_mod, "multiprocessing"), (
            "engine must not import multiprocessing — PGQ owns the fork"
        )

    def test_engine_source_has_no_multiprocessing_import(self):
        assert "import multiprocessing" not in inspect.getsource(engine_mod)

    def test_child_pid_equals_the_handle_we_return(self):
        """Prove the pid came from our fork, not from a wrapped Process."""
        proc = Process(fn=_getpid, name="pidcheck")
        sub = SubprocessProcess(proc, SubprocessConfig(enabled=True))
        sub.start()
        sub.monitor()
        _wait_done(proc)

        assert proc.status == ProcessStatus.COMPLETED
        assert proc.result == sub.pid
        assert sub.pid == proc._pid

    def test_child_does_not_inherit_parent_sigterm_handler(self):
        """An inherited handler would swallow terminate()'s SIGTERM.

        The child would then always sit out terminate_grace before SIGKILL.
        """
        previous = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda *_args: None)
        try:
            proc = Process(fn=_report_sigterm_handler, name="sighandler")
            sub = SubprocessProcess(proc, SubprocessConfig(enabled=True))
            sub.start()
            sub.monitor()
            _wait_done(proc)
        finally:
            signal.signal(signal.SIGTERM, previous)

        assert proc.status == ProcessStatus.COMPLETED
        assert proc.result == "SIG_DFL"


# ── Framed channel over the socketpair ───────────────────────────────────────


class TestFramedChannel:
    def test_handshake_records_peer_protocol(self):
        proc = Process(fn=lambda: 1, name="handshake")
        sub = SubprocessProcess(proc, SubprocessConfig(enabled=True))
        sub.start()
        sub.monitor()
        _wait_done(proc)

        assert proc.status == ProcessStatus.COMPLETED
        assert sub._handler is not None
        assert sub._handler.peer_protocol == PROTOCOL_VERSION

    def test_captured_output_arrives(self):
        """stdout/stderr must survive the trip, not be dropped by the reader."""
        proc = Process(fn=_print_both, name="capture")
        sub = SubprocessProcess(proc, SubprocessConfig(enabled=True, capture_output=True))
        sub.start()
        sub.monitor()
        _wait_done(proc)

        assert proc.status == ProcessStatus.COMPLETED
        assert proc.result == "done"
        # The ordering fix: output is emitted *before* the terminal frame, so
        # the reader still sees it instead of returning on RESULT first.
        assert sub.stdout is not None and "hello stdout" in sub.stdout
        assert sub.stderr is not None and "hello stderr" in sub.stderr

    def test_no_capture_leaves_output_unset(self):
        proc = Process(fn=lambda: "quiet", name="nocapture")
        sub = SubprocessProcess(proc, SubprocessConfig(enabled=True, capture_output=False))
        sub.start()
        sub.monitor()
        _wait_done(proc)

        assert proc.status == ProcessStatus.COMPLETED
        assert sub.stdout is None
        assert sub.stderr is None

    def test_failure_still_reports_and_still_sends_output(self):
        def boom():
            print("before the crash")
            raise ValueError("kaboom")

        proc = Process(fn=boom, name="boom")
        sub = SubprocessProcess(proc, SubprocessConfig(enabled=True, capture_output=True))
        sub.start()
        sub.monitor()
        _wait_done(proc)

        assert proc.status == ProcessStatus.FAILED
        assert "kaboom" in (proc.error or "")
        assert sub.stdout is not None and "before the crash" in sub.stdout

    @pytest.mark.parametrize("method", ["spawn", "forkserver"])
    def test_unsupported_start_method_warns_but_still_forks(self, method, caplog):
        """start_method selects nothing now — it must not pretend otherwise."""
        proc = Process(fn=lambda: 7, name=f"sm-{method}")
        sub = SubprocessProcess(proc, SubprocessConfig(enabled=True, start_method=method))
        with caplog.at_level(logging.WARNING, logger="slo.pugqeep.engine"):
            sub.start()
            sub.monitor()
        _wait_done(proc)

        assert proc.status == ProcessStatus.COMPLETED
        assert proc.result == 7
        assert any("start_method" in record.getMessage() for record in caplog.records), (
            "a non-fork start_method must be reported as selecting nothing"
        )
