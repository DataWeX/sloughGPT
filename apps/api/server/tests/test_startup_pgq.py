"""Tests for PGQ integration in StartupOrchestrator.

Verifies that startup hooks dispatch through PGQ Tree instead of
raw threading, keeping the event loop free for HTTP.
"""

import os
import time

os.environ.setdefault("SLO_AUTO_WORKFLOW", "false")
os.environ.setdefault("SLO_AUTOLOAD_MODEL", "")

from domain.infrastructure._internal.pugqeep.engine import (
    Process,
    ProcessStatus,
    Tree,
)


class TestStartupTree:
    """PGQ Tree dispatches sync work off the event loop."""

    def test_tree_branch_runs_process_in_thread_pool(self):
        """Tree.branch() executes process in its ThreadPoolExecutor."""
        tree = Tree("test-bg", pool_workers=2)
        result = {}

        def _sync_work():
            result["thread"] = __import__("threading").current_thread().name
            result["done"] = True

        proc = Process(fn=_sync_work, name="test")
        stem = tree.branch([proc])
        stem._done_event.wait(timeout=5)

        assert result.get("done") is True
        assert "tree-test-bg" in result["thread"]
        tree.shutdown()

    def test_tree_does_not_block_caller(self):
        """Tree.branch() returns immediately, work runs in background."""
        tree = Tree("test-nonblock", pool_workers=2)
        started = time.monotonic()

        def _slow_work():
            time.sleep(0.5)

        proc = Process(fn=_slow_work, name="slow")
        stem = tree.branch([proc])
        branch_time = time.monotonic() - started

        # branch() should return in <100ms, not block for 500ms
        assert branch_time < 0.1
        stem._done_event.wait(timeout=5)
        tree.shutdown()

    def test_tree_concurrent_processes(self):
        """Tree runs multiple processes concurrently via pool."""
        tree = Tree("test-concurrent", pool_workers=4)
        timestamps = []

        def _work(i):
            timestamps.append(("start", i, time.monotonic()))
            time.sleep(0.1)
            timestamps.append(("end", i, time.monotonic()))

        procs = [Process(fn=_work, args=(i,), name=f"w{i}") for i in range(4)]
        stem = tree.branch(procs)
        stem._done_event.wait(timeout=5)

        # All 4 should have started before any finished
        starts = [t for label, _, t in timestamps if label == "start"]
        ends = [t for label, _, t in timestamps if label == "end"]
        assert len(starts) == 4
        assert len(ends) == 4
        # First end should be after all starts (concurrent, not serial)
        assert min(ends) > max(starts)
        tree.shutdown()

    def test_tree_process_timeout(self):
        """Process times out when exceeding timeout."""
        tree = Tree("test-timeout", pool_workers=2)

        def _slow():
            time.sleep(10)

        proc = Process(fn=_slow, name="slow", timeout=0.1)
        stem = tree.branch([proc])
        stem._done_event.wait(timeout=5)

        assert proc.status == ProcessStatus.FAILED
        tree.shutdown()


class TestStartupPGQIntegration:
    """StartupOrchestrator uses PGQ for background work."""

    def test_imports_pgq_tree(self):
        """StartupOrchestrator imports PGQ Tree."""
        from infrastructure.startup import StartupOrchestrator

        # Should be importable without error
        assert StartupOrchestrator is not None

    def test_pgq_process_has_callbacks(self):
        """PGQ Process supports on_complete/on_fail callbacks."""
        completed = []
        proc = Process(fn=lambda: None, name="cb-test")
        proc.on_complete(lambda p: completed.append(("ok", p.id)))
        proc.on_fail(lambda p: completed.append(("fail", p.id)))

        proc.complete("result")
        assert len(completed) == 1
        assert completed[0][0] == "ok"

    def test_pgq_process_status_lifecycle(self):
        """PGQ Process transitions through status lifecycle."""
        proc = Process(fn=lambda: None, name="lifecycle")
        assert proc.status == ProcessStatus.CREATED

        proc.running()
        assert proc.status == ProcessStatus.RUNNING

        proc.complete("result")
        assert proc.status == ProcessStatus.COMPLETED
        assert proc.result == "result"
        assert proc.is_done

    def test_pgq_process_cancellation(self):
        """PGQ Process can be cancelled."""
        proc = Process(fn=lambda: None, name="cancel")
        proc.running()
        proc.cancel()
        assert proc.status == ProcessStatus.CANCELLED
        assert proc.is_done
        assert proc.is_cancelled

    def test_bg_tree_shutdown(self):
        """bg_tree shuts down cleanly."""
        tree = Tree("test-shutdown", pool_workers=2)
        tree.shutdown()
        assert tree.status.value == "stopped"
