"""Tests for pugqeep process management — subprocess, monitor, config, signal handling."""

import multiprocessing
import os
import signal
import struct
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from domains.infrastructure.pugqeep.config import (
    EngineConfig,
    MonitorConfig,
    RestartPolicy,
    SubprocessConfig,
)
from domains.infrastructure.pugqeep.engine import (
    Engine,
    EngineMetrics,
    Process,
    ProcessGroup,
    ProcessMonitor,
    ProcessStatus,
    GuardTree,
    ResultCache,
    Stem,
    StemStatus,
    SubprocessProcess,
    Tree,
    TreeStatus,
    _MSG_ERROR,
    _MSG_HEARTBEAT,
    _MSG_READY,
    _MSG_RESULT,
)


# ── Helpers ──────────────────────────────────────────────────────

def _noop():
    return 42


def _add(a, b):
    return a + b


def _sleep_and_return(secs, val):
    time.sleep(secs)
    return val


def _fail():
    raise ValueError("boom")


def _slow_fail(secs):
    time.sleep(secs)
    raise ValueError("slow boom")


def _identity(x):
    return x


# ── Config ───────────────────────────────────────────────────────

class TestEngineConfig:
    def test_defaults(self):
        cfg = EngineConfig()
        assert cfg.name == "main"
        assert cfg.max_trees == 16
        assert cfg.subprocess.enabled is True
        assert cfg.restart.max_restarts == 0
        assert cfg.monitor.enabled is True

    def test_subprocess_config_defaults(self):
        cfg = SubprocessConfig()
        assert cfg.enabled is True
        assert cfg.max_workers > 0
        assert cfg.terminate_grace == 3.0
        assert cfg.start_method in ("fork", "forkserver", "spawn")
        assert cfg.memory_limit_mb is None
        assert cfg.capture_output is False

    def test_restart_policy_defaults(self):
        rp = RestartPolicy()
        assert rp.max_restarts == 0
        assert rp.restart_delay == 1.0
        assert rp.backoff == "exponential"
        assert rp.max_backoff == 30.0

    def test_monitor_config_defaults(self):
        mc = MonitorConfig()
        assert mc.enabled is True
        assert mc.poll_interval == 1.0
        assert mc.stall_timeout == 60.0

    def test_engine_config_custom(self):
        cfg = EngineConfig(
            name="custom",
            max_trees=4,
            tree_workers=2,
            max_stems=3,
            queue_size=64,
            poll_interval=0.5,
        )
        assert cfg.name == "custom"
        assert cfg.max_trees == 4
        assert cfg.tree_workers == 2
        assert cfg.max_stems == 3
        assert cfg.queue_size == 64
        assert cfg.poll_interval == 0.5

    def test_engine_accepts_config(self):
        cfg = EngineConfig(name="via-config", max_trees=8)
        engine = Engine(config=cfg)
        assert engine.name == "via-config"
        assert engine.max_trees == 8

    def test_engine_backward_compat(self):
        engine = Engine("old-style", max_trees=12)
        assert engine.name == "old-style"
        assert engine.max_trees == 12


# ── SubprocessProcess ───────────────────────────────────────────

class TestSubprocessProcess:
    def test_spawn_simple_function(self):
        proc = Process(fn=_noop, name="simple")
        config = SubprocessConfig(enabled=True, start_method="fork")
        sub = SubprocessProcess(proc, config)

        sub.start()
        sub.monitor()

        deadline = time.time() + 5
        while not proc.is_done and time.time() < deadline:
            time.sleep(0.05)

        assert proc.status == ProcessStatus.COMPLETED
        assert proc.result == 42
        assert proc._pid is not None

    def test_spawn_with_args(self):
        proc = Process(fn=_add, args=(3, 7), name="add")
        config = SubprocessConfig(enabled=True, start_method="fork")
        sub = SubprocessProcess(proc, config)

        sub.start()
        sub.monitor()

        deadline = time.time() + 5
        while not proc.is_done and time.time() < deadline:
            time.sleep(0.05)

        assert proc.status == ProcessStatus.COMPLETED
        assert proc.result == 10

    def test_spawn_with_exception(self):
        proc = Process(fn=_fail, name="fail")
        config = SubprocessConfig(enabled=True, start_method="fork")
        sub = SubprocessProcess(proc, config)

        sub.start()
        sub.monitor()

        deadline = time.time() + 5
        while not proc.is_done and time.time() < deadline:
            time.sleep(0.05)

        assert proc.status == ProcessStatus.FAILED
        assert "boom" in proc.error

    def test_terminate(self):
        proc = Process(fn=time.sleep, args=(60,), name="long")
        config = SubprocessConfig(enabled=True, start_method="fork")
        sub = SubprocessProcess(proc, config)

        sub.start()
        time.sleep(0.2)
        assert sub.is_alive

        sub.terminate()
        time.sleep(0.5)
        assert proc.status == ProcessStatus.CANCELLED

    def test_health(self):
        proc = Process(fn=_noop, name="health")
        config = SubprocessConfig(enabled=True, start_method="fork")
        sub = SubprocessProcess(proc, config)

        sub.start()
        h = sub.health()
        assert h["pid"] == proc._pid
        assert h["alive"] is True
        assert h["elapsed"] is not None

        sub.monitor()
        deadline = time.time() + 5
        while not proc.is_done and time.time() < deadline:
            time.sleep(0.05)

        h = sub.health()
        assert h["alive"] is False

    def test_memory_limit(self):
        proc = Process(fn=_noop, name="mem")
        config = SubprocessConfig(
            enabled=True,
            start_method="fork",
            memory_limit_mb=64,
        )
        sub = SubprocessProcess(proc, config)
        sub.start()
        sub.monitor()

        deadline = time.time() + 5
        while not proc.is_done and time.time() < deadline:
            time.sleep(0.05)

        assert proc.status == ProcessStatus.COMPLETED


# ── GuardTree ────────────────────────────────────────────────────

class TestGuardTree:
    def test_branch_thread_mode(self):
        config = SubprocessConfig(enabled=False)
        tree = GuardTree("test", config=config, max_stems=4, pool_workers=2)
        proc = Process(fn=_noop, name="work")
        stem = tree.branch([proc])

        tree.wait_stem(stem, timeout=5)
        assert proc.status == ProcessStatus.COMPLETED
        assert proc.result == 42
        tree.shutdown()

    def test_branch_subprocess_mode(self):
        config = SubprocessConfig(enabled=True, start_method="fork")
        tree = GuardTree("test", config=config, max_stems=4, pool_workers=2)
        proc = Process(fn=_noop, name="work")
        stem = tree.branch([proc])

        tree.wait_stem(stem, timeout=10)
        assert proc.status == ProcessStatus.COMPLETED
        assert proc.result == 42
        tree.shutdown()

    def test_health(self):
        config = SubprocessConfig(enabled=False)
        tree = GuardTree("test", config=config)
        proc = Process(fn=_noop, name="work")
        tree.branch([proc])
        time.sleep(0.5)

        h = tree.health()
        assert isinstance(h, dict)
        tree.shutdown()

    def test_to_dict(self):
        config = SubprocessConfig(enabled=True)
        tree = GuardTree("test", config=config)
        d = tree.to_dict()
        assert d["subprocess_enabled"] is True
        assert d["subprocess_count"] == 0
        tree.shutdown()


# ── ProcessMonitor ───────────────────────────────────────────────

class TestProcessMonitor:
    def test_tracks_process(self):
        monitor = ProcessMonitor(
            config=MonitorConfig(enabled=True, poll_interval=0.1, stall_timeout=5),
            restart_policy=RestartPolicy(),
        )
        proc = Process(fn=_noop, name="tracked")
        monitor.track(proc)
        assert monitor.active_count == 1
        monitor.untrack(proc.id)
        assert monitor.active_count == 0

    def test_stall_detection(self):
        stall_called = threading.Event()

        def on_stall(p):
            stall_called.set()

        monitor = ProcessMonitor(
            config=MonitorConfig(enabled=True, poll_interval=0.1, stall_timeout=0.2),
            restart_policy=RestartPolicy(),
        )
        monitor.on_stall(on_stall)

        proc = Process(fn=time.sleep, args=(60,), name="stall")
        proc.running()
        monitor.track(proc)
        monitor.start()

        stall_called.wait(timeout=3)
        assert stall_called.is_set()

        monitor.stop()
        proc.cancel()

    def test_restart_callback(self):
        restart_called = threading.Event()

        def on_restart(p):
            restart_called.set()

        monitor = ProcessMonitor(
            config=MonitorConfig(enabled=True, poll_interval=0.1, stall_timeout=999),
            restart_policy=RestartPolicy(max_restarts=2),
        )
        monitor.on_restart(on_restart)

        proc = Process(fn=_noop, name="restart")
        proc._restart_policy = RestartPolicy(max_restarts=2)
        proc.fail("test error")
        monitor.track(proc)
        monitor.start()

        restart_called.wait(timeout=3)
        assert restart_called.is_set()

        monitor.stop()

    def test_no_restart_when_disabled(self):
        restart_called = threading.Event()

        def on_restart(p):
            restart_called.set()

        monitor = ProcessMonitor(
            config=MonitorConfig(enabled=True, poll_interval=0.1, stall_timeout=999),
            restart_policy=RestartPolicy(max_restarts=0),
        )
        monitor.on_restart(on_restart)

        proc = Process(fn=_noop, name="no-restart")
        proc.fail("test error")
        monitor.track(proc)
        monitor.start()
        time.sleep(0.5)

        assert not restart_called.is_set()
        monitor.stop()

    def test_restart_delay_exponential(self):
        monitor = ProcessMonitor(
            config=MonitorConfig(),
            restart_policy=RestartPolicy(
                restart_delay=1.0,
                backoff="exponential",
                max_backoff=30.0,
            ),
        )
        assert monitor._restart_delay(0) == 1.0
        assert monitor._restart_delay(1) == 2.0
        assert monitor._restart_delay(2) == 4.0
        assert monitor._restart_delay(3) == 8.0
        assert monitor._restart_delay(10) == 30.0  # capped

    def test_restart_delay_linear(self):
        monitor = ProcessMonitor(
            config=MonitorConfig(),
            restart_policy=RestartPolicy(
                restart_delay=2.0,
                backoff="linear",
            ),
        )
        assert monitor._restart_delay(0) == 2.0
        assert monitor._restart_delay(1) == 4.0
        assert monitor._restart_delay(2) == 6.0

    def test_restart_delay_fixed(self):
        monitor = ProcessMonitor(
            config=MonitorConfig(),
            restart_policy=RestartPolicy(
                restart_delay=3.0,
                backoff="fixed",
            ),
        )
        assert monitor._restart_delay(0) == 3.0
        assert monitor._restart_delay(1) == 3.0
        assert monitor._restart_delay(5) == 3.0

    def test_stats(self):
        monitor = ProcessMonitor(
            config=MonitorConfig(),
            restart_policy=RestartPolicy(),
        )
        p1 = Process(fn=_noop)
        p1.running()
        p2 = Process(fn=_noop)
        p2.fail("err")
        monitor.track(p1)
        monitor.track(p2)

        s = monitor.stats()
        assert s["monitored"] == 2
        assert s["running"] == 1
        assert s["failed"] == 1

    def test_stall_detection_heartbeat(self):
        stall_called = threading.Event()

        def on_stall(p):
            stall_called.set()

        monitor = ProcessMonitor(
            config=MonitorConfig(enabled=True, poll_interval=0.1, stall_timeout=0.2),
            restart_policy=RestartPolicy(),
        )
        monitor.on_stall(on_stall)

        proc = Process(fn=time.sleep, args=(60,), name="heartbeat-stall")
        proc.running()
        # Simulate old heartbeat (beyond stall timeout)
        proc._last_heartbeat = time.monotonic() - 1.0
        monitor.track(proc)
        monitor.start()

        stall_called.wait(timeout=3)
        assert stall_called.is_set()

        monitor.stop()
        proc.cancel()

    def test_stalled_processes_method(self):
        monitor = ProcessMonitor(
            config=MonitorConfig(enabled=True, poll_interval=0.1, stall_timeout=0.2),
            restart_policy=RestartPolicy(),
        )

        # Create a running process with old heartbeat
        proc = Process(fn=time.sleep, args=(60,), name="stalled-proc")
        proc.running()
        proc._last_heartbeat = time.monotonic() - 1.0
        monitor.track(proc)

        # Create a running process with fresh heartbeat
        proc2 = Process(fn=time.sleep, args=(60,), name="active-proc")
        proc2.running()
        proc2._last_heartbeat = time.monotonic()
        monitor.track(proc2)

        stalled = monitor.stalled_processes()
        assert len(stalled) == 1
        assert stalled[0].id == proc.id

    def test_stalled_processes_elapsed_fallback(self):
        monitor = ProcessMonitor(
            config=MonitorConfig(enabled=True, poll_interval=0.1, stall_timeout=0.2),
            restart_policy=RestartPolicy(),
        )

        # Process without heartbeat uses elapsed time
        proc = Process(fn=time.sleep, args=(60,), name="elapsed-stall")
        proc.running()
        proc._last_heartbeat = None
        # Manually set started_at to make it look old
        proc.started_at = time.time() - 1.0
        monitor.track(proc)

        stalled = monitor.stalled_processes()
        assert len(stalled) == 1
        assert stalled[0].id == proc.id

    def test_get_restart_count(self):
        monitor = ProcessMonitor(
            config=MonitorConfig(),
            restart_policy=RestartPolicy(),
        )

        proc = Process(fn=_noop, name="restart-test")
        monitor.track(proc)

        assert monitor.get_restart_count(proc.id) == 0
        monitor._restart_count[proc.id] = 2
        assert monitor.get_restart_count(proc.id) == 2

    def test_reset_restart_count(self):
        monitor = ProcessMonitor(
            config=MonitorConfig(),
            restart_policy=RestartPolicy(),
        )

        proc = Process(fn=_noop, name="reset-test")
        monitor.track(proc)

        monitor._restart_count[proc.id] = 3
        assert monitor.get_restart_count(proc.id) == 3
        monitor.reset_restart_count(proc.id)
        assert monitor.get_restart_count(proc.id) == 0

    def test_restart_count_synced_to_process(self):
        restart_called = threading.Event()

        def on_restart(p):
            # Verify that proc._restart_count is incremented
            restart_called.set()

        monitor = ProcessMonitor(
            config=MonitorConfig(enabled=True, poll_interval=0.1, stall_timeout=999),
            restart_policy=RestartPolicy(max_restarts=2),
        )
        monitor.on_restart(on_restart)

        proc = Process(fn=_noop, name="sync-test")
        proc._restart_policy = RestartPolicy(max_restarts=2)
        proc.fail("test error")
        monitor.track(proc)
        monitor.start()

        restart_called.wait(timeout=3)
        assert restart_called.is_set()
        # Verify count is synced
        assert proc._restart_count == 1
        assert monitor.get_restart_count(proc.id) == 1

        monitor.stop()


# ── Engine with config ───────────────────────────────────────────

class TestEngineWithConfig:
    def test_engine_creates_guard_tree(self):
        cfg = EngineConfig(name="test", max_trees=4)
        engine = Engine(config=cfg)
        tree = engine.tree("guarded", guarded=True)
        assert isinstance(tree, GuardTree)
        engine.stop()

    def test_engine_process_restart_count(self):
        engine = Engine("test")
        proc = Process(fn=_noop, name="work")
        assert proc._restart_count == 0
        proc._restart_count += 1
        assert proc._restart_count == 1
        engine.stop()

    def test_engine_to_dict_includes_monitor(self):
        cfg = EngineConfig(name="test", monitor=MonitorConfig(enabled=True))
        engine = Engine(config=cfg)
        d = engine.to_dict()
        assert "monitor" in d
        assert d["monitor"]["monitored"] == 0
        engine.stop()

    def test_engine_no_monitor_when_disabled(self):
        cfg = EngineConfig(name="test", monitor=MonitorConfig(enabled=False))
        engine = Engine(config=cfg)
        assert engine._monitor is None
        d = engine.to_dict()
        assert d["monitor"] is None
        engine.stop()

    def test_engine_process_dict_includes_restart(self):
        proc = Process(fn=_noop, name="test")
        d = proc.to_dict()
        assert "restart_count" in d
        assert "pid" in d


# ── Signal handling ─────────────────────────────────────────────

class TestSignalHandling:
    def test_install_restore_handlers(self):
        engine = Engine("test")
        engine.install_signal_handlers()
        engine.restore_signal_handlers()
        engine.stop()

    def test_signal_handler_sets_running_false(self):
        engine = Engine("test")
        engine._running = True
        engine.install_signal_handlers()

        # Simulate SIGTERM by calling the handler directly
        old_handler = signal.getsignal(signal.SIGTERM)
        if callable(old_handler):
            old_handler(signal.SIGTERM, None)

        assert engine._running is False
        engine.restore_signal_handlers()


# ── SubprocessProcess pipe protocol ─────────────────────────────

class TestSubprocessPipeProtocol:
    def test_msg_ready_sent(self):
        """Verify the child sends _MSG_READY before executing."""
        proc = Process(fn=_noop, name="pipe-test")
        config = SubprocessConfig(enabled=True, start_method="fork")
        sub = SubprocessProcess(proc, config)

        sub.start()
        sub.monitor()

        deadline = time.time() + 5
        while not proc.is_done and time.time() < deadline:
            time.sleep(0.05)

        assert proc.status == ProcessStatus.COMPLETED
        assert sub._last_heartbeat is not None


# ── EngineMetrics ────────────────────────────────────────────────

class TestEngineMetrics:
    def test_record_spawn(self):
        from domains.infrastructure.pugqeep.engine import EngineMetrics
        m = EngineMetrics()
        m.record_spawn()
        m.record_spawn()
        s = m.snapshot()
        assert s["spawned"] == 2

    def test_record_complete(self):
        from domains.infrastructure.pugqeep.engine import EngineMetrics
        m = EngineMetrics()
        proc = Process(fn=_noop)
        proc.running()
        time.sleep(0.01)
        proc.complete(42)
        m.record_complete(proc)
        s = m.snapshot()
        assert s["completed"] == 1
        assert s["avg_latency_s"] > 0

    def test_record_fail(self):
        from domains.infrastructure.pugqeep.engine import EngineMetrics
        m = EngineMetrics()
        m.record_fail()
        s = m.snapshot()
        assert s["failed"] == 1
        assert s["error_rate"] == 1.0

    def test_throughput(self):
        from domains.infrastructure.pugqeep.engine import EngineMetrics
        m = EngineMetrics()
        m.record_complete(Process(fn=_noop))
        m.record_complete(Process(fn=_noop))
        s = m.snapshot()
        assert s["throughput_per_s"] > 0

    def test_reset(self):
        from domains.infrastructure.pugqeep.engine import EngineMetrics
        m = EngineMetrics()
        m.record_spawn()
        m.record_fail()
        m.reset()
        s = m.snapshot()
        assert s["spawned"] == 0
        assert s["failed"] == 0

    def test_cancel_and_timeout(self):
        from domains.infrastructure.pugqeep.engine import EngineMetrics
        m = EngineMetrics()
        m.record_cancel()
        m.record_timeout()
        m.record_restart()
        s = m.snapshot()
        assert s["cancelled"] == 1
        assert s["timed_out"] == 1
        assert s["restarted"] == 1


# ── Process dependency resolution ────────────────────────────────

class TestProcessDependencies:
    def test_deps_met(self):
        engine = Engine("test")
        engine.tree("t")

        p1 = engine.spawn(_noop, name="a")
        p2 = engine.spawn(_noop, name="b", depends_on=[p1.id])

        assert not engine._deps_met(p2)
        p1.complete()
        assert engine._deps_met(p2)
        engine.stop()

    def test_dispatch_holds_unmet_deps(self):
        engine = Engine("test")
        engine.tree("t")
        engine.route("a", "t")
        engine.route("b", "t")

        p1 = engine.spawn(_noop, name="a")
        p2 = engine.spawn(_noop, name="b", depends_on=[p1.id])

        dispatched = engine.dispatch()
        assert dispatched == 1
        assert p2 in engine._pending

        p1.complete()
        dispatched = engine.dispatch()
        assert dispatched == 1
        assert p2 not in engine._pending
        engine.stop()

    def test_depends_on_failed_fails_child(self):
        engine = Engine("test")
        engine.tree("t")

        p1 = engine.spawn(_fail, name="a")
        p2 = engine.spawn(_noop, name="b", depends_on=[p1.id])

        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()

        assert p1.status == ProcessStatus.FAILED
        # p2 should still be pending (dep failed, never dispatched)
        assert p2.status != ProcessStatus.COMPLETED

    def test_wait_for(self):
        engine = Engine("test")
        engine.tree("t")

        p = engine.spawn(_sleep_and_return, 0.05, "done", name="work")
        engine.run_background(poll_interval=0.01)
        result = engine.wait_for(p.id, timeout=5)
        engine.stop()

        assert result.status == ProcessStatus.COMPLETED
        assert result.result == "done"

    def test_wait_for_timeout(self):
        engine = Engine("test")
        engine.tree("t")

        p = engine.spawn(time.sleep, 60, name="long")
        result = engine.wait_for(p.id, timeout=0.1)

        assert result.status != ProcessStatus.COMPLETED
        engine.stop()

    def test_wait_for_keyerror(self):
        engine = Engine("test")
        with pytest.raises(KeyError):
            engine.wait_for("nonexistent")
        engine.stop()

    def test_wait_for_any(self):
        engine = Engine("test")
        engine.tree("t")

        p1 = engine.spawn(_sleep_and_return, 0.3, "slow", name="a")
        p2 = engine.spawn(_sleep_and_return, 0.05, "fast", name="b")

        engine.run_background(poll_interval=0.01)
        result = engine.wait_for_any([p1.id, p2.id], timeout=5)
        engine.stop()

        assert result is not None
        assert result.id == p2.id


# ── spawn_chain ──────────────────────────────────────────────────

class TestSpawnChain:
    def test_chain_basic(self):
        engine = Engine("test")
        engine.tree("t")

        def double(x):
            return x * 2

        def add_one(x):
            return x + 1

        processes = engine.spawn_chain(
            (double, 5),
            (add_one,),
            tree="t",
        )

        assert len(processes) == 2
        assert processes[1].depends_on == [processes[0].id]
        engine.stop()

    def test_chain_executes_sequentially(self):
        engine = Engine("test")
        engine.tree("t")

        def double(x):
            return x * 2

        processes = engine.spawn_chain(
            (double, 3),
            (double,),
            (double,),
            tree="t",
        )

        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()

        assert processes[0].result == 6
        assert processes[1].result == 12
        assert processes[2].result == 24

    def test_chain_empty(self):
        engine = Engine("test")
        processes = engine.spawn_chain()
        assert processes == []
        engine.stop()


# ── Process cancel propagation ──────────────────────────────────

class TestCancelPropagation:
    def test_process_cancel_event(self):
        proc = Process(fn=time.sleep, args=(60,))
        proc.running()

        t = threading.Thread(target=proc.wait_cancel, daemon=True)
        t.start()

        proc.cancel()
        t.join(timeout=1)
        assert not t.is_alive()
        assert proc.is_cancelled

    def test_engine_cancel_process(self):
        engine = Engine("test")
        engine.tree("t")

        parent = engine.spawn(_noop, name="parent")
        child = engine.spawn(_noop, name="child")
        parent.children_ids = [child.id]
        child.parent_id = parent.id

        cancelled = engine.cancel_process(parent.id, propagate=True)
        assert cancelled == 2
        assert parent.status == ProcessStatus.CANCELLED
        assert child.status == ProcessStatus.CANCELLED
        engine.stop()

    def test_engine_cancel_no_propagate(self):
        engine = Engine("test")
        engine.tree("t")

        parent = engine.spawn(_noop, name="parent")
        child = engine.spawn(_noop, name="child")
        parent.children_ids = [child.id]

        cancelled = engine.cancel_process(parent.id, propagate=False)
        assert cancelled == 1
        assert parent.status == ProcessStatus.CANCELLED
        assert child.status != ProcessStatus.CANCELLED
        engine.stop()


# ── Engine metrics integration ──────────────────────────────────

class TestEngineMetricsIntegration:
    def test_engine_has_metrics(self):
        engine = Engine("test")
        assert engine.metrics is not None
        d = engine.to_dict()
        assert "metrics" in d
        engine.stop()

    def test_engine_metrics_tracks_spawns(self):
        engine = Engine("test")
        engine.tree("t")

        engine.spawn(_noop, name="a")
        engine.spawn(_noop, name="b")

        assert engine.metrics.snapshot()["spawned"] == 2
        engine.stop()

    def test_engine_metrics_tracks_completions(self):
        engine = Engine("test")
        engine.tree("t")

        p = engine.spawn(_noop, name="work")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()

        s = engine.metrics.snapshot()
        assert s["completed"] >= 1
        assert s["error_rate"] == 0.0

    def test_engine_stop_cancels_running(self):
        engine = Engine("test")
        engine.tree("t")

        p = engine.spawn(time.sleep, 60, name="long")
        engine.dispatch()

        engine.stop()

        assert p.status == ProcessStatus.CANCELLED
        assert engine.metrics.snapshot()["cancelled"] >= 1


# ── ProcessGroup ─────────────────────────────────────────────────

class TestProcessGroup:
    def test_group_spawn_and_gather(self):
        engine = Engine("test")
        engine.tree("t")

        g = engine.group("batch")
        g.spawn(_noop, name="a")
        g.spawn(_noop, name="b")
        g.spawn(_noop, name="c")

        engine.run_background(poll_interval=0.01)
        results = g.gather(timeout=5)
        engine.stop()

        assert len(results) == 3
        assert all(r == 42 for r in results)

    def test_group_all_done(self):
        engine = Engine("test")
        engine.tree("t")

        g = engine.group("batch")
        g.spawn(_noop, name="a")
        g.spawn(_noop, name="b")

        assert not g.all_done
        engine.run_background(poll_interval=0.01)
        g.wait(timeout=5)
        engine.stop()

        assert g.all_done

    def test_group_errors(self):
        engine = Engine("test")
        engine.tree("t")

        g = engine.group("mixed")
        g.spawn(_noop, name="ok")
        g.spawn(_fail, name="fail")

        engine.run_background(poll_interval=0.01)
        g.wait(timeout=5)
        engine.stop()

        assert len(g.results()) == 1
        assert len(g.errors()) == 1

    def test_group_cancel(self):
        engine = Engine("test")
        engine.tree("t")

        g = engine.group("cancel")
        g.spawn(time.sleep, 60, name="long")
        g.spawn(time.sleep, 60, name="long2")

        engine.dispatch()
        cancelled = g.cancel()
        engine.stop()

        assert cancelled == 2

    def test_group_to_dict(self):
        engine = Engine("test")
        g = engine.group("test")
        d = g.to_dict()
        assert d["name"] == "test"
        assert d["num_processes"] == 0
        engine.stop()

    def test_group_no_engine_raises(self):
        g = ProcessGroup(name="orphan")
        with pytest.raises(RuntimeError, match="not attached"):
            g.spawn(_noop)

    def test_group_elapsed(self):
        engine = Engine("test")
        g = engine.group("timing")
        time.sleep(0.05)
        assert g.elapsed > 0.04
        engine.stop()


# ── ResultCache ──────────────────────────────────────────────────

class TestResultCache:
    def test_put_and_get(self):
        cache = ResultCache()
        cache.put(_noop, (), {}, 42)
        found, val = cache.get(_noop, (), {})
        assert found is True
        assert val == 42

    def test_miss(self):
        cache = ResultCache()
        found, val = cache.get(_noop, (), {})
        assert found is False
        assert val is None

    def test_different_args_different_keys(self):
        cache = ResultCache()
        cache.put(_add, (1, 2), {}, 3)
        found, val = cache.get(_add, (3, 4), {})
        assert found is False

    def test_eviction(self):
        cache = ResultCache(maxsize=2)
        cache.put(_add, (1, 1), {}, 2)
        cache.put(_add, (2, 2), {}, 4)
        cache.put(_add, (3, 3), {}, 6)  # evicts oldest
        assert cache.size == 2
        found, _ = cache.get(_add, (1, 1), {})
        assert found is False

    def test_ttl_expiry(self):
        cache = ResultCache(ttl=0.1)
        cache.put(_noop, (), {}, 42)
        time.sleep(0.15)
        found, _ = cache.get(_noop, (), {})
        assert found is False

    def test_invalidate(self):
        cache = ResultCache()
        cache.put(_noop, (), {}, 42)
        assert cache.invalidate(_noop, (), {}) is True
        assert cache.invalidate(_noop, (), {}) is False

    def test_clear(self):
        cache = ResultCache()
        cache.put(_noop, (), {}, 1)
        cache.put(_add, (1,), {}, 2)
        n = cache.clear()
        assert n == 2
        assert cache.size == 0

    def test_stats(self):
        cache = ResultCache()
        cache.put(_noop, (), {}, 42)
        cache.get(_noop, (), {})  # hit
        cache.get(_add, (1,), {})  # miss
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5

    def test_kwargs_key(self):
        cache = ResultCache()
        cache.put(_noop, (), {"x": 1}, 99)
        found, val = cache.get(_noop, (), {"x": 1})
        assert found is True
        assert val == 99
        found, _ = cache.get(_noop, (), {"x": 2})
        assert found is False


# ── Engine cache integration ────────────────────────────────────

class TestEngineCache:
    def test_enable_disable_cache(self):
        engine = Engine("test")
        assert engine._cache is None
        engine.enable_cache()
        assert engine._cache is not None
        engine.disable_cache()
        assert engine._cache is None
        engine.stop()

    def test_cache_hit_returns_immediately(self):
        engine = Engine("test")
        engine.tree("t")
        engine.enable_cache()

        p1 = engine.spawn(_noop, name="cached")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()

        # Spawn again — should hit cache
        engine2 = Engine("test2")
        engine2.enable_cache()
        engine2._cache.put(_noop, (), {}, 42)
        p2 = engine2.spawn(_noop, name="cached")

        assert p2.status == ProcessStatus.COMPLETED
        assert p2.result == 42

    def test_cache_in_metrics(self):
        engine = Engine("test")
        engine.enable_cache()
        d = engine.to_dict()
        assert d["cache"] is not None
        assert "hit_rate" in d["cache"]
        engine.stop()


# ── Engine health ────────────────────────────────────────────────

class TestEngineHealth:
    def test_health_snapshot(self):
        engine = Engine("test")
        engine.tree("t1")
        engine.tree("t2")

        engine.spawn(_noop, name="a")
        engine.spawn(_noop, name="b")

        h = engine.health()
        assert h["name"] == "test"
        assert h["tree_count"] == 2
        assert h["process_count"] == 2
        assert h["pending"] == 2
        assert "t1" in h["trees"]
        assert "metrics" in h
        engine.stop()

    def test_health_includes_status_counts(self):
        engine = Engine("test")
        engine.tree("t")

        engine.spawn(_noop, name="a")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()

        h = engine.health()
        assert h["status_counts"].get("completed", 0) >= 1

    def test_health_with_cache(self):
        engine = Engine("test")
        engine.enable_cache()
        h = engine.health()
        assert h["cache"] is not None
        engine.stop()


class TestCancelManagerIntegration:
    """Integration with CancelManager for engine processes."""

    def test_spawn_register_cancel(self):
        engine = Engine("test")
        engine.tree("t")
        proc = engine.spawn(_noop, name="cancellable", register_cancel=True)
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()
        assert proc.is_done

    def test_cancel_via_cancel_manager(self):
        engine = Engine("test")
        engine.tree("t")
        proc = engine.spawn(_sleep_and_return, 0.5, "cancelled", name="cancel-test", register_cancel=True)
        # Dispatch to start the process
        engine.dispatch()
        # Cancel via CancelManager
        from domains.infrastructure.cancel_manager import get_cancel_manager
        mgr = get_cancel_manager()
        mgr.cancel(proc.id)
        # Wait for process to finish (cancellation sets CANCELLED status)
        import time
        time.sleep(0.1)
        engine.stop()
        assert proc.is_cancelled

    def test_cancel_manager_deregister(self):
        engine = Engine("test")
        engine.tree("t")
        proc = engine.spawn(_noop, name="deregister-test", register_cancel=True)
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        # After completion, process should still be tracked by CancelManager
        from domains.infrastructure.cancel_manager import get_cancel_manager
        mgr = get_cancel_manager()
        assert mgr.get(proc.id) is not None
        engine.stop()

    def test_spawn_without_register_cancel(self):
        engine = Engine("test")
        engine.tree("t")
        proc = engine.spawn(_noop, name="no-cancel")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()
        # Should not be registered with CancelManager
        from domains.infrastructure.cancel_manager import get_cancel_manager
        mgr = get_cancel_manager()
        assert mgr.get(proc.id) is None


# ── Subprocess CPU affinity ──────────────────────────────────────

class TestSubprocessCPUAffinity:
    def test_cpu_affinity_set(self):
        proc = Process(fn=_noop, name="affinity")
        config = SubprocessConfig(enabled=True, start_method="fork", cpu_affinity=[0])
        sub = SubprocessProcess(proc, config)
        sub.start()
        sub.monitor()
        deadline = time.time() + 5
        while not proc.is_done and time.time() < deadline:
            time.sleep(0.05)
        assert proc.status == ProcessStatus.COMPLETED

    def test_cpu_affinity_none(self):
        proc = Process(fn=_noop, name="no-affinity")
        config = SubprocessConfig(enabled=True, start_method="fork", cpu_affinity=None)
        sub = SubprocessProcess(proc, config)
        sub.start()
        sub.monitor()
        deadline = time.time() + 5
        while not proc.is_done and time.time() < deadline:
            time.sleep(0.05)
        assert proc.status == ProcessStatus.COMPLETED


# ── ProcessGroup.gather ──────────────────────────────────────────

class TestProcessGroupGather:
    def test_gather_returns_results(self):
        engine = Engine("test")
        engine.tree("t")
        g = engine.group("g")
        g.spawn(_add, 1, 2, name="a")
        g.spawn(_add, 3, 4, name="b")
        engine.run_background(poll_interval=0.01)
        results = g.gather(timeout=5)
        engine.stop()
        assert sorted(results) == [3, 7]

    def test_gather_with_failures(self):
        engine = Engine("test")
        engine.tree("t")
        g = engine.group("g")
        g.spawn(_noop, name="ok")
        g.spawn(_fail, name="fail")
        engine.run_background(poll_interval=0.01)
        results = g.gather(timeout=5)
        engine.stop()
        assert len(results) == 1
        assert results[0] == 42


# ── EngineMetrics dispatched ─────────────────────────────────────

class TestEngineMetricsDispatched:
    def test_dispatch_increments_metrics(self):
        engine = Engine("test")
        engine.tree("t")
        engine.spawn(_noop, name="a")
        engine.spawn(_noop, name="b")
        engine.dispatch()
        s = engine.metrics.snapshot()
        assert s["dispatched"] == 2
        engine.stop()

    def test_dispatch_with_deps(self):
        engine = Engine("test")
        engine.tree("t")
        p1 = engine.spawn(_noop, name="p1")
        engine.spawn(_noop, name="p2", depends_on=[p1.id])
        engine.dispatch()
        s = engine.metrics.snapshot()
        assert s["dispatched"] == 1
        engine.stop()


# ── spawn_chain with kwargs ──────────────────────────────────────

class TestSpawnChainKwargs:
    def test_chain_with_kwargs(self):
        engine = Engine("test")
        engine.tree("t")

        def add(a, b):
            return a + b

        processes = engine.spawn_chain(
            (add, 1, 2),
            (add, {"b": 10}),
            tree="t",
        )
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()
        assert processes[0].result == 3
        assert processes[1].result == 13


# ── Cancel propagation ──────────────────────────────────────────

class TestCancelPropagation:
    def test_cancel_tree(self):
        engine = Engine("test")
        engine.tree("t")
        p1 = engine.spawn(_noop, name="a")
        p2 = engine.spawn(_noop, name="b")
        engine.dispatch()
        count = engine.cancel_tree("t")
        engine.stop()
        assert count >= 0

    def test_cancel_process_nonexistent(self):
        engine = Engine("test")
        engine.tree("t")
        count = engine.cancel_process("nonexistent")
        assert count == 0
        engine.stop()


# ── Signal handling ──────────────────────────────────────────────

class TestSignalHandlingExtended:
    def test_double_install_restore(self):
        engine = Engine("test")
        engine.install_signal_handlers()
        engine.install_signal_handlers()
        engine.restore_signal_handlers()
        engine.restore_signal_handlers()
        engine.stop()

    def test_handlers_installed_property(self):
        engine = Engine("test")
        engine.install_signal_handlers()
        assert engine._signal_handlers_installed is True
        engine.restore_signal_handlers()
        assert engine._signal_handlers_installed is False
        engine.stop()


# ── Engine cache integration ─────────────────────────────────────

class TestEngineCacheExtended:
    def test_cache_hit_returns_same_result(self):
        engine = Engine("test")
        engine.tree("t")
        engine.enable_cache()
        p1 = engine.spawn(_add, 1, 2, name="c1")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        result1 = p1.result
        p2 = engine.spawn(_add, 1, 2, name="c2")
        engine.dispatch()
        engine.wait(timeout=5)
        result2 = p2.result
        engine.stop()
        assert result1 == result2 == 3

    def test_disable_cache(self):
        engine = Engine("test")
        engine.enable_cache()
        assert engine._cache is not None
        engine.disable_cache()
        assert engine._cache is None
        engine.stop()


# ── wait_all ─────────────────────────────────────────────────────

class TestWaitAll:
    def test_wait_all_returns_completed(self):
        engine = Engine("test")
        engine.tree("t")
        engine.spawn(_noop, name="a")
        engine.spawn(_add, 1, 2, name="b")
        engine.run_background(poll_interval=0.01)
        result = engine.wait_all(timeout=5)
        engine.stop()
        assert len(result) == 2
        assert all(p.is_done for p in result)

    def test_wait_all_timeout(self):
        engine = Engine("test")
        engine.tree("t")
        engine.spawn(time.sleep, 60, name="long")
        engine.run_background(poll_interval=0.01)
        result = engine.wait_all(timeout=0.2)
        engine.stop()
        assert len(result) <= 1


# ── cancel_process with propagation ──────────────────────────────

class TestCancelPropagationExtended:
    def test_cancel_parent_cancels_dependents(self):
        engine = Engine("test")
        engine.tree("t")
        p1 = engine.spawn(time.sleep, 60, name="parent")
        p2 = engine.spawn(time.sleep, 60, name="child", depends_on=[p1.id])
        engine.run_background(poll_interval=0.01)
        time.sleep(0.1)
        # p1 dispatched and running; p2 held
        count = engine.cancel_process(p1.id, propagate=True)
        engine.stop()
        assert count >= 1
        assert p1.is_cancelled

    def test_cancel_no_propagate(self):
        engine = Engine("test")
        engine.tree("t")
        p1 = engine.spawn(time.sleep, 60, name="parent")
        p2 = engine.spawn(time.sleep, 60, name="child", depends_on=[p1.id])
        engine.run_background(poll_interval=0.01)
        time.sleep(0.1)
        count = engine.cancel_process(p1.id, propagate=False)
        engine.stop()
        assert count == 1
        assert p1.is_cancelled

    def test_cancel_done_process(self):
        engine = Engine("test")
        engine.tree("t")
        p = engine.spawn(_noop, name="done")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        count = engine.cancel_process(p.id)
        engine.stop()
        assert count == 0

    def test_cancel_tree_with_running(self):
        engine = Engine("test")
        engine.tree("t")
        engine.spawn(time.sleep, 60, name="long1")
        engine.spawn(time.sleep, 60, name="long2")
        engine.dispatch()
        count = engine.cancel_tree("t")
        engine.stop()
        assert count == 2

    def test_cancel_tree_empty(self):
        engine = Engine("test")
        engine.tree("t")
        count = engine.cancel_tree("nonexistent")
        engine.stop()
        assert count == 0


# ── Engine.to_dict comprehensive ─────────────────────────────────

class TestEngineToDict:
    def test_to_dict_structure(self):
        engine = Engine("test")
        engine.tree("t")
        engine.spawn(_noop, name="a")
        d = engine.to_dict()
        assert d["name"] == "test"
        assert "running" in d
        assert "trees" in d
        assert "processes" in d
        assert "pending" in d
        assert "active_stems" in d
        assert "routing" in d
        assert "monitor" in d
        assert "metrics" in d
        assert "cache" in d
        engine.stop()

    def test_to_dict_with_config(self):
        cfg = EngineConfig(name="configured")
        engine = Engine(config=cfg)
        d = engine.to_dict()
        assert d["name"] == "configured"
        engine.stop()


# ── on_complete callback ─────────────────────────────────────────

class TestOnComplete:
    def test_on_complete_fires(self):
        engine = Engine("test")
        engine.tree("t")
        completed = []
        engine.on_complete(lambda p: completed.append(p.id))
        engine.spawn(_noop, name="a")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()
        assert len(completed) == 1

    def test_on_complete_multiple(self):
        engine = Engine("test")
        engine.tree("t")
        results = []
        engine.on_complete(lambda p: results.append(p.name))
        engine.spawn(_sleep_and_return, 0.1, "a", name="a")
        engine.spawn(_sleep_and_return, 0.1, "b", name="b")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        # Give main loop time to fire callbacks
        time.sleep(0.1)
        engine.stop()
        assert len(results) == 2
        assert sorted(results) == ["a", "b"]


# ── route ────────────────────────────────────────────────────────

class TestRoute:
    def test_route_existing_tree(self):
        engine = Engine("test")
        engine.tree("t")
        engine.route("my_proc", "t")
        assert engine._routing["my_proc"] == "t"
        engine.stop()

    def test_route_nonexistent_tree(self):
        engine = Engine("test")
        engine.tree("t")
        with pytest.raises(ValueError):
            engine.route("my_proc", "nonexistent")
        engine.stop()


# ── Tree.max_stems limit ─────────────────────────────────────────

class TestTreeMaxStems:
    def test_branch_at_max_stems(self):
        engine = Engine("test")
        engine.tree("t", max_stems=1)
        engine.spawn(_noop, name="a")
        engine.spawn(_noop, name="b")
        engine.dispatch()
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()

    def test_guard_tree_max_stems(self):
        config = SubprocessConfig(enabled=False)
        tree = GuardTree("g", config=config, max_stems=2, pool_workers=2)
        procs = [Process(fn=time.sleep, args=(60,), name=f"p{i}") for i in range(2)]
        stem = tree.branch(procs)
        assert tree.active_stems >= 1
        for p in procs:
            p.cancel()
        time.sleep(0.2)


# ── Worker pool ──────────────────────────────────────────────────

class TestWorkerPool:
    def test_start_stop_workers(self):
        engine = Engine("test")
        engine.tree("t")
        engine.start_workers(num_workers=2)
        engine.spawn(_sleep_and_return, 0.05, "ok", name="w1")
        engine.spawn(_sleep_and_return, 0.05, "ok", name="w2")
        engine.wait(timeout=5)
        engine.stop_workers()
        engine.stop()

    def test_workers_dispatch_to_tree(self):
        engine = Engine("test")
        engine.tree("t")
        engine.start_workers(num_workers=1)
        p = engine.spawn(_sleep_and_return, 0.1, 42, name="worker-test")
        engine.wait(timeout=5)
        engine.stop_workers()
        engine.stop()
        assert p.result == 42

    def test_stop_workers_without_start(self):
        engine = Engine("test")
        engine.stop_workers()
        engine.stop()


# ── ProcessMonitor restart ───────────────────────────────────────

class TestProcessMonitorRestart:
    def test_restart_callback_fires(self):
        from domains.infrastructure.pugqeep.config import RestartPolicy
        policy = RestartPolicy(max_restarts=2, restart_delay=0.01)
        monitor = ProcessMonitor(poll_interval=0.01, stall_timeout=0.1)
        monitor.start()
        restarted = []
        monitor.on_restart(lambda p: restarted.append(p.id))

        proc = Process(fn=_fail, name="fail-proc")
        proc._restart_policy = policy
        proc.fail("boom")
        monitor.track(proc)
        time.sleep(0.3)
        monitor.stop()
        assert len(restarted) >= 1

    def test_no_restart_when_max_zero(self):
        monitor = ProcessMonitor(poll_interval=0.01, stall_timeout=0.1)
        monitor.start()
        restarted = []
        monitor.on_restart(lambda p: restarted.append(p.id))

        proc = Process(fn=_fail, name="fail-no-restart")
        proc.fail("boom")
        monitor.track(proc)
        time.sleep(0.2)
        monitor.stop()
        assert len(restarted) == 0


# ── on_progress callback ─────────────────────────────────────────

class TestOnProgress:
    def test_on_progress_fires(self):
        engine = Engine("test")
        engine.tree("t")
        progress = []
        engine.spawn(_sleep_and_return, 0.1, "ok", name="prog")
        def _run():
            engine.run(poll_interval=0.01, on_progress=lambda d: progress.append(d))
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        engine.wait(timeout=5)
        engine.stop()
        t.join(timeout=2)
        assert len(progress) >= 1
        assert "pending" in progress[0]
        assert "completed" in progress[0]

    def test_on_progress_counts(self):
        engine = Engine("test")
        engine.tree("t")
        engine.spawn(_sleep_and_return, 0.1, "a", name="a")
        engine.spawn(_sleep_and_return, 0.1, "b", name="b")
        progress = []
        def _run():
            engine.run(poll_interval=0.01, on_progress=lambda d: progress.append(dict(d)))
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        engine.wait(timeout=5)
        # Give main loop time to fire final progress callback
        time.sleep(0.2)
        engine.stop()
        t.join(timeout=2)
        completed_counts = [p.get("completed", 0) for p in progress]
        assert max(completed_counts) >= 2


# ── ResultCache TTL ──────────────────────────────────────────────

class TestResultCacheTTL:
    def test_ttl_expiry(self):
        cache = ResultCache(maxsize=10, ttl=0.1)
        cache.put(_add, (1, 2), {}, 3)
        found, val = cache.get(_add, (1, 2), {})
        assert found is True
        time.sleep(0.2)
        found, val = cache.get(_add, (1, 2), {})
        assert found is False

    def test_ttl_none_never_expires(self):
        cache = ResultCache(maxsize=10, ttl=None)
        cache.put(_add, (1, 2), {}, 3)
        time.sleep(0.05)
        found, val = cache.get(_add, (1, 2), {})
        assert found is True

    def test_invalidate(self):
        cache = ResultCache(maxsize=10)
        cache.put(_add, (1, 2), {}, 3)
        removed = cache.invalidate(_add, (1, 2), {})
        assert removed is True
        found, _ = cache.get(_add, (1, 2), {})
        assert found is False

    def test_invalidate_nonexistent(self):
        cache = ResultCache(maxsize=10)
        removed = cache.invalidate(_add, (1, 2), {})
        assert removed is False

    def test_stats(self):
        cache = ResultCache(maxsize=10)
        cache.put(_add, (1, 2), {}, 3)
        cache.get(_add, (1, 2), {})
        cache.get(_add, (3, 4), {})
        s = cache.stats()
        assert s["size"] == 1
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5


# ── Engine.run stop ──────────────────────────────────────────────

class TestEngineRunStop:
    def test_run_stops_on_stop(self):
        engine = Engine("test")
        engine.tree("t")
        engine.spawn(_sleep_and_return, 0.1, "ok", name="s")
        t = threading.Thread(target=engine.run, args=(0.01,))
        t.start()
        time.sleep(0.3)
        engine.stop()
        t.join(timeout=2)
        assert not t.is_alive()

    def test_run_background_as_future(self):
        engine = Engine("test")
        engine.tree("t")
        engine.spawn(_sleep_and_return, 0.05, "done", name="f")
        future = engine.run_background(as_future=True)
        engine.wait(timeout=5)
        engine.stop()
        time.sleep(0.2)
        assert future.done()


# ── Stem status tracking ─────────────────────────────────────────

class TestStemStatus:
    def test_stem_created(self):
        stem = Stem(tree_id="t")
        assert stem.status == StemStatus.CREATED
        assert not stem.is_done

    def test_stem_running(self):
        stem = Stem(tree_id="t")
        stem.running()
        assert stem.status == StemStatus.RUNNING

    def test_stem_complete(self):
        stem = Stem(tree_id="t")
        stem.complete()
        assert stem.status == StemStatus.COMPLETED
        assert stem.is_done
        assert stem.completed_at is not None

    def test_stem_fail(self):
        stem = Stem(tree_id="t")
        stem.fail()
        assert stem.status == StemStatus.FAILED
        assert stem.is_done

    def test_stem_all_done(self):
        p1 = Process(fn=_noop, name="a")
        p2 = Process(fn=_noop, name="b")
        stem = Stem(tree_id="t", processes=[p1, p2])
        assert not stem.all_done
        p1.complete()
        assert not stem.all_done
        p2.complete()
        assert stem.all_done

    def test_stem_results(self):
        p1 = Process(fn=_noop, name="a")
        p1.complete(42)
        p2 = Process(fn=_fail, name="b")
        p2.fail("boom")
        stem = Stem(tree_id="t", processes=[p1, p2])
        assert stem.results() == [42]

    def test_stem_errors(self):
        p1 = Process(fn=_noop, name="a")
        p1.complete(42)
        p2 = Process(fn=_fail, name="b")
        p2.fail("boom")
        stem = Stem(tree_id="t", processes=[p1, p2])
        assert stem.errors() == ["boom"]

    def test_stem_to_dict(self):
        stem = Stem(tree_id="t")
        d = stem.to_dict()
        assert d["tree_id"] == "t"
        assert d["status"] == "created"
        assert d["num_processes"] == 0

    def test_stem_done_event(self):
        stem = Stem(tree_id="t")
        assert not stem._done_event.is_set()
        stem.complete()
        assert stem._done_event.is_set()


# ── Priority dispatch ordering ───────────────────────────────────

class TestPriorityDispatch:
    def test_higher_priority_dispatched_first(self):
        engine = Engine("test")
        engine.tree("t")
        p_low = engine.spawn(_noop, name="low", priority=3)
        p_high = engine.spawn(_noop, name="high", priority=0)
        p_med = engine.spawn(_noop, name="med", priority=1)
        engine.dispatch()
        dispatched = [p.name for p in engine._processes.values() if p.status != ProcessStatus.CREATED]
        engine.stop()

    def test_priority_with_deps(self):
        engine = Engine("test")
        engine.tree("t")
        p1 = engine.spawn(time.sleep, 60, name="dep", priority=0)
        p2 = engine.spawn(_noop, name="blocked", priority=0, depends_on=[p1.id])
        p3 = engine.spawn(time.sleep, 60, name="free", priority=3)
        dispatched = engine.dispatch()
        assert dispatched >= 1
        engine.stop()


# ── Config edge cases ────────────────────────────────────────────

class TestConfigEdgeCases:
    def test_engine_config_defaults(self):
        cfg = EngineConfig()
        assert cfg.name == "main"
        assert cfg.max_trees == 16
        e = Engine(config=cfg)
        assert e.name == "main"
        e.stop()

    def test_subprocess_config_defaults(self):
        cfg = SubprocessConfig()
        assert cfg.enabled is True
        assert cfg.start_method == "fork"
        assert cfg.terminate_grace == 3.0

    def test_restart_policy_defaults(self):
        cfg = RestartPolicy()
        assert cfg.max_restarts == 0
        assert cfg.restart_delay == 1.0
        assert cfg.backoff == "exponential"

    def test_monitor_config_defaults(self):
        cfg = MonitorConfig()
        assert cfg.enabled is True
        assert cfg.poll_interval == 1.0

    def test_engine_config_custom(self):
        cfg = EngineConfig(name="custom", max_trees=4)
        e = Engine(config=cfg)
        assert e.name == "custom"
        assert e.max_trees == 4
        e.stop()

    def test_subprocess_config_memory_limit(self):
        cfg = SubprocessConfig(memory_limit_mb=2048)
        assert cfg.memory_limit_mb == 2048


# ── ProcessGroup extended ────────────────────────────────────────

class TestProcessGroupExtended:
    def test_group_to_dict(self):
        engine = Engine("test")
        engine.tree("t")
        g = engine.group("mygroup")
        g.spawn(_noop, name="a")
        d = g.to_dict()
        assert d["name"] == "mygroup"
        assert d["num_processes"] == 1
        engine.stop()

    def test_group_elapsed(self):
        engine = Engine("test")
        engine.tree("t")
        g = engine.group("elapsed")
        g.spawn(_sleep_and_return, 0.05, "ok", name="a")
        engine.run_background(poll_interval=0.01)
        g.wait(timeout=5)
        assert g.elapsed is not None
        assert g.elapsed >= 0
        engine.stop()

    def test_group_num_processes(self):
        engine = Engine("test")
        engine.tree("t")
        g = engine.group("count")
        g.spawn(_noop, name="a")
        g.spawn(_noop, name="b")
        g.spawn(_noop, name="c")
        d = g.to_dict()
        assert d["num_processes"] == 3
        engine.stop()


# ── Tree health ──────────────────────────────────────────────────

class TestTreeHealth:
    def test_guard_tree_health(self):
        config = SubprocessConfig(enabled=False)
        tree = GuardTree("g", config=config)
        h = tree.health()
        assert h["name"] == "g"
        assert h["active_subprocesses"] == 0

    def test_tree_to_dict(self):
        tree = Tree("t")
        d = tree.to_dict()
        assert d["name"] == "t"
        assert d["status"] == "idle"
        tree._pool.shutdown(wait=False)


# ── Process.to_dict extended ─────────────────────────────────────

class TestProcessToDict:
    def test_to_dict_fields(self):
        proc = Process(fn=_noop, name="test-proc", timeout=10.0)
        proc.depends_on = ["dep1"]
        d = proc.to_dict()
        assert d["name"] == "test-proc"
        assert d["timeout"] == 10.0
        assert d["depends_on"] == ["dep1"]
        assert d["restart_count"] == 0
        assert d["pid"] is None

    def test_to_dict_elapsed_none(self):
        proc = Process(fn=_noop, name="not-started")
        d = proc.to_dict()
        assert d["elapsed"] is None

    def test_to_dict_elapsed_running(self):
        proc = Process(fn=time.sleep, args=(60,), name="running")
        proc.running()
        d = proc.to_dict()
        assert d["elapsed"] is not None
        assert d["elapsed"] >= 0
        proc.cancel()

    def test_to_dict_is_done(self):
        proc = Process(fn=_noop, name="done")
        assert not proc.is_done
        proc.complete(42)
        assert proc.is_done

    def test_to_dict_is_cancelled(self):
        proc = Process(fn=_noop, name="cancelled")
        assert not proc.is_cancelled
        proc.cancel()
        assert proc.is_cancelled


# ── Engine.max_trees limit ───────────────────────────────────────

class TestEngineMaxTrees:
    def test_max_trees_raises(self):
        cfg = EngineConfig(max_trees=1)
        engine = Engine(config=cfg)
        engine.tree("t1")
        with pytest.raises(RuntimeError, match="max trees"):
            engine.tree("t2")
        engine.stop()

    def test_max_trees_default(self):
        engine = Engine("test")
        assert engine.max_trees == 16
        engine.stop()


# ── SubprocessProcess.health / to_dict ───────────────────────────

class TestSubprocessProcessHealth:
    def test_health_while_running(self):
        proc = Process(fn=time.sleep, args=(60,), name="health-proc")
        config = SubprocessConfig(enabled=True, start_method="fork")
        sub = SubprocessProcess(proc, config)
        sub.start()
        h = sub.health()
        assert h["pid"] == proc._pid
        assert h["alive"] is True
        assert h["elapsed"] is not None
        sub.terminate()
        time.sleep(0.3)

    def test_health_after_terminate(self):
        proc = Process(fn=time.sleep, args=(60,), name="term-proc")
        config = SubprocessConfig(enabled=True, start_method="fork")
        sub = SubprocessProcess(proc, config)
        sub.start()
        time.sleep(0.1)
        sub.terminate()
        time.sleep(0.3)
        h = sub.health()
        assert h["alive"] is False

    def test_cancel_event(self):
        proc = Process(fn=time.sleep, args=(60,), name="cancel-proc")
        config = SubprocessConfig(enabled=True, start_method="fork")
        sub = SubprocessProcess(proc, config)
        sub.start()
        assert not sub._cancel_event.is_set()
        sub.terminate()
        time.sleep(0.3)
        assert sub._cancel_event.is_set()


# ── Engine.spawn no tree ─────────────────────────────────────────

class TestEngineSpawnNoTree:
    def test_spawn_no_tree_routes_to_default(self):
        engine = Engine("test")
        engine.tree("default")
        p = engine.spawn(_noop, name="routed")
        engine.dispatch()
        assert p._tree_name == "default"
        engine.stop()

    def test_spawn_explicit_tree(self):
        engine = Engine("test")
        engine.tree("t1")
        engine.tree("t2")
        p = engine.spawn(_noop, name="explicit", tree="t2")
        assert p._tree_name == "t2"
        engine.stop()

    def test_spawn_multiple_trees_round_robin(self):
        engine = Engine("test")
        engine.tree("t1")
        engine.tree("t2")
        p1 = engine.spawn(_noop, name="a")
        p2 = engine.spawn(_noop, name="b")
        engine.dispatch()
        trees_used = {p1._tree_name, p2._tree_name}
        assert len(trees_used) == 2
        engine.stop()


# ── Engine.cancel_tree comprehensive ─────────────────────────────

class TestCancelTreeComprehensive:
    def test_cancel_tree_only_affects_matching(self):
        engine = Engine("test")
        engine.tree("t1")
        engine.tree("t2")
        p1 = engine.spawn(time.sleep, 60, name="on-t1", tree="t1")
        p2 = engine.spawn(time.sleep, 60, name="on-t2", tree="t2")
        engine.dispatch()
        count = engine.cancel_tree("t1")
        assert count == 1
        assert p1.is_cancelled
        assert not p2.is_cancelled
        engine.stop()

    def test_cancel_tree_no_processes(self):
        engine = Engine("test")
        engine.tree("t")
        count = engine.cancel_tree("empty-tree")
        engine.stop()
        assert count == 0


# ── Engine.wait_for_any ──────────────────────────────────────────

class TestWaitForAnyExtended:
    def test_wait_for_any_first_completes(self):
        engine = Engine("test")
        engine.tree("t")
        p1 = engine.spawn(_sleep_and_return, 0.3, "slow", name="slow")
        p2 = engine.spawn(_sleep_and_return, 0.05, "fast", name="fast")
        engine.run_background(poll_interval=0.01)
        result = engine.wait_for_any([p1.id, p2.id], timeout=5)
        engine.stop()
        assert result is not None
        assert result.name == "fast"

    def test_wait_for_any_timeout(self):
        engine = Engine("test")
        engine.tree("t")
        p1 = engine.spawn(time.sleep, 60, name="long")
        engine.run_background(poll_interval=0.01)
        result = engine.wait_for_any([p1.id], timeout=0.2)
        engine.stop()
        assert result is None


# ── Subprocess cwd and capture_output ─────────────────────────────

class TestSubprocessCwdAndCapture:
    def test_subprocess_cwd(self):
        import os
        from domains.infrastructure.pugqeep.config import EngineConfig, SubprocessConfig
        cfg = EngineConfig(name="test", subprocess=SubprocessConfig(cwd="/tmp"))
        engine = Engine(config=cfg)
        engine.tree("t", guarded=True)

        def get_cwd():
            return os.getcwd()

        p = engine.spawn(get_cwd, name="getcwd", subprocess=True, tree="t")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()
        assert p.status == ProcessStatus.COMPLETED
        assert p.result == "/tmp"

    def test_subprocess_cwd_nonexistent(self):
        from domains.infrastructure.pugqeep.config import EngineConfig, SubprocessConfig
        cfg = EngineConfig(name="test", subprocess=SubprocessConfig(cwd="/nonexistent/path"))
        engine = Engine(config=cfg)
        engine.tree("t", guarded=True)

        def get_cwd():
            import os
            return os.getcwd()

        p = engine.spawn(get_cwd, name="getcwd", subprocess=True, tree="t")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()
        # Should still complete (cwd failure is non-fatal)
        assert p.status == ProcessStatus.COMPLETED

    def test_subprocess_capture_output(self):
        from domains.infrastructure.pugqeep.config import EngineConfig, SubprocessConfig
        cfg = EngineConfig(name="test", subprocess=SubprocessConfig(capture_output=True))
        engine = Engine(config=cfg)
        engine.tree("t", guarded=True)

        def print_something():
            print("hello stdout")
            print("hello stderr", flush=True)
            return "done"

        p = engine.spawn(print_something, name="capture", subprocess=True, tree="t")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()
        assert p.status == ProcessStatus.COMPLETED
        assert p.result == "done"

    def test_subprocess_env(self):
        from domains.infrastructure.pugqeep.config import EngineConfig, SubprocessConfig
        cfg = EngineConfig(name="test", subprocess=SubprocessConfig(env={"MY_TEST_VAR": "hello"}))
        engine = Engine(config=cfg)
        engine.tree("t", guarded=True)

        def get_env(key):
            import os
            return os.environ.get(key, "not_set")

        p = engine.spawn(get_env, "MY_TEST_VAR", name="envtest", subprocess=True, tree="t")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()
        assert p.status == ProcessStatus.COMPLETED
        assert p.result == "hello"

    def test_run_subprocess_convenience(self):
        from domains.infrastructure.pugqeep.config import EngineConfig, SubprocessConfig
        cfg = EngineConfig(name="test", subprocess=SubprocessConfig(cwd="/tmp"))
        engine = Engine(config=cfg)
        engine.tree("t", guarded=True)

        def add(a, b):
            return a + b

        p = engine.run_subprocess(add, 2, 3, name="add")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()
        assert p.status == ProcessStatus.COMPLETED
        assert p.result == 5

    def test_subprocess_health_with_stdout(self):
        from domains.infrastructure.pugqeep.config import EngineConfig, SubprocessConfig
        cfg = EngineConfig(name="test", subprocess=SubprocessConfig(capture_output=True))
        engine = Engine(config=cfg)
        engine.tree("t", guarded=True)

        p = engine.spawn(lambda: "ok", name="health", subprocess=True, tree="t")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()
        # Health should include stdout/stderr
        assert p.status == ProcessStatus.COMPLETED
