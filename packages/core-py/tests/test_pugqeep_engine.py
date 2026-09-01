"""Comprehensive tests for PGQ Engine — all pure logic, no mocks."""

import json
import os
import tempfile
import threading
import time

import pytest

from domains.infrastructure.pugqeep.config import EngineConfig, MonitorConfig, RestartPolicy, SubprocessConfig
from domains.infrastructure.pugqeep.engine import (
    Engine,
    EngineMetrics,
    Process,
    ProcessGroup,
    ProcessMonitor,
    ProcessStatus,
    ResultCache,
    SchedulingPolicy,
    Stem,
    StemStatus,
    Tree,
    TreeStatus,
)


# ── Helpers ──────────────────────────────────────────────────────

def _noop():
    return "ok"


def _sleep_and_return(secs, value):
    time.sleep(secs)
    return value


def _fail():
    raise RuntimeError("boom")


def _identity(x):
    return x


def _add(a, b):
    return a + b


# ══════════════════════════════════════════════════════════════════
# Process lifecycle
# ══════════════════════════════════════════════════════════════════

class TestProcess:
    def test_created_by_default(self):
        p = Process(fn=_noop)
        assert p.status == ProcessStatus.CREATED
        assert not p.is_done
        assert p.result is None
        assert p.error is None

    def test_ready(self):
        p = Process(fn=_noop)
        p.ready()
        assert p.status == ProcessStatus.READY

    def test_running_records_start_time(self):
        p = Process(fn=_noop)
        before = time.time()
        p.running()
        assert p.status == ProcessStatus.RUNNING
        assert p.started_at is not None
        assert p.started_at >= before

    def test_complete_sets_result_and_time(self):
        p = Process(fn=_noop)
        p.running()
        before = time.time()
        p.complete("result")
        assert p.status == ProcessStatus.COMPLETED
        assert p.result == "result"
        assert p.is_done
        assert p.completed_at is not None
        assert p.completed_at >= before

    def test_fail_sets_error(self):
        p = Process(fn=_noop)
        p.running()
        p.fail("error msg")
        assert p.status == ProcessStatus.FAILED
        assert p.error == "error msg"
        assert p.is_done

    def test_cancel_sets_status(self):
        p = Process(fn=_noop)
        p.cancel()
        assert p.status == ProcessStatus.CANCELLED
        assert p.is_done
        assert p.is_cancelled

    def test_cancel_sets_event(self):
        p = Process(fn=_noop)
        assert not p._cancel_event.is_set()
        p.cancel()
        assert p._cancel_event.is_set()

    def test_is_done_for_all_terminal_states(self):
        for status in (ProcessStatus.COMPLETED, ProcessStatus.FAILED, ProcessStatus.CANCELLED):
            p = Process(fn=_noop)
            p.status = status
            assert p.is_done

    def test_is_not_done_for_non_terminal(self):
        for status in (ProcessStatus.CREATED, ProcessStatus.READY, ProcessStatus.RUNNING, ProcessStatus.WAITING):
            p = Process(fn=_noop)
            p.status = status
            assert not p.is_done

    def test_elapsed_none_before_start(self):
        p = Process(fn=_noop)
        assert p.elapsed is None

    def test_elapsed_after_start(self):
        p = Process(fn=_noop)
        p.running()
        time.sleep(0.01)
        assert p.elapsed > 0

    def test_elapsed_after_complete(self):
        p = Process(fn=_noop)
        p.running()
        time.sleep(0.01)
        p.complete()
        e = p.elapsed
        assert e is not None
        assert e > 0

    def test_emit_streaming(self):
        p = Process(fn=_noop)
        p.emit(1)
        p.emit(2)
        p.emit(3)
        assert p.stream_results == [1, 2, 3]

    def test_on_complete_callback(self):
        p = Process(fn=_noop)
        results = []
        p.on_complete(lambda proc: results.append(proc.id))
        p.complete("x")
        assert results == [p.id]

    def test_on_fail_callback(self):
        p = Process(fn=_noop)
        errors = []
        p.on_fail(lambda proc: errors.append(proc.error))
        p.fail("err")
        assert errors == ["err"]

    def test_on_cancel_callback(self):
        p = Process(fn=_noop)
        cancelled = []
        p.on_cancel(lambda proc: cancelled.append(True))
        p.cancel()
        assert cancelled == [True]

    def test_on_stream_callback(self):
        p = Process(fn=_noop)
        values = []
        p.on_stream(lambda proc, v: values.append(v))
        p.emit("a")
        p.emit("b")
        assert values == ["a", "b"]

    def test_on_progress_callback(self):
        p = Process(fn=_noop)
        progress_updates = []
        p.on_progress(lambda proc, prog, msg: progress_updates.append((prog, msg)))
        p.report_progress(0.5, "halfway")
        assert progress_updates == [(0.5, "halfway")]

    def test_report_progress_clamps(self):
        p = Process(fn=_noop)
        p.report_progress(2.0)
        assert p.progress == 1.0
        p.report_progress(-1.0)
        assert p.progress == 0.0

    def test_report_progress_stores_message(self):
        p = Process(fn=_noop)
        p.report_progress(0.5, "msg")
        assert p.progress_message == "msg"

    def test_callback_exception_does_not_crash(self):
        p = Process(fn=_noop)
        p.on_complete(lambda proc: 1 / 0)
        p.on_fail(lambda proc: 1 / 0)
        p.on_cancel(lambda proc: 1 / 0)
        p.on_stream(lambda proc, v: 1 / 0)
        p.on_progress(lambda proc, p, m: 1 / 0)
        p.running()
        p.complete("x")
        p.status = ProcessStatus.RUNNING
        p.fail("e")
        p.status = ProcessStatus.RUNNING
        p.cancel()

    def test_to_dict_fields(self):
        p = Process(fn=_noop, name="test")
        p.running()
        p.complete("res")
        d = p.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "completed"
        assert "id" in d
        assert "elapsed" in d
        assert "restart_count" in d
        assert "is_done" in d
        assert "progress" in d
        assert "stream_count" in d

    def test_unique_ids(self):
        ids = {Process(fn=_noop).id for _ in range(100)}
        assert len(ids) == 100

    def test_default_priority(self):
        p = Process(fn=_noop)
        assert p._priority == 2

    def test_timeout_default_none(self):
        p = Process(fn=_noop)
        assert p.timeout is None

    def test_wait_cancel_timeout(self):
        p = Process(fn=_noop)
        p.running()
        start = time.time()
        p.wait_cancel(timeout=0.05)
        elapsed = time.time() - start
        assert elapsed < 0.5


# ══════════════════════════════════════════════════════════════════
# Stem
# ══════════════════════════════════════════════════════════════════

class TestStem:
    def test_all_done_when_empty(self):
        s = Stem()
        assert s.all_done

    def test_all_done_when_all_complete(self):
        p1 = Process(fn=_noop)
        p2 = Process(fn=_noop)
        p1.complete()
        p2.complete()
        s = Stem(processes=[p1, p2])
        assert s.all_done

    def test_not_all_done_when_one_running(self):
        p1 = Process(fn=_noop)
        p2 = Process(fn=_noop)
        p1.complete()
        p2.running()
        s = Stem(processes=[p1, p2])
        assert not s.all_done

    def test_results_collects_completed(self):
        p1 = Process(fn=_noop)
        p2 = Process(fn=_noop)
        p1.complete("a")
        p2.fail("err")
        s = Stem(processes=[p1, p2])
        assert s.results() == ["a"]

    def test_errors_collects_failed(self):
        p1 = Process(fn=_noop)
        p2 = Process(fn=_noop)
        p1.complete()
        p2.fail("err msg")
        s = Stem(processes=[p1, p2])
        assert s.errors() == ["err msg"]

    def test_complete_sets_status_and_time(self):
        s = Stem()
        before = time.time()
        s.complete()
        assert s.status == StemStatus.COMPLETED
        assert s.completed_at >= before

    def test_fail_sets_status_and_event(self):
        s = Stem()
        s.fail()
        assert s.status == StemStatus.FAILED
        assert s._done_event.is_set()

    def test_running_sets_status(self):
        s = Stem()
        s.running()
        assert s.status == StemStatus.RUNNING

    def test_is_done_property(self):
        s = Stem()
        assert not s.is_done
        s.complete()
        assert s.is_done

    def test_is_done_on_fail(self):
        s = Stem()
        s.fail()
        assert s.is_done

    def test_to_dict(self):
        p = Process(fn=_noop)
        s = Stem(processes=[p])
        d = s.to_dict()
        assert "id" in d
        assert d["num_processes"] == 1
        assert d["status"] == "created"


# ══════════════════════════════════════════════════════════════════
# Tree
# ══════════════════════════════════════════════════════════════════

class TestTree:
    def test_branch_executes_processes(self):
        tree = Tree("test", pool_workers=2)
        p1 = Process(fn=_sleep_and_return, args=(0.01, "a"))
        p2 = Process(fn=_sleep_and_return, args=(0.01, "b"))
        stem = tree.branch([p1, p2])
        tree.wait_stem(stem, timeout=5)
        assert p1.result == "a"
        assert p2.result == "b"
        tree.shutdown()

    def test_branch_marks_stem_complete(self):
        tree = Tree("test", pool_workers=2)
        p = Process(fn=_noop)
        stem = tree.branch([p])
        tree.wait_stem(stem, timeout=5)
        assert stem.status == StemStatus.COMPLETED
        tree.shutdown()

    def test_branch_handles_failure(self):
        tree = Tree("test", pool_workers=2)
        p = Process(fn=_fail)
        stem = tree.branch([p])
        tree.wait_stem(stem, timeout=5)
        assert stem.status == StemStatus.FAILED
        assert p.status == ProcessStatus.FAILED
        tree.shutdown()

    def test_mixed_success_and_failure(self):
        tree = Tree("test", pool_workers=2)
        p_ok = Process(fn=_noop)
        p_fail = Process(fn=_fail)
        stem = tree.branch([p_ok, p_fail])
        tree.wait_stem(stem, timeout=5)
        assert stem.status == StemStatus.FAILED
        assert p_ok.status == ProcessStatus.COMPLETED
        assert p_fail.status == ProcessStatus.FAILED
        tree.shutdown()

    def test_store_and_recall(self):
        tree = Tree("test")
        tree.store("key", "value")
        assert tree.recall("key") == "value"
        assert tree.recall("missing") is None
        tree.shutdown()

    def test_max_stems_limit(self):
        tree = Tree("test", max_stems=1, pool_workers=1)
        p1 = Process(fn=_sleep_and_return, args=(1.0, None))
        tree.branch([p1])
        with pytest.raises(RuntimeError, match="max stems"):
            p2 = Process(fn=_noop)
            tree.branch([p2])
        tree.shutdown()

    def test_active_stems_count(self):
        tree = Tree("test", pool_workers=2)
        assert tree.active_stems == 0
        p = Process(fn=_sleep_and_return, args=(0.5, None))
        stem = tree.branch([p])
        assert tree.active_stems == 1
        tree.wait_stem(stem, timeout=5)
        assert tree.active_stems == 0
        tree.shutdown()

    def test_status_transitions(self):
        tree = Tree("test", pool_workers=2)
        assert tree.status == TreeStatus.IDLE
        p = Process(fn=_sleep_and_return, args=(0.01, None))
        stem = tree.branch([p])
        assert tree.status == TreeStatus.BRANCHING
        tree.wait_stem(stem, timeout=5)
        assert tree.status == TreeStatus.IDLE
        tree.shutdown()

    def test_shutdown_sets_stopped(self):
        tree = Tree("test")
        tree.shutdown()
        assert tree.status == TreeStatus.STOPPED

    def test_to_dict(self):
        tree = Tree("mytree", max_stems=4)
        d = tree.to_dict()
        assert d["name"] == "mytree"
        assert d["max_stems"] == 4
        assert "status" in d
        assert "active_stems" in d
        tree.shutdown()

    def test_store_overwrites(self):
        tree = Tree("test")
        tree.store("k", 1)
        tree.store("k", 2)
        assert tree.recall("k") == 2
        tree.shutdown()

    def test_recall_none_default(self):
        tree = Tree("test")
        assert tree.recall("anything") is None
        tree.shutdown()


# ══════════════════════════════════════════════════════════════════
# EngineMetrics
# ══════════════════════════════════════════════════════════════════

class TestEngineMetrics:
    def test_initial_snapshot(self):
        m = EngineMetrics()
        s = m.snapshot()
        assert s["spawned"] == 0
        assert s["completed"] == 0
        assert s["failed"] == 0
        assert s["cancelled"] == 0
        assert s["dispatched"] == 0

    def test_record_spawn(self):
        m = EngineMetrics()
        m.record_spawn()
        m.record_spawn()
        assert m.snapshot()["spawned"] == 2

    def test_record_complete(self):
        m = EngineMetrics()
        p = Process(fn=_noop)
        p.running()
        time.sleep(0.01)
        p.complete()
        m.record_complete(p)
        s = m.snapshot()
        assert s["completed"] == 1
        assert s["avg_latency_s"] > 0

    def test_record_fail(self):
        m = EngineMetrics()
        m.record_fail()
        assert m.snapshot()["failed"] == 1

    def test_record_cancel(self):
        m = EngineMetrics()
        m.record_cancel()
        assert m.snapshot()["cancelled"] == 1

    def test_record_timeout(self):
        m = EngineMetrics()
        m.record_timeout()
        assert m.snapshot()["timed_out"] == 1

    def test_record_restart(self):
        m = EngineMetrics()
        m.record_restart()
        assert m.snapshot()["restarted"] == 1

    def test_record_dispatch(self):
        m = EngineMetrics()
        m.record_dispatch(5)
        assert m.snapshot()["dispatched"] == 5

    def test_record_dispatch_default(self):
        m = EngineMetrics()
        m.record_dispatch()
        assert m.snapshot()["dispatched"] == 1

    def test_record_memory(self):
        m = EngineMetrics()
        m.record_memory(1000)
        m.record_memory(2000)
        s = m.snapshot()
        assert s["total_memory_bytes"] == 3000
        assert s["peak_memory_bytes"] == 2000

    def test_peak_memory_tracking(self):
        m = EngineMetrics()
        m.record_memory(100)
        m.record_memory(50)
        assert m.snapshot()["peak_memory_bytes"] == 100

    def test_error_rate(self):
        m = EngineMetrics()
        m.record_complete()
        m.record_fail()
        s = m.snapshot()
        assert s["error_rate"] == pytest.approx(0.5)

    def test_throughput(self):
        m = EngineMetrics()
        m.record_complete()
        time.sleep(0.01)
        s = m.snapshot()
        assert s["throughput_per_s"] > 0

    def test_reset(self):
        m = EngineMetrics()
        m.record_spawn()
        m.record_complete()
        m.record_fail()
        m.reset()
        s = m.snapshot()
        assert s["spawned"] == 0
        assert s["completed"] == 0
        assert s["failed"] == 0

    def test_thread_safety(self):
        m = EngineMetrics()
        errors = []

        def _worker():
            try:
                for _ in range(100):
                    m.record_spawn()
                    m.record_complete()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert m.snapshot()["spawned"] == 400


# ══════════════════════════════════════════════════════════════════
# ResultCache
# ══════════════════════════════════════════════════════════════════

class TestResultCache:
    def test_put_and_get(self):
        cache = ResultCache(maxsize=10)
        cache.put(_noop, (), {}, "result")
        hit, val = cache.get(_noop, (), {})
        assert hit is True
        assert val == "result"

    def test_get_miss(self):
        cache = ResultCache(maxsize=10)
        hit, val = cache.get(_noop, (1,), {})
        assert hit is False
        assert val is None

    def test_lru_eviction(self):
        cache = ResultCache(maxsize=2)
        cache.put(_noop, (1,), {}, "a")
        cache.put(_noop, (2,), {}, "b")
        cache.put(_noop, (3,), {}, "c")
        assert cache.size == 2
        hit, val = cache.get(_noop, (1,), {})
        assert hit is False

    def test_ttl_expiry(self):
        cache = ResultCache(ttl=0.01)
        cache.put(_noop, (), {}, "val")
        time.sleep(0.02)
        hit, val = cache.get(_noop, (), {})
        assert hit is False
        assert val is None

    def test_invalidate_specific(self):
        cache = ResultCache()
        cache.put(_noop, (1,), {}, "a")
        cache.put(_noop, (2,), {}, "b")
        removed = cache.invalidate(_noop, (1,), {})
        assert removed is True
        hit, _ = cache.get(_noop, (1,), {})
        assert hit is False
        hit, _ = cache.get(_noop, (2,), {})
        assert hit is True

    def test_invalidate_by_name(self):
        cache = ResultCache()
        cache.put(_noop, (), {}, "a")
        cache.put(_sleep_and_return, (1,), {}, "b")
        removed = cache.invalidate(_noop)
        assert removed is True
        assert cache.size == 1

    def test_invalidate_all(self):
        cache = ResultCache()
        cache.put(_noop, (), {}, "a")
        cache.put(_noop, (1,), {}, "b")
        removed = cache.invalidate()
        assert removed is True
        assert cache.size == 0

    def test_clear(self):
        cache = ResultCache()
        cache.put(_noop, (), {}, "a")
        cache.put(_noop, (1,), {}, "b")
        count = cache.clear()
        assert count == 2
        assert cache.size == 0

    def test_stats(self):
        cache = ResultCache(maxsize=10, ttl=5.0)
        cache.put(_noop, (), {}, "a")
        cache.get(_noop, (), {})
        cache.get(_noop, (1,), {})
        s = cache.stats()
        assert s["size"] == 1
        assert s["maxsize"] == 10
        assert s["ttl"] == 5.0
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == pytest.approx(0.5)

    def test_size_property(self):
        cache = ResultCache()
        assert cache.size == 0
        cache.put(_noop, (), {}, "a")
        assert cache.size == 1


# ══════════════════════════════════════════════════════════════════
# ProcessMonitor
# ══════════════════════════════════════════════════════════════════

class TestProcessMonitor:
    def test_track_and_active_count(self):
        mon = ProcessMonitor()
        p = Process(fn=_noop)
        mon.track(p)
        assert mon.active_count == 1

    def test_untrack(self):
        mon = ProcessMonitor()
        p = Process(fn=_noop)
        mon.track(p)
        mon.untrack(p.id)
        assert mon.active_count == 0

    def test_untrack_nonexistent(self):
        mon = ProcessMonitor()
        mon.untrack("nope")

    def test_on_stall_callback(self):
        mon = ProcessMonitor(stall_timeout=0.01)
        stalled = []
        mon.on_stall(lambda p: stalled.append(p.id))
        p = Process(fn=_noop)
        p.running()
        p.started_at = time.time() - 100
        mon.track(p)
        mon.start()
        time.sleep(0.05)
        mon.stop()
        assert p.id in stalled

    def test_on_restart_callback(self):
        policy = RestartPolicy(max_restarts=3)
        mon = ProcessMonitor(restart_policy=policy)
        restarted = []
        mon.on_restart(lambda p: restarted.append(p.id))
        p = Process(fn=_noop)
        p.running()
        p.fail("err")
        mon.track(p)
        mon.start()
        time.sleep(0.05)
        mon.stop()
        assert p.id in restarted

    def test_restart_delay_exponential(self):
        policy = RestartPolicy(restart_delay=1.0, backoff="exponential", max_backoff=30.0)
        mon = ProcessMonitor(restart_policy=policy)
        assert mon._restart_delay(0) == 1.0
        assert mon._restart_delay(1) == 2.0
        assert mon._restart_delay(2) == 4.0

    def test_restart_delay_linear(self):
        policy = RestartPolicy(restart_delay=1.0, backoff="linear", max_backoff=30.0)
        mon = ProcessMonitor(restart_policy=policy)
        assert mon._restart_delay(0) == 1.0
        assert mon._restart_delay(1) == 2.0
        assert mon._restart_delay(2) == 3.0

    def test_restart_delay_fixed(self):
        policy = RestartPolicy(restart_delay=2.0, backoff="fixed", max_backoff=30.0)
        mon = ProcessMonitor(restart_policy=policy)
        assert mon._restart_delay(0) == 2.0
        assert mon._restart_delay(5) == 2.0

    def test_restart_delay_max_backoff(self):
        policy = RestartPolicy(restart_delay=10.0, backoff="exponential", max_backoff=15.0)
        mon = ProcessMonitor(restart_policy=policy)
        assert mon._restart_delay(5) == 15.0

    def test_restart_delay_no_policy(self):
        mon = ProcessMonitor()
        assert mon._restart_delay(0) == 1.0

    def test_start_stop(self):
        mon = ProcessMonitor()
        mon.start()
        assert mon._running
        mon.stop()
        assert not mon._running

    def test_start_idempotent(self):
        mon = ProcessMonitor()
        mon.start()
        t1 = mon._thread
        mon.start()
        assert mon._thread is t1
        mon.stop()

    def test_stats(self):
        mon = ProcessMonitor()
        p1 = Process(fn=_noop)
        p1.running()
        p2 = Process(fn=_noop)
        p2.running()
        mon.track(p1)
        mon.track(p2)
        s = mon.stats()
        assert s["monitored"] == 2
        assert s["running"] == 2

    def test_stalled_processes(self):
        mon = ProcessMonitor(stall_timeout=0.01)
        p = Process(fn=_noop)
        p.running()
        p.started_at = time.time() - 100
        mon.track(p)
        stalled = mon.stalled_processes()
        assert len(stalled) == 1
        assert stalled[0].id == p.id

    def test_stalled_with_heartbeat(self):
        mon = ProcessMonitor(stall_timeout=0.01)
        p = Process(fn=_noop)
        p.running()
        p._last_heartbeat = time.monotonic() - 100
        mon.track(p)
        stalled = mon.stalled_processes()
        assert len(stalled) == 1

    def test_not_stalled_recent(self):
        mon = ProcessMonitor(stall_timeout=10.0)
        p = Process(fn=_noop)
        p.running()
        p.started_at = time.time()
        mon.track(p)
        stalled = mon.stalled_processes()
        assert len(stalled) == 0

    def test_get_restart_count(self):
        mon = ProcessMonitor()
        assert mon.get_restart_count("x") == 0
        mon._restart_count["x"] = 3
        assert mon.get_restart_count("x") == 3

    def test_reset_restart_count(self):
        mon = ProcessMonitor()
        mon._restart_count["x"] = 5
        mon.reset_restart_count("x")
        assert mon.get_restart_count("x") == 0


# ══════════════════════════════════════════════════════════════════
# ProcessGroup
# ══════════════════════════════════════════════════════════════════

class TestProcessGroup:
    def test_add(self):
        g = ProcessGroup("g")
        p = Process(fn=_noop)
        g.add(p)
        assert g.num_processes == 1

    def test_all_done_empty(self):
        g = ProcessGroup("g")
        assert g.all_done

    def test_all_done(self):
        g = ProcessGroup("g")
        p = Process(fn=_noop)
        p.complete()
        g.add(p)
        assert g.all_done

    def test_not_all_done(self):
        g = ProcessGroup("g")
        p = Process(fn=_noop)
        p.running()
        g.add(p)
        assert not g.all_done

    def test_results(self):
        g = ProcessGroup("g")
        p1 = Process(fn=_noop)
        p2 = Process(fn=_noop)
        p1.complete("a")
        p2.fail("e")
        g.add(p1)
        g.add(p2)
        assert g.results() == ["a"]

    def test_errors(self):
        g = ProcessGroup("g")
        p1 = Process(fn=_noop)
        p1.fail("err")
        g.add(p1)
        assert g.errors() == ["err"]

    def test_cancel(self):
        g = ProcessGroup("g")
        p1 = Process(fn=_noop)
        p1.running()
        p2 = Process(fn=_noop)
        p2.complete()
        g.add(p1)
        g.add(p2)
        count = g.cancel()
        assert count == 1
        assert p1.is_cancelled
        assert p2.status == ProcessStatus.COMPLETED

    def test_elapsed_no_starts(self):
        g = ProcessGroup("g")
        e = g.elapsed
        assert e >= 0

    def test_elapsed(self):
        g = ProcessGroup("g")
        p = Process(fn=_noop)
        p.running()
        time.sleep(0.01)
        p.complete()
        g.add(p)
        assert g.elapsed > 0

    def test_gather(self):
        g = ProcessGroup("g")
        engine = Engine("test")
        engine.tree("t")
        p1 = engine.spawn(_sleep_and_return, 0.01, "a")
        p2 = engine.spawn(_sleep_and_return, 0.01, "b")
        g.add(p1)
        g.add(p2)
        engine.run_background(poll_interval=0.01)
        results = g.gather(timeout=5)
        engine.stop()
        assert sorted(results) == ["a", "b"]

    def test_to_dict(self):
        g = ProcessGroup("g")
        d = g.to_dict()
        assert d["name"] == "g"
        assert d["num_processes"] == 0
        assert "status_counts" in d


# ══════════════════════════════════════════════════════════════════
# Engine: core lifecycle
# ══════════════════════════════════════════════════════════════════

class TestEngineCore:
    def test_spawn_and_branch(self):
        engine = Engine("test")
        engine.tree("t1")
        p = engine.spawn(_sleep_and_return, 0.01, "done")
        stem = engine.branch("t1", [p])
        engine.get_tree("t1").wait_stem(stem, timeout=5)
        assert p.result == "done"
        engine.stop()

    def test_spawn_with_explicit_tree(self):
        engine = Engine("test")
        engine.tree("data")
        engine.tree("compute")
        p = engine.spawn(_noop, tree="compute")
        assert p._tree_name == "compute"
        engine.stop()

    def test_branch_on_missing_tree_raises(self):
        engine = Engine("test")
        with pytest.raises(ValueError, match="not found"):
            engine.branch("nope", [])

    def test_list_processes(self):
        engine = Engine("test")
        engine.tree("t")
        p1 = engine.spawn(_noop)
        p2 = engine.spawn(_noop)
        all_procs = engine.list_processes()
        assert len(all_procs) == 2
        engine.stop()

    def test_list_processes_by_status(self):
        engine = Engine("test")
        p1 = engine.spawn(_noop)
        p2 = engine.spawn(_noop)
        p1.complete()
        completed = engine.list_processes(status=ProcessStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].id == p1.id
        engine.stop()

    def test_get_process(self):
        engine = Engine("test")
        p = engine.spawn(_noop)
        assert engine.get_process(p.id) is p
        assert engine.get_process("nope") is None
        engine.stop()

    def test_max_trees_limit(self):
        engine = Engine("test", max_trees=2)
        engine.tree("a")
        engine.tree("b")
        with pytest.raises(RuntimeError, match="max trees"):
            engine.tree("c")
        engine.stop()

    def test_reset(self):
        engine = Engine("test")
        engine.tree("t")
        engine.spawn(_noop)
        engine.reset()
        assert len(engine.list_processes()) == 0
        assert len(engine._pending) == 0
        engine.stop()

    def test_health(self):
        engine = Engine("test")
        engine.tree("t")
        h = engine.health()
        assert h["name"] == "test"
        assert h["tree_count"] == 1
        assert "status_counts" in h
        engine.stop()

    def test_to_dict(self):
        engine = Engine("test")
        engine.tree("t1")
        engine.route("work", "t1")
        engine.spawn(_noop, name="work")
        d = engine.to_dict()
        assert d["name"] == "test"
        assert "trees" in d
        assert d["processes"] == 1
        assert d["routing"] == {"work": "t1"}
        engine.stop()

    def test_summary(self):
        engine = Engine("test")
        engine.tree("t")
        engine.spawn(_noop)
        s = engine.summary()
        assert "Engine 'test'" in s
        assert "processes" in s
        engine.stop()

    def test_set_scheduling(self):
        engine = Engine("test")
        engine.set_scheduling(SchedulingPolicy.ROUND_ROBIN)
        assert engine._scheduling_policy == SchedulingPolicy.ROUND_ROBIN
        engine.stop()

    def test_metrics_property(self):
        engine = Engine("test")
        assert isinstance(engine.metrics, EngineMetrics)
        engine.stop()

    def test_list_trees(self):
        engine = Engine("test")
        engine.tree("a")
        engine.tree("b")
        assert sorted(engine.list_trees()) == ["a", "b"]
        engine.stop()


# ══════════════════════════════════════════════════════════════════
# Engine: dispatch mode
# ══════════════════════════════════════════════════════════════════

class TestEngineDispatch:
    def test_dispatch_routes_by_name(self):
        engine = Engine("test")
        engine.tree("data")
        engine.tree("compute")
        engine.route("load", "data")
        engine.route("run", "compute")
        p1 = engine.spawn(_noop, name="load")
        p2 = engine.spawn(_noop, name="run")
        dispatched = engine.dispatch()
        assert dispatched == 2
        assert p1._tree_name == "data"
        assert p2._tree_name == "compute"
        engine.stop()

    def test_dispatch_round_robin_ungrouped(self):
        engine = Engine("test")
        engine.tree("a")
        engine.tree("b")
        p1 = engine.spawn(_noop, name="unrouted")
        p2 = engine.spawn(_noop, name="unrouted")
        engine.dispatch()
        trees = {p1._tree_name, p2._tree_name}
        assert trees == {"a", "b"}
        engine.stop()

    def test_dispatch_explicit_tree_overrides_routing(self):
        engine = Engine("test")
        engine.tree("default")
        engine.tree("special")
        engine.route("task", "default")
        p = engine.spawn(_noop, name="task", tree="special")
        engine.dispatch()
        assert p._tree_name == "special"
        engine.stop()

    def test_dispatch_empty_returns_zero(self):
        engine = Engine("test")
        assert engine.dispatch() == 0
        engine.stop()

    def test_dispatch_batches_large_groups(self):
        engine = Engine("test")
        engine.tree("t", pool_workers=1)
        engine.route("work", "t")
        engine._dispatch_batch_size = 2
        procs = [engine.spawn(_noop, name="work") for _ in range(5)]
        dispatched = engine.dispatch()
        assert dispatched == 5
        engine.stop()

    def test_dispatch_holds_unmet_deps(self):
        engine = Engine("test")
        engine.tree("t")
        dep = engine.spawn(_noop)
        child = engine.spawn(_noop, depends_on=[dep.id])
        dispatched = engine.dispatch()
        assert dispatched == 1
        assert child in engine._pending
        engine.stop()

    def test_dispatch_when_deps_met(self):
        engine = Engine("test")
        engine.tree("t")
        dep = engine.spawn(_noop)
        dep.complete()
        child = engine.spawn(_noop, depends_on=[dep.id])
        dispatched = engine.dispatch()
        assert dispatched == 2
        assert dep not in engine._pending
        assert child not in engine._pending
        engine.stop()

    def test_dispatch_batch(self):
        engine = Engine("test")
        engine.tree("t")
        for _ in range(5):
            engine.spawn(_noop)
        dispatched = engine.dispatch_batch(max_count=3)
        assert dispatched == 3
        assert len(engine._pending) == 2
        engine.stop()

    def test_dispatch_batch_default_size(self):
        engine = Engine("test")
        engine.tree("t")
        for _ in range(20):
            engine.spawn(_noop)
        dispatched = engine.dispatch_batch()
        assert dispatched == 8
        engine.stop()


# ══════════════════════════════════════════════════════════════════
# Engine: run loop
# ══════════════════════════════════════════════════════════════════

class TestEngineRun:
    def test_run_dispatches_pending(self):
        engine = Engine("test")
        engine.tree("t")
        engine.route("work", "t")
        p = engine.spawn(_sleep_and_return, 0.01, "done", name="work")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()
        assert p.result == "done"
        assert p.status == ProcessStatus.COMPLETED

    def test_run_background_is_non_blocking(self):
        engine = Engine("test")
        engine.tree("t")
        thread = engine.run_background(poll_interval=0.05)
        assert thread.is_alive()
        engine.stop()
        thread.join(timeout=2)
        assert not thread.is_alive()

    def test_run_background_as_future(self):
        engine = Engine("test")
        engine.tree("t")
        future = engine.run_background(poll_interval=0.05, as_future=True)
        assert future is not None
        engine.stop()
        future.result(timeout=2)

    def test_on_complete_fires(self):
        engine = Engine("test")
        engine.tree("t")
        engine.route("work", "t")
        completed = []
        engine.on_complete(lambda p: completed.append(p.id))
        p = engine.spawn(_sleep_and_return, 0.01, "x", name="work")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()
        assert p.id in completed

    def test_on_complete_fires_for_failure(self):
        engine = Engine("test")
        engine.tree("t")
        engine.route("fail", "t")
        results = []
        engine.on_complete(lambda p: results.append(p.status))
        p = engine.spawn(_fail, name="fail")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()
        assert ProcessStatus.FAILED in results

    def test_on_complete_callback_error_does_not_crash(self):
        engine = Engine("test")
        engine.tree("t")
        engine.route("work", "t")

        def bad_callback(p):
            raise ValueError("callback error")

        engine.on_complete(bad_callback)
        p = engine.spawn(_noop, name="work")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        engine.stop()
        assert p.is_done

    def test_on_progress_callback(self):
        engine = Engine("test")
        engine.tree("t")
        engine.route("work", "t")
        progress_data = []

        def on_progress(data):
            progress_data.append(data)

        p = engine.spawn(_sleep_and_return, 0.01, "x", name="work")

        def _run():
            engine.run(poll_interval=0.01, on_progress=on_progress)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        engine.wait(timeout=5)
        engine.stop()
        t.join(timeout=2)
        assert len(progress_data) > 0
        assert "pending" in progress_data[0]

    def test_wait_all(self):
        engine = Engine("test")
        engine.tree("t")
        p1 = engine.spawn(_sleep_and_return, 0.01, "a")
        p2 = engine.spawn(_sleep_and_return, 0.01, "b")
        engine.branch("t", [p1, p2])
        done = engine.wait_all(timeout=5)
        engine.stop()
        assert len(done) == 2
        assert all(p.is_done for p in done)

    def test_get_completed(self):
        engine = Engine("test")
        engine.tree("t")
        engine.route("work", "t")
        p = engine.spawn(_sleep_and_return, 0.01, "done", name="work")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=5)
        completed = engine.get_completed()
        assert len(completed) == 1
        assert completed[0].id == p.id
        engine.stop()

    def test_wait_for(self):
        engine = Engine("test")
        engine.tree("t")
        p = engine.spawn(_sleep_and_return, 0.01, "done")
        engine.branch("t", [p])
        result = engine.wait_for(p.id, timeout=5)
        engine.stop()
        assert result.result == "done"

    def test_wait_for_missing_raises(self):
        engine = Engine("test")
        with pytest.raises(KeyError, match="not found"):
            engine.wait_for("nope")
        engine.stop()

    def test_wait_for_timeout(self):
        engine = Engine("test")
        engine.tree("t")
        p = engine.spawn(_sleep_and_return, 20.0, "slow")
        engine.branch("t", [p])
        result = engine.wait_for(p.id, timeout=0.01)
        engine.stop()
        assert result.status != ProcessStatus.COMPLETED

    def test_wait_for_any(self):
        engine = Engine("test")
        engine.tree("t")
        p1 = engine.spawn(_sleep_and_return, 0.5, "slow")
        p2 = engine.spawn(_sleep_and_return, 0.01, "fast")
        engine.branch("t", [p1, p2])
        result = engine.wait_for_any([p1.id, p2.id], timeout=5)
        engine.stop()
        assert result is not None
        assert result.result == "fast"


# ══════════════════════════════════════════════════════════════════
# Engine: cancellation
# ══════════════════════════════════════════════════════════════════

class TestEngineCancellation:
    def test_cancel_process(self):
        engine = Engine("test")
        p = engine.spawn(_noop)
        count = engine.cancel_process(p.id)
        assert count == 1
        assert p.is_cancelled
        engine.stop()

    def test_cancel_process_with_children(self):
        engine = Engine("test")
        parent = engine.spawn(_noop)
        child = engine.spawn(_noop)
        parent.children_ids = [child.id]
        count = engine.cancel_process(parent.id, propagate=True)
        assert count == 2
        assert parent.is_cancelled
        assert child.is_cancelled
        engine.stop()

    def test_cancel_process_no_propagate(self):
        engine = Engine("test")
        parent = engine.spawn(_noop)
        child = engine.spawn(_noop)
        parent.children_ids = [child.id]
        count = engine.cancel_process(parent.id, propagate=False)
        assert count == 1
        assert parent.is_cancelled
        assert not child.is_cancelled
        engine.stop()

    def test_cancel_process_nonexistent(self):
        engine = Engine("test")
        count = engine.cancel_process("nope")
        assert count == 0
        engine.stop()

    def test_cancel_tree(self):
        engine = Engine("test")
        engine.tree("t")
        p1 = engine.spawn(_noop, tree="t")
        p2 = engine.spawn(_noop, tree="t")
        p3 = engine.spawn(_noop, tree="other")
        count = engine.cancel_tree("t")
        assert count == 2
        assert p1.is_cancelled
        assert p2.is_cancelled
        assert not p3.is_cancelled
        engine.stop()

    def test_cancel_all(self):
        engine = Engine("test")
        p1 = engine.spawn(_noop)
        p2 = engine.spawn(_noop)
        count = engine.cancel_all()
        assert count == 2
        engine.stop()

    def test_cancel_all_by_status(self):
        engine = Engine("test")
        p1 = engine.spawn(_noop)
        p1.complete()
        p2 = engine.spawn(_noop)
        p2.running()
        count = engine.cancel_all(status=ProcessStatus.RUNNING)
        assert count == 1
        assert p2.is_cancelled
        assert p1.status == ProcessStatus.COMPLETED
        engine.stop()

    def test_stop_cancels_pending(self):
        engine = Engine("test")
        p = engine.spawn(_noop)
        engine.stop()
        assert p.is_cancelled


# ══════════════════════════════════════════════════════════════════
# Engine: dependency graph
# ══════════════════════════════════════════════════════════════════

class TestEngineDependencies:
    def test_dependency_graph(self):
        engine = Engine("test")
        p1 = engine.spawn(_noop, name="a")
        p2 = engine.spawn(_noop, name="b", depends_on=[p1.id])
        g = engine.dependency_graph()
        assert len(g["nodes"]) == 2
        assert len(g["edges"]) == 1
        assert g["edges"][0]["from"] == p1.id
        assert g["edges"][0]["to"] == p2.id
        engine.stop()

    def test_dependency_graph_empty(self):
        engine = Engine("test")
        g = engine.dependency_graph()
        assert g["nodes"] == []
        assert g["edges"] == []
        engine.stop()

    def test_critical_path(self):
        engine = Engine("test")
        p1 = engine.spawn(_noop, name="a")
        p2 = engine.spawn(_noop, name="b", depends_on=[p1.id])
        p3 = engine.spawn(_noop, name="c", depends_on=[p2.id])
        path = engine.critical_path()
        assert len(path) == 3
        assert path[0] == p1.id
        assert path[2] == p3.id
        engine.stop()

    def test_critical_path_empty(self):
        engine = Engine("test")
        assert engine.critical_path() == []

    def test_orphan_processes(self):
        engine = Engine("test")
        p1 = engine.spawn(_noop, name="a")
        p2 = engine.spawn(_noop, name="b", depends_on=[p1.id])
        orphans = engine.orphan_processes()
        assert len(orphans) == 1
        assert orphans[0].id == p2.id
        engine.stop()

    def test_no_orphans_when_deps_met(self):
        engine = Engine("test")
        p1 = engine.spawn(_noop, name="a")
        p1.complete()
        p2 = engine.spawn(_noop, name="b", depends_on=[p1.id])
        orphans = engine.orphan_processes()
        assert len(orphans) == 0
        engine.stop()

    def test_spawn_chain(self):
        engine = Engine("test")
        engine.tree("t")
        chain = engine.spawn_chain(
            (_sleep_and_return, 0.01, "first"),
            (_sleep_and_return, 0.01, "second"),
            (_sleep_and_return, 0.01, "third"),
        )
        assert len(chain) == 3
        assert chain[0].depends_on == []
        assert chain[1].depends_on == [chain[0].id]
        assert chain[2].depends_on == [chain[1].id]
        engine.stop()

    def test_deps_met_with_completed_dep(self):
        engine = Engine("test")
        dep = engine.spawn(_noop)
        dep.complete()
        child = engine.spawn(_noop, depends_on=[dep.id])
        assert engine._deps_met(child)
        engine.stop()

    def test_deps_met_with_failed_dep(self):
        engine = Engine("test")
        dep = engine.spawn(_noop)
        dep.fail("err")
        child = engine.spawn(_noop, depends_on=[dep.id])
        assert not engine._deps_met(child)
        engine.stop()

    def test_deps_met_with_missing_dep(self):
        engine = Engine("test")
        child = engine.spawn(_noop, depends_on=["nonexistent"])
        assert not engine._deps_met(child)
        engine.stop()


# ══════════════════════════════════════════════════════════════════
# Engine: batch operations
# ══════════════════════════════════════════════════════════════════

class TestEngineBatch:
    def test_spawn_batch(self):
        engine = Engine("test")
        items = [
            (_noop,),
            (_add, 1, 2),
            (_sleep_and_return, 0.01, "done"),
        ]
        procs = engine.spawn_batch(items)
        assert len(procs) == 3
        engine.stop()

    def test_spawn_batch_with_kwargs(self):
        engine = Engine("test")
        items = [
            (_add, 1, 2, {"kwargs": {}}),
        ]
        procs = engine.spawn_batch(items)
        assert len(procs) == 1
        engine.stop()

    def test_spawn_batch_empty_item(self):
        engine = Engine("test")
        procs = engine.spawn_batch([(), (_noop,)])
        assert len(procs) == 1
        engine.stop()

    def test_spawn_batch_empty(self):
        engine = Engine("test")
        procs = engine.spawn_batch([])
        assert len(procs) == 0
        engine.stop()

    def test_group(self):
        engine = Engine("test")
        g = engine.group("mygroup")
        assert isinstance(g, ProcessGroup)
        assert g.name == "mygroup"
        engine.stop()


# ══════════════════════════════════════════════════════════════════
# Engine: cache integration
# ══════════════════════════════════════════════════════════════════

class TestEngineCache:
    def test_enable_disable_cache(self):
        engine = Engine("test")
        engine.enable_cache(maxsize=64, ttl=5.0)
        assert engine._cache is not None
        engine.disable_cache()
        assert engine._cache is None
        engine.stop()

    def test_cache_hit_returns_immediately(self):
        engine = Engine("test")
        engine.enable_cache()
        engine.tree("t")
        p1 = engine.spawn(_noop)
        engine.branch("t", [p1])
        engine.wait(timeout=5)
        engine._cache.put(_noop, (), {}, "cached_val")
        p2 = engine.spawn(_noop)
        assert p2.status == ProcessStatus.COMPLETED
        assert p2.result == "cached_val"
        engine.stop()


# ══════════════════════════════════════════════════════════════════
# Engine: save_state
# ══════════════════════════════════════════════════════════════════

class TestEngineSaveState:
    def test_save_state(self):
        engine = Engine("test")
        engine.tree("t")
        engine.spawn(_noop, name="task")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            engine.save_state(path)
            with open(path) as f:
                state = json.load(f)
            assert state["name"] == "test"
            assert len(state["processes"]) == 1
            assert "metrics" in state
        finally:
            os.unlink(path)
        engine.stop()


# ══════════════════════════════════════════════════════════════════
# Engine: config integration
# ══════════════════════════════════════════════════════════════════

class TestEngineConfig:
    def test_config_overrides(self):
        cfg = EngineConfig(name="from_config", max_trees=4)
        engine = Engine(config=cfg)
        assert engine.name == "from_config"
        assert engine.max_trees == 4
        engine.stop()

    def test_no_config_uses_defaults(self):
        engine = Engine("myengine", max_trees=8)
        assert engine.name == "myengine"
        assert engine.max_trees == 8
        engine.stop()


# ══════════════════════════════════════════════════════════════════
# Integration: model loading + training scenario
# ══════════════════════════════════════════════════════════════════

class TestEngineIntegration:
    def test_model_load_then_train(self):
        engine = Engine("sim")
        engine.tree("data", pool_workers=2)
        engine.tree("train", pool_workers=2)
        engine.route("load", "data")
        engine.route("epoch", "train")
        model = {"loaded": False, "loss": 5.0}

        def do_load():
            time.sleep(0.05)
            model["loaded"] = True
            return {"loaded": True}

        def do_epoch():
            model["loss"] *= 0.9
            return {"loss": model["loss"]}

        load_proc = engine.spawn(do_load, name="load")
        epoch_procs = [engine.spawn(do_epoch, name="epoch") for _ in range(3)]
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=10)
        engine.stop()
        assert load_proc.result == {"loaded": True}
        assert model["loaded"]
        for p in epoch_procs:
            assert p.status == ProcessStatus.COMPLETED
            assert p.result["loss"] < 5.0

    def test_parallel_inference_during_training(self):
        engine = Engine("sim")
        engine.tree("train", pool_workers=2)
        engine.tree("infer", pool_workers=2)
        engine.route("train", "train")
        engine.route("infer", "infer")
        results = {"train": [], "infer": []}

        def train_step():
            time.sleep(0.02)
            results["train"].append(1)
            return "trained"

        def infer_step():
            time.sleep(0.02)
            results["infer"].append(1)
            return "inferred"

        for _ in range(3):
            engine.spawn(train_step, name="train")
        for _ in range(3):
            engine.spawn(infer_step, name="infer")
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=10)
        engine.stop()
        assert len(results["train"]) == 3
        assert len(results["infer"]) == 3

    def test_chained_operations(self):
        def step_a():
            time.sleep(0.01)
            return "load"

        def step_b(prev):
            time.sleep(0.01)
            return "process"

        def step_c(prev):
            time.sleep(0.01)
            return "output"

        engine = Engine("sim")
        engine.tree("t")
        chain = engine.spawn_chain(
            (step_a,),
            (step_b,),
            (step_c,),
        )
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=10)
        engine.stop()
        assert chain[0].result == "load"
        assert chain[1].result == "process"
        assert chain[2].result == "output"

    def test_dependency_cascade(self):
        engine = Engine("sim")
        engine.tree("t")
        a = engine.spawn(_noop, name="a")
        b = engine.spawn(_noop, name="b", depends_on=[a.id])
        c = engine.spawn(_noop, name="c", depends_on=[b.id])
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=10)
        engine.stop()
        assert a.status == ProcessStatus.COMPLETED
        assert b.status == ProcessStatus.COMPLETED
        assert c.status == ProcessStatus.COMPLETED

    def test_spawn_chain_with_kwargs(self):
        engine = Engine("sim")
        engine.tree("t")
        chain = engine.spawn_chain(
            (_sleep_and_return, 0.01, "first"),
            (_identity,),
        )
        engine.branch("t", [chain[0]])
        engine.run_background(poll_interval=0.01)
        engine.wait(timeout=10)
        engine.stop()
        assert chain[0].result == "first"


# ══════════════════════════════════════════════════════════════════
# Process timeout
# ══════════════════════════════════════════════════════════════════

class TestProcessTimeout:
    def test_timeout_fails_process(self):
        tree = Tree("test", pool_workers=2)
        p = Process(fn=time.sleep, args=(10.0,), timeout=0.05)
        stem = tree.branch([p])
        tree.wait_stem(stem, timeout=5)
        assert p.status == ProcessStatus.FAILED
        assert "timed out" in p.error
        tree.shutdown()

    def test_no_timeout_completes(self):
        tree = Tree("test", pool_workers=2)
        p = Process(fn=_sleep_and_return, args=(0.01, "ok"), timeout=5.0)
        stem = tree.branch([p])
        tree.wait_stem(stem, timeout=5)
        assert p.status == ProcessStatus.COMPLETED
        assert p.result == "ok"
        tree.shutdown()


# ── Block Quantization Tests ─────────────────────────────────────

class TestBlockQuantization:
    """Tests for Q4_K and Q8 block quantization."""

    def test_q4_compress_decompress_roundtrip(self):
        import numpy as np
        from domains.infrastructure.pugqeep.compressor import PointCompressor
        comp = PointCompressor()
        W = np.random.randn(256, 256).astype(np.float32) * 0.02
        pt = comp.compress_block_q4(W.flatten(), "test_q4")
        recon = comp.decompress_block_q4(pt)
        cos = np.dot(W.flatten(), recon) / (np.linalg.norm(W) * np.linalg.norm(recon))
        assert cos > 0.99, f"Q4 cosine too low: {cos}"

    def test_q8_compress_decompress_roundtrip(self):
        import numpy as np
        from domains.infrastructure.pugqeep.compressor import PointCompressor
        comp = PointCompressor()
        W = np.random.randn(256, 256).astype(np.float32) * 0.02
        pt = comp.compress_block_q8(W.flatten(), "test_q8")
        recon = comp.decompress_block_q8(pt)
        cos = np.dot(W.flatten(), recon) / (np.linalg.norm(W) * np.linalg.norm(recon))
        assert cos > 0.999, f"Q8 cosine too low: {cos}"

    def test_q4_ratio(self):
        import numpy as np
        from domains.infrastructure.pugqeep.compressor import PointCompressor
        comp = PointCompressor()
        W = np.random.randn(768, 768).astype(np.float32) * 0.02
        pt = comp.compress_block_q4(W.flatten(), "test_q4")
        raw = W.nbytes
        compressed = pt.nbytes()
        ratio = raw / compressed
        assert ratio > 4.0, f"Q4 ratio too low: {ratio}"

    def test_q8_ratio(self):
        import numpy as np
        from domains.infrastructure.pugqeep.compressor import PointCompressor
        comp = PointCompressor()
        W = np.random.randn(768, 768).astype(np.float32) * 0.02
        pt = comp.compress_block_q8(W.flatten(), "test_q8")
        raw = W.nbytes
        compressed = pt.nbytes()
        ratio = raw / compressed
        assert ratio > 2.5, f"Q8 ratio too low: {ratio}"

    def test_q4_generate(self):
        import numpy as np
        from domains.infrastructure.pugqeep.point import Point
        from domains.infrastructure.pugqeep.compressor import PointCompressor
        comp = PointCompressor()
        W = np.random.randn(1000).astype(np.float32) * 0.02
        pt = comp.compress_block_q4(W, "test")
        gen = pt.generate(len(W))
        cos = np.dot(W, gen) / (np.linalg.norm(W) * np.linalg.norm(gen))
        assert cos > 0.99

    def test_q8_generate(self):
        import numpy as np
        from domains.infrastructure.pugqeep.compressor import PointCompressor
        comp = PointCompressor()
        W = np.random.randn(1000).astype(np.float32) * 0.02
        pt = comp.compress_block_q8(W, "test")
        gen = pt.generate(len(W))
        cos = np.dot(W, gen) / (np.linalg.norm(W) * np.linalg.norm(gen))
        assert cos > 0.999

    def test_q4_to_bytes_roundtrip(self):
        import numpy as np
        from domains.infrastructure.pugqeep.point import Point
        from domains.infrastructure.pugqeep.compressor import PointCompressor
        comp = PointCompressor()
        W = np.random.randn(1000).astype(np.float32) * 0.02
        pt = comp.compress_block_q4(W, "test")
        data = pt.to_bytes()
        pt2 = Point.from_bytes(data, "test")
        assert pt2.function_type == "block_q4"
        gen = pt2.generate(len(W))
        cos = np.dot(W, gen) / (np.linalg.norm(W) * np.linalg.norm(gen))
        assert cos > 0.99

    def test_q8_to_bytes_roundtrip(self):
        import numpy as np
        from domains.infrastructure.pugqeep.point import Point
        from domains.infrastructure.pugqeep.compressor import PointCompressor
        comp = PointCompressor()
        W = np.random.randn(1000).astype(np.float32) * 0.02
        pt = comp.compress_block_q8(W, "test")
        data = pt.to_bytes()
        pt2 = Point.from_bytes(data, "test")
        assert pt2.function_type == "block_q8"
        gen = pt2.generate(len(W))
        cos = np.dot(W, gen) / (np.linalg.norm(W) * np.linalg.norm(gen))
        assert cos > 0.999

    def test_q4_to_dict_roundtrip(self):
        import numpy as np
        from domains.infrastructure.pugqeep.point import Point
        from domains.infrastructure.pugqeep.compressor import PointCompressor
        comp = PointCompressor()
        W = np.random.randn(1000).astype(np.float32) * 0.02
        pt = comp.compress_block_q4(W, "test")
        d = pt.to_dict()
        pt2 = Point.from_dict(d)
        assert pt2.function_type == "block_q4"
        gen = pt2.generate(len(W))
        cos = np.dot(W, gen) / (np.linalg.norm(W) * np.linalg.norm(gen))
        assert cos > 0.99

    def test_q4_batch_inference(self):
        """Test Q4 quantization on realistic model weights."""
        import numpy as np
        from domains.infrastructure.pugqeep.compressor import PointCompressor
        comp = PointCompressor()

        def make_weight(shape):
            w = np.random.randn(*shape).astype(np.float32) * 0.02
            u = np.random.randn(shape[0], 1).astype(np.float32) * 0.01
            v = np.random.randn(1, shape[1]).astype(np.float32) * 0.01
            return w + u @ v

        W1 = make_weight((768, 3072))
        W2 = make_weight((3072, 768))

        def forward(x, W1, W2):
            h = np.maximum(0, x @ W1)
            return h @ W2

        x = np.random.randn(1, 768).astype(np.float32)
        out_orig = forward(x, W1, W2)

        pt1 = comp.compress_block_q4(W1.flatten(), "W1")
        pt2 = comp.compress_block_q4(W2.flatten(), "W2")
        W1_r = comp.decompress_block_q4(pt1).reshape(W1.shape)
        W2_r = comp.decompress_block_q4(pt2).reshape(W2.shape)

        out_q4 = forward(x, W1_r, W2_r)
        cos = np.dot(out_orig.flatten(), out_q4.flatten()) / (
            np.linalg.norm(out_orig) * np.linalg.norm(out_q4))
        assert cos > 0.99, f"Q4 batch inference cosine: {cos}"

    def test_q4_vs_vq_accuracy(self):
        """Q4 should beat VQ k=32 WITHOUT residual (both lossy)."""
        import numpy as np
        from domains.infrastructure.pugqeep.compressor import PointCompressor
        from domains.infrastructure.pugqeep.config import CompressorConfig
        comp = PointCompressor()
        W = np.random.randn(768, 3072).astype(np.float32) * 0.02
        flat = W.flatten()

        # VQ k=32 without residual (residual_threshold=0.0 → never stores residual)
        vq_comp = PointCompressor(CompressorConfig(n_clusters=32), residual_threshold=0.0)
        pt_vq = vq_comp.compress_cluster(flat, "vq", 32)
        recon_vq = pt_vq.generate(len(flat))
        cos_vq = np.dot(flat, recon_vq) / (np.linalg.norm(flat) * np.linalg.norm(recon_vq))

        # Q4
        pt_q4 = comp.compress_block_q4(flat, "q4")
        recon_q4 = comp.decompress_block_q4(pt_q4)
        cos_q4 = np.dot(flat, recon_q4) / (np.linalg.norm(flat) * np.linalg.norm(recon_q4))

        assert cos_q4 > cos_vq, f"Q4 ({cos_q4}) should beat VQ k=32 ({cos_vq})"

    def test_q4_small_weights(self):
        """Q4 on small arrays (e.g. bias vectors)."""
        import numpy as np
        from domains.infrastructure.pugqeep.compressor import PointCompressor
        comp = PointCompressor()
        b = np.zeros(768, dtype=np.float32)
        b[0] = 0.1
        b[-1] = -0.1
        pt = comp.compress_block_q4(b, "bias")
        recon = comp.decompress_block_q4(pt)
        assert np.allclose(b, recon, atol=1e-2)

    def test_q4_handles_padding(self):
        """Q4 handles arrays not divisible by 32."""
        import numpy as np
        from domains.infrastructure.pugqeep.compressor import PointCompressor
        comp = PointCompressor()
        for size in [1, 15, 33, 100, 1000]:
            W = np.random.randn(size).astype(np.float32) * 0.02
            pt = comp.compress_block_q4(W, "test")
            recon = comp.decompress_block_q4(pt)
            assert len(recon) == len(W), f"Size mismatch: {len(recon)} != {len(W)}"
            cos = np.dot(W, recon) / (np.linalg.norm(W) * np.linalg.norm(recon))
            assert cos > 0.95, f"Q4 small array cosine: {cos} for size {size}"

    def test_block_quant_strategy_in_tree(self):
        """BlockQuantStrategy integrates with Tree."""
        import numpy as np
        from domains.infrastructure.pugqeep.tree import Tree
        from domains.infrastructure.pugqeep.config import TreeConfig
        config = TreeConfig(name="test_block_q4", method="block_q4")
        tree = Tree("test_block_q4", config=config)
        data = {
            "W1": np.random.randn(256, 256).astype(np.float32) * 0.02,
            "W2": np.random.randn(256, 256).astype(np.float32) * 0.02,
        }
        stats = tree.load_data(data)
        assert stats["num_items"] == 2
        for name, raw in data.items():
            decompressed = tree.get_data(name)
            cos = np.dot(raw.flatten(), decompressed.flatten()) / (
                np.linalg.norm(raw) * np.linalg.norm(decompressed))
            assert cos > 0.99, f"Tree block_q4 accuracy: {cos} for {name}"
        tree.library.clear()

    def test_block_q8_strategy_in_tree(self):
        """BlockQuantStrategy Q8 integrates with Tree."""
        import numpy as np
        from domains.infrastructure.pugqeep.tree import Tree
        from domains.infrastructure.pugqeep.config import TreeConfig
        config = TreeConfig(name="test_block_q8", method="block_q8")
        tree = Tree("test_block_q8", config=config)
        data = {
            "W1": np.random.randn(256, 256).astype(np.float32) * 0.02,
        }
        stats = tree.load_data(data)
        assert stats["num_items"] == 1
        decompressed = tree.get_data("W1")
        raw = data["W1"]
        cos = np.dot(raw.flatten(), decompressed.flatten()) / (
            np.linalg.norm(raw) * np.linalg.norm(decompressed))
        assert cos > 0.999
        tree.library.clear()
