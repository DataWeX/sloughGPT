"""Tests for domains.infrastructure.pugqeep — EvictionPolicy, Tier, ProcessStatus, StemStatus, TreeStatus, TaskStatus, TaskPriority, Task, CacheEntry, Process, etc."""

from domains.infrastructure.pugqeep.cache import EvictionPolicy, Tier, CacheEntry, CacheStats, MemoryStore, HotStore, DiskStore, TieredCache
from domains.infrastructure.pugqeep.engine import (
    ProcessStatus, StemStatus, TreeStatus, Process, Stem, Tree,
    EngineMetrics, ResultCache, SchedulingPolicy,
)
from domains.infrastructure.pugqeep.task_queue import TaskStatus, TaskPriority, Task, TaskQueue
import time
import numpy as np
import pytest
import threading


# ── EvictionPolicy ─────────────────────────────────────────────────────


class TestEvictionPolicy:
    def test_all_members(self):
        assert len(EvictionPolicy) == 2

    def test_values(self):
        assert EvictionPolicy.LRU.value == "lru"
        assert EvictionPolicy.LFU.value == "lfu"

    def test_from_value(self):
        assert EvictionPolicy("lru") is EvictionPolicy.LRU
        assert EvictionPolicy("lfu") is EvictionPolicy.LFU

    def test_is_enum(self):
        from enum import Enum
        assert issubclass(EvictionPolicy, Enum)

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            EvictionPolicy("fifo")

    def test_iteration(self):
        names = [e.name for e in EvictionPolicy]
        assert "LRU" in names
        assert "LFU" in names


# ── Tier ───────────────────────────────────────────────────────────────


class TestTier:
    def test_all_members(self):
        assert len(Tier) == 3

    def test_values(self):
        assert Tier.DISK.value == "disk"
        assert Tier.HOT.value == "hot"
        assert Tier.MEMORY.value == "memory"

    def test_from_value(self):
        assert Tier("disk") is Tier.DISK
        assert Tier("hot") is Tier.HOT
        assert Tier("memory") is Tier.MEMORY

    def test_ordering(self):
        assert Tier.DISK.value == "disk"
        assert Tier.HOT.value == "hot"
        assert Tier.MEMORY.value == "memory"

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            Tier("ssd")

    def test_iteration(self):
        values = [t.value for t in Tier]
        assert len(values) == 3


# ── ProcessStatus ──────────────────────────────────────────────────────


class TestProcessStatus:
    def test_all_members(self):
        assert len(ProcessStatus) == 7

    def test_values(self):
        assert ProcessStatus.CREATED.value == "created"
        assert ProcessStatus.RUNNING.value == "running"
        assert ProcessStatus.CANCELLED.value == "cancelled"

    def test_ready_value(self):
        assert ProcessStatus.READY.value == "ready"

    def test_waiting_value(self):
        assert ProcessStatus.WAITING.value == "waiting"

    def test_completed_value(self):
        assert ProcessStatus.COMPLETED.value == "completed"

    def test_failed_value(self):
        assert ProcessStatus.FAILED.value == "failed"

    def test_all_statuses_unique(self):
        values = [s.value for s in ProcessStatus]
        assert len(values) == len(set(values))

    def test_from_value(self):
        assert ProcessStatus("created") is ProcessStatus.CREATED
        assert ProcessStatus("failed") is ProcessStatus.FAILED

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            ProcessStatus("unknown")


# ── StemStatus ─────────────────────────────────────────────────────────


class TestStemStatus:
    def test_all_members(self):
        assert len(StemStatus) == 4

    def test_values(self):
        assert StemStatus.CREATED.value == "created"
        assert StemStatus.COMPLETED.value == "completed"

    def test_running_value(self):
        assert StemStatus.RUNNING.value == "running"

    def test_failed_value(self):
        assert StemStatus.FAILED.value == "failed"

    def test_all_statuses_unique(self):
        values = [s.value for s in StemStatus]
        assert len(values) == len(set(values))

    def test_from_value(self):
        assert StemStatus("running") is StemStatus.RUNNING


# ── TreeStatus ─────────────────────────────────────────────────────────


class TestTreeStatus:
    def test_all_members(self):
        assert len(TreeStatus) == 3

    def test_values(self):
        assert TreeStatus.IDLE.value == "idle"
        assert TreeStatus.BRANCHING.value == "branching"

    def test_stopped_value(self):
        assert TreeStatus.STOPPED.value == "stopped"

    def test_all_statuses_unique(self):
        values = [s.value for s in TreeStatus]
        assert len(values) == len(set(values))

    def test_from_value(self):
        assert TreeStatus("idle") is TreeStatus.IDLE


# ── TaskStatus ─────────────────────────────────────────────────────────


class TestTaskStatus:
    def test_all_members(self):
        assert len(TaskStatus) == 5

    def test_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.CANCELLED.value == "cancelled"

    def test_completed_value(self):
        assert TaskStatus.COMPLETED.value == "completed"

    def test_failed_value(self):
        assert TaskStatus.FAILED.value == "failed"

    def test_all_statuses_unique(self):
        values = [s.value for s in TaskStatus]
        assert len(values) == len(set(values))

    def test_from_value(self):
        assert TaskStatus("pending") is TaskStatus.PENDING


# ── TaskPriority ───────────────────────────────────────────────────────


class TestTaskPriority:
    def test_all_members(self):
        assert len(TaskPriority) == 4

    def test_values(self):
        assert TaskPriority.LOW.value == 0
        assert TaskPriority.NORMAL.value == 1
        assert TaskPriority.URGENT.value == 3

    def test_high_value(self):
        assert TaskPriority.HIGH.value == 2

    def test_ordering(self):
        assert TaskPriority.LOW.value < TaskPriority.NORMAL.value < TaskPriority.HIGH.value < TaskPriority.URGENT.value

    def test_from_value(self):
        assert TaskPriority(0) is TaskPriority.LOW
        assert TaskPriority(3) is TaskPriority.URGENT

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            TaskPriority(99)


# ── Task ───────────────────────────────────────────────────────────────


class TestTask:
    def test_defaults(self):
        t = Task()
        assert t.status == TaskStatus.PENDING
        assert t.priority == TaskPriority.NORMAL
        assert t.data is None

    def test_custom(self):
        t = Task(name="test", priority=TaskPriority.HIGH, data={"key": "val"})
        assert t.name == "test"
        assert t.priority == TaskPriority.HIGH

    def test_to_dict(self):
        t = Task(name="t1", data={"a": 1}, priority=TaskPriority.URGENT)
        d = t.to_dict()
        assert d["name"] == "t1"
        assert d["data"] == {"a": 1}
        assert d["priority"] == 3
        assert d["status"] == "pending"
        assert "id" in d
        assert "created_at" in d

    def test_from_dict(self):
        t = Task(name="t1", data="x", priority=TaskPriority.HIGH, tree_id="tree1")
        d = t.to_dict()
        t2 = Task.from_dict(d)
        assert t2.name == "t1"
        assert t2.data == "x"
        assert t2.priority == TaskPriority.HIGH
        assert t2.tree_id == "tree1"

    def test_from_dict_roundtrip(self):
        t = Task(name="rt", data={"nested": [1, 2]}, priority=TaskPriority.LOW, retries=2)
        d = t.to_dict()
        t2 = Task.from_dict(d)
        assert t2.name == t.name
        assert t2.data == t.data
        assert t2.retries == t.retries

    def test_unique_ids(self):
        t1 = Task()
        t2 = Task()
        assert t1.id != t2.id

    def test_default_id_length(self):
        t = Task()
        assert len(t.id) == 12

    def test_retries_default(self):
        t = Task()
        assert t.retries == 0
        assert t.max_retries == 3

    def test_tree_id_default(self):
        t = Task()
        assert t.tree_id is None

    def test_result_default(self):
        t = Task()
        assert t.result is None

    def test_error_default(self):
        t = Task()
        assert t.error is None

    def test_metadata_default(self):
        t = Task()
        assert t.metadata == {}

    def test_to_dict_has_all_keys(self):
        t = Task()
        d = t.to_dict()
        expected_keys = {"id", "name", "data", "status", "priority", "tree_id",
                         "result", "error", "created_at", "started_at",
                         "completed_at", "retries", "max_retries", "metadata"}
        assert expected_keys == set(d.keys())

    def test_to_dict_status_value(self):
        t = Task(status=TaskStatus.COMPLETED)
        d = t.to_dict()
        assert d["status"] == "completed"

    def test_to_dict_priority_value(self):
        t = Task(priority=TaskPriority.LOW)
        d = t.to_dict()
        assert d["priority"] == 0

    def test_from_dict_minimal(self):
        d = {"id": "abc", "name": "test", "status": "pending", "priority": 1}
        t = Task.from_dict(d)
        assert t.id == "abc"
        assert t.name == "test"

    def test_from_dict_with_error(self):
        d = {"id": "x", "status": "failed", "priority": 1, "error": "oops"}
        t = Task.from_dict(d)
        assert t.error == "oops"

    def test_from_dict_with_timestamps(self):
        d = {"id": "x", "status": "completed", "priority": 1,
             "started_at": 1.0, "completed_at": 2.0}
        t = Task.from_dict(d)
        assert t.started_at == 1.0
        assert t.completed_at == 2.0

    def test_from_dict_with_metadata(self):
        d = {"id": "x", "status": "pending", "priority": 1, "metadata": {"key": "val"}}
        t = Task.from_dict(d)
        assert t.metadata == {"key": "val"}


# ── Process ────────────────────────────────────────────────────────────


class TestProcess:
    def test_defaults(self):
        p = Process(fn=lambda: None)
        assert p.status == ProcessStatus.CREATED
        assert p.name == ""
        assert p.result is None
        assert p.error is None

    def test_custom(self):
        p = Process(fn=lambda: None, name="test")
        assert p.name == "test"

    def test_unique_ids(self):
        p1 = Process(fn=lambda: None)
        p2 = Process(fn=lambda: None)
        assert p1.id != p2.id

    def test_ready(self):
        p = Process(fn=lambda: None)
        p.ready()
        assert p.status == ProcessStatus.READY

    def test_running(self):
        p = Process(fn=lambda: None)
        p.running()
        assert p.status == ProcessStatus.RUNNING
        assert p.started_at is not None

    def test_complete(self):
        p = Process(fn=lambda: None)
        p.running()
        p.complete(result=42)
        assert p.status == ProcessStatus.COMPLETED
        assert p.result == 42
        assert p.completed_at is not None

    def test_fail(self):
        p = Process(fn=lambda: None)
        p.running()
        p.fail("oops")
        assert p.status == ProcessStatus.FAILED
        assert p.error == "oops"

    def test_cancel(self):
        p = Process(fn=lambda: None)
        p.running()
        p.cancel()
        assert p.status == ProcessStatus.CANCELLED
        assert p.is_cancelled

    def test_emit(self):
        p = Process(fn=lambda: None)
        p.emit("chunk1")
        p.emit("chunk2")
        assert p.stream_results == ["chunk1", "chunk2"]

    def test_on_complete(self):
        p = Process(fn=lambda: None)
        results = []
        p.on_complete(lambda proc: results.append(proc.result))
        p.running()
        p.complete(result="done")
        assert results == ["done"]

    def test_on_fail(self):
        p = Process(fn=lambda: None)
        results = []
        p.on_fail(lambda proc: results.append(proc.error))
        p.running()
        p.fail("err")
        assert results == ["err"]

    def test_on_cancel(self):
        p = Process(fn=lambda: None)
        results = []
        p.on_cancel(lambda proc: results.append("cancelled"))
        p.running()
        p.cancel()
        assert results == ["cancelled"]

    def test_on_stream(self):
        p = Process(fn=lambda: None)
        results = []
        p.on_stream(lambda proc, val: results.append(val))
        p.emit(42)
        assert results == [42]

    def test_on_progress(self):
        p = Process(fn=lambda: None)
        results = []
        p.on_progress(lambda proc, prog, msg: results.append((prog, msg)))
        p.report_progress(0.5, "halfway")
        assert results == [(0.5, "halfway")]

    def test_progress_clamping(self):
        p = Process(fn=lambda: None)
        p.report_progress(2.0, "over")
        assert p.progress == 1.0
        p.report_progress(-1.0, "under")
        assert p.progress == 0.0

    def test_elapsed_none_before_start(self):
        p = Process(fn=lambda: None)
        assert p.elapsed is None

    def test_elapsed_after_start(self):
        p = Process(fn=lambda: None)
        p.running()
        assert p.elapsed is not None
        assert p.elapsed >= 0

    def test_is_done(self):
        p = Process(fn=lambda: None)
        assert not p.is_done
        p.running()
        p.complete()
        assert p.is_done

    def test_is_done_on_fail(self):
        p = Process(fn=lambda: None)
        p.running()
        p.fail("e")
        assert p.is_done

    def test_is_done_on_cancel(self):
        p = Process(fn=lambda: None)
        p.running()
        p.cancel()
        assert p.is_done

    def test_to_dict(self):
        p = Process(fn=lambda: None, name="test")
        d = p.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "created"
        assert "id" in d
        assert "elapsed" in d

    def test_on_complete_callback_exception(self):
        """Exception in callback should not crash process."""
        p = Process(fn=lambda: None)
        p.on_complete(lambda proc: 1 / 0)
        p.running()
        p.complete(result="ok")  # should not raise

    def test_on_fail_callback_exception(self):
        p = Process(fn=lambda: None)
        p.on_fail(lambda proc: 1 / 0)
        p.running()
        p.fail("e")  # should not raise

    def test_on_cancel_callback_exception(self):
        p = Process(fn=lambda: None)
        p.on_cancel(lambda proc: 1 / 0)
        p.running()
        p.cancel()  # should not raise

    def test_on_stream_callback_exception(self):
        p = Process(fn=lambda: None)
        p.on_stream(lambda proc, val: 1 / 0)
        p.emit(42)  # should not raise

    def test_on_progress_callback_exception(self):
        p = Process(fn=lambda: None)
        p.on_progress(lambda proc, prog, msg: 1 / 0)
        p.report_progress(0.5, "x")  # should not raise

    def test_to_dict_has_required_keys(self):
        p = Process(fn=lambda: None)
        d = p.to_dict()
        assert "id" in d
        assert "status" in d
        assert "elapsed" in d
        assert "is_done" in d
        assert "is_cancelled" in d

    def test_stream_results_returns_copy(self):
        p = Process(fn=lambda: None)
        p.emit("a")
        r1 = p.stream_results
        p.emit("b")
        r2 = p.stream_results
        assert len(r1) == 1
        assert len(r2) == 2

    def test_progress_message(self):
        p = Process(fn=lambda: None)
        p.report_progress(0.5, "loading model")
        assert p.progress_message == "loading model"

    def test_progress_default(self):
        p = Process(fn=lambda: None)
        assert p.progress == 0.0
        assert p.progress_message == ""

    def test_multiple_callbacks(self):
        p = Process(fn=lambda: None)
        results1 = []
        results2 = []
        p.on_complete(lambda proc: results1.append(proc.result))
        p.on_complete(lambda proc: results2.append(proc.result))
        p.running()
        p.complete(result="done")
        assert results1 == ["done"]
        assert results2 == ["done"]

    def test_children_ids_default(self):
        p = Process(fn=lambda: None)
        assert p.children_ids == []

    def test_parent_id_default(self):
        p = Process(fn=lambda: None)
        assert p.parent_id is None

    def test_timeout_default(self):
        p = Process(fn=lambda: None)
        assert p.timeout is None

    def test_depends_on_default(self):
        p = Process(fn=lambda: None)
        assert p.depends_on == []


# ── Stem ───────────────────────────────────────────────────────────────


class TestStem:
    def test_defaults(self):
        s = Stem()
        assert s.status == StemStatus.CREATED
        assert s.processes == []

    def test_running(self):
        s = Stem()
        s.running()
        assert s.status == StemStatus.RUNNING

    def test_complete(self):
        s = Stem()
        s.complete()
        assert s.status == StemStatus.COMPLETED
        assert s.completed_at is not None

    def test_fail(self):
        s = Stem()
        s.fail()
        assert s.status == StemStatus.FAILED
        assert s.completed_at is not None

    def test_is_done(self):
        s = Stem()
        assert not s.is_done
        s.complete()
        assert s.is_done

    def test_all_done(self):
        p1 = Process(fn=lambda: None)
        p2 = Process(fn=lambda: None)
        s = Stem(processes=[p1, p2])
        assert not s.all_done
        p1.running()
        p1.complete()
        assert not s.all_done
        p2.running()
        p2.complete()
        assert s.all_done

    def test_results(self):
        p1 = Process(fn=lambda: None)
        p1.running()
        p1.complete(result=1)
        p2 = Process(fn=lambda: None)
        p2.running()
        p2.fail("e")
        s = Stem(processes=[p1, p2])
        assert s.results() == [1]

    def test_errors(self):
        p1 = Process(fn=lambda: None)
        p1.running()
        p1.fail("err1")
        p2 = Process(fn=lambda: None)
        p2.running()
        p2.complete()
        s = Stem(processes=[p1, p2])
        assert s.errors() == ["err1"]

    def test_to_dict(self):
        s = Stem()
        d = s.to_dict()
        assert "id" in d
        assert d["status"] == "created"

    def test_to_dict_has_all_keys(self):
        s = Stem()
        d = s.to_dict()
        assert "id" in d
        assert "tree_id" in d
        assert "num_processes" in d
        assert "created_at" in d
        assert "completed_at" in d

    def test_all_done_empty(self):
        s = Stem()
        assert s.all_done

    def test_results_empty(self):
        s = Stem()
        assert s.results() == []

    def test_errors_empty(self):
        s = Stem()
        assert s.errors() == []

    def test_is_done_on_fail(self):
        s = Stem()
        s.fail()
        assert s.is_done

    def test_unique_ids(self):
        s1 = Stem()
        s2 = Stem()
        assert s1.id != s2.id

    def test_tree_id(self):
        s = Stem(tree_id="mytree")
        d = s.to_dict()
        assert d["tree_id"] == "mytree"


# ── Tree ───────────────────────────────────────────────────────────────


class TestTree:
    def test_init(self):
        t = Tree("test")
        assert t.name == "test"
        assert t.status == TreeStatus.IDLE

    def test_store_recall(self):
        t = Tree("test")
        t.store("key", "value")
        assert t.recall("key") == "value"

    def test_recall_missing(self):
        t = Tree("test")
        assert t.recall("missing") is None

    def test_active_stems(self):
        t = Tree("test")
        assert t.active_stems == 0

    def test_to_dict(self):
        t = Tree("test")
        d = t.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "idle"

    def test_shutdown(self):
        t = Tree("test")
        t.shutdown()
        assert t.status == TreeStatus.STOPPED

    def test_to_dict_has_all_keys(self):
        t = Tree("test")
        d = t.to_dict()
        assert "name" in d
        assert "status" in d
        assert "active_stems" in d
        assert "max_stems" in d
        assert "graph_keys" in d

    def test_store_overwrite(self):
        t = Tree("test")
        t.store("k", "v1")
        t.store("k", "v2")
        assert t.recall("k") == "v2"

    def test_recall_after_store(self):
        t = Tree("test")
        t.store("a", 1)
        t.store("b", 2)
        assert t.recall("a") == 1
        assert t.recall("b") == 2

    def test_max_stems_default(self):
        t = Tree("test")
        assert t.max_stems == 8

    def test_custom_max_stems(self):
        t = Tree("test", max_stems=16)
        assert t.max_stems == 16


# ── EngineMetrics ──────────────────────────────────────────────────────


class TestEngineMetrics:
    def test_init(self):
        m = EngineMetrics()
        s = m.snapshot()
        assert s["spawned"] == 0
        assert s["completed"] == 0
        assert s["failed"] == 0

    def test_record_spawn(self):
        m = EngineMetrics()
        m.record_spawn()
        m.record_spawn()
        assert m.snapshot()["spawned"] == 2

    def test_record_complete(self):
        m = EngineMetrics()
        m.record_complete()
        assert m.snapshot()["completed"] == 1

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

    def test_record_dispatch(self):
        m = EngineMetrics()
        m.record_dispatch(5)
        assert m.snapshot()["dispatched"] == 5

    def test_record_memory(self):
        m = EngineMetrics()
        m.record_memory(1024)
        assert m.snapshot()["peak_memory_bytes"] == 1024

    def test_record_memory_peak(self):
        m = EngineMetrics()
        m.record_memory(100)
        m.record_memory(50)
        assert m.snapshot()["peak_memory_bytes"] == 100

    def test_reset(self):
        m = EngineMetrics()
        m.record_spawn()
        m.record_complete()
        m.reset()
        s = m.snapshot()
        assert s["spawned"] == 0
        assert s["completed"] == 0

    def test_error_rate(self):
        m = EngineMetrics()
        m.record_complete()
        m.record_complete()
        m.record_fail()
        s = m.snapshot()
        assert s["error_rate"] == pytest.approx(1 / 3)

    def test_record_restart(self):
        m = EngineMetrics()
        m.record_restart()
        assert m.snapshot()["restarted"] == 1

    def test_snapshot_has_all_keys(self):
        m = EngineMetrics()
        s = m.snapshot()
        assert "spawned" in s
        assert "completed" in s
        assert "failed" in s
        assert "cancelled" in s
        assert "timed_out" in s
        assert "restarted" in s
        assert "dispatched" in s
        assert "avg_latency_s" in s
        assert "throughput_per_s" in s
        assert "error_rate" in s
        assert "peak_memory_bytes" in s

    def test_total_memory_accumulates(self):
        m = EngineMetrics()
        m.record_memory(100)
        m.record_memory(200)
        assert m.snapshot()["total_memory_bytes"] == 300


# ── ResultCache ────────────────────────────────────────────────────────


class TestResultCache:
    def test_put_and_get(self):
        cache = ResultCache(maxsize=10)
        fn = lambda: None
        fn.__name__ = "test_fn"
        cache.put(fn, (1, 2), {"a": 1}, "result")
        hit, val = cache.get(fn, (1, 2), {"a": 1})
        assert hit is True
        assert val == "result"

    def test_miss(self):
        cache = ResultCache(maxsize=10)
        fn = lambda: None
        fn.__name__ = "test_fn"
        hit, val = cache.get(fn, (1,), {})
        assert hit is False
        assert val is None

    def test_maxsize_eviction(self):
        cache = ResultCache(maxsize=2)
        fn = lambda: None
        fn.__name__ = "fn"
        cache.put(fn, (1,), {}, "v1")
        cache.put(fn, (2,), {}, "v2")
        cache.put(fn, (3,), {}, "v3")
        assert cache.size <= 2

    def test_clear(self):
        cache = ResultCache(maxsize=10)
        fn = lambda: None
        fn.__name__ = "fn"
        cache.put(fn, (1,), {}, "v")
        count = cache.clear()
        assert count == 1
        assert cache.size == 0

    def test_invalidate_specific(self):
        cache = ResultCache(maxsize=10)
        fn = lambda: None
        fn.__name__ = "fn"
        cache.put(fn, (1,), {}, "v")
        assert cache.invalidate(fn, (1,), {}) is True
        assert cache.size == 0

    def test_invalidate_not_found(self):
        cache = ResultCache(maxsize=10)
        fn = lambda: None
        fn.__name__ = "fn"
        assert cache.invalidate(fn, (1,), {}) is False

    def test_invalidate_all(self):
        cache = ResultCache(maxsize=10)
        fn = lambda: None
        fn.__name__ = "fn"
        cache.put(fn, (1,), {}, "v")
        assert cache.invalidate() is True

    def test_invalidate_by_fn(self):
        cache = ResultCache(maxsize=10)
        fn1 = lambda: None
        fn1.__name__ = "fn1"
        fn2 = lambda: None
        fn2.__name__ = "fn2"
        cache.put(fn1, (1,), {}, "v1")
        cache.put(fn2, (2,), {}, "v2")
        assert cache.invalidate(fn1) is True
        assert cache.size == 1

    def test_stats(self):
        cache = ResultCache(maxsize=10)
        fn = lambda: None
        fn.__name__ = "fn"
        cache.put(fn, (1,), {}, "v")
        cache.get(fn, (1,), {})
        cache.get(fn, (2,), {})
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1

    def test_ttl_expired(self):
        cache = ResultCache(maxsize=10, ttl=0.01)
        fn = lambda: None
        fn.__name__ = "fn"
        cache.put(fn, (1,), {}, "v")
        time.sleep(0.02)
        hit, val = cache.get(fn, (1,), {})
        assert hit is False

    def test_size_property(self):
        cache = ResultCache(maxsize=10)
        assert cache.size == 0
        fn = lambda: None
        fn.__name__ = "fn"
        cache.put(fn, (1,), {}, "v")
        assert cache.size == 1

    def test_invalidate_all_empty(self):
        cache = ResultCache(maxsize=10)
        assert cache.invalidate() is False

    def test_invalidate_fn_not_found(self):
        cache = ResultCache(maxsize=10)
        fn = lambda: None
        fn.__name__ = "fn"
        assert cache.invalidate(fn) is False

    def test_stats_has_all_keys(self):
        cache = ResultCache(maxsize=10)
        s = cache.stats()
        assert "size" in s
        assert "maxsize" in s
        assert "ttl" in s
        assert "hits" in s
        assert "misses" in s
        assert "hit_rate" in s

    def test_ttl_none(self):
        cache = ResultCache(maxsize=10, ttl=None)
        assert cache.ttl is None

    def test_maxsize_property(self):
        cache = ResultCache(maxsize=5)
        assert cache.maxsize == 5

    def test_clear_empty(self):
        cache = ResultCache(maxsize=10)
        assert cache.clear() == 0


# ── CacheEntry ─────────────────────────────────────────────────────────


class TestCacheEntry:
    def test_touch(self):
        e = CacheEntry(key="k", tier=Tier.MEMORY)
        old_access = e.last_accessed
        time.sleep(0.001)
        e.touch()
        assert e.access_count == 1
        assert e.last_accessed >= old_access

    def test_is_expired_no_ttl(self):
        e = CacheEntry(key="k", tier=Tier.MEMORY)
        assert not e.is_expired()

    def test_is_expired_with_ttl(self):
        e = CacheEntry(key="k", tier=Tier.MEMORY, ttl=0.01)
        time.sleep(0.02)
        assert e.is_expired()

    def test_not_expired_within_ttl(self):
        e = CacheEntry(key="k", tier=Tier.MEMORY, ttl=10.0)
        assert not e.is_expired()

    def test_pinned_default(self):
        e = CacheEntry(key="k", tier=Tier.MEMORY)
        assert not e.pinned

    def test_pinned_set(self):
        e = CacheEntry(key="k", tier=Tier.MEMORY, pinned=True)
        assert e.pinned

    def test_touch_multiple(self):
        e = CacheEntry(key="k", tier=Tier.MEMORY)
        e.touch()
        e.touch()
        e.touch()
        assert e.access_count == 3

    def test_size_bytes_default(self):
        e = CacheEntry(key="k", tier=Tier.MEMORY)
        assert e.size_bytes == 0

    def test_size_bytes_custom(self):
        e = CacheEntry(key="k", tier=Tier.MEMORY, size_bytes=1024)
        assert e.size_bytes == 1024

    def test_data_default(self):
        e = CacheEntry(key="k", tier=Tier.MEMORY)
        assert e.data is None

    def test_created_at_populated(self):
        e = CacheEntry(key="k", tier=Tier.MEMORY)
        assert e.created_at > 0

    def test_key(self):
        e = CacheEntry(key="mykey", tier=Tier.DISK)
        assert e.key == "mykey"


# ── CacheStats ─────────────────────────────────────────────────────────


class TestCacheStats:
    def test_hit_rate_zero(self):
        s = CacheStats()
        assert s.hit_rate == 0.0

    def test_hit_rate(self):
        s = CacheStats(hits=3, misses=1)
        assert s.hit_rate == pytest.approx(0.75)

    def test_to_dict(self):
        s = CacheStats(hits=5, misses=3)
        d = s.to_dict()
        assert d["hits"] == 5
        assert d["misses"] == 3
        assert d["hit_rate"] == pytest.approx(5 / 8)

    def test_defaults(self):
        s = CacheStats()
        assert s.hits == 0
        assert s.misses == 0
        assert s.evictions == 0
        assert s.promotions == 0
        assert s.demotions == 0

    def test_to_dict_has_all_keys(self):
        s = CacheStats()
        d = s.to_dict()
        assert "hits" in d
        assert "misses" in d
        assert "hit_rate" in d
        assert "evictions" in d
        assert "promotions" in d
        assert "demotions" in d
        assert "disk" in d
        assert "hot" in d
        assert "memory" in d

    def test_per_tier_counts(self):
        s = CacheStats(disk_hits=1, hot_hits=2, memory_hits=3)
        d = s.to_dict()
        assert d["disk"]["hits"] == 1
        assert d["hot"]["hits"] == 2
        assert d["memory"]["hits"] == 3


# ── MemoryStore ────────────────────────────────────────────────────────


class TestMemoryStore:
    def test_put_get(self):
        s = MemoryStore()
        s.put("k", "v")
        assert s.get("k") == "v"

    def test_get_missing(self):
        s = MemoryStore()
        assert s.get("missing") is None

    def test_remove(self):
        s = MemoryStore()
        s.put("k", "v")
        assert s.remove("k") is True
        assert s.get("k") is None

    def test_remove_missing(self):
        s = MemoryStore()
        assert s.remove("missing") is False

    def test_list_keys(self):
        s = MemoryStore()
        s.put("a", 1)
        s.put("b", 2)
        assert set(s.list_keys()) == {"a", "b"}

    def test_exists(self):
        s = MemoryStore()
        s.put("k", "v")
        assert s.exists("k")
        assert not s.exists("missing")

    def test_evict_lru(self):
        s = MemoryStore(max_size_bytes=100)
        s.put("a", "val_a", size_bytes=50)
        s.put("b", "val_b", size_bytes=50)
        evicted = s.evict_lru(10)
        assert len(evicted) >= 1

    def test_evict_lfu(self):
        s = MemoryStore(max_size_bytes=100)
        s.put("a", "val_a", size_bytes=50)
        s.put("b", "val_b", size_bytes=50)
        evicted = s.evict_lfu(10, {"a": 0, "b": 5})
        assert len(evicted) >= 1

    def test_evict_lfu_no_need(self):
        s = MemoryStore(max_size_bytes=1000)
        s.put("a", "val_a", size_bytes=10)
        evicted = s.evict_lfu(10, {"a": 1})
        assert evicted == []

    def test_put_overwrite(self):
        s = MemoryStore()
        s.put("k", "v1")
        s.put("k", "v2")
        assert s.get("k") == "v2"

    def test_size_bytes(self):
        s = MemoryStore()
        assert s.size_bytes() == 0
        s.put("k", "v", size_bytes=100)
        assert s.size_bytes() == 100

    def test_lru_ordering(self):
        s = MemoryStore(max_size_bytes=100)
        s.put("a", 1, size_bytes=40)
        s.put("b", 2, size_bytes=40)
        s.put("c", 3, size_bytes=40)
        # Access "a" to make it recent
        s.get("a")
        # Now "b" is the oldest — evict to free 30 bytes
        evicted = s.evict_lru(30)
        assert len(evicted) >= 1
        assert evicted[0][0] == "b"


# ── HotStore ───────────────────────────────────────────────────────────


class TestHotStore:
    def test_put_get(self):
        s = HotStore()
        s.put("k", "v")
        assert s.get("k") == "v"

    def test_remove(self):
        s = HotStore()
        s.put("k", "v")
        assert s.remove("k") is True

    def test_list_keys(self):
        s = HotStore()
        s.put("a", 1)
        assert s.list_keys() == ["a"]

    def test_exists(self):
        s = HotStore()
        s.put("k", "v")
        assert s.exists("k")

    def test_size_bytes(self):
        s = HotStore()
        assert s.size_bytes() == 0

    def test_get_missing(self):
        s = HotStore()
        assert s.get("missing") is None

    def test_put_with_size(self):
        s = HotStore()
        s.put("k", "v", size_bytes=100)
        assert s.size_bytes() == 100


# ── DiskStore ──────────────────────────────────────────────────────────


class TestDiskStore:
    def test_put_get_ndarray(self, tmp_path):
        s = DiskStore(tmp_path)
        arr = np.array([1.0, 2.0, 3.0])
        s.put("k", arr)
        result = s.get("k")
        np.testing.assert_array_equal(result, arr)

    def test_put_get_json(self, tmp_path):
        s = DiskStore(tmp_path)
        s.put("k", {"a": 1})
        result = s.get("k")
        assert result == {"a": 1}

    def test_remove(self, tmp_path):
        s = DiskStore(tmp_path)
        s.put("k", "v")
        assert s.remove("k") is True
        assert s.get("k") is None

    def test_remove_missing(self, tmp_path):
        s = DiskStore(tmp_path)
        assert s.remove("missing") is False

    def test_list_keys(self, tmp_path):
        s = DiskStore(tmp_path)
        s.put("a", 1)
        s.put("b", 2)
        assert set(s.list_keys()) == {"a", "b"}

    def test_exists(self, tmp_path):
        s = DiskStore(tmp_path)
        s.put("k", "v")
        assert s.exists("k")
        assert not s.exists("missing")

    def test_size_bytes(self, tmp_path):
        s = DiskStore(tmp_path)
        s.put("k", "v")
        assert s.size_bytes() > 0

    def test_get_missing_key(self, tmp_path):
        s = DiskStore(tmp_path)
        assert s.get("nonexistent") is None

    def test_put_with_meta(self, tmp_path):
        s = DiskStore(tmp_path)
        s.put("k", {"a": 1}, meta={"tag": "test"})
        result = s.get("k")
        assert result == {"a": 1}

    def test_special_chars_in_key(self, tmp_path):
        s = DiskStore(tmp_path)
        s.put("key/with\\backslash", "v")
        assert s.exists("key/with\\backslash")

    def test_overwrite(self, tmp_path):
        s = DiskStore(tmp_path)
        s.put("k", "v1")
        s.put("k", "v2")
        assert s.get("k") == "v2"


# ── TaskQueue ──────────────────────────────────────────────────────────


class TestTaskQueue:
    def test_submit_and_next(self):
        q = TaskQueue()
        t = Task(name="t1")
        q.submit(t)
        next_t = q.next()
        assert next_t is not None
        assert next_t.status == TaskStatus.RUNNING

    def test_next_empty(self):
        q = TaskQueue()
        assert q.next() is None

    def test_priority_ordering(self):
        q = TaskQueue()
        q.submit(Task(name="low", priority=TaskPriority.LOW))
        q.submit(Task(name="urgent", priority=TaskPriority.URGENT))
        q.submit(Task(name="normal", priority=TaskPriority.NORMAL))
        t1 = q.next()
        t2 = q.next()
        t3 = q.next()
        assert t1.name == "urgent"
        assert t2.name == "normal"
        assert t3.name == "low"

    def test_complete(self):
        q = TaskQueue()
        t = Task(name="t1")
        q.submit(t)
        q.next()
        result = q.complete(t.id, result="done")
        assert result.status == TaskStatus.COMPLETED
        assert result.result == "done"

    def test_fail_and_retry(self):
        q = TaskQueue()
        t = Task(name="t1", max_retries=1)
        q.submit(t)
        q.next()
        q.fail(t.id, "err")
        assert t.status == TaskStatus.PENDING
        assert t.retries == 1

    def test_fail_no_retry(self):
        q = TaskQueue()
        t = Task(name="t1", max_retries=0)
        q.submit(t)
        q.next()
        q.fail(t.id, "err")
        assert t.status == TaskStatus.FAILED

    def test_cancel(self):
        q = TaskQueue()
        t = Task(name="t1")
        q.submit(t)
        q.cancel(t.id)
        assert t.status == TaskStatus.CANCELLED

    def test_cancel_all(self):
        q = TaskQueue()
        q.submit(Task(name="t1"))
        q.submit(Task(name="t2"))
        count = q.cancel_all()
        assert count == 2

    def test_submit_batch(self):
        q = TaskQueue()
        tasks = q.submit_batch([
            {"name": "a"},
            {"name": "b"},
        ])
        assert len(tasks) == 2

    def test_pause_resume(self):
        q = TaskQueue()
        q.pause()
        q.submit(Task(name="t1"))
        assert q.next() is None
        q.resume()
        t = q.next()
        assert t is not None

    def test_stats(self):
        q = TaskQueue()
        q.submit(Task(name="t1"))
        s = q.stats()
        assert s["total"] == 1
        assert s["pending"] == 1

    def test_clear_completed(self):
        q = TaskQueue()
        t = Task(name="t1")
        q.submit(t)
        q.next()
        q.complete(t.id)
        count = q.clear_completed()
        assert count == 1

    def test_save_load(self, tmp_path):
        q = TaskQueue(name="test", storage_dir=tmp_path)
        q.submit(Task(name="t1"))
        q.save()
        loaded = TaskQueue.load(tmp_path / "test.tasks.json")
        assert loaded.name == "test"

    def test_on_complete_callback(self):
        q = TaskQueue()
        results = []
        q.on_complete(lambda t: results.append(t.name))
        t = Task(name="t1")
        q.submit(t)
        q.next()
        q.complete(t.id)
        assert "t1" in results

    def test_submit_batch_with_priority(self):
        q = TaskQueue()
        tasks = q.submit_batch([
            {"name": "a", "priority": 3},
            {"name": "b"},
        ], priority=TaskPriority.LOW)
        assert tasks[0].priority.value == 3

    def test_register_handler(self):
        q = TaskQueue()
        q.register_handler("train", lambda t: "done")
        assert "train" in q._handlers

    def test_wait_all(self):
        q = TaskQueue()
        q.submit(Task(name="t1"))
        t = q.next()
        q.complete(t.id)
        result = q.wait_all(timeout=1.0)
        assert len(result) == 1

    def test_retry_all(self):
        q = TaskQueue()
        # Submit tasks and fail them with max_retries=0 so they stay FAILED
        t1 = Task(name="a", max_retries=0)
        t2 = Task(name="b", max_retries=0)
        q.submit(t1)
        q.submit(t2)
        q.next()
        q.next()
        q.fail(t1.id, "e1")
        q.fail(t2.id, "e2")
        count = q.retry_all(reset_retries=True)
        assert count == 2

    def test_task_priority_to_int(self):
        assert TaskQueue._task_priority_to_int(TaskPriority.URGENT) == 0
        assert TaskQueue._task_priority_to_int(TaskPriority.HIGH) == 1
        assert TaskQueue._task_priority_to_int(TaskPriority.NORMAL) == 2
        assert TaskQueue._task_priority_to_int(TaskPriority.LOW) == 3

    def test_submit_over_max_size(self):
        q = TaskQueue(max_size=2)
        q.submit(Task(name="a"))
        q.submit(Task(name="b"))
        with pytest.raises(ValueError, match="full"):
            q.submit(Task(name="c"))

    def test_list_tasks_all(self):
        q = TaskQueue()
        q.submit(Task(name="a"))
        q.submit(Task(name="b"))
        tasks = q.list_tasks()
        assert len(tasks) == 2

    def test_list_tasks_by_status(self):
        q = TaskQueue()
        t = Task(name="a")
        q.submit(t)
        q.next()
        q.complete(t.id)
        pending = q.list_tasks(status=TaskStatus.PENDING)
        completed = q.list_tasks(status=TaskStatus.COMPLETED)
        assert len(pending) == 0
        assert len(completed) == 1

    def test_cancel_many(self):
        q = TaskQueue()
        t1 = Task(name="a")
        t2 = Task(name="b")
        q.submit(t1)
        q.submit(t2)
        results = q.cancel_many([t1.id, t2.id])
        assert len(results) == 2
        assert all(r.status == TaskStatus.CANCELLED for r in results)

    def test_retry_from_completed(self):
        q = TaskQueue()
        t = Task(name="t1")
        q.submit(t)
        q.next()
        q.complete(t.id)
        assert t.status == TaskStatus.COMPLETED
        retried = q.retry(t.id)
        assert retried.status == TaskStatus.PENDING

    def test_wait_for(self):
        q = TaskQueue()
        t = Task(name="t1")
        q.submit(t)
        q.next()
        q.complete(t.id, result="done")
        result = q.wait_for(t.id, timeout=1.0)
        assert result.status == TaskStatus.COMPLETED

    def test_wait_for_timeout(self):
        q = TaskQueue()
        t = Task(name="t1")
        q.submit(t)
        result = q.wait_for(t.id, timeout=0.01)
        assert result is None or result.status in (TaskStatus.PENDING, TaskStatus.RUNNING)

    def test_submit_many(self):
        q = TaskQueue()
        tasks = q.submit_many([Task(name="a"), Task(name="b")])
        assert len(tasks) == 2

    def test_get_task(self):
        q = TaskQueue()
        t = Task(name="t1")
        q.submit(t)
        assert q.get_task(t.id) is t

    def test_get_task_missing(self):
        q = TaskQueue()
        assert q.get_task("nonexistent") is None

    def test_stats_has_all_keys(self):
        q = TaskQueue()
        s = q.stats()
        assert "name" in s
        assert "total" in s
        assert "pending" in s
        assert "running" in s
        assert "completed" in s
        assert "failed" in s
        assert "cancelled" in s
        assert "paused" in s
        assert "handlers" in s

    def test_complete_nonexistent(self):
        q = TaskQueue()
        result = q.complete("nonexistent")
        assert result is None

    def test_fail_nonexistent(self):
        q = TaskQueue()
        result = q.fail("nonexistent", "err")
        assert result is None

    def test_retry_nonexistent(self):
        q = TaskQueue()
        assert q.retry("nonexistent") is None

    def test_retry_pending_task(self):
        q = TaskQueue()
        t = Task(name="t1", max_retries=0)
        q.submit(t)
        # Task is still pending, retry should reset it
        retried = q.retry(t.id, reset_retries=True)
        assert retried.status == TaskStatus.PENDING

    def test_callback_exception(self):
        q = TaskQueue()
        q.on_complete(lambda t: 1 / 0)
        t = Task(name="t1")
        q.submit(t)
        q.next()
        q.complete(t.id)  # should not raise

    def test_default_name(self):
        q = TaskQueue()
        assert q.name == "default"

    def test_submit_batch_with_data(self):
        q = TaskQueue()
        tasks = q.submit_batch([
            {"name": "a", "data": {"key": "val"}},
            {"name": "b", "tree_id": "tree1"},
        ])
        assert tasks[0].data == {"key": "val"}
        assert tasks[1].tree_id == "tree1"

    def test_submit_batch_with_metadata(self):
        q = TaskQueue()
        tasks = q.submit_batch([
            {"name": "a", "metadata": {"tag": "test"}},
        ])
        assert tasks[0].metadata == {"tag": "test"}
