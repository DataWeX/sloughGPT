"""Coverage tests for domains/shell/kernel_scheduler.py."""

import time

from domains.shell.kernel_process import Process, ProcessState, Priority
from domains.shell.kernel_scheduler import Scheduler


def _proc(pid, priority=Priority.NORMAL, state=ProcessState.CREATED):
    return Process(pid=pid, name=f"p{pid}", priority=priority, state=state)


def test_initial_state():
    s = Scheduler()
    assert s.current_pid is None
    assert s.current_process is None
    assert s.process_count == 0
    assert s.active_count == 0
    assert s.queue_sizes() == {"realtime": 0, "high": 0, "normal": 0, "low": 0, "idle": 0}


def test_current_process_returns_process():
    s = Scheduler()
    p = _proc(1)
    s.add(p)
    s.tick()
    assert s.current_pid == 1
    assert s.current_process is p


def test_set_callbacks():
    s = Scheduler()
    seen = []
    s.set_callbacks(on_complete=lambda proc, result: seen.append(("c", proc, result)))
    s.set_callbacks(on_interrupt=lambda proc: seen.append(("i", proc)))
    assert "complete" in s._callbacks
    assert "interrupt" in s._callbacks


def test_add_transitions_created_to_ready():
    s = Scheduler()
    p = _proc(1)
    s.add(p)
    assert p.state == ProcessState.READY
    assert s._queues["high"] == [1]


def test_add_keeps_non_created_state():
    s = Scheduler()
    p = _proc(1, state=ProcessState.STOPPED)
    s.add(p)
    assert p.state == ProcessState.STOPPED
    assert s._queues["high"] == [1]


def test_add_by_priority_queue():
    s = Scheduler()
    s.add(_proc(1, Priority.CRITICAL))
    s.add(_proc(2, Priority.HIGH))
    s.add(_proc(3, Priority.NORMAL))
    s.add(_proc(4, Priority.LOW))
    s.add(_proc(5, Priority.IDLE))
    assert s.queue_sizes()["realtime"] == 2
    assert s.queue_sizes()["high"] == 1
    assert s.queue_sizes()["normal"] == 1
    assert s.queue_sizes()["low"] == 1
    assert s.queue_sizes()["idle"] == 0


def test_remove_missing_returns_none():
    s = Scheduler()
    assert s.remove(999) is None


def test_remove_clears_queue_and_current():
    s = Scheduler()
    p = _proc(1)
    s.add(p)
    s.tick()
    removed = s.remove(1)
    assert removed is p
    assert s.process_count == 0
    assert s.current_pid is None


def test_get():
    s = Scheduler()
    assert s.get(1) is None
    p = _proc(1)
    s.add(p)
    assert s.get(1) is p


def test_list_all():
    s = Scheduler()
    s.add(_proc(1))
    s.add(_proc(2))
    assert len(s.list_all()) == 2


def test_tick_skips_stale_pid():
    s = Scheduler()
    s._queues["normal"] = [99]
    assert s.tick() is None
    assert s.current_pid is None


def test_tick_skips_stopped():
    s = Scheduler()
    s.add(_proc(1, state=ProcessState.STOPPED))
    assert s.tick() is None


def test_tick_sleeping_requeue_and_expiry():
    s = Scheduler()
    p1 = _proc(1)
    p2 = _proc(2)
    s.add(p1)
    s.add(p2)
    s.wait_for(1, timeout=10)
    out = s.tick()
    assert out is p2
    assert s._queues["high"] == [1]
    s.complete(2)
    s._sleeping[1] = time.time() - 1
    out2 = s.tick()
    assert out2 is p1
    assert 1 not in s._sleeping


def test_tick_returns_current_running():
    s = Scheduler()
    p = _proc(1)
    s.add(p)
    s.tick()
    assert s.tick() is p


def test_tick_priority_order():
    s = Scheduler()
    s.add(_proc(1, Priority.NORMAL))
    s.add(_proc(2, Priority.CRITICAL))
    out = s.tick()
    assert out.pid == 2


def test_deps_satisfied():
    s = Scheduler()
    dep = _proc(1)
    dep.state = ProcessState.STOPPED
    s.add(dep)
    p = _proc(2)
    p.metadata["deps"] = [1]
    s.add(p)
    assert s._deps_satisfied(p) is True
    dep2 = _proc(3)
    s.add(dep2)
    p2 = _proc(4)
    p2.metadata["deps"] = [3]
    s.add(p2)
    assert s._deps_satisfied(p2) is False


def test_deps_satisfied_no_metadata():
    s = Scheduler()
    p = _proc(1)
    assert s._deps_satisfied(p) is True


def test_wait_for_sets_waiting():
    s = Scheduler()
    p = _proc(1)
    s.add(p)
    s.wait_for(1, timeout=5)
    assert p.state == ProcessState.WAITING
    assert 1 in s._sleeping


def test_wait_for_missing_proc_still_sleeps():
    s = Scheduler()
    s.wait_for(99, timeout=0)
    assert 99 in s._sleeping
    assert s._sleeping[99] == float("inf")


def test_wake():
    s = Scheduler()
    p = _proc(1)
    s.add(p)
    s.wait_for(1, timeout=10)
    s.wake(1)
    assert p.state == ProcessState.READY
    assert 1 not in s._sleeping
    assert s._queues["high"] == [1, 1]


def test_wake_not_waiting_still_requeues():
    s = Scheduler()
    p = _proc(1, state=ProcessState.READY)
    s.add(p)
    s._sleeping[1] = 123.0
    s.wake(1)
    assert 1 not in s._sleeping
    assert s._queues["high"] == [1]


def test_complete_missing_proc():
    s = Scheduler()
    s.complete(999, {"x": 1})


def test_complete_basic():
    s = Scheduler()
    p = _proc(1)
    s.add(p)
    s.tick()
    s.complete(1, {"ok": True})
    assert p.state == ProcessState.ZOMBIE
    assert p.result == {"ok": True}
    assert s.current_pid is None


def test_complete_invokes_callback():
    s = Scheduler()
    seen = []
    s.set_callbacks(on_complete=lambda proc, result: seen.append((proc, result)))
    p = _proc(1)
    s.add(p)
    s.complete(1, "done")
    assert seen == [(p, "done")]


def test_complete_callback_raising_is_swallowed():
    s = Scheduler()
    s.set_callbacks(on_complete=lambda proc, result: (_ for _ in ()).throw(RuntimeError("boom")))
    p = _proc(1)
    s.add(p)
    s.complete(1, "done")
    assert p.state == ProcessState.ZOMBIE


def test_reap_specific():
    s = Scheduler()
    p = _proc(1)
    s.add(p)
    s.complete(1)
    reaped = s.reap(1)
    assert reaped is p
    assert s.process_count == 0


def test_reap_specific_not_zombie_returns_none():
    s = Scheduler()
    p = _proc(1)
    s.add(p)
    assert s.reap(1) is None
    assert s.reap(999) is None


def test_reap_all():
    s = Scheduler()
    p1 = _proc(1)
    p2 = _proc(2)
    s.add(p1)
    s.add(p2)
    s.complete(1)
    p2.state = ProcessState.STOPPED
    reaped = s.reap()
    assert len(reaped) == 2
    assert s.process_count == 0


def test_finish():
    s = Scheduler()
    p = _proc(1)
    s.add(p)
    s.tick()
    s._finish(1)
    assert p.state == ProcessState.STOPPED
    assert s.current_pid is None
    s._finish(999)


def test_stats():
    s = Scheduler()
    p = _proc(1)
    s.add(p)
    s.tick()
    s.wait_for(2, timeout=5)
    stats = s.stats()
    assert stats["total_processes"] == 1
    assert stats["total"] == 1
    assert stats["active"] == 1
    assert stats["current_pid"] == 1
    assert stats["queues"]["normal"] == 0
    assert stats["sleeping"] == 1


def test_priority_to_queue_raw_int():
    s = Scheduler()
    assert s._priority_to_queue(0) == "realtime"
    assert s._priority_to_queue(1) == "realtime"
    assert s._priority_to_queue(2) == "high"
    assert s._priority_to_queue(3) == "normal"
    assert s._priority_to_queue(4) == "low"
    assert s._priority_to_queue(5) == "idle"
    assert s._priority_to_queue(99) == "idle"
