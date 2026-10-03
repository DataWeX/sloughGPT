"""Tests for pipe.py — the single admission door in front of the execution stack.

Covers the queue's own contract (capacity, backpressure, derived bound) and
the four doors wired to it. Swirls/tree topology are deliberately absent:
they are data-side and not this module's concern.
"""

from __future__ import annotations

import threading
import time

import pytest

# One canonical root throughout. The domain.infrastructure._internal shim
# re-executes each real module under its own name (its package __init__
# overwrites __path__), so a class imported from there is NOT the class the
# engine raises -- pytest.raises would silently fail to match.
from domains.infrastructure.pugqeep import (
    DEFAULT_CAPACITY,
    Pipe,
    PipeClosed,
    ProcessQueue,
)
from domains.infrastructure.pugqeep.config import EngineConfig
from domains.infrastructure.pugqeep.engine import Engine, Pool, Process


class _Stub:
    """Minimal stand-in for Process: an id and a done flag."""

    def __init__(self, pid: str) -> None:
        self.id = pid
        self.is_done = False


# ── the bound is derived, never declared ─────────────────────────────────────


class TestDerivedBound:
    def test_non_positive_capacity_rejected(self):
        with pytest.raises(ValueError):
            ProcessQueue(maxsize=0)
        with pytest.raises(ValueError):
            ProcessQueue(maxsize=-1)

    def test_diameter_is_read_off_maxsize(self):
        queue = ProcessQueue(maxsize=7)
        assert queue.diameter() == 7
        assert queue.diameter() == queue.maxsize
        assert queue.usage() == 0
        assert queue.headroom() == 7

    def test_three_views_of_one_queue(self):
        pipe = Pipe("t", capacity=5)
        assert pipe.diameter() == 5
        assert pipe.usage() == 0
        assert pipe.headroom() == 5
        # diameter() must not be a separate field that can drift from the queue
        assert pipe.diameter() == pipe.queue.maxsize
        pipe.admit(_Stub("a"))
        assert (pipe.diameter(), pipe.usage(), pipe.headroom()) == (5, 1, 4)

    def test_headroom_is_derived_after_every_admission(self):
        pipe = Pipe("t", capacity=3)
        for i in range(3):
            pipe.admit(_Stub(str(i)))
            assert pipe.headroom() == 3 - (i + 1)
        assert pipe.headroom() == 0

    def test_peak_is_observed_not_configured(self):
        queue = ProcessQueue(maxsize=4)
        stubs = [_Stub(str(i)) for i in range(4)]
        for stub in stubs:
            queue.put(stub)
        assert queue.peak_usage == 4
        for stub in stubs:
            stub.is_done = True
        # peak survives retirement; the stack empties but the high-water mark stays
        assert queue.usage() == 0
        assert queue.peak_usage == 4
        assert queue.diameter() == 4

    def test_stats_carries_the_whole_bound(self):
        queue = ProcessQueue(maxsize=2)
        queue.put(_Stub("a"))
        assert queue.stats() == {
            "diameter": 2,
            "usage": 1,
            "headroom": 1,
            "peak": 1,
            "waits": 0,
            "closed": False,
        }


# ── admission: capacity hits block, they never reject ────────────────────────


class TestAdmission:
    def test_fills_to_capacity(self):
        queue = ProcessQueue(maxsize=3)
        for i in range(3):
            assert queue.put(_Stub(str(i))) is True
        assert queue.usage() == 3
        assert queue.headroom() == 0

    def test_double_admit_does_not_double_count(self):
        """Engine.spawn and Pool.branch both sit in front of this queue."""
        queue = ProcessQueue(maxsize=2)
        stub = _Stub("a")
        assert queue.put(stub) is True
        assert queue.put(stub) is True
        assert queue.usage() == 1

    def test_admitting_without_an_id_is_a_type_error(self):
        queue = ProcessQueue(maxsize=1)
        with pytest.raises(TypeError):
            queue.put(object())

    def test_full_blocks_then_release_admits(self):
        queue = ProcessQueue(maxsize=1)
        first = _Stub("a")
        queue.put(first)

        outcome: list[bool] = []
        worker = threading.Thread(target=lambda: outcome.append(queue.put(_Stub("b"), timeout=5)))
        worker.start()
        time.sleep(0.2)
        assert outcome == [], "full must block, not reject"

        queue.release(first)
        worker.join(timeout=5)
        assert outcome == [True]
        assert queue.usage() == 1

    def test_finished_process_retires_without_an_explicit_release(self):
        """Nobody has to remember to call release(): usage() retires on read."""
        queue = ProcessQueue(maxsize=1)
        stub = _Stub("a")
        queue.put(stub)
        stub.is_done = True
        assert queue.usage() == 0
        assert queue.headroom() == 1
        assert queue.put(_Stub("b")) is True

    def test_blocked_producer_wakes_on_completion_with_no_release_call(self):
        """The producer re-sweeps on its own clock — no notification to miss."""
        queue = ProcessQueue(maxsize=1)
        first = _Stub("a")
        queue.put(first)

        outcome: list[bool] = []
        worker = threading.Thread(target=lambda: outcome.append(queue.put(_Stub("b"), timeout=5)))
        worker.start()
        time.sleep(0.2)
        assert outcome == []

        first.is_done = True  # completion only; release() is never called
        worker.join(timeout=5)
        assert outcome == [True], "capacity must free itself when a process finishes"

    def test_unreadable_process_frees_its_slot_rather_than_leaking(self):
        class _Broken:
            id = "broken"

            @property
            def is_done(self):
                raise RuntimeError("unreadable")

        queue = ProcessQueue(maxsize=1)
        queue.put(_Broken())
        assert queue.usage() == 0, "an unreadable process must not wedge the queue"

    def test_timeout_returns_false_instead_of_waiting_forever(self):
        queue = ProcessQueue(maxsize=1)
        queue.put(_Stub("a"))
        assert queue.put(_Stub("b"), timeout=0.15) is False
        assert queue.blocked_waits >= 1

    def test_close_wakes_every_blocked_producer(self):
        queue = ProcessQueue(maxsize=1)
        queue.put(_Stub("a"))

        outcome: list[bool] = []
        worker = threading.Thread(target=lambda: outcome.append(queue.put(_Stub("b"))))
        worker.start()
        time.sleep(0.2)
        assert outcome == []

        queue.close()
        worker.join(timeout=5)
        assert outcome == [False], "close must unblock, or stop() becomes a hang"
        assert queue.put(_Stub("c")) is False

    def test_open_re_arms_after_close(self):
        queue = ProcessQueue(maxsize=2)
        queue.close()
        assert queue.put(_Stub("a")) is False
        queue.open()
        assert queue.closed is False
        assert queue.put(_Stub("a")) is True

    def test_release_and_clear(self):
        queue = ProcessQueue(maxsize=3)
        a, b = _Stub("a"), _Stub("b")
        queue.put(a)
        queue.put(b)
        assert queue.release(a) is True
        assert queue.release(a) is False, "releasing twice must not over-free"
        assert queue.usage() == 1
        queue.clear()
        assert queue.usage() == 0


# ── Pipe: the façade over one queue ──────────────────────────────────────────


class TestPipe:
    def test_owns_contains_and_release(self):
        pipe = Pipe("t", capacity=4)
        stub = _Stub("x")
        pipe.admit(stub)
        assert pipe.owns(stub)
        assert stub in pipe
        assert pipe.release(stub) is True
        assert not pipe.owns(stub)

    def test_repr_reports_the_bound(self):
        pipe = Pipe("t", capacity=9)
        assert "t" in repr(pipe)
        assert "0/9" in repr(pipe)

    def test_close_open_and_flag(self):
        pipe = Pipe("t")
        assert not pipe.is_closed
        pipe.close()
        assert pipe.is_closed
        pipe.open()
        assert not pipe.is_closed


# ── the four doors ───────────────────────────────────────────────────────────


class TestEngineDoors:
    def test_capacity_comes_from_config(self):
        engine = Engine(config=EngineConfig(name="t", queue_size=5))
        try:
            assert engine._pipe.diameter() == 5
        finally:
            engine.stop()

    def test_default_capacity_without_config(self):
        engine = Engine("t")
        try:
            assert engine._pipe.diameter() == DEFAULT_CAPACITY
        finally:
            engine.stop()

    def test_spawn_admits_before_registering(self):
        engine = Engine("t")
        try:
            proc = engine.spawn(lambda: 1, name="x")
            assert engine._pipe.usage() == 1
            assert proc in engine._pipe
            assert proc.id in engine._processes
        finally:
            engine.stop()

    def test_spawn_after_stop_refuses_without_registering(self):
        """Admission is the first mutation, so a refusal must leave no trace."""
        engine = Engine("t")
        engine.stop()
        with pytest.raises(PipeClosed):
            engine.spawn(lambda: 1)
        assert engine._pipe.usage() == 0
        assert engine._processes == {}

    def test_spawn_blocks_at_capacity_then_proceeds(self):
        engine = Engine(config=EngineConfig(name="t", queue_size=2))
        try:
            first = engine.spawn(lambda: 1)
            engine.spawn(lambda: 2)
            assert engine._pipe.usage() == 2

            outcome: list[object] = []
            worker = threading.Thread(target=lambda: outcome.append(engine.spawn(lambda: 3)))
            worker.start()
            time.sleep(0.2)
            assert outcome == [], "spawn must apply backpressure, not reject"

            first.complete(1)
            worker.join(timeout=5)
            assert len(outcome) == 1, "spawn must proceed once capacity frees"
            assert engine._pipe.usage() <= 2
        finally:
            engine.stop()

    def test_engine_pools_share_one_pipe(self):
        """One queue per admission path — not one per pool."""
        engine = Engine("t")
        try:
            plain = engine.pool("plain")
            guarded = engine.pool("guarded", guarded=True)
            assert plain._pipe is engine._pipe
            assert guarded._pipe is engine._pipe
            assert plain._owns_pipe is False
            assert guarded._owns_pipe is False
        finally:
            engine.stop()

    def test_dispatch_does_not_double_admit(self):
        engine = Engine("t")
        try:
            engine.pool("a")
            engine.spawn(lambda: 42, pool="a")
            assert engine._pipe.usage() == 1
            engine.dispatch()
            # spawn admitted it; Pool.branch must pass through, not add a second slot
            assert engine._pipe.usage() <= 1
        finally:
            engine.stop()

    def test_engine_branch_admits_processes_that_never_passed_spawn(self):
        """Engine.branch takes arbitrary Process objects — a door of its own."""
        engine = Engine("t")
        try:
            engine.pool("a")
            proc = Process(fn=lambda: 7, name="direct")
            engine.branch("a", [proc])
            assert engine._pipe.usage() <= 1
        finally:
            engine.stop()

    def test_stop_closes_the_pipe_and_dispatch_re_arms_it(self):
        engine = Engine("t")
        try:
            engine.stop()
            assert engine._pipe.is_closed
            engine.dispatch()
            assert not engine._pipe.is_closed
        finally:
            engine.stop()


class TestStandalonePoolDoor:
    def test_mints_and_owns_its_own_pipe(self):
        pool = Pool("solo")
        try:
            assert pool._owns_pipe is True
            assert pool._pipe.name == "pool:solo"
        finally:
            pool.shutdown()

    def test_branch_admits(self):
        pool = Pool("solo")
        try:
            # Gate the work: an instantly-finishing lambda retires itself
            # before the assertion reads usage(), which would look like a
            # failed admission rather than a completed one.
            gate = threading.Event()
            proc = Process(fn=gate.wait, args=(5,), name="b")
            pool.branch([proc])
            assert pool._pipe.owns(proc)
            assert pool._pipe.usage() == 1
            gate.set()
        finally:
            pool.shutdown()

    def test_shutdown_closes_only_a_pipe_it_owns(self):
        owned = Pool("owned")
        owned.shutdown()
        assert owned._pipe.is_closed

        engine = Engine("t")
        try:
            shared = engine.pool("shared")
            engine.stop()
            # Engine.stop() shuts pools down; an injected pipe is the Engine's
            # and its lifetime is the Engine's, already closed by stop().
            assert shared._owns_pipe is False
        finally:
            engine.stop()

    def test_branch_rolls_back_when_the_pipe_refuses(self):
        pool = Pool("solo")
        pool._pipe.close()
        try:
            with pytest.raises(PipeClosed):
                pool.branch([Process(fn=lambda: 1, name="x")])
            assert pool._pipe.usage() == 0
        finally:
            pool.shutdown()

    def test_rollback_frees_only_what_this_call_added(self):
        """A process admitted elsewhere must survive this call's failure."""
        pool = Pool("solo")
        preexisting = Process(fn=lambda: 0, name="pre")
        pool._pipe.admit(preexisting)
        pool._pipe.close()
        try:
            with pytest.raises(PipeClosed):
                pool.branch([preexisting, Process(fn=lambda: 1, name="fresh")])
            # only the slot this call added went away; the pre-existing one stayed
            assert pool._pipe.owns(preexisting)
            assert pool._pipe.usage() == 1
            assert list(pool._pipe.queue._items.values()) == [preexisting]
        finally:
            pool._pipe.open()
            pool.shutdown()
