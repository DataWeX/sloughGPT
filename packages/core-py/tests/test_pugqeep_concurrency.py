"""Races on Engine's shared bookkeeping: _processes, _pending, _completed.

Each test pins a defect that was reachable from ordinary use, not a
hypothetical one:

* dispatch() rebuilt _pending from a snapshot taken at entry, so a spawn()
  landing while it classified was erased from the queue. The process stayed
  registered in _processes, nothing ever dispatched it, and wait_for() ran out
  its timeout.
* Iterating _processes/_pools live while another thread spawned raised
  RuntimeError: dictionary changed size during iteration.
* get_completed() copied then cleared _completed, dropping anything appended
  between the two statements.
"""

import threading
import time

from domain.infrastructure._internal.pugqeep.engine import Engine


def _noop():
    return "ok"


def _sleep(secs):
    time.sleep(secs)
    return secs


# ════════════════════════════════════════════════════════════════
# dispatch() must merge, never replace
# ════════════════════════════════════════════════════════════════


def test_dispatch_preserves_spawns_arriving_mid_classification():
    """A spawn() between dispatch()'s snapshot and its write-back must survive."""
    engine = Engine("race")
    engine.pool("t")

    anchor = engine.spawn(_noop)  # no deps -> dispatchable
    held = engine.spawn(_noop, depends_on=[anchor.id])  # dep unmet -> held

    injected = []
    real_deps_met = engine._deps_met

    def racing_deps_met(proc):
        if not injected:
            # Exactly the window: dispatch() has already snapshotted _pending,
            # and this spawn() appends to the live list before the write-back.
            injected.append(engine.spawn(_noop))
        return real_deps_met(proc)

    engine._deps_met = racing_deps_met

    try:
        engine.dispatch()
        pending = {p.id for p in engine._pend()}
        assert held.id in pending, "held work disappeared from _pending"
        assert injected[0].id in pending, "spawn() during dispatch was dropped"
        assert anchor.id not in pending, "dispatchable work was never dispatched"
    finally:
        engine._deps_met = real_deps_met
        engine.stop()


def test_dispatch_hammer_loses_no_process():
    """Spawning from several threads while dispatch() runs never orphans work.

    A process is orphaned when it is neither queued, nor dispatched (pool
    chosen), nor finished -- it exists, reports as pending, and will never run.
    """
    engine = Engine("race")
    engine.pool("t")

    anchor = engine.spawn(_noop)
    threads = []
    errors = []

    def spawner(count):
        try:
            for i in range(count):
                if i % 2:
                    # stays queued: dependency never completes
                    engine.spawn(_noop, depends_on=[anchor.id])
                else:
                    engine.spawn(_noop)
        except Exception as e:  # pragma: no cover - the failure under test
            errors.append(e)

    for _ in range(4):
        t = threading.Thread(target=spawner, args=(50,))
        t.start()
        threads.append(t)

    # Bounded on purpose: a spawner parked on a full pipe has to surface as a
    # failure, not leave the suite wedged.
    deadline = time.time() + 30
    while any(t.is_alive() for t in threads) and time.time() < deadline:
        engine.dispatch()
    for t in threads:
        t.join(timeout=30)
    stuck = [t for t in threads if t.is_alive()]

    try:
        assert not stuck, f"{len(stuck)} spawner(s) wedged -- pipe not draining"
        queued = {p.id for p in engine._pend()}
        orphaned = [
            p
            for p in engine._procs()
            if p.id not in queued and not p.is_done and p._pool_name is None
        ]
        assert not errors, errors
        assert not orphaned, f"orphaned processes: {[p.name for p in orphaned]}"
        assert len(engine._procs()) == 201  # anchor + 4 * 50
    finally:
        engine.stop()


# ════════════════════════════════════════════════════════════════
# iterating while spawning
# ════════════════════════════════════════════════════════════════


def test_concurrent_spawn_and_iterators_never_raise():
    """list_processes/health/summary run while other threads spawn."""
    engine = Engine("race")
    engine.pool("t", max_stems=8, pool_workers=4)
    # Capacity is 128 and admission blocks: without something draining, a
    # 300-process spawn test measures the pipe, not the iterators.
    engine.run_background(poll_interval=0.001)

    errors = []
    done_reading = threading.Event()

    def spawner():
        try:
            for _ in range(100):
                engine.spawn(_noop)
        except Exception as e:  # pragma: no cover - the failure under test
            errors.append(e)
        finally:
            done_reading.set()

    def reader():
        try:
            while not done_reading.is_set():
                engine.list_processes()
                engine.health()
                engine.summary()
                engine.wait_all(timeout=0)
        except Exception as e:  # pragma: no cover - the failure under test
            errors.append(e)

    spawners = [threading.Thread(target=spawner) for _ in range(3)]
    readers = [threading.Thread(target=reader) for _ in range(3)]
    for t in spawners + readers:
        t.start()
    for t in spawners:
        t.join(timeout=30)
    for t in readers:
        t.join(timeout=10)
    stuck = [t for t in spawners + readers if t.is_alive()]

    try:
        assert not stuck, f"{len(stuck)} thread(s) wedged"
        assert not errors, errors
        assert len(engine.list_processes()) == 300
    finally:
        engine.stop()


# ════════════════════════════════════════════════════════════════
# Pool capacity
# ════════════════════════════════════════════════════════════════


def test_concurrent_branch_never_exceeds_max_stems():
    """branch() must hold max_stems even though it releases _lock to admit.

    The lock is deliberately dropped around the blocking admit(), so the
    capacity check and the registration cannot share a critical section.
    Without a reservation two threads pass the check together, both register,
    and a Pool built for 4 stems ends up running more -- the one guarantee
    this package is supposed to make.

    The barrier matters: spawned separately, each caller spends its first
    microseconds in engine.spawn() and arrives after earlier callers have
    already registered, so the check passes cleanly and the race never
    shows. Releasing them together lands every caller in branch() before any
    one of them has reached the registration.
    """
    engine = Engine("race")
    pool = engine.pool("t", max_stems=4, pool_workers=2)
    max_stems = pool.max_stems
    attempts = 12

    # Spawn first, outside the race: only branch() is under test.
    procs = [engine.spawn(_sleep, 30.0) for _ in range(attempts)]
    gate = threading.Barrier(attempts, timeout=15)
    tallied = {"ok": 0, "full": 0}
    tally = threading.Lock()
    failures: list[BaseException] = []

    def attempt(p):
        # 30s sleep: no stem retires mid-test, so capacity freed by _execute
        # finishing cannot let extra branches legitimately succeed and mask
        # the race.
        try:
            gate.wait()
            pool.branch([p])
            with tally:
                tallied["ok"] += 1
        except RuntimeError:
            with tally:
                tallied["full"] += 1
        except Exception as e:  # pragma: no cover - barrier/infra trouble
            failures.append(e)

    threads = [threading.Thread(target=attempt, args=(p,)) for p in procs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    stuck = [t for t in threads if t.is_alive()]

    try:
        assert not failures, failures
        assert not stuck, f"{len(stuck)} branch() call(s) wedged"
        # Behavioral checks first: they name the actual violation, and they
        # hold for any implementation. An unfixed Pool reports "5 stems
        # admitted on a max_stems=4 pool" here rather than tripping over the
        # bookkeeping field below.
        assert tallied["ok"] <= max_stems, (
            f"{tallied['ok']} stems admitted on a max_stems={max_stems} pool"
        )
        assert len(pool._stems) <= max_stems, (
            f"{len(pool._stems)} live stems on a max_stems={max_stems} pool"
        )
        # Then the accounting the fix itself relies on.
        assert len(pool._stems) + pool._reserved <= max_stems, (
            f"{len(pool._stems)} stems + {pool._reserved} reserved exceeds max_stems={max_stems}"
        )
        assert pool._reserved == 0, "a reservation was never given back"
    finally:
        engine.stop()


# ════════════════════════════════════════════════════════════════
# _completed drain
# ════════════════════════════════════════════════════════════════


def test_get_completed_drain_loses_nothing():
    """Draining completed work while run() records more of it loses none."""
    engine = Engine("race")
    engine.pool("t", max_stems=8, pool_workers=4)
    total = 60
    procs = [engine.spawn(_sleep, 0.005, name=f"w{i}") for i in range(total)]
    engine.run_background(poll_interval=0.001)

    drained = []
    deadline = time.time() + 30
    try:
        while len(drained) < total and time.time() < deadline:
            drained.extend(engine.get_completed())
        engine.wait(timeout=30)
        # a final drain for whatever run() recorded after the loop
        drained.extend(engine.get_completed())

        ids = [p.id for p in drained]
        assert len(ids) == len(set(ids)), "a process was handed back twice"
        assert len(set(ids)) == total, f"drained {len(set(ids))} of {total}"
    finally:
        engine.stop()
    assert all(p.is_done for p in procs)
