"""
Tests for the SQLite-backed JobStore crash-recovery semantics.

Covers the heartbeat format contract (heartbeats are persisted via
``datetime.now().isoformat()`` with a 'T' separator — stale detection must
compare in the SAME format, not an SQLite ``datetime(?, 'unixepoch')`` string,
which silently never fires), stale 'recovering' detection, and the recoverable
list (interrupted + failed + stale recovering, never an actively-recovered row).
"""

from datetime import datetime, timedelta

from training.job_store import JobStore


def _mk_store(tmp_path):
    return JobStore(str(tmp_path / "training_jobs.db"))


def _create(store, job_id, status="pending", heartbeat=None, crashed=0):
    store.create(job_id, f"job {job_id}", {"model": "sloughgpt"}, "shakespeare")
    kwargs = {"status": status}
    if heartbeat is not None:
        kwargs["last_heartbeat"] = heartbeat
    if crashed:
        kwargs["crashed"] = crashed
    if kwargs:
        store.update(job_id, **kwargs)
    return store.get(job_id)


def _old_heartbeat(seconds=600):
    return (datetime.now() - timedelta(seconds=seconds)).isoformat()


# ── detect_crashed_jobs (heartbeat-format regression) ────────────────────────


def test_detect_crashed_jobs_stale_running(tmp_path):
    store = _mk_store(tmp_path)
    _create(store, "a", status="running", heartbeat=_old_heartbeat())
    _create(store, "b", status="running", heartbeat=datetime.now().isoformat())

    crashed = store.detect_crashed_jobs(timeout_seconds=300)
    ids = [j["id"] for j in crashed]

    assert ids == ["a"]


def test_detect_crashed_jobs_no_rows(tmp_path):
    store = _mk_store(tmp_path)
    _create(store, "a", status="running", heartbeat=datetime.now().isoformat())

    assert store.detect_crashed_jobs(timeout_seconds=300) == []


def test_detect_crashed_jobs_stale_recovering(tmp_path):
    store = _mk_store(tmp_path)
    _create(store, "a", status="recovering", heartbeat=_old_heartbeat())
    _create(store, "b", status="recovering", heartbeat=datetime.now().isoformat())

    crashed = store.detect_crashed_jobs(timeout_seconds=300)
    assert [j["id"] for j in crashed] == ["a"]


def test_detect_crashed_jobs_ignores_non_running_statuses(tmp_path):
    store = _mk_store(tmp_path)
    _create(store, "a", status="interrupted", heartbeat=_old_heartbeat())
    _create(store, "b", status="failed", heartbeat=_old_heartbeat())
    _create(store, "c", status="completed", heartbeat=_old_heartbeat())

    assert store.detect_crashed_jobs(timeout_seconds=300) == []


# ── mark_recovering ──────────────────────────────────────────────────────────


def test_mark_recovering_sets_fresh_heartbeat_and_clears_crashed(tmp_path):
    store = _mk_store(tmp_path)
    _create(store, "a", status="interrupted", heartbeat=_old_heartbeat(), crashed=1)

    store.mark_recovering("a")
    row = store.get("a")

    assert row["status"] == "recovering"
    assert row["crashed"] == 0
    assert not JobStore.is_stale_heartbeat(row)
    assert store.get_recoverable_jobs() == []


# ── is_stale_heartbeat ───────────────────────────────────────────────────────


def test_is_stale_heartbeat(tmp_path):
    store = _mk_store(tmp_path)
    _create(store, "a", status="running", heartbeat=datetime.now().isoformat())
    _create(store, "b", status="running", heartbeat=_old_heartbeat())
    _create(store, "c", status="running")
    store.update("c", last_heartbeat=None)  # NULL heartbeat

    assert not JobStore.is_stale_heartbeat(store.get("a"))
    assert JobStore.is_stale_heartbeat(store.get("b"))
    assert JobStore.is_stale_heartbeat(store.get("c"))
    assert JobStore.is_stale_heartbeat({})  # missing field


def test_is_stale_heartbeat_garbage_treated_stale(tmp_path):
    store = _mk_store(tmp_path)
    _create(store, "a", status="running", heartbeat="not-a-timestamp")

    assert JobStore.is_stale_heartbeat(store.get("a"))


# ── get_recoverable_jobs ─────────────────────────────────────────────────────


def test_get_recoverable_jobs_includes_interrupted_and_failed(tmp_path):
    store = _mk_store(tmp_path)
    _create(store, "interrupted", status="interrupted")
    _create(store, "failed", status="failed")
    _create(store, "completed", status="completed")
    _create(store, "running", status="running", heartbeat=datetime.now().isoformat())

    ids = [j["id"] for j in store.get_recoverable_jobs()]
    assert sorted(ids) == ["failed", "interrupted"]


def test_get_recoverable_jobs_includes_stale_recovering(tmp_path):
    store = _mk_store(tmp_path)
    _create(store, "stale", status="recovering", heartbeat=_old_heartbeat())

    ids = [j["id"] for j in store.get_recoverable_jobs()]
    assert ids == ["stale"]


def test_get_recoverable_jobs_excludes_active_recovering(tmp_path):
    store = _mk_store(tmp_path)
    _create(store, "a", status="interrupted", heartbeat=_old_heartbeat())
    store.mark_recovering("a")

    assert [j["id"] for j in store.get_recoverable_jobs()] == []
