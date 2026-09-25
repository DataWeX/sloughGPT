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


def _create(store, job_id, status="pending", heartbeat=None, crashed=0, data_path=None):
    store.create(job_id, f"job {job_id}", {"model": "sloughgpt"}, "shakespeare")
    kwargs = {"status": status}
    if data_path is not None:
        kwargs["data_path"] = data_path
    if heartbeat is not None:
        kwargs["last_heartbeat"] = heartbeat
    if crashed:
        kwargs["crashed"] = crashed
    if kwargs:
        store.update(job_id, **kwargs)
    return store.get(job_id)


def _corpus(tmp_path):
    """A dataset the recovery endpoint would accept (real file on disk)."""
    data_file = tmp_path / "corpus.txt"
    data_file.write_text("hello world\n" * 10, encoding="utf-8")
    return str(data_file)


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
    data_path = _corpus(tmp_path)
    _create(store, "interrupted", status="interrupted", data_path=data_path)
    _create(store, "failed", status="failed", data_path=data_path)
    _create(store, "completed", status="completed", data_path=data_path)
    _create(store, "running", status="running", heartbeat=datetime.now().isoformat())

    ids = [j["id"] for j in store.get_recoverable_jobs()]
    assert sorted(ids) == ["failed", "interrupted"]


def test_get_recoverable_jobs_includes_stale_recovering(tmp_path):
    store = _mk_store(tmp_path)
    _create(
        store,
        "stale",
        status="recovering",
        heartbeat=_old_heartbeat(),
        data_path=_corpus(tmp_path),
    )

    ids = [j["id"] for j in store.get_recoverable_jobs()]
    assert ids == ["stale"]


def test_get_recoverable_jobs_hides_jobs_the_endpoint_would_reject(tmp_path):
    # POST /recovery/recover/{id} 422s these (no dataset recorded / a trainer
    # this path cannot restart) — offering them as "Recoverable" is a dead
    # Resume button.
    store = _mk_store(tmp_path)
    data_path = _corpus(tmp_path)
    _create(store, "no_dataset", status="interrupted")
    _create(store, "distill", status="failed", data_path=data_path)
    store.update("distill", config={"type": "distill"})
    _create(store, "ok", status="interrupted", data_path=data_path)

    assert [j["id"] for j in store.get_recoverable_jobs()] == ["ok"]


def test_get_recoverable_jobs_excludes_active_recovering(tmp_path):
    store = _mk_store(tmp_path)
    _create(store, "a", status="interrupted", heartbeat=_old_heartbeat())
    store.mark_recovering("a")

    assert [j["id"] for j in store.get_recoverable_jobs()] == []


# ── discard_live (list auto-purge must keep durable history) ─────────────────


def test_discard_live_drops_memory_only_keeps_store_row(tmp_path):
    from training.job_store import PersistentTrainingJobs

    store = _mk_store(tmp_path)
    _create(store, "keep", status="completed")

    jobs = PersistentTrainingJobs()
    jobs._live["keep"] = {"id": "keep", "status": "completed"}
    jobs.discard_live("keep")

    assert "keep" not in jobs._live
    assert store.get("keep") is not None
    assert store.get("keep")["status"] == "completed"


# ── store_row_to_job (API shape aliases for durable-only rows) ───────────────


def test_store_row_to_job_maps_checkpoint_epochs_model(tmp_path):
    store = _mk_store(tmp_path)
    store.create(
        "hist",
        "old run",
        {"model": "gpt2", "method": "distill"},
        "shakespeare",
    )
    store.update(
        "hist",
        status="interrupted",
        checkpoint_path="models/auto-training/hist/checkpoint.soul",
        total_epochs=5,
        progress=42,
    )
    # Simulate auto-purge: only the durable row remains.
    from training.job_store import PersistentTrainingJobs

    jobs = PersistentTrainingJobs()
    # Point the wrapper at this tmp store.
    jobs._store = lambda: store  # type: ignore[method-assign]

    rows = jobs.values()
    assert len(rows) == 1
    row = rows[0]
    assert row["checkpoint"] == "models/auto-training/hist/checkpoint.soul"
    assert row["epochs"] == 5
    assert row["model"] == "gpt2"
    assert row["method"] == "distill"
    assert row["status"] == "interrupted"
    # store.get also returns the API shape (used by recover + single GET).
    got = store.get("hist")
    assert got["checkpoint"] == "models/auto-training/hist/checkpoint.soul"
    assert got["epochs"] == 5
    assert got["model"] == "gpt2"


def test_store_row_to_job_prefers_existing_checkpoint_field(tmp_path):
    store = _mk_store(tmp_path)
    store.create("a", "a", {"model": "gpt2"}, "ds")
    store.update("a", checkpoint="already-set.soul", checkpoint_path="other.soul")
    row = store.get("a")
    assert row["checkpoint"] == "already-set.soul"
    assert row["checkpoint_path"] == "other.soul"


# ── created_at stamping (the history UI's sort key) ─────────────────────────
# Most job-creation paths assign training_jobs[id] = {...} without created_at;
# the registry stamps it at that single choke point so every job is sortable.


def test_registry_setitem_injects_created_at(tmp_path):
    from training.job_store import PersistentTrainingJobs

    store = _mk_store(tmp_path)
    jobs = PersistentTrainingJobs()
    jobs._store = lambda: store  # type: ignore[method-assign]

    job = {"id": "n1", "name": "new", "status": "pending"}
    jobs["n1"] = job

    assert job["created_at"]
    assert store.get("n1")["created_at"] == job["created_at"]


def test_registry_setitem_preserves_existing_created_at(tmp_path):
    from training.job_store import PersistentTrainingJobs

    store = _mk_store(tmp_path)
    jobs = PersistentTrainingJobs()
    jobs._store = lambda: store  # type: ignore[method-assign]

    job = {"id": "n2", "name": "rerun", "status": "running", "created_at": "2026-01-01T00:00:00"}
    jobs["n2"] = job

    assert job["created_at"] == "2026-01-01T00:00:00"
    assert store.get("n2")["created_at"] == "2026-01-01T00:00:00"


# ── _job_summary created_at fallback (frontend crash regression) ────────────
# _job_summary strips None fields, so a job with no timestamps omits created_at
# entirely; the summary must fall back to started_at/updated_at, and the
# frontend renders '—' when the key is still absent.


def test_job_summary_created_at_falls_back_to_started_at():
    from training.jobs_api import _job_summary

    summary = _job_summary(
        {"id": "1", "status": "failed", "error": "OOM", "started_at": "2026-01-02T03:04:05"}
    )
    assert summary["created_at"] == "2026-01-02T03:04:05"


def test_job_summary_created_at_falls_back_to_updated_at():
    from training.jobs_api import _job_summary

    summary = _job_summary({"id": "1", "status": "completed", "updated_at": "2026-02-03T04:05:06"})
    assert summary["created_at"] == "2026-02-03T04:05:06"


def test_job_summary_omits_created_at_when_no_timestamps():
    from training.jobs_api import _job_summary

    summary = _job_summary({"id": "3"})
    assert "created_at" not in summary
