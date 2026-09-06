"""Tests for domains.agents.run_history — AgentRunStore MogDB persistence.

Covers: start, append_log, set_tasks, complete, fail, get, list_runs, clear,
pruning, singleton, safe_id validation. Uses temp directory for MogDB.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.agents.run_history import (
    AgentRunStore, _new_run_id, set_mogdb_path, reset_mogdb, reset_agent_run_store,
)


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "agent_runs")
    set_mogdb_path(db_path)
    reset_agent_run_store()
    yield AgentRunStore(db_path=db_path, max_runs=5)
    reset_agent_run_store()
    reset_mogdb()


class TestNewRunId:
    def test_format(self):
        rid = _new_run_id()
        parts = rid.split("_")
        assert parts[0] == "run"
        assert len(parts) >= 4

    def test_unique(self):
        ids = {_new_run_id() for _ in range(50)}
        assert len(ids) == 50


class TestAgentRunStore:
    def test_start(self, store):
        run_id = store.start(goal="Test goal")
        assert run_id.startswith("run_")
        record = store.get(run_id)
        assert record["goal"] == "Test goal"
        assert record["status"] == "running"

    def test_start_with_context(self, store):
        run_id = store.start(goal="G", context="ctx")
        record = store.get(run_id)
        assert record["context"] == "ctx"

    def test_append_log(self, store):
        run_id = store.start(goal="G")
        store.append_log(run_id, "Step 1")
        record = store.get(run_id)
        assert any("Step 1" in log for log in record["logs"])

    def test_append_log_nonexistent(self, store):
        store.append_log("nonexistent", "msg")

    def test_set_tasks(self, store):
        run_id = store.start(goal="G")
        tasks = [{"name": "t1", "status": "completed"}, {"name": "t2", "status": "pending"}]
        store.set_tasks(run_id, tasks)
        record = store.get(run_id)
        assert record["completed_count"] == 1
        assert record["failed_count"] == 0

    def test_complete(self, store):
        run_id = store.start(goal="G")
        store.complete(run_id, response="done", tasks=[{"status": "completed"}])
        record = store.get(run_id)
        assert record["status"] == "completed"
        assert record["response"] == "done"
        assert record["finished_at"] is not None

    def test_fail(self, store):
        run_id = store.start(goal="G")
        store.fail(run_id, error="boom")
        record = store.get(run_id)
        assert record["status"] == "failed"
        assert record["error"] == "boom"

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent") is None

    def test_get_unsafe_id(self, store):
        assert store.get("../../etc/passwd") is None

    def test_list_runs(self, store):
        store.start(goal="G1")
        store.start(goal="G2")
        runs = store.list_runs()
        assert len(runs) == 2

    def test_list_runs_limit(self, store):
        for i in range(10):
            store.start(goal=f"G{i}")
        runs = store.list_runs(limit=3)
        assert len(runs) == 3

    def test_clear(self, store):
        store.start(goal="G1")
        store.start(goal="G2")
        removed = store.clear()
        assert removed == 2
        assert store.list_runs() == []

    def test_pruning(self, tmp_path):
        db_path = str(tmp_path / "prune_test")
        set_mogdb_path(db_path)
        s = AgentRunStore(db_path=db_path, max_runs=3)
        ids = [s.start(goal=f"G{i}") for i in range(5)]
        runs = s.list_runs()
        assert len(runs) <= 3
        reset_agent_run_store()

    def test_safe_id(self, store):
        assert store._safe_id("run_123") is True
        assert store._safe_id("../../etc") is False
        assert store._safe_id("") is False

    def test_list_runs_newest_first(self, store):
        rid1 = store.start(goal="first")
        rid2 = store.start(goal="second")
        rid3 = store.start(goal="third")
        runs = store.list_runs()
        assert len(runs) == 3
        assert runs[0]["id"] == rid3
        assert runs[2]["id"] == rid1

    def test_start_record_fields(self, store):
        rid = store.start(goal="test goal", context="ctx")
        record = store.get(rid)
        assert record["goal"] == "test goal"
        assert record["context"] == "ctx"
        assert record["status"] == "running"
        assert record["response"] == ""
        assert record["error"] == ""
        assert len(record["logs"]) == 1

    def test_append_log_noop_missing(self, store):
        store.append_log("nonexistent_run", "msg")

    def test_complete_with_tasks(self, store):
        rid = store.start(goal="test")
        store.complete(rid, response="done", tasks=[
            {"id": "t1", "status": "completed"},
        ])
        record = store.get(rid)
        assert record["status"] == "completed"
        assert record["response"] == "done"
        assert record["finished_at"] is not None
        assert record["completed_count"] == 1

    def test_fail_records_error(self, store):
        rid = store.start(goal="test")
        store.fail(rid, error="crashed")
        record = store.get(rid)
        assert record["status"] == "failed"
        assert record["error"] == "crashed"
        assert record["finished_at"] is not None

    def test_list_runs_empty(self, store):
        runs = store.list_runs()
        assert runs == []

    def test_clear_removes_all(self, store):
        store.start(goal="a")
        store.start(goal="b")
        store.start(goal="c")
        removed = store.clear()
        assert removed == 3
        assert store.list_runs() == []

    def test_prune_on_start(self, tmp_path):
        db_path = str(tmp_path / "prune_start_test")
        set_mogdb_path(db_path)
        s = AgentRunStore(db_path=db_path, max_runs=3)
        ids = []
        for i in range(5):
            ids.append(s.start(goal=f"run {i}"))
        runs = s.list_runs()
        assert len(runs) == 3
        assert s.get(ids[0]) is None
        assert s.get(ids[1]) is None
        reset_agent_run_store()
