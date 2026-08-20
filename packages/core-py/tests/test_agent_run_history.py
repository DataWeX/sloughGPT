"""Tests for domains.agents.run_history — AgentRunStore file-backed persistence."""

import json
import os
import tempfile

import pytest
from domains.agents.run_history import AgentRunStore, _new_run_id, reset_agent_run_store


@pytest.fixture
def store():
    """Create a temporary AgentRunStore for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = AgentRunStore(directory=tmpdir, max_runs=10)
        yield s


class TestNewRunId:
    def test_format(self):
        rid = _new_run_id()
        parts = rid.split("_")
        assert parts[0] == "run"
        assert len(parts) >= 4  # run + date + time + pid + counter

    def test_unique(self):
        ids = {_new_run_id() for _ in range(50)}
        assert len(ids) == 50


class TestSafeId:
    def test_valid(self):
        s = AgentRunStore()
        assert s._safe_id("run_20260101_120000_1_000001") is True

    def test_empty(self):
        s = AgentRunStore()
        assert s._safe_id("") is False

    def test_path_traversal(self):
        s = AgentRunStore()
        assert s._safe_id("../../../etc/passwd") is False


class TestAgentRunStoreLifecycle:
    def test_start_returns_id(self, store):
        rid = store.start(goal="test goal")
        assert rid is not None
        assert rid.startswith("run_")

    def test_start_creates_file(self, store):
        rid = store.start(goal="test goal")
        assert os.path.exists(store._path(rid))

    def test_start_record_fields(self, store):
        rid = store.start(goal="test goal", context="ctx")
        record = store.get(rid)
        assert record["goal"] == "test goal"
        assert record["context"] == "ctx"
        assert record["status"] == "running"
        assert record["response"] == ""
        assert record["error"] == ""
        assert len(record["logs"]) == 1

    def test_append_log(self, store):
        rid = store.start(goal="test")
        store.append_log(rid, "Step 1")
        store.append_log(rid, "Step 2")
        record = store.get(rid)
        assert len(record["logs"]) == 3  # start + 2 appends

    def test_append_log_noop_missing(self, store):
        store.append_log("nonexistent_run", "msg")  # should not raise

    def test_set_tasks(self, store):
        rid = store.start(goal="test")
        tasks = [
            {"id": "t1", "status": "completed"},
            {"id": "t2", "status": "failed"},
            {"id": "t3", "status": "running"},
        ]
        store.set_tasks(rid, tasks)
        record = store.get(rid)
        assert len(record["tasks"]) == 3
        assert record["completed_count"] == 1
        assert record["failed_count"] == 1

    def test_complete(self, store):
        rid = store.start(goal="test")
        store.complete(rid, response="done", tasks=[
            {"id": "t1", "status": "completed"},
        ])
        record = store.get(rid)
        assert record["status"] == "completed"
        assert record["response"] == "done"
        assert record["finished_at"] is not None
        assert record["completed_count"] == 1

    def test_fail(self, store):
        rid = store.start(goal="test")
        store.fail(rid, error="crashed")
        record = store.get(rid)
        assert record["status"] == "failed"
        assert record["error"] == "crashed"
        assert record["finished_at"] is not None


class TestAgentRunStoreQueries:
    def test_get_existing(self, store):
        rid = store.start(goal="test")
        record = store.get(rid)
        assert record is not None
        assert record["id"] == rid

    def test_get_missing(self, store):
        assert store.get("nonexistent") is None

    def test_get_bad_id(self, store):
        assert store.get("../../../etc/passwd") is None

    def test_list_runs_newest_first(self, store):
        rid1 = store.start(goal="first")
        rid2 = store.start(goal="second")
        rid3 = store.start(goal="third")
        runs = store.list_runs()
        assert len(runs) == 3
        assert runs[0]["id"] == rid3  # newest first
        assert runs[2]["id"] == rid1

    def test_list_runs_limit(self, store):
        for i in range(5):
            store.start(goal=f"run {i}")
        runs = store.list_runs(limit=3)
        assert len(runs) == 3

    def test_list_runs_empty_dir(self, store):
        runs = store.list_runs()
        assert runs == []


class TestAgentRunStoreClear:
    def test_clear_removes_all(self, store):
        store.start(goal="a")
        store.start(goal="b")
        store.start(goal="c")
        removed = store.clear()
        assert removed == 3
        assert store.list_runs() == []


class TestAgentRunStorePrune:
    def test_prune_on_start(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = AgentRunStore(directory=tmpdir, max_runs=3)
            ids = []
            for i in range(5):
                ids.append(s.start(goal=f"run {i}"))
            runs = s.list_runs()
            assert len(runs) == 3
            # oldest runs pruned
            assert s.get(ids[0]) is None
            assert s.get(ids[1]) is None
