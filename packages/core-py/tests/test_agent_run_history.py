"""
Tests for the agent run history store (file-backed orchestration records).
"""

import pytest

from domains.agents.run_history import AgentRunStore, reset_agent_run_store


@pytest.fixture
def store(tmp_path):
    return AgentRunStore(directory=str(tmp_path / "agent_runs"), max_runs=5)


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_agent_run_store()
    yield
    reset_agent_run_store()


class TestStart:
    def test_start_returns_id_and_creates_record(self, store):
        run_id = store.start("Research AI agents", "context here")
        assert run_id.startswith("run_")
        record = store.get(run_id)
        assert record["goal"] == "Research AI agents"
        assert record["context"] == "context here"
        assert record["status"] == "running"
        assert record["finished_at"] is None
        assert record["completed_count"] == 0
        assert record["failed_count"] == 0
        assert len(record["logs"]) == 1

    def test_start_persists_to_disk(self, store, tmp_path):
        run_id = store.start("Goal")
        assert (tmp_path / "agent_runs" / f"{run_id}.json").exists()


class TestAppendLog:
    def test_appends_log_line(self, store):
        run_id = store.start("Goal")
        store.append_log(run_id, "Planning...")
        record = store.get(run_id)
        assert len(record["logs"]) == 2
        assert record["logs"][1].endswith("Planning...")

    def test_append_unknown_run_is_noop(self, store):
        store.append_log("run_missing", "nope")
        assert store.get("run_missing") is None


class TestSetTasks:
    def test_updates_task_list_and_counts(self, store):
        run_id = store.start("Goal")
        tasks = [
            {"id": "t1", "status": "completed", "description": "a"},
            {"id": "t2", "status": "failed", "description": "b"},
            {"id": "t3", "status": "pending", "description": "c"},
        ]
        store.set_tasks(run_id, tasks)
        record = store.get(run_id)
        assert len(record["tasks"]) == 3
        assert record["completed_count"] == 1
        assert record["failed_count"] == 1


class TestComplete:
    def test_marks_completed_with_response(self, store):
        run_id = store.start("Goal")
        store.set_tasks(run_id, [
            {"id": "t1", "status": "completed"},
            {"id": "t2", "status": "failed"},
        ])
        store.complete(run_id, response="final answer")
        record = store.get(run_id)
        assert record["status"] == "completed"
        assert record["response"] == "final answer"
        assert record["finished_at"] is not None
        assert record["completed_count"] == 1
        assert record["failed_count"] == 1

    def test_complete_with_tasks_replaces(self, store):
        run_id = store.start("Goal")
        store.set_tasks(run_id, [{"id": "a", "status": "pending"}])
        store.complete(run_id, response="done", tasks=[{"id": "a", "status": "completed"}])
        record = store.get(run_id)
        assert record["tasks"] == [{"id": "a", "status": "completed"}]
        assert record["completed_count"] == 1

    def test_complete_unknown_run_is_noop(self, store):
        store.complete("run_missing", response="x")
        assert store.get("run_missing") is None


class TestFail:
    def test_marks_failed_with_error(self, store):
        run_id = store.start("Goal")
        store.fail(run_id, "boom")
        record = store.get(run_id)
        assert record["status"] == "failed"
        assert record["error"] == "boom"
        assert record["finished_at"] is not None

    def test_fail_unknown_run_is_noop(self, store):
        store.fail("run_missing", "boom")
        assert store.get("run_missing") is None


class TestSetTasks:
    def test_unknown_run_is_noop(self, store):
        store.set_tasks("run_missing", [{"id": "t", "status": "pending"}])
        assert store.get("run_missing") is None


class TestListRuns:
    def test_returns_newest_first(self, store):
        first = store.start("First")
        second = store.start("Second")
        runs = store.list_runs()
        assert runs[0]["id"] == second
        assert runs[1]["id"] == first

    def test_respects_limit(self, store):
        for i in range(3):
            store.start(f"Goal {i}")
        runs = store.list_runs(limit=2)
        assert len(runs) == 2

    def test_empty_store(self, store):
        assert store.list_runs() == []

    def test_ignores_non_json_files(self, store, tmp_path):
        (tmp_path / "agent_runs" / "notes.txt").write_text("ignore me")
        store.start("Goal")
        runs = store.list_runs()
        assert len(runs) == 1
        assert all(r["status"] == "running" for r in runs)


class TestClearAndPrune:
    def test_clear_removes_all(self, store):
        store.start("A")
        store.start("B")
        assert store.clear() == 2
        assert store.list_runs() == []

    def test_prune_bounds_run_count(self, store):
        for i in range(8):
            store.start(f"Goal {i}")
        runs = store.list_runs()
        assert len(runs) <= 5

    def test_unsafe_run_id_rejected(self, store):
        assert store.get("../../etc/passwd") is None

    def test_clear_swallows_os_error(self, store, monkeypatch):
        store.start("A")

        def boom(path):
            raise OSError("permission denied")

        monkeypatch.setattr("os.remove", boom)
        assert store.clear() == 0

    def test_prune_swallows_os_error(self, store, monkeypatch):
        for i in range(8):
            store._save(f"run_test_{i}", {"id": f"run_test_{i}", "status": "running"})

        def boom(path):
            raise OSError("permission denied")

        monkeypatch.setattr("os.remove", boom)
        store._prune()
        assert len(store.list_runs()) == 8


class TestSingleton:
    def test_get_agent_run_store_returns_singleton(self):
        from domains.agents.run_history import get_agent_run_store
        a = get_agent_run_store()
        b = get_agent_run_store()
        assert a is b
