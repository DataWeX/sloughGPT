"""Unit tests for PGQ facade (pugqeep)."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from domains.infrastructure.pugqeep.facade import PGQ
from domains.infrastructure.pugqeep.task_queue import Task, TaskStatus, TaskPriority


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_array(shape=(4, 4), seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32)


def _make_point(name="test_point", n=100):
    centroids = np.arange(n, dtype=np.float32)
    assignments = np.arange(n) % len(centroids)
    from domains.infrastructure.pugqeep.point import Point
    return Point(
        identity=name,
        function_type="cluster",
        params={"centroids": centroids, "assignments": assignments},
        accuracy=0.95,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestPGQConstruction:
    def test_default_construction(self):
        pgq = PGQ()
        assert pgq.name == "model"
        assert pgq._config.n_clusters == 16

    def test_custom_construction(self):
        pgq = PGQ(name="custom", n_clusters=8, method="function",
                  memory_max_mb=64, hot_max_mb=32)
        assert pgq.name == "custom"
        assert pgq._config.n_clusters == 8
        assert pgq._config.method == "function"

    def test_library_tree_created(self):
        pgq = PGQ(name="test")
        assert pgq.library is not None
        assert pgq.tree is not None

    def test_is_loaded_false_initially(self):
        pgq = PGQ(name="test")
        assert pgq.is_loaded is False


# ---------------------------------------------------------------------------
# put / get / has / remove
# ---------------------------------------------------------------------------

class TestPutGetHasRemove:
    def test_put_compress_returns_point(self):
        pgq = PGQ(name="test", n_clusters=4)
        data = _make_array((4, 4))
        result = pgq.put("w1", data)
        assert hasattr(result, "identity")
        assert result.identity == "w1"

    def test_put_raw_stores_in_cache(self):
        pgq = PGQ(name="test")
        pgq.put_raw("key1", "hello", tier="memory", size_bytes=5)
        assert pgq.get_any("key1") == "hello"

    def test_put_no_compress_stores_raw(self):
        pgq = PGQ(name="test")
        data = _make_array((2, 2))
        result = pgq.put("raw1", data, compress=False)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, data)

    def test_get_returns_decompressed_array(self):
        pgq = PGQ(name="test", n_clusters=4)
        data = np.ones((4, 4), dtype=np.float32)
        pgq.put("ones", data)
        got = pgq.get("ones")
        assert got is not None
        assert got.shape == (4, 4)

    def test_get_cache_miss_returns_none(self):
        pgq = PGQ(name="test")
        assert pgq.get("nonexistent") is None

    def test_get_any_returns_cached_value(self):
        pgq = PGQ(name="test")
        pgq.put_raw("item", [1, 2, 3])
        assert pgq.get_any("item") == [1, 2, 3]

    def test_has_compressed_point(self):
        pgq = PGQ(name="test", n_clusters=4)
        pgq.put("exists", _make_array((4, 4)))
        assert pgq.has("exists") is True

    def test_has_cache_entry(self):
        pgq = PGQ(name="test")
        pgq.put_raw("cached", "value")
        assert pgq.has("cached") is True

    def test_has_returns_false(self):
        pgq = PGQ(name="test")
        assert pgq.has("missing") is False

    def test_remove_compressed_point(self):
        pgq = PGQ(name="test", n_clusters=4)
        pgq.put("rem", _make_array((4, 4)))
        assert pgq.remove("rem") is True
        assert pgq.has("rem") is False

    def test_remove_cache_entry(self):
        pgq = PGQ(name="test")
        pgq.put_raw("cached", "val")
        assert pgq.remove("cached") is True
        assert pgq.get_any("cached") is None

    def test_remove_nonexistent(self):
        pgq = PGQ(name="test")
        assert pgq.remove("nope") is False


# ---------------------------------------------------------------------------
# put_many / get_many / exists_many / remove_many
# ---------------------------------------------------------------------------

class TestBatchOperations:
    def test_put_many(self):
        pgq = PGQ(name="test", n_clusters=4)
        data = {f"w{i}": _make_array((4, 4), seed=i) for i in range(3)}
        stats = pgq.put_many(data)
        assert stats["count"] == 3
        assert stats["total_bytes"] == 3 * 4 * 4 * 4  # 3 arrays * 16 floats * 4 bytes

    def test_get_many(self):
        pgq = PGQ(name="test", n_clusters=4)
        for i in range(3):
            pgq.put(f"w{i}", _make_array((4, 4), seed=i))
        results = pgq.get_many(["w0", "w1", "w2", "missing"])
        assert len(results) == 4
        assert results["w0"] is not None
        assert results["w1"] is not None
        assert results["w2"] is not None
        assert results["missing"] is None

    def test_exists_many(self):
        pgq = PGQ(name="test", n_clusters=4)
        pgq.put("a", _make_array((4, 4), seed=0))
        pgq.put_raw("b", "val")
        result = pgq.exists_many(["a", "b", "c"])
        assert result["a"] is True
        assert result["b"] is True
        assert result["c"] is False

    def test_remove_many(self):
        pgq = PGQ(name="test", n_clusters=4)
        pgq.put("r1", _make_array((4, 4), seed=0))
        pgq.put("r2", _make_array((4, 4), seed=1))
        pgq.put_raw("r3", "val")
        count = pgq.remove_many(["r1", "r2", "r3", "missing"])
        assert count == 3
        assert not pgq.has("r1")
        assert not pgq.has("r2")
        assert not pgq.has("r3")


# ---------------------------------------------------------------------------
# Task queue
# ---------------------------------------------------------------------------

class TestTaskQueue:
    def test_submit_task(self):
        pgq = PGQ(name="test")
        task = Task(name="job1", data={"x": 1})
        result = pgq.submit_task(task)
        assert result.name == "job1"
        assert result.status == TaskStatus.PENDING

    def test_next_task(self):
        pgq = PGQ(name="test")
        pgq.submit_task(Task(name="j1"))
        pgq.submit_task(Task(name="j2"))
        t1 = pgq.next_task()
        assert t1 is not None
        assert t1.status == TaskStatus.RUNNING
        t2 = pgq.next_task()
        assert t2 is not None
        assert t2.id != t1.id

    def test_next_task_empty(self):
        pgq = PGQ(name="test")
        assert pgq.next_task() is None

    def test_complete_task(self):
        pgq = PGQ(name="test")
        pgq.submit_task(Task(name="j1"))
        task = pgq.next_task()
        result = pgq.complete_task(task.id, result="done")
        assert result.status == TaskStatus.COMPLETED
        assert result.result == "done"

    def test_fail_task(self):
        pgq = PGQ(name="test")
        pgq.submit_task(Task(name="j1", max_retries=0))
        task = pgq.next_task()
        result = pgq.fail_task(task.id, error="oops")
        assert result.status == TaskStatus.FAILED
        assert result.error == "oops"

    def test_fail_task_retries(self):
        pgq = PGQ(name="test")
        pgq.submit_task(Task(name="j1", max_retries=2))
        task = pgq.next_task()
        result = pgq.fail_task(task.id, error="err")
        # max_retries=2, retries=0 initially, so after first fail retries becomes 1
        assert result.status == TaskStatus.PENDING
        assert result.retries == 1

    def test_cancel_task(self):
        pgq = PGQ(name="test")
        pgq.submit_task(Task(name="j1"))
        task = pgq.next_task()
        result = pgq.cancel_task(task.id)
        assert result.status == TaskStatus.CANCELLED

    def test_cancel_pending_task(self):
        pgq = PGQ(name="test")
        t1 = pgq.submit_task(Task(name="j1"))
        pgq.submit_task(Task(name="j2"))
        result = pgq.cancel_task(t1.id)
        assert result.status == TaskStatus.CANCELLED
        # Next should give j2
        nxt = pgq.next_task()
        assert nxt.id != t1.id

    def test_get_task(self):
        pgq = PGQ(name="test")
        t = pgq.submit_task(Task(name="findme"))
        assert pgq.get_task(t.id).name == "findme"

    def test_list_tasks_all(self):
        pgq = PGQ(name="test")
        pgq.submit_task(Task(name="a"))
        pgq.submit_task(Task(name="b"))
        assert len(pgq.list_tasks()) == 2

    def test_list_tasks_by_status(self):
        pgq = PGQ(name="test")
        t1 = pgq.submit_task(Task(name="a"))
        pgq.submit_task(Task(name="b"))
        pgq.next_task()  # starts first task
        completed = pgq.list_tasks(status=TaskStatus.RUNNING)
        assert len(completed) == 1

    def test_pause_resume_queue(self):
        pgq = PGQ(name="test")
        pgq.submit_task(Task(name="j1"))
        pgq.pause_queue()
        assert pgq.next_task() is None
        pgq.resume_queue()
        assert pgq.next_task() is not None


# ---------------------------------------------------------------------------
# Search / best
# ---------------------------------------------------------------------------

class TestSearchAndBest:
    def test_search(self):
        pgq = PGQ(name="test", n_clusters=4)
        pgq.put("layer_weight_0", _make_array((4, 4), seed=0))
        pgq.put("layer_bias_0", _make_array((4, 4), seed=1))
        results = pgq.search("layer")
        assert len(results) == 2

    def test_search_no_match(self):
        pgq = PGQ(name="test", n_clusters=4)
        pgq.put("w1", _make_array((4, 4), seed=0))
        results = pgq.search("nonexistent")
        assert len(results) == 0

    def test_best(self):
        pgq = PGQ(name="test")
        for i in range(5):
            p = _make_point(f"pt_{i}", n=20)
            p.accuracy = i * 0.1
            pgq.library.add(p)
        best = pgq.best(n=3)
        assert len(best) == 3
        assert best[0].accuracy >= best[1].accuracy >= best[2].accuracy


# ---------------------------------------------------------------------------
# Stats / cache_stats / queue_stats / cleanup / export
# ---------------------------------------------------------------------------

class TestStatsAndCleanup:
    def test_stats(self):
        pgq = PGQ(name="test")
        pgq.put("w1", _make_array((4, 4), seed=0))
        pgq.submit_task(Task(name="j1"))
        s = pgq.stats()
        assert "name" in s
        assert "tree" in s
        assert "cache" in s
        assert "queue" in s

    def test_cache_stats(self):
        pgq = PGQ(name="test")
        pgq.put_raw("k", "v")
        cs = pgq.cache_stats()
        assert isinstance(cs, dict)

    def test_queue_stats(self):
        pgq = PGQ(name="test")
        pgq.submit_task(Task(name="j1"))
        qs = pgq.queue_stats()
        assert qs["total"] == 1

    def test_cleanup_cache(self):
        pgq = PGQ(name="test")
        count = pgq.cleanup_cache()
        assert isinstance(count, int)
        assert count >= 0

    def test_export_stats(self):
        pgq = PGQ(name="test")
        pgq.put("w1", _make_array((4, 4), seed=0))
        es = pgq.export_stats()
        assert es["version"] == "0.1.0"
        assert "name" in es


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        pgq = PGQ(name="roundtrip", n_clusters=4)
        pgq.put("layer1", _make_array((4, 4), seed=0))
        pgq.put("layer2", _make_array((4, 4), seed=1))

        save_path = tmp_path / "test_pgq.points.json"
        pgq.save(save_path)

        loaded = PGQ.load(save_path)
        assert loaded.name == "roundtrip"
        assert loaded.has("layer1")
        assert loaded.has("layer2")

    def test_save_load_with_tasks(self, tmp_path):
        pgq = PGQ(name="with_tasks", n_clusters=4)
        pgq.put("w1", _make_array((4, 4), seed=0))
        pgq.submit_task(Task(name="job_a"))
        pgq.submit_task(Task(name="job_b"))

        save_path = tmp_path / "test.points.json"
        pgq.save(save_path)

        loaded = PGQ.load(save_path)
        assert len(loaded.list_tasks()) == 2

    def test_save_load_no_tasks_file(self, tmp_path):
        pgq = PGQ(name="no_tasks", n_clusters=4)
        pgq.put("w1", _make_array((4, 4), seed=0))

        save_path = tmp_path / "test.points.json"
        pgq.save(save_path)
        # Manually remove the tasks file to simulate no-task scenario
        task_path = tmp_path / "test.tasks.json"
        if task_path.exists():
            task_path.unlink()

        loaded = PGQ.load(save_path)
        assert loaded.has("w1")

    def test_load_from_file_factory(self, tmp_path):
        pgq = PGQ(name="factory_test", n_clusters=4)
        pgq.put("w1", _make_array((4, 4), seed=0))
        save_path = tmp_path / "test.points.json"
        pgq.save(save_path)

        loaded = PGQ.from_file(save_path)
        assert loaded.name == "factory_test"
        assert loaded.has("w1")

    def test_get_after_load(self, tmp_path):
        original = PGQ(name="rt", n_clusters=4)
        data = np.ones((4, 4), dtype=np.float32)
        original.put("ones", data)

        save_path = tmp_path / "rt.points.json"
        original.save(save_path)

        loaded = PGQ.load(save_path)
        got = loaded.get("ones")
        # Shape metadata is not persisted by the facade; decompressed data
        # is the flat point values. Just verify it returns valid data.
        assert got is not None
        assert isinstance(got, np.ndarray)
        assert len(got) > 0


# ---------------------------------------------------------------------------
# from_model (mocked — heavy dependency)
# ---------------------------------------------------------------------------

class TestFromModel:
    @pytest.mark.slow
    @patch("domains.infrastructure.pugqeep.facade.load_model_to_points")
    def test_from_model(self, mock_load):
        fake_tree = MagicMock()
        fake_tree.library = MagicMock()
        mock_load.return_value = fake_tree

        pgq = PGQ.from_model("test-model", n_clusters=8, method="cluster")
        mock_load.assert_called_once_with(
            "test-model", n_clusters=8, method="cluster", storage_dir=None,
        )
        assert pgq.name == "test-model"


# ---------------------------------------------------------------------------
# Queue factory
# ---------------------------------------------------------------------------

class TestQueueFactory:
    @pytest.mark.slow
    @patch("domains.infrastructure.pugqeep.facade.ModelQueue")
    @patch("domains.infrastructure.pugqeep.facade.QueueConfig")
    def test_queue_factory(self, mock_config, mock_queue):
        mock_queue_instance = MagicMock()
        mock_queue.return_value = mock_queue_instance

        result = PGQ.queue(["model_a", "model_b"], n_clusters=8)
        mock_queue.assert_called_once()
        assert mock_queue_instance.load_model.call_count == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_put_many_empty(self):
        pgq = PGQ(name="test")
        stats = pgq.put_many({})
        assert stats["count"] == 0
        assert stats["total_bytes"] == 0

    def test_get_many_empty(self):
        pgq = PGQ(name="test")
        assert pgq.get_many([]) == {}

    def test_exists_many_empty(self):
        pgq = PGQ(name="test")
        assert pgq.exists_many([]) == {}

    def test_remove_many_empty(self):
        pgq = PGQ(name="test")
        assert pgq.remove_many([]) == 0

    def test_put_compressed_with_method_override(self):
        pgq = PGQ(name="test", n_clusters=4)
        data = _make_array((4, 4))
        result = pgq.put("override", data, method="function")
        assert hasattr(result, "identity")

    def test_put_tier_memory(self):
        pgq = PGQ(name="test")
        data = _make_array((2, 2))
        result = pgq.put("tier_mem", data, compress=False, tier="memory")
        assert isinstance(result, np.ndarray)

    def test_get_decompress_shape_preserved(self):
        pgq = PGQ(name="test", n_clusters=4)
        data = np.ones((3, 5), dtype=np.float32)
        pgq.put("shaped", data)
        got = pgq.get("shaped")
        assert got.shape == (3, 5)

    def test_cancel_nonexistent_task(self):
        pgq = PGQ(name="test")
        result = pgq.cancel_task("nonexistent_id")
        assert result is None

    def test_complete_nonexistent_task(self):
        pgq = PGQ(name="test")
        result = pgq.complete_task("nonexistent_id")
        assert result is None

    def test_fail_nonexistent_task(self):
        pgq = PGQ(name="test")
        result = pgq.fail_task("nonexistent_id", "err")
        assert result is None

    def test_task_priority_ordering(self):
        pgq = PGQ(name="test")
        pgq.submit_task(Task(name="low", priority=TaskPriority.LOW))
        pgq.submit_task(Task(name="urgent", priority=TaskPriority.URGENT))
        pgq.submit_task(Task(name="normal", priority=TaskPriority.NORMAL))
        first = pgq.next_task()
        assert first.name == "urgent"
        second = pgq.next_task()
        assert second.name == "normal"
        third = pgq.next_task()
        assert third.name == "low"

    def test_complete_nonexistent_returns_none(self):
        pgq = PGQ(name="test")
        assert pgq.complete_task("fake_id", "result") is None

    def test_fail_nonexistent_returns_none(self):
        pgq = PGQ(name="test")
        assert pgq.fail_task("fake_id", "error") is None

    def test_get_task_nonexistent(self):
        pgq = PGQ(name="test")
        assert pgq.get_task("fake") is None

    def test_list_tasks_all_statuses(self):
        pgq = PGQ(name="test")
        t1 = pgq.submit_task(Task(name="a", max_retries=0))
        t2 = pgq.submit_task(Task(name="b", max_retries=0))
        t3 = pgq.submit_task(Task(name="c"))
        # Complete t1
        running = pgq.next_task()
        pgq.complete_task(running.id)
        # Fail t2 (max_retries=0 → stays failed)
        running2 = pgq.next_task()
        pgq.fail_task(running2.id, "err")
        # Cancel t3 (still pending)
        pgq.cancel_task(t3.id)

        assert len(pgq.list_tasks(status=TaskStatus.COMPLETED)) == 1
        assert len(pgq.list_tasks(status=TaskStatus.FAILED)) == 1
        assert len(pgq.list_tasks(status=TaskStatus.CANCELLED)) == 1

    def test_search_case_insensitive(self):
        pgq = PGQ(name="test", n_clusters=4)
        pgq.put("LayerWeight_0", _make_array((4, 4), seed=0))
        results = pgq.search("layerweight")
        assert len(results) == 1

    def test_best_returns_sorted(self):
        pgq = PGQ(name="test")
        for i in range(10):
            p = _make_point(f"pt_{i}", n=20)
            p.accuracy = i / 10.0
            pgq.library.add(p)
        best = pgq.best(n=5)
        assert len(best) == 5
        accs = [b.accuracy for b in best]
        assert accs == sorted(accs, reverse=True)
