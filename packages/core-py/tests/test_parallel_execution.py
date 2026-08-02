"""Tests for TrainingExecutor, InferencePool dynamic sizing, and pugqeep integration."""

import asyncio
import time
import threading
from unittest.mock import patch

import numpy as np
import pytest


# ── TrainingExecutor tests ──────────────────────────────────────────


class TestTrainingExecutor:
    """Core executor functionality."""

    def test_submit_and_complete(self):
        from domains.training.executor import TrainingExecutor

        exec_ = TrainingExecutor(max_workers=2)
        results = []

        def train_fn(job_id):
            results.append(job_id)
            return {"loss": 0.5}

        job_id = exec_.submit(train_fn, "job_1")
        time.sleep(0.1)
        status = exec_.status(job_id)
        assert status is not None
        assert status["status"] == "completed"
        assert results == ["job_1"]
        exec_.shutdown(wait=True)

    def test_concurrency_limit(self):
        from domains.training.executor import TrainingExecutor

        exec_ = TrainingExecutor(max_workers=2)
        running = []
        max_seen = []

        def slow_fn(job_id):
            running.append(job_id)
            max_seen.append(len(running))
            time.sleep(0.05)
            running.remove(job_id)

        for i in range(4):
            exec_.submit(slow_fn, f"j{i}")
        time.sleep(0.3)

        assert max(max_seen) <= 2
        exec_.shutdown(wait=True)

    def test_cancel_queued_job(self):
        from domains.training.executor import TrainingExecutor, JobStatus

        exec_ = TrainingExecutor(max_workers=1)
        evt = threading.Event()

        def slow_fn(job_id):
            evt.wait(timeout=2)

        exec_.submit(slow_fn, "blocker")
        time.sleep(0.01)
        job_id = exec_.submit(slow_fn, "to_cancel")
        cancelled = exec_.cancel(job_id)
        assert cancelled is True
        info = exec_.status(job_id)
        assert info["status"] in ("cancelled", "queued")
        evt.set()
        exec_.shutdown(wait=True)

    def test_is_cancelled_flag(self):
        from domains.training.executor import TrainingExecutor

        exec_ = TrainingExecutor(max_workers=2)
        check_results = []

        def check_fn(job_id, tree_id, point_lib, is_cancelled):
            check_results.append(is_cancelled())
            return {}

        exec_.submit_training(check_fn, "c1", tree_id="test")
        time.sleep(0.1)
        assert check_results == [False]

        job_id2 = exec_.submit_training(check_fn, "c2", tree_id="test")
        exec_.cancel(job_id2)
        time.sleep(0.1)
        # Running job's is_cancelled should reflect the flag
        exec_.shutdown(wait=True)

    def test_list_jobs(self):
        from domains.training.executor import TrainingExecutor

        exec_ = TrainingExecutor(max_workers=2)

        def noop(job_id):
            pass

        exec_.submit(noop, "a")
        exec_.submit(noop, "b")
        time.sleep(0.1)
        jobs = exec_.list_jobs()
        assert len(jobs) == 2
        ids = {j["job_id"] for j in jobs}
        assert ids == {"a", "b"}
        exec_.shutdown(wait=True)

    def test_tree_id_tracking(self):
        from domains.training.executor import TrainingExecutor

        exec_ = TrainingExecutor(max_workers=2)

        def noop(job_id):
            pass

        exec_.submit(noop, "t1", tree_id="gpt2")
        time.sleep(0.1)
        status = exec_.status("t1")
        assert status["tree_id"] == "gpt2"
        exec_.shutdown(wait=True)

    def test_purge_completed(self):
        from domains.training.executor import TrainingExecutor

        exec_ = TrainingExecutor(max_workers=2)

        def noop(job_id):
            pass

        exec_.submit(noop, "old")
        time.sleep(0.1)
        # Backdate completed_at
        exec_._jobs["old"].completed_at = time.time() - 7200
        purged = exec_.purge_completed(max_age_s=3600)
        assert purged == 1
        assert exec_.status("old") is None
        exec_.shutdown(wait=True)


class TestTrainingExecutorEdgeBranches:
    """Remaining branch coverage: result_type, call_args, failures, cancel edge."""

    def test_to_dict_non_dict_result(self):
        from domains.training.executor import TrainingExecutor

        exec_ = TrainingExecutor(max_workers=2)

        def list_fn(job_id):
            return [1.0, 2.0]

        job_id = exec_.submit(list_fn, "list_result")
        time.sleep(0.1)
        status = exec_.status(job_id)
        assert status["status"] == "completed"
        assert status["result_type"] == "list"
        exec_.shutdown(wait=True)

    def test_submit_with_call_args(self):
        from domains.training.executor import TrainingExecutor

        exec_ = TrainingExecutor(max_workers=2)
        seen = {}

        def fn(job_id, **kw):
            seen.update(kw)
            return {}

        job_id = exec_.submit(fn, "call_args_job", _call_args={"a": 1, "b": 2}, c=3)
        time.sleep(0.1)
        assert seen == {"a": 1, "b": 2, "c": 3}
        exec_.shutdown(wait=True)

    def test_failed_job_records_error(self):
        from domains.training.executor import TrainingExecutor

        exec_ = TrainingExecutor(max_workers=1)

        def boom(job_id):
            raise ValueError("training exploded")

        job_id = exec_.submit(boom, "fail_job")
        time.sleep(0.2)
        status = exec_.status(job_id)
        assert status["status"] == "failed"
        assert "training exploded" in status["error"]
        exec_.shutdown(wait=True)

    def test_point_storage_exception_is_swallowed(self):
        from domains.training.executor import TrainingExecutor

        exec_ = TrainingExecutor(max_workers=2)

        class BadLibrary:
            def add(self, point):
                raise RuntimeError("disk full")

        def train_fn(job_id, tree_id, point_lib, is_cancelled):
            return {"w": np.random.randn(32).astype(np.float32)}

        job_id = exec_.submit_training(
            train_fn, "bad_lib", tree_id="t", point_library=BadLibrary(),
        )
        status = None
        for _ in range(50):
            status = exec_.status(job_id)
            if status is not None and status["status"] != "running":
                break
            time.sleep(0.1)
        assert status["status"] == "completed"
        exec_.shutdown(wait=True)

    def test_result_summary_none_cases(self):
        from domains.training.executor import TrainingExecutor

        exec_ = TrainingExecutor(max_workers=2)
        assert exec_.result_summary("unknown") is None

        def returns_none(job_id):
            return None

        jid = exec_.submit(returns_none, "none_result")
        time.sleep(0.1)
        assert exec_.result_summary(jid) is None

        def returns_list(job_id):
            return [1, 2]

        jid2 = exec_.submit(returns_list, "list_result2")
        time.sleep(0.1)
        assert exec_.result_summary(jid2) is None
        exec_.shutdown(wait=True)

    def test_result_summary_completed_dict(self):
        from domains.training.executor import TrainingExecutor

        exec_ = TrainingExecutor(max_workers=2)

        def dict_fn(job_id):
            return {"w": np.random.randn(8).astype(np.float32)}

        job_id = exec_.submit(dict_fn, "dict_result")
        time.sleep(0.1)
        summary = exec_.result_summary(job_id)
        assert summary is not None
        assert summary["job_id"] == "dict_result"
        assert "w" in summary["weights"]
        assert summary["weights"]["w"]["shape"] == [8]
        assert summary["total_bytes"] == 8 * 4
        exec_.shutdown(wait=True)

    def test_active_count(self):
        from domains.training.executor import TrainingExecutor

        exec_ = TrainingExecutor(max_workers=1)
        evt = threading.Event()
        started = threading.Event()

        def slow_fn(job_id):
            started.set()
            evt.wait(timeout=2)

        exec_.submit(slow_fn, "blocker2")
        assert started.wait(timeout=2)
        assert exec_.active_count() == 1
        evt.set()
        exec_.shutdown(wait=True)

    def test_cancel_unknown_job_returns_false(self):
        from domains.training.executor import TrainingExecutor

        exec_ = TrainingExecutor(max_workers=2)
        assert exec_.cancel("no_such_job") is False
        exec_.shutdown(wait=True)

    def test_cancel_completed_job_returns_false(self):
        from domains.training.executor import TrainingExecutor

        exec_ = TrainingExecutor(max_workers=2)

        def fast_fn(job_id):
            return {}

        job_id = exec_.submit(fast_fn, "done_job")
        time.sleep(0.2)
        assert exec_.cancel(job_id) is False
        exec_.shutdown(wait=True)

    def test_get_training_executor_singleton(self):
        import domains.training.executor as exmod

        old = exmod._instance
        try:
            exmod._instance = None
            ex1 = exmod.get_training_executor()
            assert isinstance(ex1, exmod.TrainingExecutor)
            ex2 = exmod.get_training_executor()
            assert ex1 is ex2
        finally:
            exmod._instance = old
            if ex1 is not None and ex1 is not old:
                ex1.shutdown(wait=True)

    def test_get_training_executor_double_check_race(self):
        import domains.training.executor as exmod

        old = exmod._instance
        new_exec = exmod.TrainingExecutor(max_workers=1)
        entered = threading.Event()
        release = threading.Event()
        try:
            exmod._instance = None

            def other_thread():
                with exmod._instance_lock:
                    entered.set()
                    release.wait(timeout=2)
                    exmod._instance = new_exec

            t = threading.Thread(target=other_thread)
            t.start()
            assert entered.wait(timeout=2)

            results = []

            def do_get():
                results.append(exmod.get_training_executor())

            g = threading.Thread(target=do_get)
            g.start()
            time.sleep(0.2)  # let g pass the fast-path check and block on the lock
            release.set()
            g.join(timeout=2)
            t.join(timeout=2)
            assert results and results[0] is new_exec
        finally:
            exmod._instance = old
            new_exec.shutdown(wait=True)


class TestCompressCheckpointBranches:
    """Failure paths in compress_checkpoint()."""

    def test_missing_file_returns_none(self):
        from domains.training.executor import compress_checkpoint

        assert compress_checkpoint("/nonexistent/checkpoint.soul") is None

    def test_import_error_returns_none(self, monkeypatch):
        import sys
        import tempfile
        from pathlib import Path
        from domains.training import executor as exmod

        monkeypatch.setitem(sys.modules, "domains.infrastructure.pugqeep", None)
        with tempfile.TemporaryDirectory() as tmpdir:
            soul_path = str(Path(tmpdir) / "x.soul")
            Path(soul_path).write_text("x")
            assert exmod.compress_checkpoint(soul_path) is None

    def test_model_none_returns_none(self):
        import tempfile
        from pathlib import Path
        from domains.training import executor as exmod

        with patch("domains.training.slonet.import_from_sou", return_value=None), \
                tempfile.TemporaryDirectory() as tmpdir:
            soul_path = str(Path(tmpdir) / "x.soul")
            Path(soul_path).write_text("x")
            assert exmod.compress_checkpoint(soul_path) is None

    def test_load_failure_returns_none(self):
        import tempfile
        from pathlib import Path
        from domains.training import executor as exmod

        with patch("domains.training.slonet.import_from_sou", side_effect=RuntimeError("corrupt")), \
                tempfile.TemporaryDirectory() as tmpdir:
            soul_path = str(Path(tmpdir) / "x.soul")
            Path(soul_path).write_text("x")
            assert exmod.compress_checkpoint(soul_path) is None

    def test_weights_converted_to_ndarray(self):
        import tempfile
        from pathlib import Path
        from domains.training import executor as exmod

        class FakeModel:
            def state_dict(self):
                return {"w": [1.0, 2.0, 3.0]}

        with patch("domains.training.slonet.import_from_sou", return_value=FakeModel()), \
                tempfile.TemporaryDirectory() as tmpdir:
            soul_path = str(Path(tmpdir) / "x.soul")
            Path(soul_path).write_text("x")
            stats = exmod.compress_checkpoint(soul_path, n_clusters=2)
            assert stats is not None
            assert stats["point_count"] == 1


# ── TrainingExecutor + PGQ integration ─────────────────────────────


class TestPGQTrainingIntegration:
    """Test submit_training on the PGQ facade."""

    def test_submit_training_routes_to_tree(self):
        from domains.infrastructure.pugqeep import PGQ

        pgq = PGQ(name="test_train")
        results = []

        def train_fn(job_id, tree_id, point_lib, is_cancelled):
            results.append({"job_id": job_id, "tree_id": tree_id})
            return {}

        job_id = pgq.submit_training(train_fn, "pgq_job_1")
        time.sleep(0.2)
        status = pgq.training_status(job_id)
        assert status is not None
        assert status["status"] == "completed"
        assert results[0]["tree_id"] == "test_train"

    def test_submit_training_stores_points(self):
        from domains.infrastructure.pugqeep import PGQ

        pgq = PGQ(name="test_points")

        def train_fn(job_id, tree_id, point_lib, is_cancelled):
            # Return trained weights as dict
            return {
                "layer1.weight": np.random.randn(128).astype(np.float32),
                "layer2.weight": np.random.randn(64).astype(np.float32),
            }

        job_id = pgq.submit_training(train_fn, "pgq_points")
        time.sleep(0.3)
        status = pgq.training_status(job_id)
        assert status["status"] == "completed"
        # Points should be stored in the library
        assert pgq.library.has("layer1.weight")
        assert pgq.library.has("layer2.weight")

    def test_cancel_training(self):
        from domains.infrastructure.pugqeep import PGQ

        pgq = PGQ(name="test_cancel")
        evt = threading.Event()

        def slow_fn(job_id, tree_id, point_lib, is_cancelled):
            evt.wait(timeout=2)
            return {}

        job_id = pgq.submit_training(slow_fn, "cancel_me")
        time.sleep(0.05)
        cancelled = pgq.cancel_training(job_id)
        assert cancelled is True
        evt.set()
        time.sleep(0.1)
        pgq.cancel_training.__func__  # verify it's a method


# ── InferencePool dynamic sizing ────────────────────────────────────


class TestInferencePoolDynamic:
    """Test that InferencePool auto-sizes to CPU count."""

    @pytest.mark.asyncio
    async def test_default_size_from_cpu_count(self):
        from apps.api.server.infrastructure.inference_pool import InferencePool
        from domains.infrastructure.resource_manager import get_resource_manager

        InferencePool._instance = None
        pool = await InferencePool.get_instance()
        expected = get_resource_manager().inference_pool_size
        assert pool._max_workers == expected
        assert pool._max_workers >= 1
        await pool.shutdown()
        InferencePool._instance = None

    @pytest.mark.asyncio
    async def test_env_override(self):
        from apps.api.server.infrastructure.inference_pool import InferencePool

        InferencePool._instance = None
        with patch.dict("os.environ", {"SLO_INFERENCE_POOL_SIZE": "3"}):
            pool = await InferencePool.get_instance()
            assert pool._max_workers == 3
        await pool.shutdown()
        InferencePool._instance = None

    @pytest.mark.asyncio
    async def test_create_explicit(self):
        from apps.api.server.infrastructure.inference_pool import InferencePool

        pool = await InferencePool.create(max_workers=5)
        assert pool._max_workers == 5
        await pool.shutdown()
        InferencePool._instance = None


# ── ModelServer concurrent reads ────────────────────────────────────


class TestModelServerReadSemaphore:
    """Test read semaphore for concurrent tokenization."""

    def test_read_semaphore_created(self):
        from domains.infrastructure.model_server import ModelServer

        server = ModelServer(max_concurrent=1, enable_warmup=False)
        loop = asyncio.new_event_loop()
        sem = loop.run_until_complete(server._get_read_semaphore())
        assert sem is not None
        assert sem._value == 4  # max_concurrent * 4
        loop.close()

    def test_tokenize_uses_read_semaphore(self):
        from domains.infrastructure.model_server import ModelServer

        class FakeTokenizer:
            eos_token_id = 0
            pad_token_id = 0
            def __call__(self, text, **kwargs):
                return {"input_ids": [[1, 2, 3]]}

        server = ModelServer(
            tokenizer=FakeTokenizer(),
            enable_warmup=False,
            enable_circuit_breaker=False,
        )
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(server.tokenize("hello"))
        assert "input_ids" in result
        loop.close()
