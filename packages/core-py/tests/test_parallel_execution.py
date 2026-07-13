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

        InferencePool._instance = None
        pool = await InferencePool.get_instance()
        import multiprocessing
        expected = min(multiprocessing.cpu_count(), 8)
        assert pool._max_workers == expected
        await pool.shutdown()
        InferencePool._instance = None

    @pytest.mark.asyncio
    async def test_env_override(self):
        from apps.api.server.infrastructure.inference_pool import InferencePool

        InferencePool._instance = None
        with patch.dict("os.environ", {"MAN_INFERENCE_POOL_SIZE": "3"}):
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
