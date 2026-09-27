"""Tests for training/helpers.py shared utilities."""

from __future__ import annotations

import asyncio
import time

from training.helpers import _finish_job, _run_async
from training.jobs import training_jobs


class TestFinishJob:
    def test_mutates_in_place_so_runtime_reference_sees_terminal_status(self):
        """TrainingRuntime keeps a reference to the job dict and syncs IT to
        the JobStore. Replacing the dict here left the runtime syncing a stale
        'running' record — completed runs were persisted as running, flipped to
        'interrupted' on restart, and wrongly offered for resume."""
        job_id = "_test_finish_inplace_"
        training_jobs[job_id] = {"id": job_id, "status": "running", "progress": 99}
        runtime_ref = training_jobs[job_id]
        try:
            _finish_job(job_id, "completed")
            assert training_jobs[job_id]["status"] == "completed"
            assert training_jobs[job_id] is runtime_ref
            assert runtime_ref["status"] == "completed"
            assert runtime_ref["progress"] == 99
        finally:
            training_jobs.pop(job_id, None)

    def test_records_error_without_clobbering_existing_fields(self):
        job_id = "_test_finish_error_"
        training_jobs[job_id] = {"id": job_id, "status": "running", "loss": 0.5}
        try:
            _finish_job(job_id, "failed", "boom")
            assert training_jobs[job_id]["status"] == "failed"
            assert training_jobs[job_id]["error"] == "boom"
            assert training_jobs[job_id]["loss"] == 0.5
        finally:
            training_jobs.pop(job_id, None)

    def test_missing_job_is_a_noop(self):
        _finish_job("_test_finish_missing_", "completed")


class TestRunAsync:
    def test_returns_without_blocking(self):
        """A slow coroutine must not hold the caller (webhook fan-out)."""
        started = time.monotonic()

        async def slow():
            await asyncio.sleep(30)

        _run_async(slow())
        assert time.monotonic() - started < 5

    def test_coroutine_still_runs(self):
        """Fire-and-forget still executes the coroutine."""
        done = []

        async def quick():
            done.append(True)

        _run_async(quick())
        deadline = time.monotonic() + 10
        while not done and time.monotonic() < deadline:
            time.sleep(0.05)
        assert done == [True]

    def test_coro_exception_swallowed(self):
        """Failures never propagate to the caller."""

        async def boom():
            raise RuntimeError("nope")

        _run_async(boom())  # must not raise
