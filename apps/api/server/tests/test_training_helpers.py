"""Tests for training/helpers.py shared utilities."""

from __future__ import annotations

import asyncio
import time

from training.helpers import _run_async


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
