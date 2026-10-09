"""SyncRunner tests — persistent loop, sync→async adapter."""

from __future__ import annotations

import asyncio

from avion.sync import SyncRunner, get_default_runner


class TestSyncRunner:
    def test_run_returns_value(self):
        async def add(a: int, b: int) -> int:
            return a + b

        with SyncRunner() as runner:
            assert runner.run(add(2, 3)) == 5

    def test_loop_persists_across_runs(self):
        async def which() -> asyncio.AbstractEventLoop:
            return asyncio.get_running_loop()

        with SyncRunner() as runner:
            first = runner.run(which())
            second = runner.run(which())
            assert first is second
            assert first is runner.loop

    def test_interleaved_coroutines_share_state(self):
        """State created in one run is visible in the next (same loop)."""
        store: dict[str, str] = {}

        async def set_key() -> None:
            store["key"] = "value"
            await asyncio.sleep(0)

        async def get_key() -> str | None:
            await asyncio.sleep(0)
            return store.get("key")

        with SyncRunner() as runner:
            runner.run(set_key())
            assert runner.run(get_key()) == "value"

    def test_close_is_idempotent(self):
        runner = SyncRunner()
        runner.run(asyncio.sleep(0))
        runner.close()
        runner.close()  # second close must not raise

    def test_context_manager_closes_loop(self):
        with SyncRunner() as runner:
            loop = runner.loop
            runner.run(asyncio.sleep(0))
        assert loop.is_closed()

    def test_close_cancels_pending_tasks(self):
        async def forever() -> None:
            await asyncio.sleep(3600)

        runner = SyncRunner()
        runner.loop.create_task(forever())
        runner.close()  # must not hang on the 1h sleep
        assert runner._loop is None


class TestDefaultRunner:
    def test_default_runner_is_reused(self):
        from avion.sync import run

        first = get_default_runner()
        assert run(asyncio.sleep(0)) is None
        assert get_default_runner() is first

    def test_default_runner_not_closed_by_sync_runner(self):
        with SyncRunner() as runner:
            runner.run(asyncio.sleep(0))
        assert not get_default_runner().loop.is_closed()
