"""InferencePool.run_generator — loop-responsiveness regression tests.

Card 20261029_031406: run_generator's cleanup did thread.join(timeout=5)
inline inside the async generator's finally — up to 5s of event-loop stall
per stream teardown when the worker thread is still busy (wedged inference,
early consumer break). The loop must keep serving other coroutines while
teardown waits for the worker.
"""

from __future__ import annotations

import asyncio
import threading
import time

from infrastructure.inference_pool import InferencePool


class TestRunGeneratorLoopResponsiveness:
    async def test_teardown_join_does_not_block_event_loop(self):
        pool = InferencePool(max_workers=2)
        worker_release = threading.Event()

        def slow_worker():
            # Worker still busy when the consumer stops early: teardown must
            # join it (bounded), but must NOT stall the loop while waiting.
            yield 1
            worker_release.wait(timeout=3)
            yield 2

        try:
            stream = pool.run_generator(slow_worker)
            first = await anext(stream)
            assert first == 1

            t0 = time.monotonic()
            close_task = asyncio.create_task(stream.aclose())
            # While teardown waits on the worker, a concurrent coroutine on
            # THIS loop must still be serviced promptly.
            await asyncio.sleep(0.1)
            loop_stayed_free_for = time.monotonic() - t0
            worker_release.set()
            await close_task

            # RED: the finally does thread.join(timeout=5) inline, so the
            # sleep above only runs after the worker finishes (~3s). GREEN:
            # the join runs off-loop (asyncio.to_thread) and the sleep
            # completes in ~0.1s while teardown is still waiting.
            assert loop_stayed_free_for < 1.0
        finally:
            worker_release.set()
            await pool.shutdown()
