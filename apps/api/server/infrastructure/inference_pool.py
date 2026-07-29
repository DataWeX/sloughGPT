"""
Dedicated thread pool for CPU-bound inference work.

Eliminates per-request thread creation by maintaining a fixed-size pool
of worker threads. Async routes submit CPU-bound work (tokenization,
model generation, decoding) to this pool via ``loop.run_in_executor``.

Usage::

    pool = InferencePool.get_instance()
    result = await pool.run(model.generate, input_ids)
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

logger = logging.getLogger("slo.infrastructure.inference_pool")


class InferencePool:
    """Fixed-size thread pool for CPU-bound inference operations.

    Provides semaphore-based backpressure (max concurrent = pool size)
    and per-task timeouts.
    """

    _instance: Optional[InferencePool] = None
    _lock = asyncio.Lock()

    def __init__(self, max_workers: int = 4, queue_timeout: float = 30.0):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="inference",
        )
        self._semaphore = asyncio.Semaphore(max_workers)
        self._queue_timeout = queue_timeout
        self._max_workers = max_workers
        logger.info("InferencePool created (workers=%d)", max_workers, extra={"tag": "INFRA"})

    @classmethod
    async def get_instance(cls) -> InferencePool:
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    from domains.infrastructure.resource_manager import get_resource_manager
                    rm = get_resource_manager()
                    import os
                    max_workers = int(os.environ.get(
                        "SLO_INFERENCE_POOL_SIZE",
                        rm.inference_pool_size,
                    ))
                    cls._instance = cls(max_workers=max_workers)
        return cls._instance

    @classmethod
    async def create(cls, max_workers: int) -> InferencePool:
        pool = cls(max_workers=max_workers)
        cls._instance = pool
        return pool

    async def run(
        self,
        fn: Callable[..., Any],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """Submit a synchronous function to the thread pool.

        Args:
            fn: Synchronous callable (e.g. model.generate, tokenizer.encode).
            *args: Positional arguments for fn.
            timeout: Per-call timeout in seconds. Falls back to default.
            **kwargs: Keyword arguments for fn.

        Returns:
            The return value of fn.

        Raises:
            TimeoutError: If execution exceeds timeout.
        """
        acquired = await self._semaphore.acquire()
        if not acquired:
            raise RuntimeError("InferencePool: semaphore acquire returned False")
        try:
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs)),
                timeout=timeout or self._queue_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("InferencePool task timed out (fn=%s)", getattr(fn, "__name__", str(fn)), extra={"tag": "INFRA"})
            raise TimeoutError("Inference task timed out")
        finally:
            self._semaphore.release()

    async def run_generator(
        self,
        fn: Callable[..., Any],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ):
        """Run a synchronous generator in the thread pool, yielding results.

        For streaming generation — runs the generator function in a thread
        and yields items as they're produced. Semaphore held for the
        generator's lifetime.
        """
        import queue
        from threading import Thread

        result_queue: queue.Queue = queue.Queue(maxsize=64)
        error_container: list[BaseException] = []

        acquired = await self._semaphore.acquire()
        if not acquired:
            raise RuntimeError("InferencePool: semaphore acquire returned False")

        def _run():
            try:
                gen = fn(*args, **kwargs)
                for item in gen:
                    result_queue.put(item)
            except BaseException as e:
                error_container.append(e)
            finally:
                result_queue.put(_SENTINEL)

        thread = Thread(target=_run, daemon=True)
        thread.start()

        deadline = time.monotonic() + (timeout or self._queue_timeout)

        try:
            while thread.is_alive() or not result_queue.empty():
                if error_container:
                    raise error_container[0]
                if time.monotonic() > deadline:
                    raise TimeoutError("Inference generator timed out")
                try:
                    item = result_queue.get(timeout=0.02)
                except queue.Empty:
                    await asyncio.sleep(0)
                    continue
                if item is _SENTINEL:
                    break
                yield item
        finally:
            thread.join(timeout=5)
            self._semaphore.release()

    async def shutdown(self):
        self._executor.shutdown(wait=True, cancel_futures=False)
        logger.info("InferencePool shut down (workers=%d)", self._max_workers, extra={"tag": "INFRA"})


_SENTINEL = object()
