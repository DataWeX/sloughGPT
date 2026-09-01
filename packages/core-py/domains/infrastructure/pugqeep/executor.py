"""
ParallelExecutor — shared threading logic for batch compress/decompress.

Single place for ProducerConsumerQueue coordination.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Dict, List, Tuple


class ParallelExecutor:
    """Executes a function over items in parallel via ProducerConsumerQueue.

    Args:
        num_workers: Thread count. -1 = cpu_count.
        queue_factor: Queue size = num_workers * queue_factor.
        timeout: Max seconds to wait for queue drain.
    """

    __slots__ = ('_num_workers', '_queue_factor', '_timeout')

    def __init__(self, num_workers: int = -1, queue_factor: int = 4,
                 timeout: float = 30.0):
        if num_workers < 0:
            num_workers = os.cpu_count() or 4
        self._num_workers = num_workers
        self._queue_factor = queue_factor
        self._timeout = timeout

    @property
    def num_workers(self) -> int:
        return self._num_workers

    def run(self, items: List[Tuple[str, Any]], handler: Callable[[Any], None],
            name: str = "parallel") -> None:
        """Run handler on each item in parallel.

        Args:
            items: List of items to process.
            handler: Callable that processes one item.
            name: Queue name for debugging.
        """
        from domains.infrastructure.producer_consumer import ProducerConsumerQueue

        q = ProducerConsumerQueue(
            maxsize=self._num_workers * self._queue_factor,
            num_consumers=self._num_workers,
            handler=handler,
            name=name,
        )
        q.start()
        try:
            for item in items:
                q.put(item)
            while not q.empty:
                time.sleep(0.01)
        finally:
            q.stop(timeout=self._timeout)

    def map(self, items: List[Tuple[str, Any]],
            fn: Callable[[Any], Any],
            name: str = "parallel") -> Dict[str, Any]:
        """Map fn over items in parallel, return {key: result}.

        Args:
            items: List of (key, value) pairs.
            fn: Transform function applied to each value.
            name: Queue name for debugging.

        Returns:
            Dict mapping keys to results.
        """
        results: Dict[str, Any] = {}
        lock = threading.Lock()

        def process(item):
            key, value = item
            result = fn(value)
            with lock:
                results[key] = result

        self.run(items, process, name)
        return results
