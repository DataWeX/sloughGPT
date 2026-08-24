from __future__ import annotations

import logging
import time
import threading
from typing import Iterator

from .filters import Filter, FilterChain
from .sources import Record, Source
from .stores import Store

logger = logging.getLogger(__name__)


class Collector:
    def __init__(self, source: Source, store: Store, filters: list[Filter] | None = None):
        self.source = source
        self.store = store
        self.chain = FilterChain(filters or [])
        self.stats = {"collected": 0, "filtered": 0, "errors": 0}

    def collect(self) -> int:
        before = self.store.count()
        for record in self.source.read():
            try:
                if self.chain.accept(record):
                    self.store.write(record)
                    self.stats["collected"] += 1
                else:
                    self.stats["filtered"] += 1
            except Exception as e:
                self.stats["errors"] += 1
                logger.debug("Collection error: %s", e)
        return self.store.count() - before

    def read(self) -> Iterator[Record]:
        for record in self.source.read():
            if self.chain.accept(record):
                yield record

    def collect_continuous(self, interval: float = 60.0, max_rounds: int | None = None):
        rounds = 0
        while max_rounds is None or rounds < max_rounds:
            self.collect()
            rounds += 1
            time.sleep(interval)


class ParallelCollector:
    def __init__(self, collectors: list[Collector]):
        self._collectors = collectors
        self.stats = {"total_collected": 0, "sources": {}}

    def collect(self) -> int:
        total = 0
        for collector in self._collectors:
            count = collector.collect()
            total += count
            self.stats["sources"][collector.source.name] = collector.stats
        self.stats["total_collected"] = total
        return total

    def collect_threaded(self) -> int:
        results = [0] * len(self._collectors)
        def run(i, c):
            results[i] = c.collect()
        threads = []
        for i, collector in enumerate(self._collectors):
            t = threading.Thread(target=run, args=(i, collector))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        total = sum(results)
        for i, collector in enumerate(self._collectors):
            self.stats["sources"][collector.source.name] = collector.stats
        self.stats["total_collected"] = total
        return total

    def collect_continuous(self, interval: float = 60.0, max_rounds: int | None = None):
        rounds = 0
        while max_rounds is None or rounds < max_rounds:
            self.collect_threaded()
            rounds += 1
            time.sleep(interval)

    @property
    def store(self):
        if self._collectors:
            return self._collectors[0].store
        return None


class BatchCollector:
    def __init__(self, source: Source, store: Store, filters: list[Filter] | None = None,
                 batch_size: int = 100, max_retries: int = 3, retry_delay: float = 1.0):
        self.source = source
        self.store = store
        self.chain = FilterChain(filters or [])
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.stats = {"collected": 0, "filtered": 0, "errors": 0, "retries": 0, "batches": 0}

    def collect(self) -> int:
        before = self.store.count()
        batch = []
        for record in self.source.read():
            try:
                if self.chain.accept(record):
                    batch.append(record)
                    self.stats["collected"] += 1
                else:
                    self.stats["filtered"] += 1
            except Exception as e:
                self.stats["errors"] += 1
                logger.debug("Collection error in batch collector: %s", e)

            if len(batch) >= self.batch_size:
                self._write_batch(batch)
                batch = []

        if batch:
            self._write_batch(batch)

        return self.store.count() - before

    def _write_batch(self, batch: list[Record]):
        for attempt in range(self.max_retries):
            try:
                for record in batch:
                    self.store.write(record)
                self.stats["batches"] += 1
                return
            except Exception as e:
                if attempt < self.max_retries - 1:
                    self.stats["retries"] += 1
                    logger.debug("Batch write retry %d/%d: %s", attempt + 1, self.max_retries, e)
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    self.stats["errors"] += len(batch)
                    logger.warning("Batch write failed after %d retries: %s", self.max_retries, e)
