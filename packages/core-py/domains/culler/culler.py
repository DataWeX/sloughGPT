from __future__ import annotations

import time
from typing import Iterator

from .filters import Filter, FilterChain
from .sources import Record, Source
from .stores import Store


class Culler:
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
            except Exception:
                self.stats["errors"] += 1
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
