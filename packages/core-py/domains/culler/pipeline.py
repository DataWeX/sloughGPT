from __future__ import annotations

import time
from typing import Iterator

from .culler import Culler
from .filters import Filter, FilterChain
from .sources import Record, Source
from .stores import Store


class CullerPipeline:
    def __init__(self, source: Source, store: Store, filters: list[Filter] | None = None, name: str = ""):
        self.name = name or f"{source.name}->{store.name}"
        self.source = source
        self.store = store
        self.filters = filters or []
        self._culler = Culler(source, store, self.filters)

    def collect(self) -> int:
        return self._culler.collect()

    def read(self) -> Iterator[Record]:
        return self._culler.read()

    def collect_continuous(self, interval: float = 60.0, max_rounds: int | None = None):
        self._culler.collect_continuous(interval, max_rounds)

    @property
    def stats(self) -> dict:
        return {
            "source": self.source.name,
            "store": self.store.name,
            "pipeline": self.name,
            "culler": dict(self._culler.stats),
            "filter_chain": dict(self._culler.chain.stats),
            "store_count": self.store.count(),
        }
