from __future__ import annotations

from typing import Iterator

from .filters import Filter
from .pipeline import CollectionPipeline
from .sources import Record, Source
from .stores import Store


class CollectionRegistry:
    def __init__(self):
        self._sources: dict[str, Source] = {}
        self._stores: dict[str, Store] = {}
        self._filters: dict[str, Filter] = {}
        self._pipelines: dict[str, CollectionPipeline] = {}

    def register_source(self, name: str, source: Source):
        self._sources[name] = source

    def register_store(self, name: str, store: Store):
        self._stores[name] = store

    def register_filter(self, name: str, f: Filter):
        self._filters[name] = f

    def get_source(self, name: str) -> Source | None:
        return self._sources.get(name)

    def get_store(self, name: str) -> Store | None:
        return self._stores.get(name)

    def get_filter(self, name: str) -> Filter | None:
        return self._filters.get(name)

    def create_pipeline(self, name: str, source_name: str, store_name: str, filter_names: list[str] | None = None) -> CollectionPipeline | None:
        source = self._sources.get(source_name)
        store = self._stores.get(store_name)
        if source is None or store is None:
            return None
        filters = []
        if filter_names:
            for fn in filter_names:
                f = self._filters.get(fn)
                if f is not None:
                    filters.append(f)
        pipeline = CollectionPipeline(source, store, filters, name=name)
        self._pipelines[name] = pipeline
        return pipeline

    def get_pipeline(self, name: str) -> CollectionPipeline | None:
        return self._pipelines.get(name)

    def remove_pipeline(self, name: str) -> bool:
        """Remove a pipeline by name. Returns True if it existed."""
        if name in self._pipelines:
            del self._pipelines[name]
            return True
        return False

    def collect(self, pipeline_name: str) -> int:
        pipeline = self._pipelines.get(pipeline_name)
        if pipeline is None:
            return 0
        return pipeline.collect()

    def list_sources(self) -> list[str]:
        return list(self._sources.keys())

    def list_stores(self) -> list[str]:
        return list(self._stores.keys())

    def list_filters(self) -> list[str]:
        return list(self._filters.keys())

    def list_pipelines(self) -> list[str]:
        return list(self._pipelines.keys())

    def stats(self) -> dict:
        return {
            "sources": self.list_sources(),
            "stores": self.list_stores(),
            "filters": self.list_filters(),
            "pipelines": {name: p.stats for name, p in self._pipelines.items()},
        }


_default_registry: CollectionRegistry | None = None


def get_registry() -> CollectionRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = CollectionRegistry()
    return _default_registry
