"""Registry of searchable stores.

Adapters self-register; the service queries the registry. Adding a store
to system-wide search = write an adapter + register it — no endpoint or
UI change.
"""

from __future__ import annotations

from domain.search.protocol import Searchable

_registry: SearchRegistry | None = None


class SearchRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, Searchable] = {}

    def register(self, adapter: Searchable) -> None:
        if not isinstance(adapter, Searchable):
            raise TypeError(f"{type(adapter).__name__} does not satisfy the Searchable protocol")
        if adapter.store in self._adapters:
            raise ValueError(f"store already registered: {adapter.store}")
        self._adapters[adapter.store] = adapter

    def get(self, store: str) -> Searchable | None:
        return self._adapters.get(store)

    def adapters(self, stores: list[str] | None = None) -> list[Searchable]:
        if stores is None:
            return list(self._adapters.values())
        missing = [s for s in stores if s not in self._adapters]
        if missing:
            raise KeyError(f"unknown stores: {', '.join(missing)}")
        return [self._adapters[s] for s in stores]

    def store_names(self) -> list[str]:
        return list(self._adapters.keys())


def get_registry() -> SearchRegistry:
    """Process-wide registry singleton."""
    global _registry
    if _registry is None:
        _registry = SearchRegistry()
        _register_default_stores(_registry)
    return _registry


def _register_default_stores(registry: SearchRegistry) -> None:
    """Wire built-in adapters. Failures degrade to fewer stores, never
    to a broken search — a store that cannot even import is simply not
    registered (and therefore never reported, by design: registration
    is a deploy-time fact, runtime failures go to ``partial``)."""
    from domain.search.adapters.datasets import DatasetsAdapter
    from domain.search.adapters.files import FilesAdapter
    from domain.search.adapters.filesources import default_file_adapters
    from domain.search.adapters.kb import KBAdapter
    from domain.search.adapters.knowledge import KnowledgeAdapter
    from domain.search.adapters.members import MembersAdapter
    from domain.search.adapters.memory import MemoryAdapter
    from domain.search.adapters.sessions import SessionsAdapter
    from domain.search.adapters.training import TrainingJobsAdapter

    adapters = [
        MembersAdapter(),
        TrainingJobsAdapter(),
        DatasetsAdapter(),
        KnowledgeAdapter(),
        KBAdapter(),
        MemoryAdapter(),
        FilesAdapter(),
        SessionsAdapter(),
        *default_file_adapters(),
    ]
    for adapter in adapters:
        registry.register(adapter)
