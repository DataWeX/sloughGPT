"""System-wide search — one query, every registered store.

Public surface:

- :class:`SearchHit` / :class:`SearchContext` / :class:`SearchResult`
- :class:`Searchable` (protocol implemented by each store adapter)
- :class:`SearchRegistry` / :func:`get_registry`
- :class:`SearchService`

Adding a store: write an adapter in ``domain/search/adapters/`` and
register it in ``_register_default_stores``; the contract suite in
``tests/api-server/`` validates it against every other.
"""

from domain.search.protocol import Capability, Searchable
from domain.search.registry import SearchRegistry, get_registry
from domain.search.service import DEFAULT_LIMIT_PER_STORE, DEFAULT_TIMEOUT_S, SearchService
from domain.search.types import SearchContext, SearchHit, SearchResult

__all__ = [
    "Capability",
    "DEFAULT_LIMIT_PER_STORE",
    "DEFAULT_TIMEOUT_S",
    "Searchable",
    "SearchContext",
    "SearchHit",
    "SearchRegistry",
    "SearchResult",
    "SearchService",
    "get_registry",
]
