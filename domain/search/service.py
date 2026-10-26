"""SearchService — concurrent fan-out over registered stores.

Owns the cross-cutting rules so adapters do not have to:

- per-store timeout (one hung store cannot stall the query);
- per-store limit (one chatty store cannot starve the rest);
- failure isolation (a raising store lands in ``partial``, never
  silently shrinks the result set);
- visibility (workspace-scoped stores are skipped without context and
  receive the context to scope their rows).
"""

from __future__ import annotations

import asyncio
import logging

from domain.search.protocol import Searchable
from domain.search.registry import SearchRegistry, get_registry
from domain.search.types import SearchContext, SearchHit, SearchResult

logger = logging.getLogger(__name__)

DEFAULT_LIMIT_PER_STORE = 20
DEFAULT_TIMEOUT_S = 2.0


class SearchService:
    def __init__(self, registry: SearchRegistry | None = None) -> None:
        self._registry = registry or get_registry()

    async def search(
        self,
        q: str,
        ctx: SearchContext,
        *,
        stores: list[str] | None = None,
        limit_per_store: int = DEFAULT_LIMIT_PER_STORE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> SearchResult:
        """Fan out to registered stores and merge their hits.

        Returns hits sorted by score desc; failed stores listed in
        ``partial``; stores deliberately not queried listed in ``skipped``.
        """
        if not q.strip():
            raise ValueError("q must not be empty")
        if limit_per_store < 1:
            raise ValueError("limit_per_store must be >= 1")

        adapters = self._registry.adapters(stores)
        result = SearchResult()
        runnable: list[Searchable] = []
        for adapter in adapters:
            if adapter.workspace_scoped and not ctx.workspace_id:
                result.skipped.append(adapter.store)
            else:
                runnable.append(adapter)

        if runnable:
            outcomes = await asyncio.gather(
                *(self._run_one(a, q, limit_per_store, ctx, timeout_s) for a in runnable)
            )
            for store, hits in outcomes:
                if hits is None:
                    result.partial.append(store)
                else:
                    result.hits.extend(hits)

        result.hits.sort(key=lambda h: (-h.score, h.title, h.store))
        return result

    async def _run_one(
        self,
        adapter: Searchable,
        q: str,
        limit: int,
        ctx: SearchContext,
        timeout_s: float,
    ) -> tuple[str, list[SearchHit] | None]:
        """Run one adapter; ``None`` signals failure (=> partial)."""
        try:
            hits = await asyncio.wait_for(adapter.search(q, limit, ctx), timeout=timeout_s)
            # Defense in depth: an adapter that overshoots its limit is
            # trimmed here so one store can never crowd out the others.
            return adapter.store, hits[:limit]
        except TimeoutError:
            logger.debug("search store timed out: %s", adapter.store)
            return adapter.store, None
        except Exception:
            logger.debug("search store failed: %s", adapter.store, exc_info=True)
            return adapter.store, None
