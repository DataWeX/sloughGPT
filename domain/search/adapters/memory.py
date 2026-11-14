"""Auto-memory adapter — the semantic memory layer (unscoped: it is a
per-process assistant memory, not workspace data)."""

from __future__ import annotations

from domain.search.matching import substring_score
from domain.search.types import SearchContext, SearchHit

_ROUTE = "route:/memory"
_TITLE_MAX = 80


class MemoryAdapter:
    store = "memory"
    capability = "indexed"
    workspace_scoped = False

    def __init__(self, service=None):
        self._service = service

    def _resolve(self):
        if self._service is None:
            from domain.memory._internal.service import get_memory_service

            self._service = get_memory_service()
        return self._service

    async def search(self, q: str, limit: int, ctx: SearchContext) -> list[SearchHit]:
        # retrieve() returns [] when memory is disabled or the query is
        # blank — that is "no matches", not failure. Store errors raise
        # and become partial upstream. No service (defensive) = same as
        # disabled: nothing searchable, not a store failure.
        service = self._resolve()
        if service is None:
            return []
        items = service.retrieve(q, limit=limit)
        hits: list[SearchHit] = []
        for idx, item in enumerate(items[:limit]):
            content = str(item.get("content") or "")
            title = content[:_TITLE_MAX]
            detail = str(item.get("topic") or "")
            score = substring_score(q, title, detail)
            if score is None:
                # Semantic matches legitimately miss the substring rule;
                # rank them mid-range rather than hiding them.
                raw = float(item.get("score") or 0.0)
                score = min(1.0, raw) if raw > 0 else 0.5
            hits.append(
                SearchHit(
                    id=str(item.get("id") or f"mem-{idx}"),
                    store=self.store,
                    title=title,
                    detail=detail,
                    score=score,
                    locator=_ROUTE,
                )
            )
        return hits
