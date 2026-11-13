"""Chat-sessions adapter — full-text over the session corpus via the
shared in-memory index (extracted from routers/inference.py so the
legacy ``GET /chat/sessions/search`` endpoint and this adapter read
one cache, not two)."""

from __future__ import annotations

from domain.search.types import SearchContext, SearchHit

_ROUTE = "route:/chat"
_DETAIL_MAX = 160


class SessionsAdapter:
    store = "sessions"
    capability = "indexed"
    workspace_scoped = False

    def __init__(self, index=None):
        self._index = index

    def _resolve(self):
        if self._index is None:
            from domain.search.session_index import get_session_index

            self._index = get_session_index()
        return self._index

    async def search(self, q: str, limit: int, ctx: SearchContext) -> list[SearchHit]:
        # The index owns matching (casefold substring over session names
        # and messages, most-recent-first) — we normalize output only.
        results = self._resolve().search(q, limit)
        hits: list[SearchHit] = []
        for idx, r in enumerate(results[:limit]):
            name = str(r.get("name") or r.get("id") or f"session-{idx}")
            matches = r.get("matches") or []
            first = matches[0] if matches else {}
            role = str(first.get("role") or "")
            snippet = str(first.get("content") or "")
            detail = f"{role}: {snippet}"[:_DETAIL_MAX] if role else snippet[:_DETAIL_MAX]
            # Index orders name matches before message matches, so a
            # session whose title matched carries role=="session" first.
            score = 1.0 if role == "session" else 0.6
            hits.append(
                SearchHit(
                    id=str(r.get("id") or f"session-{idx}"),
                    store=self.store,
                    title=name,
                    detail=detail,
                    score=score,
                    locator=_ROUTE,
                )
            )
        return hits
