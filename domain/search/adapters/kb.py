"""Knowledge-base adapter — the KnowledgeEngine (distinct from the
``knowledge`` fact store: kb is the RAG/knowledge-engine surface that
the legacy ``/kb/search`` endpoint wraps)."""

from __future__ import annotations

from domain.search.matching import substring_score
from domain.search.types import SearchContext, SearchHit

_ROUTE = "route:/kb"
_TITLE_MAX = 80


class KBAdapter:
    store = "kb"
    capability = "indexed"
    workspace_scoped = False

    def __init__(self, engine=None):
        self._engine = engine

    def _resolve(self):
        if self._engine is None:
            from domain.knowledge.engine import get_knowledge_engine

            self._engine = get_knowledge_engine()
        return self._engine

    async def search(self, q: str, limit: int, ctx: SearchContext) -> list[SearchHit]:
        # The engine owns matching and ranking (its own index); we only
        # normalize output. Failure propagates => aggregator partial.
        result = self._resolve().query(q, top_k=limit)
        entries = result.data if result and getattr(result, "success", False) else []
        hits: list[SearchHit] = []
        for idx, entry in enumerate(entries[:limit]):
            content = str(entry.get("content") or "")
            title = content[:_TITLE_MAX]
            detail = str(entry.get("topic") or "")
            # Engine scores vary by backing index; normalize for the
            # contract (0, 1] with a sensible fallback for semantic-only
            # matches our substring rule cannot see.
            score = substring_score(q, title, detail)
            if score is None:
                raw = float(entry.get("score") or 0.0)
                score = min(1.0, raw) if raw > 0 else 0.5
            hits.append(
                SearchHit(
                    id=str(entry.get("id") or f"entry-{idx}"),
                    store=self.store,
                    title=title,
                    detail=detail,
                    score=score,
                    locator=_ROUTE,
                )
            )
        return hits
