"""Knowledge adapter — facts from the knowledge store."""

from __future__ import annotations

from domain.search.matching import substring_score
from domain.search.types import SearchContext, SearchHit

_ROUTE = "route:/knowledge"


class KnowledgeAdapter:
    store = "knowledge"
    capability = "live"
    workspace_scoped = False

    def __init__(self, repo=None):
        self._repo = repo

    def _resolve(self):
        if self._repo is None:
            from domain.infrastructure import get_knowledge_repository

            self._repo = get_knowledge_repository()
        return self._repo

    async def search(self, q: str, limit: int, ctx: SearchContext) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for fact in self._resolve().list_facts():
            title = (fact.content or fact.id)[:80]
            detail = fact.topic or ""
            score = substring_score(q, title, detail)
            if score is None:
                continue
            hits.append(
                SearchHit(
                    id=fact.id,
                    store=self.store,
                    title=title,
                    detail=detail,
                    score=score,
                    locator=_ROUTE,
                )
            )
            if len(hits) >= limit:
                break
        return hits
