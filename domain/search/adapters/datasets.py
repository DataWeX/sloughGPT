"""Datasets adapter.

Dataset entities carry no workspace field (see card "un-dead workspace
search"), so this store is global — any authenticated caller can search
it, matching the existing ``/datasets/search`` endpoint's scope.
"""

from __future__ import annotations

from domain.search.matching import substring_score
from domain.search.types import SearchContext, SearchHit

_ROUTE = "route:/datasets"


class DatasetsAdapter:
    store = "datasets"
    capability = "live"
    workspace_scoped = False

    def __init__(self, repo=None):
        self._repo = repo

    def _resolve(self):
        if self._repo is None:
            from domain.infrastructure import get_dataset_repository

            self._repo = get_dataset_repository()
        return self._repo

    async def search(self, q: str, limit: int, ctx: SearchContext) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for ds in self._resolve().list():
            title = ds.name or ds.id
            detail = f"{ds.format} · {ds.record_count} records"
            score = substring_score(q, title, detail)
            if score is None:
                continue
            hits.append(
                SearchHit(
                    id=ds.id,
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
