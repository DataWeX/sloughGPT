"""Training jobs adapter — jobs for the current workspace."""

from __future__ import annotations

from domain.search.matching import substring_score
from domain.search.types import SearchContext, SearchHit

_ROUTE = "route:/training/queue"


class TrainingJobsAdapter:
    store = "training_jobs"
    capability = "live"
    workspace_scoped = True

    def __init__(self, repo=None):
        self._repo = repo

    def _resolve(self):
        if self._repo is None:
            from domain.training.repository import TrainingRepository

            self._repo = TrainingRepository()
        return self._repo

    async def search(self, q: str, limit: int, ctx: SearchContext) -> list[SearchHit]:
        if not ctx.workspace_id:
            return []
        repo = self._resolve()
        hits: list[SearchHit] = []
        # Raises on store failure — the aggregator reports us as partial
        # rather than letting an empty list masquerade as "no matches".
        for job in repo.list_by_workspace(ctx.workspace_id):
            title = getattr(job, "name", None) or job.id
            detail = f"Status: {getattr(job, 'status', 'unknown')}"
            score = substring_score(q, title, detail)
            if score is None:
                continue
            hits.append(
                SearchHit(
                    id=job.id,
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
