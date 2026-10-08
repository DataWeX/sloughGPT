"""Training jobs adapter — jobs for the current workspace."""

from __future__ import annotations

from domain.search.matching import substring_score
from domain.search.types import SearchContext, SearchHit

_ROUTE = "route:/training"


class TrainingJobsAdapter:
    store = "training_jobs"
    capability = "live"
    workspace_scoped = True

    def __init__(self, repo=None):
        self._repo = repo

    def _resolve(self):
        if self._repo is None:
            # The durable job history (same store GET /training/jobs reads).
            from training.job_store import JobStore

            self._repo = JobStore()
        return self._repo

    async def search(self, q: str, limit: int, ctx: SearchContext) -> list[SearchHit]:
        if not ctx.workspace_id:
            return []
        repo = self._resolve()
        hits: list[SearchHit] = []
        # JobStore returns plain dicts (store_row_to_job), not objects.
        # Raises on store failure — the aggregator reports us as partial
        # rather than letting an empty list masquerade as "no matches".
        for job in repo.list_by_workspace(ctx.workspace_id):
            jid = str(job.get("id") or job.get("_id") or "")
            if not jid:
                continue
            title = str(job.get("name") or jid)
            detail = f"Status: {job.get('status', 'unknown')}"
            score = substring_score(q, title, detail)
            if score is None:
                continue
            hits.append(
                SearchHit(
                    id=jid,
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
