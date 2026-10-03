"""Members adapter — workspace membership roster."""

from __future__ import annotations

from domain.search.matching import substring_score
from domain.search.types import SearchContext, SearchHit

_MEMBERS_ROUTE = "route:/workspace/members"


class MembersAdapter:
    store = "members"
    capability = "live"
    workspace_scoped = True

    def __init__(self, ws_repo=None, user_repo=None):
        self._ws_repo = ws_repo
        self._user_repo = user_repo

    def _resolve(self):
        if self._ws_repo is None or self._user_repo is None:
            from services.auth import UserRepository, WorkspaceRepository

            self._ws_repo = self._ws_repo or WorkspaceRepository()
            self._user_repo = self._user_repo or UserRepository()
        return self._ws_repo, self._user_repo

    async def search(self, q: str, limit: int, ctx: SearchContext) -> list[SearchHit]:
        if not ctx.workspace_id:
            return []
        ws_repo, user_repo = self._resolve()
        hits: list[SearchHit] = []
        for m in ws_repo.list_members(ctx.workspace_id):
            u = user_repo.get(m.user_id)
            title = u.username if u else m.user_id
            detail = f"Role: {m.role.value}"
            score = substring_score(q, title, detail)
            if score is None:
                continue
            hits.append(
                SearchHit(
                    id=m.id,
                    store=self.store,
                    title=title,
                    detail=detail,
                    score=score,
                    locator=_MEMBERS_ROUTE,
                )
            )
            if len(hits) >= limit:
                break
        return hits
