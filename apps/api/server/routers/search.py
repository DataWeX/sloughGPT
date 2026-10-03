"""Search API — system-wide general search.

One endpoint over every store registered in ``domain.search``. Query
semantics (matching, ranking, limits, visibility) live in the core;
this module is transport only: auth, parameter validation, envelope.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

from fastapi import APIRouter, Depends, Query
from infrastructure.auth import require_auth_if_enabled
from schemas.common import raise_error, success_response

from domain.search import (
    DEFAULT_LIMIT_PER_STORE,
    DEFAULT_TIMEOUT_S,
    SearchContext,
    SearchService,
)
from services.auth import User, UserRepository, WorkspaceRepository

logger = logging.getLogger(__name__)

_service = SearchService()


async def search_system(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    limit: int = Query(DEFAULT_LIMIT_PER_STORE, ge=1, le=100, description="Max hits per store"),
    workspace_id: str | None = Query(
        None, description="Workspace scope — required for workspace-scoped stores"
    ),
    stores: list[str] | None = Query(None, description="Restrict to specific stores (repeatable)"),
    auth_user: dict = Depends(require_auth_if_enabled),
) -> dict:
    """Search across registered stores.

    Returns normalized hits sorted by score; ``partial`` lists stores
    that failed or timed out; ``skipped`` lists stores not queried
    (e.g. workspace-scoped stores without a workspace in context).
    """
    # Auth disabled => require_auth_if_enabled returns None (its
    # documented anonymous mode): an identity-less caller is valid and
    # search runs unscoped. Auth enabled => the dependency guarantees a
    # payload and the identity must resolve.
    user: User | None = None
    if auth_user is not None:
        user = UserRepository().get(auth_user.get("sub", ""))
        if not user:
            raise_error("User not found", "E_NOT_FOUND", status_code=404)
    if not q.strip():
        # min_length catches "" but not whitespace-only input; blank is
        # a validation error, not a 500 from the core.
        raise_error("q must not be blank", "E_VALIDATION", status_code=422)

    if workspace_id:
        ws = WorkspaceRepository()
        if not ws.get(workspace_id):
            raise_error("Workspace not found", "E_NOT_FOUND", status_code=404)
        if user is not None:
            # Identity present => membership gate before any store runs.
            # Anonymous (auth off) => open deployment, existence only.
            member = ws.get_member(workspace_id, user.id)
            if not member and not user.is_admin:
                raise_error("Access denied", "E_AUTH_MISSING", status_code=403)

    ctx = SearchContext(
        workspace_id=workspace_id or "",
        user_id=user.id if user else "",
        is_admin=user.is_admin if user else False,
    )
    result = await _service.search(
        q,
        ctx,
        stores=stores,
        limit_per_store=limit,
        timeout_s=DEFAULT_TIMEOUT_S,
    )
    return success_response(
        data={
            "hits": [asdict(h) for h in result.hits],
            "partial": result.partial,
            "skipped": result.skipped,
            "query": q,
        }
    )


router = APIRouter(prefix="/search", tags=["search"])
router.add_api_route("", search_system, methods=["GET"])
