"""Datasets Search Router — federated dataset discovery (descriptor-projected).

One capability, ``datasets.search_federated``: a single query fans out to
GitHub, HuggingFace, Kaggle and Open Library, normalized to
``{source, id, title, description, stats}`` with per-provider status.
A provider failure or timeout degrades to an ``ok: false`` provider row —
it never fails the whole search (discovery must degrade, not break).

Projected from a ``ToolSpec`` descriptor via ``create_router`` (verb, auth,
validation, envelope and OpenAPI metadata all derived). Crosses the sync-core
seam with ``asyncio.to_thread`` — ``FederatedSearch`` is sync by design and
parallelizes providers with a ``ThreadPoolExecutor`` internally.

Tier: external utility tooling (a site surface of the app) — this route never
enters the chat tool registry, the ``domain.tools`` engine, or the inference
UI's tool list.
"""

from __future__ import annotations

import asyncio

from infrastructure.contract import RouteSpec, create_router

from domain.agents import ToolParam, ToolSpec
from domain.training import FederatedSearch


async def search_federated(q: str, limit: int = 10, sources: str | None = None) -> dict:
    """Run one query across dataset registries; partial results by design.

    Args:
        q: Natural-language search string (e.g. "shakespeare dataset").
        limit: Per-provider result cap (default 10).
        sources: Optional comma-separated provider subset
            (``github,huggingface,kaggle,openlibrary``).

    Returns:
        ``{results: [{source, id, title, description, stats}], providers:
        [{source, ok, count, error, elapsed_ms}]}``.
    """
    wanted = [s.strip() for s in sources.split(",") if s.strip()] if sources else None
    searcher = FederatedSearch()
    return await asyncio.to_thread(searcher.search, q, limit, wanted)


SPEC = ToolSpec(
    name="datasets.search_federated",
    description=(
        "Federated dataset discovery: one query across GitHub, HuggingFace, "
        "Kaggle and Open Library; normalized results with per-provider status, "
        "partial on provider failure."
    ),
    parameters=[
        ToolParam(
            name="q", type="string", description="Natural-language search query", required=True
        ),
        ToolParam(name="limit", type="integer", description="Per-provider result cap (1-50)"),
        ToolParam(
            name="sources",
            type="string",
            description="Comma-separated provider subset (github,huggingface,kaggle,openlibrary)",
        ),
    ],
    execute=search_federated,
    auth_scope="public",
    idempotent=True,
    result={
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "description": {"type": ["string", "null"]},
                        "stats": {"type": "object"},
                    },
                },
            },
            "providers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "ok": {"type": "boolean"},
                        "count": {"type": "integer"},
                        "error": {"type": ["string", "null"]},
                        "elapsed_ms": {"type": "integer"},
                    },
                },
            },
        },
    },
    version="1",
)

router = create_router(
    SPEC,
    RouteSpec(
        path="/search/all",
        method="GET",
        summary="Federated dataset search across GitHub, HuggingFace, Kaggle, Open Library",
    ),
    search_federated,
    prefix="/datasets",
    tags=("datasets",),
)
