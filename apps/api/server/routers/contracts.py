"""Contracts Router — the contract registry's own read path.

Pilot capability: the first route emitted by ``create_router`` instead of a
hand-written endpoint. One descriptor (``SPEC``), one projection (verb, auth,
validation, envelope and OpenAPI metadata all derived from it), and boot
registration by importing this module — see ``docs/TRANSPORT_PROJECTIONS.md``.

Tier: external utility tooling (an operator/introspection surface of the app).
It is a site route, so it never enters the chat tool registry, the
``domain.tools`` engine, or anything the model reasons over.

GET /contracts is the crossing named in AGENTS.md "Endpoint & Transport Rule" —
a read-only descriptor inventory, deliberately not a god-endpoint: it lists the
contract *declarations*, it does not execute them.
"""

from __future__ import annotations

from typing import Any

from infrastructure.contract import REGISTRY, RouteSpec, create_router

from domain.agents import ToolParam, ToolSpec


async def list_contracts(prefix: str | None = None) -> dict[str, Any]:
    """Return every capability crossing registered through the descriptor."""
    contracts = REGISTRY.routes(prefix=prefix)
    return {
        "contracts": contracts,
        "count": len(contracts),
        "coverage": (
            "descriptor-projected routes only — grandfathered hand-written routers "
            "register directly with main.py until migrated"
        ),
    }


SPEC = ToolSpec(
    name="contracts.list",
    description="List capability contracts registered through the descriptor mechanism.",
    parameters=[
        ToolParam(
            name="prefix",
            type="string",
            description="Only return contracts whose route path starts with this prefix.",
        ),
    ],
    execute=list_contracts,
    auth_scope="authenticated",
    idempotent=True,
    result={
        "type": "object",
        "properties": {
            "contracts": {"type": "array", "items": {"type": "object"}},
            "count": {"type": "integer"},
        },
    },
    version="1",
)

router = create_router(
    SPEC,
    RouteSpec(path="", method="GET", summary="List registered capability contracts"),
    list_contracts,
    prefix="/contracts",
    tags=("contracts",),
)
