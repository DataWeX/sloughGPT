"""Contract Router — the descriptor registry's own read path.

Pilot for card 57b0ea27 (deliverable 5): a **brand-new router that is itself a
projection**. Nothing here hand-writes a route, a verb, an envelope or an auth
dependency — ``create_router`` derives all of it from ``CONTRACTS_SPEC`` below.

``GET /contracts`` returns the contracts that projection registered as it built
them (``infrastructure.contract.get_contracts()``), so the endpoint documents
the projection system by executing it. Until more modules adopt
``create_router`` this is a short list; it grows exactly as fast as the
codebase stops hand-writing routes.

Deliberately *not* migrating any existing router: the 57 grandfathered ones
stay as they are and move only when next touched (AGENTS.md Endpoint &
Transport Rule 5).
"""

from __future__ import annotations

from typing import Any

from infrastructure.contract import create_router, get_contracts

from domain.agents import ToolParam, ToolSpec


async def _read_contracts(name: str | None = None) -> dict[str, Any]:
    """Return registered contracts, optionally filtered by descriptor name.

    ``execute`` and the HTTP handler are the same callable — one declaration,
    two projections (agent tool-call and HTTP GET).
    """
    contracts = get_contracts()
    if name:
        needle = name.lower()
        contracts = [c for c in contracts if needle in c["name"].lower()]
    return {"contracts": contracts, "count": len(contracts)}


CONTRACTS_SPEC = ToolSpec(
    name="contracts/list",
    description="List every descriptor-projected HTTP contract registered at boot",
    parameters=[
        ToolParam(
            name="name",
            type="string",
            description="Case-insensitive substring to filter descriptor names",
            required=False,
        ),
    ],
    execute=_read_contracts,
    idempotent=True,
    auth_scope="public",
    version="1",
)

router = create_router(CONTRACTS_SPEC, "contracts", _read_contracts)
