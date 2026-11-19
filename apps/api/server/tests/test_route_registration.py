"""Regression tests for the single router assembler (route_registration).

The bug class this pins (card ``20261119_105501``): production
``_phase6_routers`` skipped a manifest router when *any* registered path
``startswith(router.prefix)``. The manifest imports ``agent_routines``
(prefix ``/agents``) before ``agents`` (same prefix), so the whole
11-route ``agents`` router was dropped — live ``GET /agents`` 404'd (573
hits in 40 minutes) while ``scripts/check_contract.py`` and the test
harness, both assembling with unconditional ``include_router``, kept
reporting the routes as served. Three assemblers, one divergent; the spec
lied and the suite stayed green.

Invariants pinned here:

* two routers sharing a prefix both register;
* registration is idempotent — a second pass adds no routes;
* partial overlap includes the router and logs a WARNING;
* every declared path of every boot router is served by the assembled app
  (declared-vs-served parity, the check the gate now enforces too);
* ``GET /agents`` resolves on an app assembled exactly like production.
"""

import logging

from fastapi import APIRouter, FastAPI
from infrastructure.route_registration import (
    _app_paths,
    declared_paths,
    register_routers,
)


def _paths(app: FastAPI) -> set[str]:
    # FastAPI >= 0.141 stores includes as path-less _IncludedRouter wrappers;
    # _app_paths resolves them (reading route.path alone sees nothing).
    return _app_paths(app)


def _router(prefix: str, route: str) -> APIRouter:
    router = APIRouter(prefix=prefix)

    @router.get(route)
    def _handler():
        return {}

    return router


def test_shared_prefix_both_register():
    """Two routers sharing ``/agents`` must BOTH land on the app.

    Prefix sharing is not ownership — this is the exact shape of the
    agent_routines/agents collision that 404'd ``GET /agents``.
    """
    app = FastAPI()
    first = _router("/agents", "/routines/stream")
    second = _router("/agents", "")  # GET /agents itself

    result = register_routers(app, [first, second])

    assert len(result.included) == 2
    assert result.skipped == []
    assert {"/agents", "/agents/routines/stream"} <= _paths(app)


def test_registration_is_idempotent():
    """A second pass must skip every router and add no routes."""
    app = FastAPI()
    routers = [_router("/agents", ""), _router("/foo", "/bar")]

    first = register_routers(app, routers)
    route_count = len(app.routes)
    second = register_routers(app, routers)

    assert len(first.included) == 2
    assert second.included == []
    assert len(second.skipped) == 2
    assert len(app.routes) == route_count


def test_partial_overlap_includes_and_warns(caplog):
    """A router with any unserved path is registered; overlap warns.

    Dropping the router on partial overlap would lose its unserved routes
    — the over-skip defect — so inclusion is mandatory and the shadowing
    is surfaced at WARNING instead of swallowed.
    """
    app = FastAPI()

    @app.get("/agents/legacy")
    def legacy():
        return {}

    router = APIRouter(prefix="/agents")

    @router.get("/legacy")
    def dup():
        return {}

    @router.get("/fresh")
    def fresh():
        return {}

    with caplog.at_level(logging.WARNING, logger="slo.routers"):
        result = register_routers(app, [router])

    assert result.included == [router]
    assert "/agents/fresh" in _paths(app)
    assert any("overlaps" in record.message for record in caplog.records)


def test_every_boot_router_declared_path_is_served():
    """Declared-vs-served parity for the real boot router set.

    ``check_contract.py --runtime`` enforces the same invariant; this is
    its test-level twin, so a regression fails in seconds without the gate.
    """
    from routers import get_all_routers

    app = FastAPI()
    routers = get_all_routers()
    register_routers(app, routers)

    served = _paths(app)
    missing = {
        getattr(router, "prefix", None) or repr(router): [
            p for p in declared_paths(router) if p not in served
        ]
        for router in routers
    }
    missing = {name: paths for name, paths in missing.items() if paths}

    assert not missing, f"declared but not served: {missing}"


def test_get_agents_route_resolves():
    """``GET /agents`` must exist on a production-shaped assembly.

    The live server 404'd this route for 40 minutes while every gate
    stayed green; openapi() is the definitive served table (it resolves
    FastAPI >= 0.141's lazy ``_IncludedRouter`` wrappers).
    """
    from routers import get_all_routers

    app = FastAPI()
    register_routers(app, get_all_routers())

    agents = app.openapi()["paths"].get("/agents", {})

    assert "get" in agents, "GET /agents missing — agent list journey 404s"
    assert "post" in agents, "POST /agents missing — agent create journey 404s"
