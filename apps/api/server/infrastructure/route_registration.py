"""One assembler for feature routers — production, the gate, and the tests.

``_phase6_routers`` used to skip a manifest router when *any* registered path
``startswith(router.prefix)``. The generated manifest imports ``agent_routines``
(prefix ``/agents``) before ``agents`` (same prefix), so including the first
claimed ``/agents`` and the whole second router — 11 routes: ``GET|POST
/agents``, ``/agents/{id}``, ``/agents/runs*``, ``/agents/execute``,
``/agents/orchestrate`` — was skipped. Result: **404 in production** (573 hits
in 40 minutes, card ``20261119_105501``) while the contract gate and the test
harness, which both assembled the app with unconditional ``include_router``,
kept reporting those routes as served. Three assemblers, one of them
divergent — the spec lied and the suite stayed green.

Invariants encoded here:

* **Ownership is a route path, not a prefix.** Two routers may share a prefix;
  what matters is whether a path a router *declares* is already served.
* **Registration is idempotent.** A router whose declared paths are *all*
  already served is skipped, so the boot retry (``_phase6_routers``' three
  attempts) and repeated test setup never double-register.
* **Over-skipping is the bug class this removes.** A router with any *unserved*
  path is always included; a partial overlap is logged at WARNING (the served
  duplicate shadows the new one) instead of silently dropping the rest.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, NamedTuple

logger = logging.getLogger("slo.routers")

__all__ = ["RegistrationResult", "_app_paths", "declared_paths", "register_routers"]


class RegistrationResult(NamedTuple):
    """What registration did: routers put on the app, and why others were not."""

    included: list[Any]
    skipped: list[str]


def declared_paths(router: Any) -> list[str]:
    """Every path ``router`` would put on the app, prefix resolved.

    FastAPI stores the prefixed path on routes added through a router that was
    constructed with a prefix, but not on every construction style, so the
    prefix is applied defensively — the same normalisation
    ``scripts/check_contract.py`` uses for its inventory.

    FastAPI >= 0.141 wraps a previously included router in a lazy
    ``_IncludedRouter`` (no ``path`` of its own, an ``original_router``
    attribute instead), so nested includes are recursed rather than skipped.
    """
    prefix = getattr(router, "prefix", "") or ""
    paths: list[str] = []
    for route in getattr(router, "routes", None) or []:
        inner = getattr(route, "original_router", None)
        if inner is not None:
            paths.extend(declared_paths(inner))
            continue
        path = getattr(route, "path", None)
        if not path:
            continue  # mounts / malformed entries carry no comparable path
        if prefix and not path.startswith(prefix):
            path = prefix + path
        paths.append(path)
    return paths


def _app_paths(app: Any) -> set[str]:
    """Paths already served on ``app``, resolving lazy ``_IncludedRouter``.

    FastAPI >= 0.141 appends a path-less ``_IncludedRouter`` wrapper to
    ``app.routes`` instead of flattening; reading only ``route.path`` would
    make every previously included router invisible and defeat idempotency
    (the boot retry would double-register).
    """
    served: set[str] = set()
    for route in getattr(app, "routes", None) or []:
        inner = getattr(route, "original_router", None)
        if inner is not None:
            served.update(declared_paths(inner))
            continue
        path = getattr(route, "path", None)
        if path:
            served.add(path)
    return served


def register_routers(app: Any, routers: Iterable[Any]) -> RegistrationResult:
    """Put every not-yet-served router on ``app`` — the one registration path.

    ``infrastructure/startup.py`` (boot), ``scripts/check_contract.py`` (spec)
    and ``tests/test_support.py`` (test client) all call this, so all three
    describe the same route table. Assembling the app any other way is how the
    spec came to list routes the server 404'd.
    """
    served = _app_paths(app)
    included: list[Any] = []
    skipped: list[str] = []

    for router in routers:
        paths = declared_paths(router)
        name = getattr(router, "prefix", None) or repr(router)
        if not paths:
            skipped.append(f"{name}: declares no routes")
            logger.debug("route registration: skipping %s — declares no routes", name)
            continue

        unserved = [p for p in paths if p not in served]
        if not unserved:
            # Everything it declares is already on the app: an idempotent
            # re-registration (boot retry, second test pass), not a conflict.
            skipped.append(f"{name}: already registered")
            logger.debug("route registration: %s already registered", name)
            continue

        overlap = [p for p in paths if p in served]
        if overlap:
            # Register anyway — dropping the router would lose `unserved` —
            # but say so: the earlier route wins the match on these paths.
            logger.warning(
                "route registration: %s overlaps already-served %s — registering "
                "the rest; the earlier routes shadow the duplicates",
                name,
                ", ".join(overlap),
            )

        app.include_router(router)
        included.append(router)
        served.update(paths)

    return RegistrationResult(included, skipped)
