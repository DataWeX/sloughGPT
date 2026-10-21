"""
Bootstrap Router - one-shot boot fan-out read-model.

GET /bootstrap returns the landing journey's first-paint domain reads in a
single response, composed in-process from the member endpoints' own
handlers — the route table is the registry, resolved per request, so the
aggregate never duplicates member logic and members appear automatically
as the startup phases register them.

Registered pre-lifespan like /health: during the 25-40s model-load phase
it answers with a partial payload (unregistered members land in ``errors``
with "not_ready"), and a failing member degrades to ``null`` + an error
entry — the aggregate never 503s and never gates on readiness.

Additive by contract: the individual endpoints stay for the mobile BFF,
SDK, monitoring and polling consumers; streams, polling loops, telemetry
and client-state docstore KV reads are out of scope by construction.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi import params as fparams
from fastapi.encoders import jsonable_encoder
from fastapi.params import Depends as DependsParam
from fastapi.routing import APIRoute
from infrastructure.auth import require_auth_if_enabled
from schemas.common import endpoint, success_response

# key -> (path, method): the measured first-paint domain reads of the
# landing journey (prod-build inventory, card 342d729f, 2026-10-21).
_BOOTSTRAP_MEMBERS: dict[str, tuple[str, str]] = {
    "health": ("/health", "GET"),
    "startup_progress": ("/health/startup-progress", "GET"),
    "consciousness_status": ("/consciousness/status", "GET"),
    "souls": ("/souls", "GET"),
    "sessions": ("/chat/sessions", "GET"),
    "training_jobs": ("/training/jobs", "GET"),
    "turbo_status": ("/training/turbo/status", "GET"),
    "knowledge_stats": ("/knowledge/stats", "GET"),
    "feedback_stats": ("/feedback/stats/summary", "GET"),
    "meta_weights_stats": ("/meta-weights/stats", "GET"),
    "datasets": ("/datasets", "GET"),
}


def _unwrap(resp: Any) -> Any:
    """Unwrap the shared ``{status, data}`` envelope.

    The bootstrap payload carries the same ``data`` slices the frontend's
    http-client hands to controllers today, so existing types stay valid.
    """
    data = jsonable_encoder(resp)
    if isinstance(data, dict) and "status" in data and "data" in data:
        return data["data"]
    return data


async def _member_kwargs(fn: Any, request: Request, auth_user: Any) -> dict[str, Any]:
    """Kwargs for calling a member handler directly (no FastAPI DI).

    Mirrors what FastAPI itself would inject: the request where a signature
    asks for one, the aggregate's route-resolved ``auth_user`` for
    ``require_auth_if_enabled`` params (same caller identity — the route
    itself enforces auth, so members never see a synthetic user), the plain
    default of Query/Path/Header params, and a loud failure for anything
    else required rather than a guess.
    """
    kwargs: dict[str, Any] = {}
    for name, param in inspect.signature(fn).parameters.items():
        default = param.default
        if name in ("request", "req"):
            kwargs[name] = request
        elif isinstance(default, DependsParam):
            if default.dependency is require_auth_if_enabled:
                kwargs[name] = auth_user
            else:
                raise TypeError(f"member requires unresolvable dependency {name!r}")
        elif isinstance(default, (fparams.Query, fparams.Path, fparams.Header, fparams.Cookie)):
            kwargs[name] = default.default
        elif default is inspect.Parameter.empty:
            raise TypeError(f"required parameter {name!r} cannot be inferred")
        else:
            kwargs[name] = default
    return kwargs


def _iter_routes(routes, prefix: str = ""):
    """Flatten the route tree to (full_path, route) pairs.

    FastAPI >= 0.141 includes routers lazily (_IncludedRouter wrappers
    without .path), older versions flatten eagerly — this walks both.
    """
    for route in routes:
        if isinstance(route, APIRoute):
            yield prefix + route.path, route
        else:
            orig = getattr(route, "original_router", None)
            if orig is not None:
                ctx = getattr(route, "include_context", None)
                yield from _iter_routes(orig.routes, prefix + (getattr(ctx, "prefix", "") or ""))


def _get_route(request: Request, path: str, method: str):
    for route_path, route in _iter_routes(request.app.routes):
        if route_path == path and method in (getattr(route, "methods", None) or set()):
            return route
    return None


class BootstrapRouter:
    def __init__(self):
        self.router = APIRouter(tags=["bootstrap"])
        self.router.add_api_route("/bootstrap", self.bootstrap, methods=["GET"])

    @endpoint("bootstrap.get")
    async def bootstrap(
        self,
        request: Request,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """One request for the landing page's first-paint domain reads.

        Members run concurrently (each member handler owns its own thread
        offload and TTL cache), so total time is the slowest member, not
        the sum.  Response: ``data.members`` maps every member key to the
        same payload its individual endpoint serves; unavailable or failing
        members are ``null`` with an entry in ``data.errors``.
        """

        async def call(key: str, path: str, method: str) -> tuple[str, Any, str | None]:
            route = _get_route(request, path, method)
            if route is None:
                return key, None, "not_ready"
            try:
                fn = route.endpoint
                kwargs = await _member_kwargs(fn, request, auth_user)
                # @endpoint-wrapped async handlers look sync to
                # iscoroutinefunction — unwrap before classifying; a sync
                # wrapper that returns a coroutine is awaited after the
                # thread offload as a safety net.
                if inspect.iscoroutinefunction(inspect.unwrap(fn)):
                    resp = await fn(**kwargs)
                else:
                    resp = await asyncio.to_thread(lambda: fn(**kwargs))
                    if inspect.isawaitable(resp):
                        resp = await resp
                return key, _unwrap(resp), None
            except Exception as exc:  # per-member degradation — never 503
                return key, None, f"{type(exc).__name__}: {exc}"

        results = await asyncio.gather(
            *(call(key, path, method) for key, (path, method) in _BOOTSTRAP_MEMBERS.items())
        )
        members: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for key, data, err in results:
            members[key] = data
            if err:
                errors[key] = err
        return success_response(data={"members": members, "errors": errors})


router = BootstrapRouter().router
