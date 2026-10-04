"""Descriptor-projected transport — emit a FastAPI route fragment from a ToolSpec.

Card: 57b0ea27 (Descriptor-projected transport).
Doctrine: the *Endpoint & Transport Rule* in root AGENTS.md; the pattern in
``docs/TRANSPORT_PROJECTIONS.md``.

The contract is declared once, per module, as a ``ToolSpec``. This module is
the HTTP *projection* of that declaration — verb, request model, envelope,
auth scope and the ``x-contract`` OpenAPI marker are all derived, so none of
them get re-decided per handler the way a hand-written router re-decides them.

    router = create_router(spec, "doctor/report", read_report)   # GET  (idempotent)
    router = create_router(spec, "doctor/check",  run_check)     # POST (action)

Layering: this lives on the server side. ``domain`` stays capability-only and
must never import FastAPI — the dependency points one way, server -> domain.

Annotation note
---------------
FastAPI reads the endpoint's real signature, so the per-spec Pydantic model has
to be attached after the function is defined. ``endpoint.__annotations__`` is
therefore assigned explicitly: with ``from __future__ import annotations`` a
closure reference like ``q: _model`` would be stringified and unresolvable.
Tested both paths; assigning real objects is the one that works under PEP 563.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel, create_model
from schemas.common import classify_and_raise, success_response

from domain.agents._internal.tools import ToolSpec
from infrastructure.auth import require_auth_if_enabled

logger = logging.getLogger("slo.contract")

_READ = "GET"
_WRITE = "POST"

# JSON Schema type -> python type used to build the request model.
_JSON_TO_PY: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def resolve_method(spec: ToolSpec, override: str | None = None) -> str:
    """Verb from idempotency: a safe/idempotent read is GET, an action is POST.

    An explicit override wins, and is logged — a descriptor that says one thing
    and a route that does another is exactly the drift this whole module exists
    to stop, so the disagreement has to be visible in the log.
    """
    if override is not None:
        verb = override.upper()
        logger.info(
            "create_router: explicit method override %s for %r (descriptor idempotent=%s)",
            verb,
            spec.name,
            spec.idempotent,
        )
        return verb
    return _READ if spec.idempotent else _WRITE


def build_request_model(spec: ToolSpec) -> type[BaseModel] | None:
    """Derive the request model from ``spec.parameters`` (None when it takes none).

    Raises ValueError at *router-build* time for an unsupported type, so a
    descriptor mistake fails at boot instead of at request time.
    """
    if not spec.parameters:
        return None

    fields: dict[str, Any] = {}
    for p in spec.parameters:
        py_type = _JSON_TO_PY.get(p.type)
        if py_type is None:
            raise ValueError(
                f"create_router: {spec.name!r} parameter {p.name!r} has unsupported "
                f"type {p.type!r} — add it to _JSON_TO_PY"
            )
        # Optional params must be annotated Optional: FastAPI passes an absent
        # query/body field explicitly as None, and a bare `str` field rejects
        # None (the default is only unvalidated when left untouched).
        fields[p.name] = (py_type, ...) if p.required else (py_type | None, None)

    model_name = "".join(part.capitalize() for part in spec.name.split("/"))
    return create_model(f"{model_name}Request", **fields)  # type: ignore[call-overload]


def check_handler_matches(spec: ToolSpec, handler: Callable[..., Any]) -> None:
    """Fail fast if the handler needs an argument the descriptor does not declare.

    A 422 catches a *bad request*; it cannot catch a handler and its own
    descriptor disagreeing. That mismatch is a boot-time defect, so it is
    checked when the router is built, never per request.
    """
    sig = inspect.signature(handler)
    required = [
        p.name
        for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]
    declared = {p.name for p in spec.parameters}
    missing = [name for name in required if name not in declared]
    if missing:
        raise ValueError(
            f"create_router: handler for {spec.name!r} requires {missing!r}, "
            f"which the descriptor does not declare (declared={sorted(declared)!r})"
        )


async def _dispatch(spec: ToolSpec, handler: Callable[..., Any], data: dict[str, Any]) -> Any:
    """Run the handler and shape the one shared success/error story."""
    try:
        result = handler(**data)
        if inspect.isawaitable(result):
            result = await result
    except Exception as e:  # noqa: BLE001 - classify_and_raise re-raises as AppError
        classify_and_raise(e, source=f"router.{spec.name}")
        raise  # unreachable; keeps the non-None path honest for type checkers
    return success_response(data=result)


def create_router(
    spec: ToolSpec,
    route: str,
    handler: Callable[..., Any],
    *,
    method: str | None = None,
    prefix: str = "",
    tags: list[str] | None = None,
) -> APIRouter:
    """Project one descriptor into one route fragment.

    Args:
        spec: the single source of truth (params, auth scope, idempotency, version).
        route: path fragment, e.g. ``"doctor/report"``.
        handler: called with the descriptor's parameters as keyword arguments.
        method: explicit verb override; logged when it disagrees with the
            descriptor's idempotency.
        prefix: router prefix, e.g. ``"/tools"``.
        tags: OpenAPI tags; defaults to the route's first path segment.

    Returns:
        An ``APIRouter`` holding exactly one route, ready for ``include_router``.
    """
    verb = resolve_method(spec, override=method)
    model = build_request_model(spec)
    check_handler_matches(spec, handler)

    # strip() first: "   " has no slashes, so a slash-only strip would silently
    # accept it and mint a route whose path is "/   ".
    clean = route.strip().strip("/")
    if not clean:
        raise ValueError(f"create_router: {spec.name!r} has an empty route")
    path = "/" + clean
    if tags is None:
        tags = [clean.split("/")[0]]

    openapi_extra: dict[str, Any] = {
        "x-contract": {
            "name": spec.name,
            "version": spec.version,
            "auth_scope": spec.auth_scope,
            "idempotent": spec.idempotent,
            "params": spec.params,
            "result": spec.result,
            "emitted_by": "create_router",
        }
    }

    router = APIRouter(prefix=prefix, tags=tags)

    if model is None:

        async def endpoint(auth_user: dict = Depends(require_auth_if_enabled)) -> Any:
            return await _dispatch(spec, handler, {})

        input_name = None
    elif verb == _READ:

        async def endpoint(
            query=Depends(), auth_user: dict = Depends(require_auth_if_enabled)
        ) -> Any:
            return await _dispatch(spec, handler, query.model_dump())

        input_name = "query"
    else:

        async def endpoint(
            payload=Body(...), auth_user: dict = Depends(require_auth_if_enabled)
        ) -> Any:
            return await _dispatch(spec, handler, payload.model_dump())

        input_name = "payload"

    annotations: dict[str, Any] = {"auth_user": dict, "return": Any}
    if input_name is not None and model is not None:
        annotations[input_name] = model
    endpoint.__annotations__ = annotations

    router.add_api_route(
        path,
        endpoint,
        methods=[verb],
        openapi_extra=openapi_extra,
        summary=spec.description or spec.name,
    )
    return router
