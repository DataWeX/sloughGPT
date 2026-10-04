"""Contract → transport projection: one descriptor, every crossing declared from it.

``create_router(spec, route, handler)`` emits a FastAPI route fragment from a
capability's ``ToolSpec`` — verb, auth, request validation, response envelope and
error classification are *derived*, never re-decided per file. See
``docs/TRANSPORT_PROJECTIONS.md`` for the design and AGENTS.md
"Endpoint & Transport Rule" for the policy.

Placement: the server/transport layer only. The descriptor stays
capability-only — HTTP facts (``path``/``method``) live in :class:`RouteSpec`,
so transport never leaks back into the domain contract and ``domain`` keeps no
FastAPI import.

Two tool tiers, one contract: this module builds *site* routes. Model-facing
tools keep going through ``domain.agents.ToolRegistry`` — one descriptor may
feed both projections (the same ``execute`` callable), but the chat tool list
and these routes are never mixed.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, Body, Depends, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, create_model
from schemas.common import classify_and_raise, success_response

from domain.agents import ToolSpec
from infrastructure.auth import require_auth_if_enabled

logger = logging.getLogger("slo.contract")

__all__ = [
    "REGISTRY",
    "ContractRegistry",
    "RouteEntry",
    "RouteSpec",
    "create_router",
    "derive_method",
]

CapabilityHandler = Callable[..., Coroutine[Any, Any, Any]]

#: The verb a descriptor-derived route gets when ``RouteSpec.method`` is silent:
#: idempotent reads are safe to retry and cache (GET), actions are not (POST).
_READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_KNOWN_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"})
_JSON_TO_PY: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def derive_method(spec: ToolSpec) -> str:
    """Default verb from the descriptor's idempotency — a default, not a decree.

    An explicit ``RouteSpec.method`` overrides it: verb semantics are
    safety-relevant (retry + cache behaviour), so the override is stated at the
    call site and logged, never silently applied.
    """
    return "GET" if spec.idempotent else "POST"


@dataclass(frozen=True)
class RouteSpec:
    """Projection facts — how one capability crosses *this* transport.

    Kept separate from ``ToolSpec`` on purpose: the descriptor declares what the
    capability is, this declares how HTTP carries it. Neither imports the other's
    vocabulary.
    """

    path: str
    method: str | None = None
    summary: str = ""
    tags: tuple[str, ...] = ()
    # Explicit success status for this crossing (e.g. 201 for creates).
    # None keeps FastAPI's default 200 — existing projections are unaffected.
    status_code: int | None = None


@dataclass(frozen=True)
class RouteEntry:
    """One registered capability crossing — the row ``GET /contracts`` serves."""

    name: str
    version: str
    auth_scope: str
    idempotent: bool
    description: str
    method: str
    path: str
    operation_id: str
    module: str
    params: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "auth_scope": self.auth_scope,
            "idempotent": self.idempotent,
            "description": self.description,
            "method": self.method,
            "path": self.path,
            "operation_id": self.operation_id,
            "module": self.module,
            "params": self.params,
            "result": self.result,
        }


class ContractRegistry:
    """Boot-time registry of descriptor-projection routes.

    Owns the two invariants a projected transport must hold: no two capabilities
    claim the same ``(method, path)``, and no two operations claim the same
    ``operation_id``. Violations raise at import time — a route collision should
    break boot, not discover itself at 404 in production.
    """

    def __init__(self) -> None:
        self._routes: dict[tuple[str, str], RouteEntry] = {}
        self._operation_ids: dict[str, tuple[str, str]] = {}

    def register(self, entry: RouteEntry) -> None:
        key = (entry.method, entry.path)
        existing = self._routes.get(key)
        if existing is not None:
            raise ValueError(
                f"duplicate contract route {entry.method} {entry.path}: "
                f"'{existing.name}' ({existing.module}) already owns it"
            )
        self._routes[key] = entry

    def claim_operation_id(self, base: str, method: str, path: str) -> str:
        """Reserve a unique operation_id, disambiguating repeats with a suffix."""
        candidate = base
        n = 2
        while candidate in self._operation_ids:
            other = self._operation_ids[candidate]
            if other == (method, path):
                return candidate
            candidate = f"{base}_{n}"
            n += 1
        self._operation_ids[candidate] = (method, path)
        return candidate

    def routes(self, prefix: str | None = None) -> list[dict[str, Any]]:
        """Sorted contract inventory (what ``GET /contracts`` returns)."""
        entries = sorted(self._routes.values(), key=lambda e: (e.path, e.method))
        if prefix:
            entries = [e for e in entries if e.path.startswith(prefix)]
        return [e.as_dict() for e in entries]

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._routes


REGISTRY = ContractRegistry()


def _join(prefix: str, path: str) -> str:
    """Effective route path — exactly FastAPI's ``prefix + path`` join.

    Mirrored deliberately: any normalization here would make the registry key
    differ from the route FastAPI actually serves, and the runtime cross-check
    would start lying.
    """
    return f"{prefix}{path}" or "/"


def _coerce(json_type: str, value: str) -> Any:
    """Cast one query-string value to the contract's declared JSON Schema type."""
    if json_type == "string":
        return value
    if json_type == "integer":
        return int(value)
    if json_type == "number":
        return float(value)
    if json_type == "boolean":
        low = value.lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"expected a boolean, got {value!r}")
    if json_type in {"object", "array"}:
        return json.loads(value)
    return value


def _query_dependency(spec: ToolSpec):
    """Build the read-side dependency: validate query params against the contract.

    The descriptor's ``parameters`` *are* the query schema — strings arrive as
    strings, numbers/booleans/objects are coerced to their declared types, and
    a missing required or unparseable value is a 422, not a downstream
    ``ValueError``. Keys outside the contract are ignored (cache-busters etc.).
    """

    async def dependency(request: Request) -> dict[str, Any]:
        raw = dict(request.query_params)
        out: dict[str, Any] = {}
        errors: list[dict[str, Any]] = []
        for p in spec.parameters:
            if p.name not in raw:
                if p.required:
                    errors.append(
                        {"loc": ("query", p.name), "msg": "Field required", "type": "missing"}
                    )
                continue
            try:
                out[p.name] = _coerce(p.type, raw[p.name])
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(
                    {
                        "loc": ("query", p.name),
                        "msg": f"Invalid value for {p.type}: {exc}",
                        "type": "value_error",
                    }
                )
        if errors:
            raise RequestValidationError(errors)
        return out

    return dependency


def _request_model(spec: ToolSpec) -> type[BaseModel]:
    """Generate the write-side body model from the descriptor's parameters."""
    fields: dict[str, tuple[Any, Any]] = {}
    for p in spec.parameters:
        ann = _JSON_TO_PY.get(p.type, Any)
        default: Any = ... if p.required else None
        fields[p.name] = (ann, default)
    schema_name = f"{spec.name.replace('.', '_')}_v{spec.version}"
    return create_model(schema_name, **fields)  # type: ignore[call-overload]


def _check_handler_accepts(spec: ToolSpec, handler: CapabilityHandler) -> None:
    """Boot-time invariant: the handler can receive every declared parameter."""
    try:
        sig = inspect.signature(handler)
    except (TypeError, ValueError):  # builtins / C callables — nothing to check
        return
    params = sig.parameters
    if any(p.name == "request" for p in spec.parameters):
        raise ValueError(
            f"'request' is reserved by the projection (injected raw Request) — "
            f"rename the '{spec.name}' parameter"
        )
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    missing = [p.name for p in spec.parameters if p.name not in params]
    if missing:
        raise ValueError(
            f"handler for '{spec.name}' does not accept declared parameter(s) {missing}; "
            f"the contract is the call signature — adapt the handler, not the descriptor"
        )


def _openapi_extra(spec: ToolSpec, method: str) -> dict[str, Any]:
    """Operation-level contract metadata: ``x-contract`` + projected parameters.

    Read-side parameters are validated by our own dependency (FastAPI never
    sees them as function parameters), so they are published here — otherwise
    the OpenAPI view of the contract would silently omit what it enforces.
    """
    extra: dict[str, Any] = {
        "x-contract": {
            "name": spec.name,
            "version": spec.version,
            "auth_scope": spec.auth_scope,
            "idempotent": spec.idempotent,
        }
    }
    if method in _READ_METHODS and spec.parameters:
        extra["parameters"] = [
            {
                "name": p.name,
                "in": "query",
                "required": p.required,
                "description": p.description,
                "schema": {"type": p.type},
            }
            for p in spec.parameters
        ]
    return extra


def _build_endpoint(
    spec: ToolSpec,
    handler: CapabilityHandler,
    method: str,
    source: str,
) -> CapabilityHandler:
    """Return the FastAPI endpoint function for one projected route.

    Shape: validate (contract-derived) → call → envelope (``success_response``)
    → classify (``classify_and_raise``). The try/except is async by design: the
    awaited handler raises *inside* it, so classification always fires.
    """
    wants_request = _wants_request(handler)
    authed = (spec.auth_scope or "public") != "public"

    async def _dispatch(payload: Any, request: Request) -> dict[str, Any]:
        try:
            data: dict[str, Any] = (
                payload.model_dump() if isinstance(payload, BaseModel) else dict(payload or {})
            )
            if wants_request:
                data["request"] = request
            result = handler(**data)
            if inspect.isawaitable(result):
                result = await result
            return success_response(data=result)
        except Exception as exc:
            classify_and_raise(exc, source=source)

    if method in _READ_METHODS:
        dep = _query_dependency(spec)
        if authed:

            async def read_endpoint(
                request: Request,
                payload: dict[str, Any] = Depends(dep),
                auth_user: dict | None = Depends(require_auth_if_enabled),
            ) -> dict[str, Any]:
                return await _dispatch(payload, request)

        else:

            async def read_endpoint(  # noqa: F811 — scoped twin, public variant
                request: Request,
                payload: dict[str, Any] = Depends(dep),
            ) -> dict[str, Any]:
                return await _dispatch(payload, request)

        return read_endpoint

    model = _request_model(spec)
    # A contract with no *required* parameter does not demand a body: a
    # descriptor-projected DELETE/POST carries its identity in the path (or
    # nothing at all), and bodyless writes must not 422. Body(...) stays for
    # any contract that declares a required field — existing projections are
    # byte-identical (they all declare required params or are reads).
    body: Any = Body(...) if any(p.required for p in spec.parameters) else Body(None)
    if authed:

        async def write_endpoint(
            request: Request,
            payload: Any = body,
            auth_user: dict | None = Depends(require_auth_if_enabled),
        ) -> dict[str, Any]:
            return await _dispatch(payload, request)

    else:

        async def write_endpoint(  # noqa: F811 — scoped twin, public variant
            request: Request,
            payload: Any = body,
        ) -> dict[str, Any]:
            return await _dispatch(payload, request)

    # Contract-derived body schema: FastAPI validates against it *and* publishes
    # it in the OpenAPI request body (same declaration the model sees).
    write_endpoint.__annotations__["payload"] = model
    return write_endpoint


def _wants_request(handler: CapabilityHandler) -> bool:
    """Whether the handler takes the raw ``Request`` (auth, client ip, streaming)."""
    try:
        return "request" in inspect.signature(handler).parameters
    except (TypeError, ValueError):
        return False


def create_router(
    spec: ToolSpec,
    route: RouteSpec | Sequence[tuple[RouteSpec, CapabilityHandler]],
    handler: CapabilityHandler | None = None,
    *,
    prefix: str = "",
    tags: Sequence[str] = (),
    registry: ContractRegistry | None = None,
) -> APIRouter:
    """Project a capability descriptor into a FastAPI route fragment.

    One declaration feeds everything: verb (``derive_method``), auth (the
    descriptor's ``auth_scope`` → ``require_auth_if_enabled``), request
    validation (``spec.parameters``), response envelope, error classification and
    OpenAPI metadata. A module registers by calling this once at import — that
    *is* the boot-time registration: no central endpoint list, no hand-laid route.

    Args:
        spec: The capability contract (model-facing and site-facing both).
        route: A single ``RouteSpec``, or ``[(RouteSpec, handler), ...]`` for a
            capability crossing more than once.
        handler: The capability callable (required for a single ``RouteSpec``).
            Must accept every declared parameter; ``request`` is injected when
            declared.

    Returns:
        An ``APIRouter`` for the module to expose as ``router``.

    Raises:
        ValueError: unknown verb, handler/contract mismatch, or a duplicate
            ``(method, path)`` — at import time, not first request.
    """
    reg = registry if registry is not None else REGISTRY

    if isinstance(route, RouteSpec):
        if handler is None:
            raise ValueError("create_router(spec, route, handler): handler is required")
        entries: list[tuple[RouteSpec, CapabilityHandler]] = [(route, handler)]
    else:
        entries = list(route)
        if any(h is None for _, h in entries):
            raise ValueError("create_router: every entry needs a handler")

    if not spec.name or not spec.version or not spec.auth_scope:
        raise ValueError(
            f"incomplete descriptor {spec.name!r}: name, version and auth_scope are the contract"
        )

    router = APIRouter(prefix=prefix, tags=list(tags))
    source = f"contract.{spec.name}"

    for route_spec, route_handler in entries:
        method = (route_spec.method or derive_method(spec)).upper()
        if method not in _KNOWN_METHODS:
            raise ValueError(f"{spec.name}: unknown HTTP method {method!r}")
        derived = derive_method(spec)
        if method != derived:
            logger.info(
                "%s declares %s %s explicitly (idempotent=%s derives %s)",
                spec.name,
                method,
                _join(prefix, route_spec.path),
                spec.idempotent,
                derived,
            )

        _check_handler_accepts(spec, route_handler)
        path = _join(prefix, route_spec.path)
        operation_id = reg.claim_operation_id(f"{spec.name}.{method.lower()}", method, path)
        reg.register(
            RouteEntry(
                name=spec.name,
                version=spec.version,
                auth_scope=spec.auth_scope,
                idempotent=spec.idempotent,
                description=spec.description,
                method=method,
                path=path,
                operation_id=operation_id,
                module=getattr(route_handler, "__module__", source),
                params=spec.params,
                result=spec.result,
            )
        )

        endpoint = _build_endpoint(spec, route_handler, method, source)
        router.add_api_route(
            route_spec.path,
            endpoint,
            methods=[method],
            summary=route_spec.summary or spec.description,
            description=spec.description,
            operation_id=operation_id,
            tags=list(route_spec.tags) if route_spec.tags else None,
            status_code=route_spec.status_code,
            responses={
                422: {"description": "Contract violation (missing or mistyped parameter)"},
            },
            openapi_extra=_openapi_extra(spec, method),
        )
        logger.debug("projected %s %s -> %s", method, path, spec.name)

    return router
