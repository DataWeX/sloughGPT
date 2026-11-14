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
import re
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, Union, get_args, get_origin

from fastapi import APIRouter, Body, Depends, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, create_model
from schemas.common import classify_and_raise, success_response

from domain.agents import ToolParam, ToolSpec
from infrastructure.auth import require_auth_if_enabled

logger = logging.getLogger("slo.contract")

__all__ = [
    "REGISTRY",
    "ContractRegistry",
    "RouteEntry",
    "RouteSpec",
    "create_router",
    "derive_method",
    "get_contracts",
    "params_from_model",
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
    #: Raw-wire facet (ruling 2026-11-05, card ``20261104_151035``): ``True``
    #: wraps the handler result in the standard ``{status, data}`` envelope;
    #: ``False`` returns it untouched — for wires that must stay byte-identical
    #: (SSE ``StreamingResponse`` passthrough, endpoints whose callers parse a
    #: bare payload). The default keeps the pilot projection unchanged.
    envelope: bool = True
    #: Input carrier override: ``"query"`` validates the declared parameters as
    #: query strings (even on a write verb — ``POST .../purge?status=...``),
    #: ``"body"`` as a JSON body. ``None`` derives it from the method, which is
    #: what FastAPI itself does for hand-built routes.
    input: str | None = None


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
            if existing.name == entry.name and existing.module != entry.module:
                # Same capability re-claiming its own crossing from a second
                # import identity (``training.control`` vs
                # ``apps.api.server.training.control`` — tests import and
                # patch via both, so the file executes twice). Idempotent
                # re-registration, not a collision; the first row stands.
                # Same identity re-registering is still an error: that is a
                # double projection of one route.
                return
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


def get_contracts() -> list[dict[str, Any]]:
    """Every descriptor-projected HTTP contract registered so far, sorted.

    The read-back half of the projection: ``GET /contracts`` serves the same
    inventory filtered by prefix; ``scripts/gen_ts_contracts.py`` reads it whole
    to emit the TS bindings.
    """
    return REGISTRY.routes()


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


def _query_dependency(spec: ToolSpec, path_params: frozenset[str] = frozenset()):
    """Build the read-side dependency: validate query params against the contract.

    The descriptor's ``parameters`` *are* the query schema — strings arrive as
    strings, numbers/booleans/objects are coerced to their declared types, and
    a missing required or unparseable value is a 422, not a downstream
    ``ValueError``. Keys outside the contract are ignored (cache-busters etc.).
    ``path_params`` are bound by the URL template instead, never by the query.
    """

    async def dependency(request: Request) -> dict[str, Any]:
        raw = dict(request.query_params)
        out: dict[str, Any] = {}
        errors: list[dict[str, Any]] = []
        for p in spec.parameters:
            if p.name in path_params:
                continue
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


def _request_model(spec: ToolSpec, path_params: frozenset[str] = frozenset()) -> type[BaseModel]:
    """Generate the write-side body model from the descriptor's parameters.

    Path params belong to the URL template, not the body — they are excluded
    here and bound from ``request.path_params`` at dispatch time.
    """
    fields: dict[str, tuple[Any, Any]] = {}
    for p in spec.parameters:
        if p.name in path_params:
            continue
        ann = _JSON_TO_PY.get(p.type, Any)
        default: Any = ... if p.required else None
        fields[p.name] = (ann, default)
    schema_name = f"{spec.name.replace('.', '_')}_v{spec.version}"
    return create_model(schema_name, **fields)  # type: ignore[call-overload]


_ANN_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _json_type_of(annotation: Any) -> str:
    """JSON Schema type for a field annotation — unions unwrap to their member.

    ``str | None`` documents as ``string`` (nullability is a defaulting detail,
    not a second type); unknown composites fall back to ``object``, the
    loosest honest answer in ``ToolParam.type``'s vocabulary.
    """
    if isinstance(annotation, type):
        return _ANN_TO_JSON.get(annotation, "object")
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        members = [a for a in get_args(annotation) if a is not type(None)]
        if len(members) == 1:
            return _json_type_of(members[0])
        return "object"
    if origin in (list, tuple, set, frozenset):
        return "array"
    return "object"


def params_from_model(model: type[BaseModel]) -> list[ToolParam]:
    """Derive descriptor ``parameters`` from a pydantic request model.

    Model-style routes (ruling ``c3c81a311``): the model *is* the runtime
    validator — constraints and all — while the descriptor's ``parameters``
    document its fields for the registry/TS projection. Deriving them from
    ``model_fields`` keeps one declaration (the model) feeding both views, so
    a field added to the model can never drift from the published contract.
    """
    return [
        ToolParam(
            name=name,
            type=_json_type_of(field.annotation),
            description=field.description or "",
            required=field.is_required(),
        )
        for name, field in model.model_fields.items()
    ]


def _resolve_ann(param: inspect.Parameter, handler: CapabilityHandler) -> Any:
    """The parameter's annotation as a live object.

    ``from __future__ import annotations`` turns every annotation into a
    string; model-style detection, raw-Request recognition and path coercion
    all need the real class. Evaluated in the handler's own module namespace
    (where the name was written), falling back to the raw text when it does
    not resolve.
    """
    ann = param.annotation
    if not isinstance(ann, str):
        return ann
    globs = getattr(handler, "__globals__", None)
    if not globs:
        return ann
    try:
        return eval(ann, globs)  # noqa: S307 — annotation text from our own source
    except Exception:  # noqa: BLE001 — an unresolvable name degrades to the raw text
        return ann


def _is_raw_request(param: inspect.Parameter, handler: CapabilityHandler) -> bool:
    """``request`` as the raw FastAPI Request (not a pydantic body model)."""
    ann = _resolve_ann(param, handler)
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return False
    if isinstance(ann, type):
        return issubclass(ann, Request)
    return param.name == "request"  # unannotated — the name is the convention


def _flat_params(
    sig: inspect.Signature, path_params: frozenset[str], handler: CapabilityHandler
) -> dict[str, inspect.Parameter]:
    """Handler params the *contract* must supply — excludes transport injections."""
    return {
        name: p
        for name, p in sig.parameters.items()
        if name not in path_params and name != "auth_user" and not _is_raw_request(p, handler)
    }


def _model_param(
    handler: CapabilityHandler, path_params: frozenset[str] = frozenset()
) -> tuple[str, type[BaseModel]] | None:
    """The handler's single pydantic body parameter, if it takes exactly one.

    Model-style body: the annotation *is* the validator — constraints
    (``ge/le/pattern``) stay enforced by the same class the hand-built route
    used, and the projection passes the validated instance straight through.
    """
    try:
        sig = inspect.signature(handler)
        flats = _flat_params(sig, path_params, handler)
    except (TypeError, ValueError):
        return None
    models = [
        (name, ann)
        for name, p in flats.items()
        if isinstance(ann := _resolve_ann(p, handler), type) and issubclass(ann, BaseModel)
    ]
    if len(models) == 1 and len(flats) == 1:
        return models[0]
    return None


def _wants_auth_user(handler: CapabilityHandler) -> bool:
    """Whether the handler reads the decoded token payload (``auth_user``)."""
    try:
        return "auth_user" in inspect.signature(handler).parameters
    except (TypeError, ValueError):
        return False


def _path_types(handler: CapabilityHandler, path_params: frozenset[str]) -> dict[str, str]:
    """JSON Schema type per path placeholder, from the handler's annotation."""
    if not path_params:
        return {}
    try:
        params = inspect.signature(handler).parameters
    except (TypeError, ValueError):
        return dict.fromkeys(path_params, "string")
    out: dict[str, str] = {}
    for name in path_params:
        ann = _resolve_ann(params[name], handler) if name in params else None
        out[name] = _ANN_TO_JSON.get(ann, "string")
    return out


def _check_handler_accepts(
    spec: ToolSpec, handler: CapabilityHandler, path_params: frozenset[str] = frozenset()
) -> None:
    """Boot-time invariant: the handler can receive every declared parameter."""
    try:
        sig = inspect.signature(handler)
    except (TypeError, ValueError):  # builtins / C callables — nothing to check
        return
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return  # **kwargs absorbs anything the contract sends
    flats = _flat_params(sig, path_params, handler)
    has_model = any(
        isinstance(ann := _resolve_ann(p, handler), type) and issubclass(ann, BaseModel)
        for p in flats.values()
    )
    if has_model and _model_param(handler, path_params) is None:
        raise ValueError(
            f"handler for '{spec.name}' mixes a pydantic body parameter with flat "
            f"parameters; the projection takes one body style per route — split "
            f"the handler or declare the fields flat"
        )
    missing_path = sorted(name for name in path_params if name not in params)
    if missing_path:
        raise ValueError(
            f"handler for '{spec.name}' does not accept path parameter(s) "
            f"{missing_path} declared in the route template; the path is the "
            f"call signature — adapt the handler, not the RouteSpec"
        )
    if _model_param(handler, path_params) is not None:
        # Model-style: spec.parameters document the model's fields for the
        # registry/TS projection; the model itself validates them at runtime.
        return
    if any(p.name == "request" for p in spec.parameters):
        raise ValueError(
            f"'request' is reserved by the projection (injected raw Request) — "
            f"rename the '{spec.name}' parameter"
        )
    missing = [p.name for p in spec.parameters if p.name not in params]
    if missing:
        raise ValueError(
            f"handler for '{spec.name}' does not accept declared parameter(s) {missing}; "
            f"the contract is the call signature — adapt the handler, not the descriptor"
        )


def _openapi_extra(
    spec: ToolSpec,
    path_params: frozenset[str] = frozenset(),
    path_types: dict[str, str] | None = None,
    input_style: str = "query",
) -> dict[str, Any]:
    """Operation-level contract metadata: ``x-contract`` + projected parameters.

    Read-side (and ``input="query"``) parameters are validated by our own
    dependency — FastAPI never sees them as function parameters — so they are
    published here; otherwise the OpenAPI view of the contract would silently
    omit what it enforces. Path placeholders are published as ``in: path`` for
    the same reason: the endpoint signature does not carry them.
    """
    extra: dict[str, Any] = {
        "x-contract": {
            "name": spec.name,
            "version": spec.version,
            "auth_scope": spec.auth_scope,
            "idempotent": spec.idempotent,
        }
    }
    parameters: list[dict[str, Any]] = [
        {
            "name": name,
            "in": "path",
            "required": True,
            "schema": {"type": (path_types or {}).get(name, "string")},
        }
        for name in sorted(path_params)
    ]
    if input_style == "query" and spec.parameters:
        parameters.extend(
            {
                "name": p.name,
                "in": "query",
                "required": p.required,
                "description": p.description,
                "schema": {"type": p.type},
            }
            for p in spec.parameters
            if p.name not in path_params
        )
    if parameters:
        extra["parameters"] = parameters
    return extra


def _build_endpoint(
    spec: ToolSpec,
    handler: CapabilityHandler,
    source: str,
    *,
    path_params: frozenset[str] = frozenset(),
    path_types: dict[str, str] | None = None,
    envelope: bool = True,
    input_style: str = "query",
) -> CapabilityHandler:
    """Return the FastAPI endpoint function for one projected route.

    Shape: validate (contract-derived) → call → envelope → classify
    (``classify_and_raise``). The try/except is async by design: the awaited
    handler raises *inside* it, so classification always fires.

    Binding order: payload (body model or query dict) → path placeholders →
    raw ``Request`` → ``auth_user`` — exactly what the hand-built signature
    declared, now derived from the descriptor and the handler's annotations.
    ``envelope=False`` returns the handler result untouched (raw-wire facets
    keep ``StreamingResponse`` passthrough and bare payload shapes working).
    """
    wants_request = _wants_request(handler)
    wants_auth_user = _wants_auth_user(handler)
    authed = (spec.auth_scope or "public") != "public"
    model_style = _model_param(handler, path_params)
    types = path_types or {}
    # A body exists only when there is something to validate: a no-arg write
    # gets no payload parameter at all (hand-built ones ignore a sent body).
    # That is also the only way to keep FastAPI from auto-declaring a 422 for
    # a validation that cannot fail — it 422s on *any* body field, even an
    # optional empty one, while a no-param hand-built route serves 200 only.
    has_body = model_style is not None or any(p.name not in path_params for p in spec.parameters)

    def _bind(payload: Any, request: Request, auth_user: dict | None) -> dict[str, Any]:
        if model_style is not None:
            data: dict[str, Any] = {model_style[0]: payload} if payload is not None else {}
        elif isinstance(payload, BaseModel):
            data = payload.model_dump()
        else:
            data = dict(payload or {})
        for name in path_params:
            raw = request.path_params.get(name, "")
            try:
                data[name] = _coerce(types.get(name, "string"), raw)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RequestValidationError(
                    [
                        {
                            "loc": ("path", name),
                            "msg": f"Invalid value for {types.get(name, 'string')}: {exc}",
                            "type": "value_error",
                        }
                    ]
                ) from exc
        if wants_request:
            data["request"] = request
        if wants_auth_user:
            data["auth_user"] = auth_user
        return data

    async def _dispatch(payload: Any, request: Request, auth_user: dict | None) -> Any:
        try:
            data = _bind(payload, request, auth_user)
            result = handler(**data)
            if inspect.isawaitable(result):
                result = await result
            return success_response(data=result) if envelope else result
        except Exception as exc:
            classify_and_raise(exc, source=source)

    if input_style == "query" or not has_body:
        # Query-carried — or nothing to carry: a no-param write rides the same
        # payload-free shape with an empty spec (its dependency returns {}).
        dep = _query_dependency(spec, path_params)
        if authed:

            async def query_endpoint(
                request: Request,
                payload: dict[str, Any] = Depends(dep),
                auth_user: dict | None = Depends(require_auth_if_enabled),
            ) -> dict[str, Any]:
                return await _dispatch(payload, request, auth_user)

        else:

            async def query_endpoint(  # noqa: F811 — scoped twin, public variant
                request: Request,
                payload: dict[str, Any] = Depends(dep),
            ) -> dict[str, Any]:
                return await _dispatch(payload, request, None)

        endpoint = query_endpoint
    else:
        body_model = (
            model_style[1] if model_style is not None else _request_model(spec, path_params)
        )
        body_default: Any = Body(...)
        if model_style is not None:
            # A model body whose handler declares a default is an OPTIONAL
            # body — FastAPI's own semantics for `req: Model = None`. The
            # hand-built route served an absent body as None → handler
            # defaults; forcing `Body(...)` here flipped /self-train/start
            # to 422 on a bodyless POST (caught by CI, card 20261101_154058).
            _model_sig = inspect.signature(handler).parameters.get(model_style[0])
            if _model_sig is not None and _model_sig.default is None:
                body_default = Body(None)
        if authed:

            async def write_endpoint(
                request: Request,
                payload: Any = body_default,
                auth_user: dict | None = Depends(require_auth_if_enabled),
            ) -> dict[str, Any]:
                return await _dispatch(payload, request, auth_user)

        else:

            async def write_endpoint(  # noqa: F811 — scoped twin, public variant
                request: Request,
                payload: Any = body_default,
            ) -> dict[str, Any]:
                return await _dispatch(payload, request, None)

        # Contract-derived body schema: FastAPI validates against it *and*
        # publishes it in the OpenAPI request body (same declaration the
        # model sees — constraints included, model-style).
        write_endpoint.__annotations__["payload"] = body_model
        endpoint = write_endpoint

    if not envelope:
        # Raw wire: no response_model — the payload shape is whatever the
        # handler returned, exactly as the hand-built route served it.
        endpoint.__annotations__.pop("return", None)
    return endpoint


def _wants_request(handler: CapabilityHandler) -> bool:
    """Whether the handler takes the raw ``Request`` (auth, client ip, streaming).

    Annotation-aware: ``request`` annotated with a pydantic model is a *body*
    parameter (model-style), not the raw Request — ``load_adapter(request:
    LoadAdapterRequest)`` must receive its validated model, not Starlette's.
    """
    try:
        param = inspect.signature(handler).parameters.get("request")
    except (TypeError, ValueError):
        return False
    if param is None:
        return False
    ann = _resolve_ann(param, handler)
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return False
    return True


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
            Must accept every declared parameter; the raw ``Request`` and the
            decoded ``auth_user`` are injected when declared, and every
            ``{placeholder}`` in the path must be an accepted parameter.

    Route-level facets: ``RouteSpec.envelope=False`` serves the handler result
    unwrapped (raw wire / SSE passthrough); ``RouteSpec.input="query"``
    validates declared parameters as query strings even on a write verb.

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

        path_params = frozenset(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", route_spec.path))
        authed = (spec.auth_scope or "public") != "public"
        _check_handler_accepts(spec, route_handler, path_params)
        if _wants_auth_user(route_handler) and not authed:
            raise ValueError(
                f"handler for '{spec.name}' reads auth_user but the descriptor "
                f"auth_scope is public — declare the scope so the projection "
                f"can enforce and forward the token"
            )
        input_style = route_spec.input or ("query" if method in _READ_METHODS else "body")
        if input_style not in {"query", "body"}:
            raise ValueError(
                f"{spec.name}: unknown input style {input_style!r} (expected query|body)"
            )
        if input_style == "query" and _model_param(route_handler, path_params):
            raise ValueError(
                f"{spec.name}: a pydantic body parameter cannot ride a "
                f"query-input route — the model is the body validator"
            )
        path_types = _path_types(route_handler, path_params)
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

        endpoint = _build_endpoint(
            spec,
            route_handler,
            source,
            path_params=path_params,
            path_types=path_types,
            envelope=route_spec.envelope,
            input_style=input_style,
        )
        # 422 only when validation can actually fail — no parameters, no path
        # placeholders and no body model serves 200 alone, exactly what the
        # hand-built twin published (FastAPI derives the same way).
        validates = (
            bool(spec.parameters)
            or bool(path_params)
            or _model_param(route_handler, path_params) is not None
        )
        router.add_api_route(
            route_spec.path,
            endpoint,
            methods=[method],
            summary=route_spec.summary or spec.description,
            description=spec.description,
            operation_id=operation_id,
            tags=list(route_spec.tags) if route_spec.tags else None,
            responses=(
                {422: {"description": "Contract violation (missing or mistyped parameter)"}}
                if validates
                else None
            ),
            openapi_extra=_openapi_extra(spec, path_params, path_types, input_style),
        )
        logger.debug("projected %s %s -> %s", method, path, spec.name)

    return router
