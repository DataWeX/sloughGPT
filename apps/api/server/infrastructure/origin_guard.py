"""First-party origin guard — who may SEND a state-changing request.

Card 20261108_142919 (premise corrected during expert review): there is no
browser→provider path to gate — the provider key lives in backend settings.
The real exposure is WHO MAY CALL: ``SLO_AUTH_REQUIRED`` defaults to off
(anonymous on every endpoint), and CORS only blocks *reading* a response,
not *sending* the request — so a cross-origin SIMPLE POST (form/text-plain,
no preflight) can execute state-changing actions today.

This guard closes that: unsafe methods (POST/PUT/PATCH/DELETE) carrying an
``Origin`` that is neither loopback, nor same-origin (netloc == Host), nor
in ``SLO_CORS_ORIGINS`` are refused with ``403 E_BAD_ORIGIN`` before they
reach a handler. Curl and any other non-browser client send no ``Origin``
and pass untouched.

Deliberate non-goals (the card's must-not-affect list):
  * ``/v1`` — the OpenAI-compatible surface external consumers integrate
    against; path exemption for ``/v1`` and ``/v1/...``.
  * safe methods (GET/HEAD/OPTIONS) — they change no state. SSE streams ride
    GET; BaseHTTPMiddleware never sees WebSocket connections at all.
  * the CORS preflight — ``CORSMiddleware`` is registered outermost in
    main.py and answers OPTIONS before any inner middleware runs; OPTIONS is
    not in the unsafe set either (belt and suspenders).

Placement: registered in ``infrastructure.middleware.get_configured_middleware``
between ``CorrelationIdMiddleware`` (outer, so the 403 envelope carries the
correlation ID) and ``ReadinessGateMiddleware``. The allow rules mirror
main.py's CORS block (untouchable under this card): ``_CORS_LOCAL_ORIGIN_REGEX``
+ the ``SLO_CORS_ORIGINS`` default list — keep the two in sync if either
changes.

``require_first_party_origin`` is the exported per-route projection of the
same rule, for descriptor/route-level use where middleware placement is wrong.
"""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlparse

from fastapi import Header
from fastapi.responses import JSONResponse
from schemas.common import error_response, raise_error
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("slo.infra.origin_guard")

_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Mirror of main.py's CORS rules (this card must not touch main.py):
#   _CORS_LOCAL_ORIGIN_REGEX + the SLO_CORS_ORIGINS default literal.
_CORS_LOCAL_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
_SLO_CORS_ORIGINS_DEFAULT = (
    "http://localhost:3000,http://localhost:5173,http://localhost:5175,"
    "http://127.0.0.1:3000,http://127.0.0.1:5173,http://127.0.0.1:5175,"
    "http://localhost:8000"
)

_V1_PREFIX = "/v1"


def _allowed_origins() -> set[str]:
    """The explicit CORS allow-list, read at call time (env-overridable)."""
    raw = os.environ.get("SLO_CORS_ORIGINS", _SLO_CORS_ORIGINS_DEFAULT)
    return {o.strip() for o in raw.split(",") if o.strip()}


def is_allowed_origin(origin: str, host: str | None) -> bool:
    """True when this ``Origin`` may send unsafe-method requests.

    Three positive rules, mirroring the CORS block:
      1. same-origin — Origin netloc equals the request Host;
      2. loopback — the local-origin regex (any localhost/127.0.0.1/[::1]
         port, env-overridable via ``SLO_CORS_LOCAL_ORIGIN_REGEX``);
      3. explicitly allowed — exact member of ``SLO_CORS_ORIGINS``.
    """
    if host:
        netloc = urlparse(origin).netloc
        if netloc and netloc.lower() == host.lower():
            return True  # same-origin page posting its own backend
    pattern = os.environ.get("SLO_CORS_LOCAL_ORIGIN_REGEX", _CORS_LOCAL_ORIGIN_REGEX)
    if re.fullmatch(pattern, origin):
        return True  # loopback, any port
    return origin in _allowed_origins()


class OriginGuardMiddleware(BaseHTTPMiddleware):
    """403 E_BAD_ORIGIN for unsafe methods with a foreign Origin."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method not in _UNSAFE_METHODS:
            return await call_next(request)
        origin = request.headers.get("origin")
        if not origin:
            return await call_next(request)  # curl / non-browser client
        path = request.url.path
        if path == _V1_PREFIX or path.startswith(_V1_PREFIX + "/"):
            return await call_next(request)  # /v1 external consumers untouched
        if is_allowed_origin(origin, request.headers.get("host")):
            return await call_next(request)

        logger.warning(
            "OriginGuard 403: %s %s blocked (Origin=%s)",
            request.method,
            path,
            origin,
        )
        # Hand-built envelope: raising AppError inside BaseHTTPMiddleware
        # escapes the exception-handler stack and surfaces as a 500.
        return JSONResponse(
            status_code=403,
            content=error_response(
                f"Cross-origin {request.method} from {origin} is not allowed",
                "E_BAD_ORIGIN",
                details={"origin": origin},
            ),
        )


def require_first_party_origin(
    origin: str | None = Header(default=None),
    host: str | None = Header(default=None),
) -> None:
    """Route-level projection of the guard (descriptor/per-route use).

    Raises ``AppError`` (403, ``E_BAD_ORIGIN``) for a foreign Origin; no
    Origin header (curl, server-to-server) passes.
    """
    if not origin:
        return
    if is_allowed_origin(origin, host):
        return
    raise_error(
        f"Cross-origin request from {origin} is not allowed",
        "E_BAD_ORIGIN",
        status_code=403,
        details={"origin": origin},
    )
