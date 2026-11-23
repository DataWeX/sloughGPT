"""
Config-drift precondition — fail closed when the client's versioned config
disagrees with the server's current document.

Card 20261108_222308 (FungoLabs gap-audit pattern #9): generated contracts are
freshness-gated at build time, but HAND-WRITTEN config/limits/flags could
silently drift — a client calibrated on config version N-1 acting against a
server already at N. FungoLabs halts the transfer with a sentence; this
middleware halts the ACTION the same way, BEFORE anything irreversible runs
(``E_CONFIG_DRIFT``, 412).

Mechanism (one check, every action route — no per-route conveyor belt):

    Unsafe-method requests MAY present ``X-Slo-Config-Version`` — the sha256
    ``config_version`` the client holds from ``GET /runtime-config`` (sibling
    card 20261108_134211; same value as that route's ETag).

    * presented == server's current ``build_document()["config_version"]``
      → proceed.
    * presented != current (stale N-1, a hash this server never issued, or an
      empty claim) → 412 with the registry's explicit refusal sentence. The
      server keeps no version history: the current hash is the only known one,
      so ONE comparison covers both card failure modes (disagreement and
      unknown version).
    * absent → no claim → pass-through. CLI/SDK/mobile/tests hold no versioned
      config; drift is a property of clients that opted into the document, and
      the web client always presents one (``apps/web/lib/runtime-config.ts``).

    GET/HEAD/OPTIONS are never checked — reads are the RECOVERY path a stale
    client refreshes through, and display-only values are out of the card's
    scope (the hash material already excludes description/since).

Layering: ``routers.runtime_config`` is imported lazily at dispatch time —
the same precedent as ``UnifiedRequestMiddleware``'s ``from routers.inference
import _get_model_status`` — so this module stays importable without the full
app (test apps build their own stacks).

Tier: transport infrastructure (same family as rate limiting and correlation
IDs) — it declares no capability and never enters the model-facing tool
registry, ``domain/tools``, or anything the model reasons over.
"""

from __future__ import annotations

from fastapi.responses import JSONResponse
from schemas.common import error_response
from starlette.middleware.base import BaseHTTPMiddleware

from domain.infrastructure._internal.errors import ERROR_REGISTRY, ErrorCode

#: Header carrying the client's held ``config_version`` (sha256 hex).
CONFIG_VERSION_HEADER = "X-Slo-Config-Version"

#: Only actions can strand a stale client in front of an irreversible call —
#: reads are how a stale client recovers, so they are never refused.
_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

#: Single source: the registry entry the whole stack renders on refusal —
#: (class_name, http_status, recoverable, user_message).
_DRIFT = ERROR_REGISTRY[ErrorCode.E_CONFIG_DRIFT]


class ConfigDriftMiddleware(BaseHTTPMiddleware):
    """Refuse an action whose presented config_version is not the current one.

    Returns the unified error envelope (``error_response``) with 412 — the
    precondition-failed status, which ``apps/lib/retry.ts`` deliberately does
    not retry (412 ∉ RETRYABLE_STATUSES): a drift refusal must never be
    replayed blindly.
    """

    async def dispatch(self, request, call_next):
        # Reads pass unconditionally — they are the recovery path.
        if request.method not in _UNSAFE_METHODS:
            return await call_next(request)
        # ``in`` (not truthiness): a PRESENTED claim must be validated even if
        # it is empty — fail closed on a malformed claim, not on its absence.
        if CONFIG_VERSION_HEADER not in request.headers:
            return await call_next(request)
        presented = request.headers[CONFIG_VERSION_HEADER]

        # Lazy import (routers-in-dispatch precedent, middleware.py line ~146):
        # the hash must be recomputed per check — staleness is the failure.
        from routers.runtime_config import build_document

        current = build_document()["config_version"]
        if presented == current:
            return await call_next(request)

        # Nothing downstream runs: the action is refused pre-handler, with the
        # card's required explicit sentence (registry user_message).
        return JSONResponse(
            status_code=_DRIFT[1],
            content=error_response(
                _DRIFT[3],
                code=ErrorCode.E_CONFIG_DRIFT.value,
                details={"client_version": presented, "server_version": current},
            ),
        )
