"""Runtime config crossing — descriptor-projected read of the flag registry.

Card 20261108_134211: the server's feature flags (versioned, ``SLO_FF_*``
env-overridable, ledgered in ``config/flag_history.jsonl``) had **zero**
exposure to the browser — no router read ``feature_flags``. This module serves
the resolved document as ONE descriptor (``runtime_config.get``) so a client
learns what is switchable on without a deploy.

Tier: external utility tooling (a site capability of the app itself). It never
enters the chat tool registry, the ``domain.tools`` engine, or anything the
model reasons over (Endpoint & Transport Rule, two tool tiers over one contract).

No secrets by construction: the payload carries resolved ``enabled`` values and
public flag metadata only — no ``env_var`` names, no settings, no provider
keys (the card's DO NOT clause).

Raw-wire facet (``envelope=False``): the cache ladder needs a real ``Response``
— ETag header, ``Cache-Control: private, max-age=30``, and a conditional 304 —
so the handler serves the document itself instead of the shared envelope;
``apps/lib/request.ts`` accepts a bare payload (it only unwraps when both
``status`` and ``data`` are present).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from infrastructure.contract import RouteSpec, create_router

from domain.agents import ToolSpec
from domain.shared import FeatureFlags


def build_document() -> dict[str, Any]:
    """The capability itself: resolved flags + the sha256 that versions them.

    ``enabled`` is the *resolved* value (an ``SLO_FF_*`` override beats the
    registered status — that is what a consumer acts on); ``status`` rides along
    so a viewer can show the registered intent next to the effective state.
    The material is sorted by name so the hash is stable across processes.
    """
    rows = sorted(
        (
            {
                "name": name,
                "status": flag["status"],
                "enabled": flag["enabled"],
                "version": flag["version"],
                "since": flag["since"],
                "description": flag["description"],
            }
            for name, flag in FeatureFlags.list_all().items()
        ),
        key=lambda row: row["name"],
    )
    material = json.dumps(
        [(row["name"], row["status"], row["version"], row["enabled"]) for row in rows],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return {"config_version": hashlib.sha256(material.encode()).hexdigest(), "flags": rows}


def _if_none_match_hits(header: str, version: str) -> bool:
    """RFC 9110 If-None-Match: ``*``, a comma list, and weak tags all count."""
    if not header:
        return False
    if header.strip() == "*":
        return True
    tags = {part.strip().removeprefix("W/").strip('"') for part in header.split(",")}
    return version in tags


def serve_runtime_config(request: Request) -> Response:
    """HTTP projection: the document plus its cache ladder (ETag / 304 / 30s)."""
    doc = build_document()
    etag = f'"{doc["config_version"]}"'
    headers = {"ETag": etag, "Cache-Control": "private, max-age=30"}
    if _if_none_match_hits(request.headers.get("if-none-match", ""), doc["config_version"]):
        return Response(status_code=304, headers=headers)
    return JSONResponse(content=doc, headers=headers)


SPEC = ToolSpec(
    name="runtime_config.get",
    description=(
        "Resolved runtime configuration for the web client: every feature flag "
        "with its registered status and env-overridden enabled state, versioned "
        "by a sha256 config_version (served as the ETag)."
    ),
    parameters=[],
    execute=build_document,
    auth_scope="authenticated",
    idempotent=True,
    result={
        "type": "object",
        "properties": {
            "config_version": {"type": "string"},
            "flags": {"type": "array", "items": {"type": "object"}},
        },
    },
    version="1",
)

router = create_router(
    SPEC,
    RouteSpec(
        path="/runtime-config",
        envelope=False,  # raw wire: headers + conditional 304 ride a real Response
        summary="Resolved feature-flag document with an ETag cache ladder",
    ),
    serve_runtime_config,
    tags=("config",),
)
