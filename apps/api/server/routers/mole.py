"""
Mole router — surface the read-only mole report to the UI.

Contract (the /mole page is built against exactly this):

    GET  /mole/report  -> data: {report: object|null, path: str, age_s: float|null}
    POST /mole/run     -> body {window_s?: float, skip?: [str]} -> data: {report: object}

``GET`` only reads the report file (``$SLO_DOCTOR_REPORT``, default
``~/.cache/slog-doctor/findings-report.json``). A missing or corrupt file
is an empty state, not an error: ``report`` comes back ``null``.

``POST`` runs the LIGHT probes only — ``http`` (read-only GETs) and ``sse``
(6s window) — and folds in the journey findings that are already on disk.
The browser journey sweep is never triggered from the API: ``run_mole``
is called with ``run_sweep=False``, so the journey probe only reads the
existing sweep report. Budget on a responsive stack: preflight 3s (which
skips ``http``/``sse`` outright when the API is dead) + SSE window 6s (plus
a best-effort ≤4s ``/errors/stream`` read) + read-only GETs against the
same API that is serving this request → well inside 15s. No retries, no
writes to the site.

The POST body is optional and so is every field in it: omitting it preserves
the historical contract exactly (6s window, nothing skipped). ``window_s`` is
bounded (1–300s) and ``skip`` must name registered probes only — both enforced
by pydantic (422), never silently ignored. ``run_sweep`` is deliberately NOT a
body field: the API never triggers the browser sweep, whatever the caller asks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, Depends
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel, Field, model_validator
from schemas.common import endpoint, success_response

logger = logging.getLogger("slo.routers.mole")

# SSE observation window for the API-initiated light run (CLI default is 9s;
# the API keeps the interactive run short).
LIGHT_WINDOW_S = 6.0


def default_report_path() -> str:
    """Resolve the report path through the mole package (``$SLO_DOCTOR_REPORT``)."""
    from domain.core import default_report_path as _default

    return _default()


def read_report(path: str) -> tuple[dict | None, float | None]:
    """Read the report file. Missing/corrupt file -> ``(None, None)``.

    ``age_s`` is seconds since the report's ``ts`` (unix seconds); it is
    ``None`` whenever the report itself is unavailable or untimestamped.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError) as exc:
        # Corrupt or unreadable report — an empty state, never a 500.
        logger.debug("mole report unreadable at %s: %s", path, exc)
        return None, None
    if not isinstance(data, dict):
        return None, None

    ts = data.get("ts")
    age_s: float | None = None
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        age_s = max(0.0, time.time() - float(ts))
    return data, age_s


class MoleRunBody(BaseModel):
    """Optional run parameters for ``POST /mole/run`` (light-run contract).

    Everything is optional — an omitted body (or omitted field) means the
    historical behaviour: 6s SSE window, nothing skipped, no browser sweep
    (``run_sweep`` is not expressible here at all).
    """

    window_s: float | None = Field(
        None, ge=1, le=300, description="SSE observation window in seconds (default 6.0)"
    )
    skip: list[str] | None = Field(
        None,
        max_length=6,
        description=(
            "Registered probe names to skip — one of: gates, benchmarks, http, sse, journey, system"
        ),
    )

    @model_validator(mode="after")
    def _known_probes_only(self):
        if self.skip:
            # Lazy import: validator runs per request, boot stays cheap.
            # Through the public facade, never domain.*._internal (router
            # facade guard) — same registry, one import site.
            from domain.core import known_probe_names

            known = set(known_probe_names())
            unknown = {name.strip() for name in self.skip if name.strip()} - known
            if unknown:
                raise ValueError(f"unknown probe(s) in skip: {', '.join(sorted(unknown))}")
        return self


class MoleRouter:
    """Report-surfacing endpoints for the Mole."""

    def __init__(self):
        self.router = APIRouter(prefix="/mole", tags=["mole"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/report", self.get_report, methods=["GET"])
        self.router.add_api_route("/run", self.run_mole, methods=["POST"])

    @endpoint("mole.report")
    async def get_report(self) -> dict:
        """Read the stored mole report (empty state when there is none)."""
        path = default_report_path()
        report, age_s = await asyncio.to_thread(read_report, path)
        return success_response(data={"report": report, "path": path, "age_s": age_s})

    @endpoint("mole.run")
    async def run_mole(
        self,
        body: MoleRunBody | None = None,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Run the light probes (http + sse + journey-from-disk), write the report.

        The browser journey sweep is never invoked here — ``run_sweep=False``
        keeps the journey probe on its read-the-file path, whatever ``body``
        asks for. ``body`` is optional (see :class:`MoleRunBody`): omitting it
        preserves the historical contract exactly — 6s window, nothing skipped.
        """
        from domain.core import run_mole as _run_mole

        window_s = LIGHT_WINDOW_S if body is None or body.window_s is None else body.window_s
        skip: tuple[str, ...] = () if body is None or body.skip is None else tuple(body.skip)
        report = await asyncio.to_thread(
            _run_mole,
            run_sweep=False,
            window_s=window_s,
            skip=skip,
            write=True,
        )
        return success_response(data={"report": report.to_dict()})


router = MoleRouter().router
