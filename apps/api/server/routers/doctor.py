"""
Site Doctor router — surface the read-only doctor report to the UI.

Contract (the /doctor page is built against exactly this):

    GET  /doctor/report  -> data: {report: object|null, path: str, age_s: float|null}
    POST /doctor/run     -> data: {report: object}

``GET`` only reads the report file (``$SLO_DOCTOR_REPORT``, default
``~/.cache/slog-doctor/findings-report.json``). A missing or corrupt file
is an empty state, not an error: ``report`` comes back ``null``.

``POST`` runs the LIGHT probes only — ``http`` (read-only GETs) and ``sse``
(6s window) — and folds in the journey findings that are already on disk.
The browser journey sweep is never triggered from the API: ``run_doctor``
is called with ``run_sweep=False``, so the journey probe only reads the
existing sweep report. Budget on a responsive stack: preflight 3s (which
skips ``http``/``sse`` outright when the API is dead) + SSE window 6s (plus
a best-effort ≤4s ``/errors/stream`` read) + read-only GETs against the
same API that is serving this request → well inside 15s. No retries, no
writes to the site.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, Depends
from infrastructure.auth import require_auth_if_enabled
from schemas.common import endpoint, success_response

logger = logging.getLogger("slo.routers.doctor")

# SSE observation window for the API-initiated light run (CLI default is 9s;
# the API keeps the interactive run short).
LIGHT_WINDOW_S = 6.0


def default_report_path() -> str:
    """Resolve the report path through the doctor package (``$SLO_DOCTOR_REPORT``)."""
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
        logger.debug("doctor report unreadable at %s: %s", path, exc)
        return None, None
    if not isinstance(data, dict):
        return None, None

    ts = data.get("ts")
    age_s: float | None = None
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        age_s = max(0.0, time.time() - float(ts))
    return data, age_s


class DoctorRouter:
    """Report-surfacing endpoints for the Site Doctor."""

    def __init__(self):
        self.router = APIRouter(prefix="/doctor", tags=["doctor"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/report", self.get_report, methods=["GET"])
        self.router.add_api_route("/run", self.run_doctor, methods=["POST"])

    @endpoint("doctor.report")
    async def get_report(self) -> dict:
        """Read the stored doctor report (empty state when there is none)."""
        path = default_report_path()
        report, age_s = await asyncio.to_thread(read_report, path)
        return success_response(data={"report": report, "path": path, "age_s": age_s})

    @endpoint("doctor.run")
    async def run_doctor(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Run the light probes (http + sse + journey-from-disk), write the report.

        The browser journey sweep is never invoked here — ``run_sweep=False``
        keeps the journey probe on its read-the-file path.
        """
        from domain.core import run_doctor as _run_doctor

        report = await asyncio.to_thread(
            _run_doctor,
            run_sweep=False,
            window_s=LIGHT_WINDOW_S,
            write=True,
        )
        return success_response(data={"report": report.to_dict()})


router = DoctorRouter().router
