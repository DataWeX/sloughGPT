"""Contract: docs/routers.md rows must exactly match the served route table.

Mirrors test_api_envelope's anti-drift pattern (070): the doc's claim is
pinned to measured reality, with a floor guard so an empty/broken route
discovery can never vacuously pass.

Route discovery mirrors startup (infrastructure/startup.py):
get_all_routers() + training.router + the four pre-lifespan main.py routers,
materialized via probe.openapi() (fastapi >=0.139 defers include_router with
lazy _IncludedRouter wrappers — app.routes alone is incomplete).
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

_METHODS = ("GET", "POST", "PUT", "DELETE", "PATCH")
_ROW_RE = re.compile(
    r"^\|\s*`(GET|POST|PUT|DELETE|PATCH)`\s*\|\s*`([^`]+)`\s*\|",
)
_FRAMEWORK = {"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}
_DOC = Path(__file__).resolve().parents[2] / "docs" / "routers.md"


def _live_routes() -> set[tuple[str, str]]:
    """Probe-app routes: everything the server would register, no lifespan."""
    import main as mainmod
    from fastapi import FastAPI
    from routers import get_all_routers
    from training.router import router as training_router

    probe = FastAPI()
    for r in get_all_routers():
        probe.include_router(r)
    probe.include_router(training_router)
    for name in (
        "_health_router",
        "_consciousness_router",
        "_status_router",
        "_dashboard_router",
    ):
        probe.include_router(getattr(mainmod, name))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paths = probe.openapi().get("paths", {})
    return {
        (m.upper(), p.rstrip("/") or "/")
        for p, ops in paths.items()
        if p not in _FRAMEWORK
        for m in ops
        if m.upper() in _METHODS
    }


def _doc_rows() -> set[tuple[str, str]]:
    rows = set()
    for line in _DOC.read_text().splitlines():
        m = _ROW_RE.match(line)
        if m:
            rows.add((m.group(1), m.group(2).rstrip("/") or "/"))
    return rows


@pytest.fixture(scope="module")
def live() -> set[tuple[str, str]]:
    routes = _live_routes()
    assert len(routes) >= 600, (
        f"route discovery found only {len(routes)} routes — probe is broken "
        "(fastapi include_router laziness changed? do not trust a passing run)"
    )
    return routes


def test_doc_has_no_stale_rows(live: set[tuple[str, str]]) -> None:
    """Every documented (method, path) must actually be served."""
    stale = sorted(_doc_rows() - live)
    assert not stale, (
        f"docs/routers.md documents {len(stale)} unserved route(s) — remove or "
        f"fix them: {stale[:20]}"
    )


def test_doc_covers_every_served_route(live: set[tuple[str, str]]) -> None:
    """Every served (method, path) must be documented."""
    missing = sorted(live - _doc_rows())
    assert not missing, (
        f"docs/routers.md is missing {len(missing)} served route(s) — add "
        f"them: {missing[:20]}"
    )
