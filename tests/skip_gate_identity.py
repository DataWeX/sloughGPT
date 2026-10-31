"""Pure identity checks for the live-stack skip gates — card 20261029_041801.

The gates in ``tests/test_integration.py`` (API) and
``tests/test_header_screenshots.py`` (web) historically checked only
PORT-OPEN, so a foreign listener (another checkout's API mid-load, another
checkout's vite on :5173) turned live tests into 120s-per-test
pytest-timeout ERRORS instead of module skips — 27 setup-timeouts in one
run, because the old probe budget (5 x 30s + 4 x 2s = 158s) itself
EXCEEDED pytest.ini's ``timeout = 120``.

Both helpers here answer one question: "is the thing on that port OURS?"
They live in this non-test module (not in the slow test files) so the
fast lane can pin them (``tests/test_skip_gate_identity.py``): the slow
modules are deselected by default (``-m "not slow"``), and importing
``test_header_screenshots`` would drag its ``pytest.importorskip``
contagion along.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

# Dev-server header apps/web/vite.config.ts emits to identify its checkout.
WEB_TREE_HEADER = "X-Slo-Web-Tree"

# Probe budget for the API skip gate (card 20261029_041801): identity, not
# port-open. Must stay well under pytest.ini's per-test timeout=120 — the
# old budget (5 attempts x 30s + 4 x 2s = 158s) EXCEEDED it, so a foreign
# listener holding :8000 turned every test into a 120s setup-timeout ERROR
# instead of a module skip. New budget: 5 x 6s + 4 x 2s = 38s max, pinned
# by tests/test_skip_gate_identity.py against pytest.ini's timeout.
PROBE_ATTEMPTS = 5
PROBE_TIMEOUT = 6
PROBE_SLEEP = 2

_REPO_ROOT = Path(__file__).resolve().parents[1]


def probe_budget_seconds() -> int:
    """Worst-case seconds the API fixture may spend probing before skipping."""
    return PROBE_ATTEMPTS * PROBE_TIMEOUT + (PROBE_ATTEMPTS - 1) * PROBE_SLEEP


def readiness_payload_ok(payload: object) -> bool:
    """True only for OUR ready API: our nested envelope says ``status: "ready"``.

    The real body from ``GET /health/ready`` is
    ``success_response(get_readiness())`` (apps/api/server/routers/health.py
    wrapping apps/api/server/controllers/health.py, envelope from
    apps/api/server/schemas/common.py)::

        {"status": "success",
         "data": {"status": "ready" | "not_ready",
                  "checks": {"app_lifecycle": ..., "model": ..., ...}}}

    Any listener can hold a port; only this exact nested envelope with
    ``data.status == "ready"`` counts as our server. A foreign service, a
    different JSON shape, or our own instance still loading (model
    ``"loading"``/``"not_loaded"`` → ``"not_ready"``) answers with anything
    else → the module SKIPS instead of failing every test against a
    listener that is not ours.
    """
    if not isinstance(payload, dict):
        return False
    if payload.get("status") != "success":
        return False
    data = payload.get("data")
    if not isinstance(data, dict):
        return False
    if data.get("status") != "ready":
        return False
    return isinstance(data.get("checks"), dict)


def web_tree_id(root: Path | None = None) -> str:
    """Identity token of a web checkout's app — mirrors vite.config.ts.

    ``apps/web/vite.config.ts`` serves this value as the
    ``X-Slo-Web-Tree`` dev-server header: ``sha256(realpath(apps/web))``
    truncated to 16 hex chars. Node's ``fs.realpathSync`` and Python's
    ``os.path.realpath`` both resolve symlinks, so the two sides agree
    byte-for-byte. A server WITHOUT the header (foreign checkout's vite,
    a pre-fix server, a static preview) or with a different value is not
    this tree → the screenshot tests skip instead of capturing a foreign
    checkout's UI.

    ``root`` defaults to this repo (``tests/..``); tests pass other roots
    to prove different checkouts produce different ids.
    """
    base = root if root is not None else _REPO_ROOT
    web_dir = os.path.realpath(str(base / "apps" / "web"))
    return hashlib.sha256(web_dir.encode("utf-8")).hexdigest()[:16]
