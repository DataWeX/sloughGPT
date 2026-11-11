"""Test-run results crossing — descriptor-projected read of the run store.

Card 20261111_110106: ``tests/core-py/test_results/*.json`` is written by the
journey/telemetry suites on every run and had zero consumers — an artifact
nobody reads produces no decisions. This module makes it readable by the
monitoring surface as one capability.

Tier: external utility tooling — an operator/site read route. It never enters
the chat tool registry, the ``domain.tools`` engine, or anything the model
reasons over (Endpoint & Transport Rule, two tool tiers over one contract).

The store is **read-only** here: this crossing only summarizes what the test
runs already wrote. A file that fails to parse is dropped whole with a warning
(fail-safe — one corrupt artifact must not take the whole surface down).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from infrastructure.contract import RouteSpec, create_router

from domain.agents import ToolSpec

logger = logging.getLogger("slo.test_results")

#: Store root, resolved at import and read at call time (tests monkeypatch it).
RESULTS_DIR = Path("tests") / "core-py" / "test_results"


def load_results(root: Path) -> list[dict[str, Any]]:
    """Summarize every ``*.json`` run artifact under *root*.

    Returns one row per parseable file, freshest first:

    ``{file, status, total, passed, failed, mtime, entries}``

    An absent directory (fresh checkout, store cleaned) is a valid state and
    yields ``[]`` — the monitoring surface shows "no runs yet", not an error.
    """
    if not root.is_dir():
        return []

    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Fail-safe: drop the corrupt file whole, keep serving the rest.
            logger.warning("test_results: dropping unparseable %s: %s", path.name, exc)
            continue
        if not isinstance(data, list):
            logger.warning("test_results: dropping %s — expected a JSON list", path.name)
            continue

        entries = [e for e in data if isinstance(e, dict)]
        passed = sum(1 for e in entries if e.get("passed") is True)
        failed = sum(1 for e in entries if e.get("passed") is False)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        rows.append(
            {
                "file": path.name,
                "status": "failed" if failed else "passed",
                "total": len(entries),
                "passed": passed,
                "failed": failed,
                "mtime": mtime,
                "entries": entries,
            }
        )

    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


def list_test_results() -> dict[str, Any]:
    """Serve the per-file run summaries (the capability's one operation)."""
    return {"files": load_results(RESULTS_DIR)}


SPEC = ToolSpec(
    name="test_results.list",
    description=(
        "Per-file summaries of the stored test-run artifacts "
        "(tests/core-py/test_results/*.json): counts, status and freshness."
    ),
    parameters=[],
    execute=list_test_results,
    auth_scope="authenticated",
    idempotent=True,
    result={
        "type": "object",
        "properties": {
            "files": {"type": "array", "items": {"type": "object"}},
        },
    },
    version="1",
)

router = create_router(
    SPEC,
    RouteSpec(
        path="/test-results",
        summary="Per-file summaries of the stored test-run artifacts",
    ),
    list_test_results,
    tags=("monitoring",),
)
