"""Test-results Router — journey evidence JSON as a descriptor-projected read.

The journey suites write per-case evidence into ``tests/core-py/test_results/
*.json`` (training journeys, e2e triggers, computer-use runs) and nothing in
the UI has ever read them (card 20261111_110106). This module projects one
descriptor — ``test_results.list`` — through ``create_router``: GET
/test-results, authenticated, enveloped, OpenAPI metadata derived. It is a
site read of artifacts the app already owns, so it stays tier (b): external
utility tooling, never entering the chat tool registry, ``domain.tools``, or
anything the model reasons over.

Two data facts the projection must not paper over:

* journey files store ``passed`` as the *strings* ``'True'``/``'False'`` —
  normalized to real booleans, ``None`` when unparseable. ``bool('False')`` is
  ``True``, so an honest conversion is the whole point of the boundary;
* ``devtools_telemetry.json`` lives in the same directory but is a different
  artifact (step/console_count/…). It is excluded by shape (items carrying a
  ``test`` key), not by filename, so a future non-run JSON cannot slip in as a
  suite either.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from infrastructure.contract import RouteSpec, create_router

from domain.shared import find_repo_root
from domain.agents import ToolSpec

#: Journey evidence directory — resolved once at import, overridable in tests.
RESULTS_DIR = find_repo_root(Path(__file__)) / "tests" / "core-py" / "test_results"


def _normalize_passed(value: object) -> bool | None:
    """Map the journey files' ``'True'``/``'False'`` strings to real booleans.

    Known strings map explicitly; anything unparseable becomes ``None`` and is
    surfaced as ``unknown`` — an honest gap, never a silent pass (and never
    ``bool(value)``, which would render every ``'False'`` as a pass).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value == "True":
            return True
        if value == "False":
            return False
        return None
    return None


def _read_suite(path: Path) -> dict[str, Any] | None:
    """Project one evidence file as a suite, or ``None`` when it is not one.

    A suite is a JSON list of objects each carrying a ``test`` key — that
    shape filter excludes ``devtools_telemetry.json`` (and any future
    non-run JSON) without hardcoding filenames. Unreadable or malformed
    files are skipped: one corrupt artifact must not fail the whole read.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except (OSError, ValueError):
        return None
    if not isinstance(raw, list) or not raw:
        return None
    if not all(isinstance(item, dict) and "test" in item for item in raw):
        return None

    items = [
        {
            "test": str(item.get("test", "")),
            "passed": _normalize_passed(item.get("passed")),
            "detail": str(item.get("detail", "") or ""),
        }
        for item in raw
    ]
    return {
        "file": path.name,
        "mtime": mtime,
        "total": len(items),
        "passed": sum(1 for i in items if i["passed"] is True),
        "failed": sum(1 for i in items if i["passed"] is False),
        "unknown": sum(1 for i in items if i["passed"] is None),
        "items": items,
    }


async def list_test_runs() -> dict[str, Any]:
    """Every journey evidence file as normalized runs, newest mtime first."""
    suites: list[dict[str, Any]] = []
    if RESULTS_DIR.is_dir():
        try:
            paths = sorted(
                RESULTS_DIR.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            paths = []
        for path in paths:
            suite = _read_suite(path)
            if suite is not None:
                suites.append(suite)
    totals = {
        key: sum(s[key] for s in suites) for key in ("total", "passed", "failed", "unknown")
    }
    return {"suites": suites, **totals}


SPEC = ToolSpec(
    name="test_results.list",
    description=(
        "Journey test-result runs from tests/core-py/test_results/*.json, "
        "normalized to real booleans (string 'False' never becomes truthy)."
    ),
    parameters=[],
    execute=list_test_runs,
    auth_scope="authenticated",
    idempotent=True,
    result={
        "type": "object",
        "properties": {
            "suites": {"type": "array", "items": {"type": "object"}},
            "total": {"type": "integer"},
            "passed": {"type": "integer"},
            "failed": {"type": "integer"},
            "unknown": {"type": "integer"},
        },
    },
    version="1",
)

router = create_router(
    SPEC,
    RouteSpec(path="", method="GET", summary="List journey test-result runs"),
    list_test_runs,
    prefix="/test-results",
    tags=("test-results",),
)
