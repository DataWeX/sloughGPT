"""The generated router mount table must match the routers/ directory.

``routers/_manifest.py`` replaces the hand-maintained ``_router_names`` list
(card 57b0ea27). Generating it removes the *hand* part; this test keeps the
*checked* part — a generated file nobody verifies is just a copy of the drift
it was meant to prevent.

Runs in the normal test gate, so a new router file added without re-running
``scripts/gen_router_manifest.py`` fails here rather than in CI review.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "scripts" / "gen_router_manifest.py"
MANIFEST = REPO / "apps" / "api" / "server" / "routers" / "_manifest.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GEN), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )


def test_manifest_is_current():
    """``--check`` exits 0 only when the manifest matches routers/*.py."""
    proc = _run("--check")
    assert proc.returncode == 0, (
        "routers/_manifest.py is stale — run `python scripts/gen_router_manifest.py`"
        f"\n--- stderr ---\n{proc.stderr}"
    )


def test_manifest_shape():
    """The generated table binds names only: no imports, no router modules.

    This is what keeps it free at cold start — reading it must not pull in a
    single router (the deferred-import win the table exists to protect).
    """
    import ast

    tree = ast.parse(MANIFEST.read_text(encoding="utf-8"))

    # only the __future__ import — nothing that could pull a router module in
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and getattr(node, "module", None) != "__future__"
    ]
    assert imports == [], f"manifest must not import anything, found {imports!r}"

    value = None
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(getattr(t, "id", "") == "ROUTER_MODULES" for t in targets):
                value = node.value
    assert isinstance(value, ast.Tuple), "ROUTER_MODULES must be a tuple literal"
    assert all(
        isinstance(elt, ast.Constant) and isinstance(elt.value, str) for elt in value.elts
    ), "ROUTER_MODULES must be a tuple of plain string literals"

    src = MANIFEST.read_text(encoding="utf-8")
    assert "GENERATED FILE" in src, "manifest lost its do-not-edit header"


def test_manifest_matches_seed_order():
    """Mount order is load-bearing: inference must precede chat.

    ``include_router`` is first-match-wins while OpenAPI is last-wins, so
    re-ordering would silently fork the spec from behaviour (the
    InferenceRouter-vs-ChatRouter ``/chat`` split).
    """
    from routers._manifest import ROUTER_MODULES

    assert "inference" in ROUTER_MODULES
    assert "chat" in ROUTER_MODULES
    assert ROUTER_MODULES.index("inference") < ROUTER_MODULES.index("chat")


def test_pre_lifespan_routers_are_excluded():
    """health/status/consciousness/dashboard are mounted in main.py directly."""
    from routers._manifest import ROUTER_MODULES

    for name in ("health", "status", "consciousness", "dashboard"):
        assert name not in ROUTER_MODULES, (
            f"{name!r} is registered pre-lifespan in main.py — including it in "
            "the manifest would register it twice"
        )
