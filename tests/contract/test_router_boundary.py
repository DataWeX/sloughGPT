"""Shared-core contract: routers translate at the edge, never into the core.

Enforces the invariant that a FastAPI router module must not import another
router's private ``_instance`` (``from routers.x import _instance``). Doing so
creates a second, silent coupling to the shared kernel that breaks like the
``TrainingEngine.get_executor`` 500 when internals move. Consumers that need a
router capability must use its public facade (e.g.
``routers.inference.build_session_metadata_index`` / ``handle_chat``).

BFF endpoints (``routers/mobile.py``) may shape responses for their shell, but
ride the same shared handlers/accessors the web client calls.
"""

from __future__ import annotations

import re
from pathlib import Path

ROUTERS_DIR = Path(__file__).resolve().parents[2] / "apps" / "api" / "server" / "routers"

_PRIVATE_INSTANCE = re.compile(
    r"from routers\.(?P<module>[a-zA-Z_][a-zA-Z0-9_]*) import\s*.*?\b(?:_instance(?:\s+as\s+\w+)?)"
)
_INSTANCE_ATTR = re.compile(r"routers\.(?P<module>[a-zA-Z_][a-zA-Z0-9_]*)\._instance\b")


def _router_file_violations() -> list[str]:
    bad: list[str] = []
    for py in sorted(ROUTERS_DIR.glob("*.py")):
        src = py.read_text(encoding="utf-8")
        for i, line in enumerate(src.splitlines(), 1):
            if _PRIVATE_INSTANCE.search(line):
                bad.append(f"{py.name}:{i}: {line.strip()}")
            if _INSTANCE_ATTR.search(line):
                bad.append(f"{py.name}:{i}: {line.strip()}")
    return bad


def test_routers_never_import_another_routers_private_instance() -> None:
    violations = _router_file_violations()
    assert violations == [], (
        "Routers must not reach into another router's private _instance "
        "(shared-core contract). Use the router's public facade instead.\n" + "\n".join(violations)
    )


def test_inference_exposes_public_facades_over_its_instance() -> None:
    """The sanctioned pattern: public functions delegating to the singleton."""
    src = (ROUTERS_DIR / "inference.py").read_text(encoding="utf-8")
    for facade in (
        "def build_session_metadata_index()",
        "async def handle_chat(req: ChatRequest)",
    ):
        assert facade in src, f"inference router must keep the public facade: {facade}"
