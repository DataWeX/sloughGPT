"""One capability, one HTTP surface — split-brain guard.

Phoneme shipped two HTTP halves over the same ``domain/voice`` capability:
``routers/phoneme.py`` (9 ``/phoneme/*`` routes) and the live phoneme group
in ``routers/multimodal.py` (``/multimodal/*-phonemes``, consumed by
``phoneme-controller.ts``). The dedicated router had zero frontend consumers
— only its own test — and was deleted (card ``20261108_152007``, 2026-11-08:
"phoneme is model capability, not a dictionary UI").

This pins the boot route surface (``get_all_routers()`` — the same contract
``infrastructure/startup.py`` registers): no ``/phoneme`` path may regrow,
and both live halves must keep serving the capability. The presence
assertions guard against a vacuous pass: ``get_all_routers()`` logs and skips
a router whose import failed, so an absent live half fails the test even
though the strays list would be empty.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[2] / "apps" / "api" / "server"
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))


def _boot_paths() -> set[str]:
    """Every path the boot manifest registers (startup.py's contract)."""
    from routers import get_all_routers

    return {
        route.path
        for router in get_all_routers()
        for route in getattr(router, "routes", [])
        if hasattr(route, "path")
    }


def test_phoneme_capability_has_no_dedicated_router_surface() -> None:
    paths = _boot_paths()
    strays = sorted(p for p in paths if p.startswith("/phoneme"))
    assert not strays, (
        "phoneme is served by /multimodal/*-phonemes + /voice/* — a dedicated "
        f"/phoneme router is the split-brain this card removed: {strays}"
    )
    assert "/multimodal/encode-phonemes" in paths, "live phoneme half missing"
    assert "/voice/tts" in paths, "live voice half missing"
