"""Dependency contract: the consciousness/dream engine must stand alone.

The engine is written against the standard library only. What used to break
that was not a direct import — it was the package facade. ``import
domain.cognition`` eagerly imported ``_internal.core``, ``_internal.processor``
and ``_internal.rag_service``, which pull in the inference stack and numpy, so
touching *any* cognition submodule paid for the whole model stack.

That is why these tests run in a SUBPROCESS with an explicitly constructed
``sys.path``: asserting isolation in-process would inherit the runner's full
path and prove nothing. It would report a clean it did not cause — the exact
false-clean this suite exists to prevent (``rg`` was once missing and an empty
sweep was reported as "0 issues").

An inner life that only runs when the model stack is importable is a demo, not
an engine. This file is what keeps it an engine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Anything that means "the model stack came with us". Extend deliberately —
# every entry is a promise about what the dream engine does NOT need.
HEAVY_PREFIXES = (
    "numpy",
    "pydantic",
    "yaml",
    "torch",
    "fastapi",
    "domain.inference",
    "domain.chat",
    "domain.companion",
    "domain.infrastructure",
    "domain.logging",
    "domain.rag",
    "domain.training",
    "domain.cognition._internal.core",
    "domain.cognition._internal.processor",
    "domain.cognition._internal.rag_service",
)


def _run_probe(source: str) -> dict:
    """Run ``source`` in a clean subprocess and return its parsed JSON verdict."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONNOUSERSITE": "1",
        # Deliberately NO PYTHONPATH: the probe builds its own sys.path, so the
        # runner's core-py/downcraft/api-server entries cannot leak in.
        "PYTHONHOME": os.environ.get("PYTHONHOME", ""),
    }
    proc = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    assert proc.returncode == 0, f"probe failed:\nSTDOUT={proc.stdout}\nSTDERR={proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


# The probe body is shared: import the engine, exercise the whole inner-life
# journey, then report which heavy modules are loaded. Kept as one string so
# the exercised behaviour and the assertion can never drift apart.
_JOURNEY_PROBE = """
import json, sys, tempfile, pathlib
sys.path.insert(0, {repo!r})

from domain.cognition._internal.consciousness.engine import ConsciousnessEngine
from domain.cognition._internal.consciousness.config import ConsciousnessConfig

HEAVY = {heavy!r}
heavy = sorted(
    m for m in sys.modules
    if m.split(".")[0] in ("numpy", "pydantic", "yaml", "torch", "fastapi")
    or m.startswith(tuple(HEAVY))
)

tmp = pathlib.Path(tempfile.mkdtemp())
eng = ConsciousnessEngine(ConsciousnessConfig(level=2, store_path=str(tmp)))
eng.process("hello", "hi there")
dream = eng.dream()
tick = eng.maybe_dream()
reflection = eng.reflect()
report = eng.self_report()
records = eng.mind_log.query(limit=0)

print(json.dumps({{
    "heavy": heavy,
    "episodes": len(eng.self_model.episodes),
    "dream_locus": dream.get("locus"),
    "dream_from_locus": dream.get("from_locus"),
    "tick_dreamed": bool(tick.get("dreamed")),
    "reflection_type": type(reflection).__name__,
    "report_keys": sorted(report.keys()),
    "records": len(records),
}}))
"""


class TestDreamEngineIsStandalone:
    """The engine runs, and runs alone."""

    def test_engine_runs_without_the_model_stack(self) -> None:
        verdict = _run_probe(_JOURNEY_PROBE.format(repo=str(REPO_ROOT), heavy=list(HEAVY_PREFIXES)))

        # (a) Functional independence — the whole journey works.
        assert verdict["episodes"] >= 1
        assert verdict["dream_locus"] == "dreaming"
        assert verdict["dream_from_locus"] == "awakening"
        assert verdict["tick_dreamed"] is True  # satiation gate fires on new material
        assert verdict["reflection_type"] == "Reflection"
        assert verdict["records"] >= 1
        for key in ("beliefs", "doubts", "reflection", "timeline", "last_dream"):
            assert key in verdict["report_keys"], f"report missing {key!r}"

        # (b) Dependency independence — nothing from the model stack came along.
        assert verdict["heavy"] == [], f"consciousness dragged in the stack: {verdict['heavy']}"

    def test_package_facade_import_is_cheap(self) -> None:
        """`import domain.cognition` must not load the model stack either."""
        source = f"""
import json, sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import domain.cognition
HEAVY = {list(HEAVY_PREFIXES)!r}
heavy = sorted(
    m for m in sys.modules
    if m.split(".")[0] in ("numpy", "pydantic", "yaml", "torch", "fastapi")
    or m.startswith(tuple(HEAVY))
)
print(json.dumps({{"heavy": heavy}}))
"""
        verdict = _run_probe(source)
        assert verdict["heavy"] == [], f"the facade pulled in the stack: {verdict['heavy']}"

    def test_lazy_facade_still_resolves_every_public_name(self) -> None:
        """The lazy facade must answer for every name it advertises.

        The package now imports on access instead of at import time, so this
        guards the 27 `from domain.cognition import X` call sites: a typo'd
        mapping would surface as an AttributeError in unrelated code.
        """
        source = f"""
import json, sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import domain.cognition as cog
missing = []
for name in cog.__all__:
    try:
        getattr(cog, name)
    except Exception as exc:  # noqa: BLE001 - report, don't mask
        missing.append(f"{{name}}: {{type(exc).__name__}}")
print(json.dumps({{"missing": missing, "dir_ok": all(n in dir(cog) for n in cog.__all__)}}))
"""
        verdict = _run_probe(source)
        assert verdict["missing"] == [], f"facade names failed to resolve: {verdict['missing']}"
        assert verdict["dir_ok"] is True, "__dir__ must advertise the lazy names"
