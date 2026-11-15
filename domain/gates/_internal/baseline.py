"""Domain-baseline gate core — catalogue drift, computed the way the script does.

Reuses `scripts/domain_baseline_rehash.py` primitives directly (port, don't
rebuild): `dir_hashes()` hashes every domain subdir *in HEAD* — the baseline
describes committed contents, so a stale on-disk leftover has no tree to hash.
The gate is the read-only half of that script's `--check` mode: it never
rewrites hashes (humans move classification; the script moves hashes only
when explicitly run without `--check`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from domain.gates import GateReport

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import domain_baseline_rehash as _rehash  # noqa: E402  (sibling script, not a package)


def check() -> GateReport:
    """Drifted / uncatalogued / ghost dirs vs domain/domain-baseline.json."""
    doc = json.loads(_rehash.BASELINE.read_text(encoding="utf-8"))
    current = _rehash.dir_hashes()
    findings: list[str] = []
    for name, h in sorted(current.items()):
        stored = doc["dirs"].get(name, {}).get("tree_hash")
        if stored is None:
            findings.append(f"{name}: uncatalogued — add it to domain-baseline.json")
        elif stored != h:
            findings.append(
                f"{name}: drifted ({stored[:8]} -> {h[:8]}) — re-review class/facets, "
                "then run `python3 scripts/domain_baseline_rehash.py`"
            )
    for name in sorted(set(doc["dirs"]) - set(current)):
        findings.append(f"{name}: ghost entry — dir gone from HEAD, delete it")
    return GateReport(name="domain-baseline", ok=not findings, findings=findings, known=[])
