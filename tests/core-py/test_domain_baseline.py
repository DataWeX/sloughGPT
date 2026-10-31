"""Domain baseline gate — the Page Rule mechanism for the `domain/` tree.

`docs/PRODUCT_ENGINEERING.md` (CCGT rule, page rule precedent) prescribed
placement in documentation alone; nothing mechanical ran it. This gate makes
`domain/domain-baseline.json` load-bearing, exactly like `page-baseline.json`:

1. catalogue completeness — every `domain/*/` on disk must be catalogued,
   and every catalogue entry must exist (no ghosts);
2. schema/evidence — class/facets/note/consumers/debt present and typed;
   entries cite their evidence;
3. shrink-only ratchet — counts of `misfiled`/`projection`/`debt` may only
   decrease vs the stored counts (new debt cannot be slipped in);
4. directory hash drift — per-dir git tree hash must match
   `git rev-parse HEAD:domain/<name>`: when a dir's committed contents
   change, its placement/class gets re-reviewed (red gate message names the
   rehash script). Same deliberate-edit discipline as the hash-chained board.

Baseline: card 20261031_083744 (user ruling: staged plan — CCGT-primary
catalogue now; physical merges ride their own cards; 4-parent re-root is a
later decision).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BASELINE = REPO / "domain" / "domain-baseline.json"
DOMAIN = REPO / "domain"
HEX40 = re.compile(r"^[0-9a-f]{40}$")

# __pycache__ is interpreter litter, never a package — excluded from the scan.
PRUNE = {"__pycache__"}


def _load() -> dict:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def _on_disk() -> list[str]:
    return sorted(p.name for p in DOMAIN.iterdir() if p.is_dir() and p.name not in PRUNE)


def _git_tree(name: str) -> str:
    res = subprocess.run(
        ["git", "rev-parse", f"HEAD:domain/{name}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        pytest.skip(f"git unavailable for tree hash ({res.stderr.strip()})")
    return res.stdout.strip()


class TestCatalogueCompleteness:
    def test_every_dir_on_disk_is_catalogued(self):
        doc = _load()
        missing = sorted(set(_on_disk()) - set(doc["dirs"]))
        assert not missing, (
            f"uncatalogued domain dirs: {missing} — add them to "
            f"domain/domain-baseline.json (class/facets/consumers/note) first"
        )

    def test_no_ghost_entries(self):
        doc = _load()
        ghosts = sorted(set(doc["dirs"]) - set(_on_disk()))
        assert not ghosts, f"catalogued dirs that no longer exist: {ghosts}"


class TestSchemaAndEvidence:
    def test_top_level_contract(self):
        doc = _load()
        assert doc["metadata"]["schema"] == 1
        assert doc["metadata"]["recorded_commit"]
        assert doc["classes"] == [
            "core-capability",
            "feature",
            "projection",
            "misfiled",
        ]
        assert doc["facets"] == ["cognitive", "core", "gateway", "training"]
        assert set(doc["ratchet"]) == {"misfiled", "projection", "debt"}

    def test_every_entry_is_complete_and_typed(self):
        doc = _load()
        problems: list[str] = []
        for name, entry in doc["dirs"].items():
            if entry["class"] not in doc["classes"]:
                problems.append(f"{name}: bad class {entry['class']!r}")
            if not entry["facets"] or not set(entry["facets"]) <= set(doc["facets"]):
                problems.append(f"{name}: facets must be non-empty subset of CCGT")
            if not isinstance(entry["consumers"], list):
                problems.append(f"{name}: consumers must be a list")
            if not entry["note"].strip():
                problems.append(f"{name}: note (evidence) missing")
            debt = entry["debt"]
            if debt is not None and (not isinstance(debt, str) or not debt.strip()):
                problems.append(f"{name}: debt must be null or a reason string")
            if not HEX40.match(entry["tree_hash"]):
                problems.append(f"{name}: tree_hash is not a 40-hex git sha")
        assert not problems, problems


class TestShrinkOnlyRatchet:
    def test_counts_never_grow(self):
        doc = _load()
        actual = {
            "misfiled": sum(1 for e in doc["dirs"].values() if e["class"] == "misfiled"),
            "projection": sum(1 for e in doc["dirs"].values() if e["class"] == "projection"),
            "debt": sum(1 for e in doc["dirs"].values() if e["debt"]),
        }
        grown = {k: (doc["ratchet"][k], v) for k, v in actual.items() if v > doc["ratchet"][k]}
        assert not grown, (
            f"ratchet counts grew (stored, actual): {grown} — debt/misfiled/"
            f"projection may only shrink; new debts need an explicit card"
        )

    def test_new_entries_cannot_be_misfiled(self):
        """class:misfiled can never appear while ratchet.misfiled=0 — a new
        dir's misplacement must not enter the catalogue at all (placement
        problems go through their own shrink/move card instead)."""
        doc = _load()
        if doc["ratchet"]["misfiled"] != 0:
            pytest.skip("grandfathered misfiled entries exist — count ratchet governs")
        offenders = [name for name, entry in doc["dirs"].items() if entry["class"] == "misfiled"]
        assert not offenders, (
            f"{offenders} classed misfiled but ratchet.misfiled=0 — move the "
            "dir out (its own card) or ratchet down via an explicit ruling"
        )


class TestDirectoryHashDrift:
    def test_dir_tree_hashes_match_head(self):
        doc = _load()
        drifted = []
        for name in _on_disk():
            stored = doc["dirs"].get(name, {}).get("tree_hash")
            current = _git_tree(name)
            if stored != current:
                drifted.append((name, stored[:8], current[:8]))
        assert not drifted, (
            "domain dirs changed since catalogue — re-review class/facets for "
            f"{[d[0] for d in drifted]} (drifted: {drifted}), then run "
            "`python3 scripts/domain_baseline_rehash.py` and commit the "
            "updated baseline with your change"
        )

    def test_recorded_commit_is_a_sha(self):
        doc = _load()
        assert HEX40.match(doc["metadata"]["recorded_commit"])
