"""Tests for scripts/check_lane_claims.py — the lane-claims advisory.

Policy: AGENTS.md "Lane claims" + the coordination-workflow skill.

The contract under test is the one the script's own docstring states —
"Advisory by default: always exit 0" — so the failure paths matter as much
as the happy path. A gate that reddens CI because a card id typo'd or a
store shape drifted would train sessions to bypass it (same reasoning as
tests/test_check_surface_scope.py). ``--strict`` is the ratchet.

Covered: path matching (the correlation the whole script rests on), dirty
path extraction, card path extraction, and both exit-code paths.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_lane_claims as gate  # noqa: E402

# ── matches: prefix/wildcard correlation ──────────────────────────────────


def test_matches_exact_and_prefix() -> None:
    assert gate.matches("scripts/check_lane_claims.py", "scripts/check_lane_claims.py")
    assert gate.matches("scripts/check_lane_claims.py", "scripts")
    assert gate.matches("packages/grib/src/grib/claims.py", "packages/grib")


def test_matches_is_boundary_anchored() -> None:
    """`scripts` must not swallow a sibling directory that merely starts with it."""
    assert not gate.matches("scriptsx/foo.py", "scripts")
    assert not gate.matches("domain/training/x.py", "domain/training/_internal")


def test_matches_wildcard() -> None:
    assert gate.matches("tests/core-py/test_a.py", "tests/core-py/*.py")
    assert not gate.matches("tests/core-py/sub/test_a.py", "tests/core-py/*.py")


# ── dirty_paths: porcelain parsing ────────────────────────────────────────


def test_dirty_paths_reads_porcelain(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "new.py").write_text("x\n")
    (tmp_path / "old.py").write_text("x\n")
    subprocess.run(["git", "add", "old.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "old.py").write_text("changed\n")
    paths = gate.dirty_paths(tmp_path)
    assert "new.py" in paths and "old.py" in paths


def test_dirty_paths_outside_git_is_empty(tmp_path: Path) -> None:
    assert gate.dirty_paths(tmp_path) == []


# ── card_paths: path tokens from the claimed card ─────────────────────────


def test_card_paths_extracts_repo_paths(tmp_path: Path) -> None:
    from grib.store import PlannerStore

    store = PlannerStore(board_dir=tmp_path, notes_dir=tmp_path)
    card = store.add_card(
        "advisory",
        description="Owns scripts/check_lane_claims.py and packages/grib/src/grib/claims.py",
    )
    tokens = gate.card_paths(tmp_path, card.id)
    assert "scripts/check_lane_claims.py" in tokens
    assert "packages/grib/src/grib/claims.py" in tokens


def test_card_paths_unknown_card_is_empty(tmp_path: Path) -> None:
    assert gate.card_paths(tmp_path, "nope") == []
    assert gate.card_paths(tmp_path, None) == []


# ── exit codes: advisory first, strict as the ratchet ─────────────────────


def _seed(tmp_path: Path, rows: list[dict]) -> None:
    (tmp_path / "claims.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


def test_clean_checkout_is_ok_and_exit_zero(tmp_path: Path, capsys) -> None:
    _seed(tmp_path, [])
    code = gate.main(["--board-dir", str(tmp_path), "--session", "ses-mine"])
    out = capsys.readouterr().out
    assert code == 0
    assert "ok" in out


def test_foreign_worktree_claim_warns_but_exits_zero(tmp_path: Path, monkeypatch, capsys) -> None:
    """Advisory contract: a real conflict still exits 0 without --strict."""
    here = "/some/checkout"
    monkeypatch.setattr(gate, "detect_git", lambda: (here, "feat/x"))
    monkeypatch.setattr(gate, "dirty_paths", lambda p: [])
    _seed(
        tmp_path,
        [
            {
                "ts": time.time(),  # fresh heartbeat — within TTL
                "event": "claim",
                "session": "ses-foreign",
                "card_id": None,
                "worktree": here,
                "branch": "feat/x",
                "ttl_min": 120,
            }
        ],
    )
    code = gate.main(["--board-dir", str(tmp_path), "--session", "ses-mine"])
    out = capsys.readouterr().out
    assert "ses-foreign" in out and "THEIR checkout" in out
    assert code == 0
    strict = gate.main(["--board-dir", str(tmp_path), "--session", "ses-mine", "--strict"])
    assert strict == 1


def test_expired_and_own_claims_are_silent(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(gate, "detect_git", lambda: ("/some/checkout", None))
    monkeypatch.setattr(gate, "dirty_paths", lambda p: [])
    _seed(
        tmp_path,
        [
            {  # heartbeat far past TTL -> expired, never blocks
                "ts": 0.0,
                "event": "claim",
                "session": "ses-dead",
                "card_id": None,
                "worktree": "/some/checkout",
                "branch": None,
                "ttl_min": 120,
            },
            {  # my own claim -> skipped
                "ts": time.time(),  # fresh heartbeat — within TTL
                "event": "claim",
                "session": "ses-mine",
                "card_id": None,
                "worktree": "/other",
                "branch": None,
                "ttl_min": 120,
            },
        ],
    )
    code = gate.main(["--board-dir", str(tmp_path), "--session", "ses-mine"])
    out = capsys.readouterr().out
    assert code == 0
    assert "ses-dead" not in out
    assert "ses-mine" not in out.replace("ses-mine)", "")  # header may echo id
    assert "conflict" not in out


def test_dirty_path_overlap_fires(tmp_path: Path, monkeypatch, capsys) -> None:
    from grib.store import PlannerStore

    store = PlannerStore(board_dir=tmp_path, notes_dir=tmp_path)
    card = store.add_card("x", description="touch scripts/check_lane_claims.py")
    monkeypatch.setattr(gate, "detect_git", lambda: ("/elsewhere", None))
    monkeypatch.setattr(gate, "dirty_paths", lambda p: ["scripts/check_lane_claims.py"])
    _seed(
        tmp_path,
        [
            {
                "ts": time.time(),  # fresh heartbeat — within TTL
                "event": "claim",
                "session": "ses-foreign",
                "card_id": card.id,
                "worktree": "/other-checkout",  # NOT here -> branch (b), not (a)
                "branch": None,
                "ttl_min": 120,
            }
        ],
    )
    code = gate.main(["--board-dir", str(tmp_path), "--session", "ses-mine"])
    out = capsys.readouterr().out
    assert code == 0
    assert "dirty path(s) match" in out
    assert "scripts/check_lane_claims.py" in out
