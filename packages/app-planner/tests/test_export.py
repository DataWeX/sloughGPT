"""
Tests for app_planner.export — live canonical->repo mirror for the GitHub
sync pipeline (board + notes + manifest).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from app_planner.export import ExportError, export
from app_planner.store import PlannerStore


def _canonical(tmp_path: Path) -> tuple[Path, Path]:
    board_dir = tmp_path / "canonical-board"
    notes_dir = tmp_path / "canonical-notes"
    notes_dir.mkdir()
    store = PlannerStore(board_dir=board_dir, notes_dir=notes_dir)
    store.add_card("sync me")
    (notes_dir / "alpha.md").write_text("# alpha\ncanonical body\n")
    (notes_dir / "beta.md").write_text("# beta\n")
    return board_dir, notes_dir


def test_export_mirrors_board_and_notes(tmp_path):
    board_dir, notes_dir = _canonical(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()

    result = export(repo, board_dir=board_dir, notes_dir=notes_dir)

    dest_board = repo / ".kanban" / "board.jsonl"
    assert dest_board.read_bytes() == (board_dir / "board.jsonl").read_bytes()
    assert (repo / ".dev-notes" / "alpha.md").read_bytes() == (notes_dir / "alpha.md").read_bytes()
    assert (repo / ".dev-notes" / "beta.md").is_file()
    assert result["note_count"] == 2
    assert result["notes_copied"] == 2
    assert result["card_count"] >= 1


def test_export_manifest_tracks_board(tmp_path):
    board_dir, notes_dir = _canonical(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()

    result = export(repo, board_dir=board_dir, notes_dir=notes_dir)

    manifest = json.loads(Path(result["manifest"]).read_text())
    board_bytes = (board_dir / "board.jsonl").read_bytes()
    assert manifest["note_count"] == 2
    assert manifest["card_count"] == result["card_count"]
    assert manifest["board_sha256"] == hashlib.sha256(board_bytes).hexdigest()
    assert manifest["exported_at"]
    assert manifest["source"] == str(board_dir)


def test_export_prunes_stale_repo_notes_but_keeps_legacy(tmp_path):
    board_dir, notes_dir = _canonical(tmp_path)
    repo = tmp_path / "repo"
    legacy = repo / ".dev-notes" / "store"
    legacy.mkdir(parents=True)
    (legacy / "notes.journal.jsonl").write_text('{"id": "old"}\n')
    orphan = repo / ".dev-notes" / "gone.md"
    orphan.write_text("# no longer canonical\n")

    result = export(repo, board_dir=board_dir, notes_dir=notes_dir)

    assert not orphan.exists(), "stale canonical-absent note must be pruned"
    assert result["notes_removed"] == 1
    # grandfathered legacy artifacts are not ours to delete
    assert (legacy / "notes.journal.jsonl").is_file()


def test_export_refuses_when_canonical_board_missing(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ExportError, match="no canonical board"):
        export(repo, board_dir=tmp_path / "nope", notes_dir=tmp_path / "notes")


def test_export_refuses_when_canonical_notes_dir_missing(tmp_path):
    """A missing notes dir must never masquerade as zero notes (that
    would delete every mirrored note on the next sync)."""
    board_dir = tmp_path / "canonical-board"
    PlannerStore(board_dir=board_dir, notes_dir=tmp_path / "unused").add_card("x")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".dev-notes").mkdir()
    (repo / ".dev-notes" / "keep.md").write_text("# keep\n")

    with pytest.raises(ExportError, match="no canonical notes dir"):
        export(repo, board_dir=board_dir, notes_dir=tmp_path / "missing")

    assert (repo / ".dev-notes" / "keep.md").is_file()


# ── union merge: the mirror must never destroy repo-only cards ────────────


def _dest_store(repo: Path) -> PlannerStore:
    return PlannerStore(board_dir=repo / ".kanban", notes_dir=repo / ".dev-notes")


def _dest_cards(repo: Path) -> dict[str, object]:
    return {c.id: c for c in _dest_store(repo).load_board().cards}


def _dest_lines(repo: Path) -> list[str]:
    raw = (repo / ".kanban" / "board.jsonl").read_text(encoding="utf-8")
    return [line for line in raw.splitlines() if line.strip()]


def test_export_preserves_repo_only_cards(tmp_path):
    """The repo copy may hold cards the canonical store never saw — sessions
    commit cards straight to the repo. A byte-copy mirror deletes them
    (observed 2026-10-31: 1136 cards -> 12 on main)."""
    board_dir, notes_dir = _canonical(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    _dest_store(repo).add_card("repo only card")

    export(repo, board_dir=board_dir, notes_dir=notes_dir)

    titles = {c.title for c in _dest_cards(repo).values()}
    assert "repo only card" in titles, "repo-only card must survive the sync"
    assert "sync me" in titles, "canonical card must still be exported"


def test_export_canonical_wins_on_shared_id(tmp_path):
    """Union means canonical decides the content of cards it owns; the repo
    copy is never the authority for a card canonical also has."""
    board_dir, notes_dir = _canonical(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    export(repo, board_dir=board_dir, notes_dir=notes_dir)

    dest = _dest_store(repo)
    board = dest.load_board()
    card_id = board.cards[0].id
    board.cards[0].title = "locally clobbered"
    board.cards[0].column = "done"
    dest.save_board(board)

    export(repo, board_dir=board_dir, notes_dir=notes_dir)

    card = _dest_cards(repo)[card_id]
    assert card.title == "sync me"
    assert card.column == "todo"


def test_export_seals_chain_after_union(tmp_path):
    """The gate rejects an unsealed board (chain_index=-1), so a union of two
    independently chained boards must be re-sealed before it is written."""
    board_dir, notes_dir = _canonical(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    _dest_store(repo).add_card("repo only card")

    export(repo, board_dir=board_dir, notes_dir=notes_dir)

    lines = _dest_lines(repo)
    assert json.loads(lines[0]).get("schema") == "planner/1", "schema header must stay line 1"

    store = _dest_store(repo)
    assert store.verify_chain() == [], "union output must be re-sealed"
    cards = store.load_board().cards
    assert [c.chain_index for c in cards] == list(range(len(cards)))
    assert len({c.id for c in cards}) == len(cards), "no duplicate card ids"


def test_export_union_is_idempotent(tmp_path):
    """Two consecutive syncs over an unchanged store produce identical bytes
    (the timer runs every ~12 min; a non-idempotent export spams commits)."""
    board_dir, notes_dir = _canonical(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    _dest_store(repo).add_card("repo only card")

    export(repo, board_dir=board_dir, notes_dir=notes_dir)
    first = (repo / ".kanban" / "board.jsonl").read_bytes()

    export(repo, board_dir=board_dir, notes_dir=notes_dir)
    second = (repo / ".kanban" / "board.jsonl").read_bytes()

    assert first == second
