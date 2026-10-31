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
