"""
Tests for app_planner.migrate — repo-local -> canonical store import.

Covers: chain verification gate, union-by-id with canonical precedence,
backup-before-write, originals untouched, dry-run writes nothing, and the
no-source error path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app_planner.migrate import MigrationError, migrate
from app_planner.store import PlannerStore


def _seed_repo_local(root: Path, chained: bool = True) -> tuple[Path, Path]:
    """Create a repo-local .kanban + .dev-notes with one note.

    *chained=True* (default) computes the hash chain, matching how a real
    repo board is persisted; *chained=False* leaves cards as plain
    add_card wrote them (chain_index=-1), covering the unchained-source path.
    """
    board_dir = root / ".kanban"
    notes_dir = root / ".dev-notes"
    store = PlannerStore(board_dir=board_dir, notes_dir=notes_dir)
    store.add_card("first card", column="todo")
    store.add_card("second card", column="wip")
    if chained:
        store.compute_chains()
    store.create_note("first note", body="note body")
    return board_dir, notes_dir


def test_migrate_imports_cards_and_notes(tmp_path):
    repo = tmp_path / "repo"
    canonical_board = tmp_path / "canonical-board"
    canonical_notes = tmp_path / "canonical-notes"
    _seed_repo_local(repo)

    result = migrate(repo, board_dir=canonical_board, notes_dir=canonical_notes)

    assert result["board"]["imported"] == 2
    assert result["notes"]["imported"] == 1
    assert result["board"]["skipped"] == 0

    dst = PlannerStore(board_dir=canonical_board, notes_dir=canonical_notes)
    titles = {c.title for c in dst.load_board().cards}
    assert titles == {"first card", "second card"}
    assert [n.title for n in dst.list_notes()] == ["first note"]
    # Union must carry a valid canonical chain.
    assert dst.verify_chain() == []


def test_migrate_canonical_wins_on_id_conflict(tmp_path):
    from dataclasses import replace

    repo = tmp_path / "repo"
    canonical_board = tmp_path / "canonical-board"
    canonical_notes = tmp_path / "canonical-notes"
    _seed_repo_local(repo)

    dst = PlannerStore(board_dir=canonical_board, notes_dir=canonical_notes)
    src = PlannerStore(board_dir=repo / ".kanban", notes_dir=repo / ".dev-notes")
    conflicting = src.load_board().cards[0]
    # Same id as a source card, different title — canonical must survive.
    dst._append_card(replace(conflicting, title="canonical version"))

    result = migrate(repo, board_dir=canonical_board, notes_dir=canonical_notes)

    assert result["board"]["imported"] == 1  # only the non-conflicting card
    assert result["board"]["skipped"] == 1
    titles = {c.title for c in dst.load_board().cards}
    assert "canonical version" in titles
    assert conflicting.title not in titles
    assert dst.verify_chain() == []


def test_migrate_accepts_unchained_source(tmp_path):
    """Plain add_card boards (chain_index=-1) are legitimate — they import
    with a fresh chain instead of being refused as 'broken'."""
    repo = tmp_path / "repo"
    canonical_board = tmp_path / "canonical-board"
    canonical_notes = tmp_path / "canonical-notes"
    _seed_repo_local(repo, chained=False)

    result = migrate(repo, board_dir=canonical_board, notes_dir=canonical_notes)

    assert result["board"]["imported"] == 2
    dst = PlannerStore(board_dir=canonical_board, notes_dir=canonical_notes)
    assert dst.verify_chain() == []


def test_migrate_broken_source_chain_refuses(tmp_path):
    repo = tmp_path / "repo"
    board_dir, _ = _seed_repo_local(repo)

    # Tamper with a card line -> chain verification must fail.
    board_file = board_dir / "board.jsonl"
    lines = board_file.read_text().splitlines()
    card_idx = next(i for i, ln in enumerate(lines) if '"title"' in ln)
    record = json.loads(lines[card_idx])
    record["title"] = "tampered"
    lines[card_idx] = json.dumps(record, sort_keys=True)
    board_file.write_text("\n".join(lines) + "\n")

    with pytest.raises(MigrationError, match="chain broken"):
        migrate(
            repo,
            board_dir=tmp_path / "canonical-board",
            notes_dir=tmp_path / "canonical-notes",
        )


def test_migrate_leaves_source_untouched(tmp_path):
    repo = tmp_path / "repo"
    board_dir, notes_dir = _seed_repo_local(repo)
    board_before = (board_dir / "board.jsonl").read_bytes()
    notes_before = (notes_dir / "notes.journal.jsonl").read_bytes()

    migrate(repo, board_dir=tmp_path / "cb", notes_dir=tmp_path / "cn")

    assert (board_dir / "board.jsonl").read_bytes() == board_before
    assert (notes_dir / "notes.journal.jsonl").read_bytes() == notes_before


def test_migrate_backs_up_existing_canonical_board(tmp_path):
    repo = tmp_path / "repo"
    canonical_board = tmp_path / "canonical-board"
    canonical_notes = tmp_path / "canonical-notes"
    _seed_repo_local(repo)

    dst = PlannerStore(board_dir=canonical_board, notes_dir=canonical_notes)
    dst.add_card("pre-existing canonical card")

    result = migrate(repo, board_dir=canonical_board, notes_dir=canonical_notes)

    assert result["backups"], "expected a backup of the pre-existing board"
    backup_path = Path(result["backups"][0])
    assert backup_path.is_file()
    # Backup preserves the pre-migration content byte-for-byte.
    assert "pre-existing canonical card" in backup_path.read_text()


def test_migrate_dry_run_writes_nothing(tmp_path):
    repo = tmp_path / "repo"
    canonical_board = tmp_path / "canonical-board"
    canonical_notes = tmp_path / "canonical-notes"
    _seed_repo_local(repo)

    result = migrate(repo, dry_run=True, board_dir=canonical_board, notes_dir=canonical_notes)

    assert result["dry_run"] is True
    assert result["board"]["imported"] == 2
    assert result["notes"]["imported"] == 1
    assert not canonical_board.exists() or not list(canonical_board.iterdir())
    assert not canonical_notes.exists() or not list(canonical_notes.iterdir())


def test_migrate_without_source_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("app_planner.config.find_project_root", lambda start=None: None)
    with pytest.raises(MigrationError, match="no repo-local planner data"):
        migrate(
            None,
            board_dir=tmp_path / "cb",
            notes_dir=tmp_path / "cn",
        )
