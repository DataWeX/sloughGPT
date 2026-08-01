"""
Tests for planner.kanban — the kanban CLI (``kanban`` / ``planner kanban``),
driven through ``cli_main`` with an explicit ``--dir`` so every command
operates on an isolated temp board.
"""

import re

import pytest

from planner.kanban import KanbanStore, cli_main, reset_kanban_store


@pytest.fixture(autouse=True)
def _isolate_store():
    reset_kanban_store()


@pytest.fixture
def board_dir(tmp_path):
    return tmp_path / "board"


def _run(board_dir, capsys, *parts):
    code = cli_main(["--dir", str(board_dir), *parts])
    return code, capsys.readouterr().out


def _add(board_dir, capsys, title, *extra):
    code, out = _run(board_dir, capsys, "add", title, *extra)
    assert code == 0, out
    m = re.search(r"Added: (\S+)", out)
    assert m, out
    return m.group(1)


def test_init_creates_board(board_dir, capsys):
    code, out = _run(board_dir, capsys, "init", "--name", "sprint-1")
    assert code == 0
    assert "sprint-1" in out
    assert "todo" in out and "in_progress" in out and "done" in out
    assert (board_dir / "board.json").exists()


def test_init_force_recreates(board_dir, capsys):
    _add(board_dir, capsys, "Stale card")
    code, out = _run(board_dir, capsys, "init", "--name", "fresh", "--force")
    assert code == 0
    assert "fresh" in out
    board = KanbanStore(board_dir=board_dir).load_board()
    assert board.name == "fresh"
    assert board.cards == []


def test_add_card_defaults(board_dir, capsys):
    card_id = _add(board_dir, capsys, "Fix boot order")
    code, out = _run(board_dir, capsys, "show", card_id)
    assert code == 0
    assert "Fix boot order" in out
    assert "column:   todo" in out
    assert "priority: medium" in out
    assert "tags:     none" in out


def test_add_card_with_fields(board_dir, capsys):
    card_id = _add(board_dir, capsys, "Ship v2", "--column", "in_progress",
                   "--priority", "high", "--tags", "kernel,os",
                   "--due", "2026-08-15", "--assignee", "mana",
                   "--desc", "release the kraken")
    code, out = _run(board_dir, capsys, "show", card_id)
    assert code == 0
    assert "column:   in_progress" in out
    assert "priority: high" in out
    assert "tags:     kernel, os" in out
    assert "assignee: mana" in out
    assert "due:      2026-08-15" in out
    assert "release the kraken" in out


def test_add_rejects_invalid_priority(board_dir):
    with pytest.raises(SystemExit) as exc:
        cli_main(["--dir", str(board_dir), "add", "Bad", "--priority", "uber"])
    assert exc.value.code == 2


def test_list_filters(board_dir, capsys):
    _add(board_dir, capsys, "Kernel task", "--column", "in_progress", "--priority", "high", "--tags", "kernel")
    _add(board_dir, capsys, "UI task", "--priority", "low", "--tags", "ui")
    code, out = _run(board_dir, capsys, "list", "--column", "in_progress")
    assert code == 0
    assert "Kernel task" in out and "UI task" not in out
    code, out = _run(board_dir, capsys, "list", "--tag", "ui")
    assert code == 0
    assert "UI task" in out and "Kernel task" not in out
    code, out = _run(board_dir, capsys, "list", "--priority", "high")
    assert code == 0
    assert "Kernel task" in out and "UI task" not in out


def test_list_empty(board_dir, capsys):
    code, out = _run(board_dir, capsys, "list")
    assert code == 0
    assert "No cards found." in out


def test_show_unknown_returns_1(board_dir, capsys):
    code, out = _run(board_dir, capsys, "show", "00000000_nope")
    assert code == 1
    assert "Card not found" in out


def test_edit_updates_fields(board_dir, capsys):
    card_id = _add(board_dir, capsys, "Old name", "--priority", "low")
    code, out = _run(board_dir, capsys, "edit", card_id,
                      "--title", "New name", "--priority", "critical", "--tags", "a,b",
                      "--assignee", "mana", "--due", "2026-09-01")
    assert code == 0
    assert "Updated:" in out
    code, out = _run(board_dir, capsys, "show", card_id)
    assert code == 0
    assert "New name" in out
    assert "priority: critical" in out
    assert "tags:     a, b" in out


def test_edit_no_changes_returns_1(board_dir, capsys):
    card_id = _add(board_dir, capsys, "T")
    code, out = _run(board_dir, capsys, "edit", card_id)
    assert code == 1
    assert "No changes specified." in out


def test_edit_unknown_returns_1(board_dir, capsys):
    code, out = _run(board_dir, capsys, "edit", "00000000_nope", "--priority", "high")
    assert code == 1
    assert "Card not found" in out


def test_move_card(board_dir, capsys):
    card_id = _add(board_dir, capsys, "Move me", "--column", "todo")
    code, out = _run(board_dir, capsys, "move", card_id, "done")
    assert code == 0
    assert "->  done" in out
    code, out = _run(board_dir, capsys, "show", card_id)
    assert "column:   done" in out


def test_move_unknown_column_returns_1(board_dir, capsys):
    card_id = _add(board_dir, capsys, "Stay")
    code, out = _run(board_dir, capsys, "move", card_id, "frozen")
    assert code == 1
    assert "Card not found" in out


def test_delete_removes_card(board_dir, capsys):
    card_id = _add(board_dir, capsys, "Doomed")
    code, out = _run(board_dir, capsys, "rm", card_id)
    assert code == 0
    assert "Deleted." in out
    code, out = _run(board_dir, capsys, "list")
    assert "No cards found." in out


def test_delete_unknown_prints_not_found(board_dir, capsys):
    code, out = _run(board_dir, capsys, "delete", "00000000_nope")
    assert code == 0
    assert "Card not found." in out


def test_board_renders_cards(board_dir, capsys):
    c1 = _add(board_dir, capsys, "First card", "--column", "in_progress")
    c2 = _add(board_dir, capsys, "Second card")
    code, out = _run(board_dir, capsys, "board")
    assert code == 0
    assert "Kanban:" in out
    assert c1 in out and c2 in out
    assert "2 card(s)" in out


def test_note_add_list_delete(board_dir, capsys):
    card_id = _add(board_dir, capsys, "Commented")
    code, out = _run(board_dir, capsys, "note", "add", card_id, "look into this", "--author", "mana")
    assert code == 0
    assert "Note added:" in out
    note_id = re.search(r"\[(\S+)\]", out).group(1)
    code, out = _run(board_dir, capsys, "note", "list", card_id)
    assert code == 0
    assert "look into this" in out
    assert "[mana]" in out
    code, out = _run(board_dir, capsys, "note", "delete", card_id, note_id)
    assert code == 0
    assert "Note deleted." in out
    code, out = _run(board_dir, capsys, "note", "list", card_id)
    assert "No notes on this card." in out


def test_note_unknown_card_returns_1(board_dir, capsys):
    code, out = _run(board_dir, capsys, "note", "add", "00000000_nope", "hi")
    assert code == 1
    assert "Card not found." in out


def test_columns_and_column_management(board_dir, capsys):
    code, out = _run(board_dir, capsys, "columns")
    assert code == 0
    assert "todo" in out and "in_progress" in out and "done" in out
    code, out = _run(board_dir, capsys, "column-add", "blocked", "--wip", "2")
    assert code == 0
    assert "blocked" in out
    code, out = _run(board_dir, capsys, "columns")
    assert "blocked" in out and "(wip: 2)" in out
    code, out = _run(board_dir, capsys, "column-rename", "blocked", "frozen")
    assert code == 0
    assert "Renamed." in out
    code, out = _run(board_dir, capsys, "columns")
    assert "frozen" in out and "blocked" not in out
    code, out = _run(board_dir, capsys, "column-rm", "frozen")
    assert code == 0
    assert "Removed." in out
    code, out = _run(board_dir, capsys, "columns")
    assert "frozen" not in out


def test_column_add_duplicate_fails(board_dir, capsys):
    code, out = _run(board_dir, capsys, "column-add", "todo")
    assert code == 1
    assert "Failed." in out


def test_archive_done(board_dir, capsys):
    keep = _add(board_dir, capsys, "Keep me", "--column", "in_progress")
    _add(board_dir, capsys, "Done one", "--column", "done")
    code, out = _run(board_dir, capsys, "archive")
    assert code == 0
    assert "Archived 1 done card(s)." in out
    code, out = _run(board_dir, capsys, "show", keep)
    assert code == 0
    code, out = _run(board_dir, capsys, "list")
    assert "Done one" not in out


def test_search_cards(board_dir, capsys):
    _add(board_dir, capsys, "Engine tuning", "--tags", "kernel", "--desc", "sweep the camshaft")
    _add(board_dir, capsys, "Paint the wall", "--tags", "ui")
    for query in ("engine", "kernel", "camshaft"):
        code, out = _run(board_dir, capsys, "search", query)
        assert code == 0
        assert "Engine tuning" in out
        assert "Paint the wall" not in out
    code, out = _run(board_dir, capsys, "search", "missing-thing")
    assert code == 0
    assert "No cards matching" in out


def test_stats(board_dir, capsys):
    _add(board_dir, capsys, "One", "--column", "done", "--priority", "high")
    _add(board_dir, capsys, "Two", "--column", "todo", "--priority", "low")
    _add(board_dir, capsys, "Three", "--column", "done", "--priority", "medium")
    code, out = _run(board_dir, capsys, "stats")
    assert code == 0
    assert "Total cards: 3" in out
    assert "done" in out and "todo" in out
    assert "high" in out and "low" in out


def test_no_command_returns_1(board_dir, capsys):
    code, _ = _run(board_dir, capsys)
    assert code == 1


def test_add_card_invalid_priority_falls_back_to_medium(board_dir):
    store = KanbanStore(board_dir=board_dir)
    card = store.add_card("Fallback", priority="uber")
    assert card.priority == "medium"


def test_find_one_ambiguous_prefix_returns_none(board_dir):
    store = KanbanStore(board_dir=board_dir)
    store.add_card("Alpha one")
    store.add_card("Alpha two")
    assert store.get_card("alpha") is None


def test_update_card_ignores_unknown_keys(board_dir):
    store = KanbanStore(board_dir=board_dir)
    card = store.add_card("Stable")
    updated = store.update_card(card.short_id, bogus="x")
    assert updated is not None
    assert not hasattr(updated, "bogus")
    assert store.get_card(card.short_id).title == "Stable"


def test_rename_column_migrates_cards(board_dir):
    store = KanbanStore(board_dir=board_dir)
    card = store.add_card("Old col", column="in_progress")
    assert store.rename_column("in_progress", "doing")
    assert store.get_card(card.short_id).column == "doing"
    assert [c.name for c in store.list_columns() if c.name == "doing"]


def test_remove_column_migrates_cards(board_dir):
    store = KanbanStore(board_dir=board_dir)
    card = store.add_card("Drop col", column="in_progress")
    assert store.remove_column("in_progress", move_to="review")
    assert store.get_card(card.short_id).column == "review"
    assert not [c for c in store.list_columns() if c.name == "in_progress"]


def test_archive_done_empty(board_dir):
    store = KanbanStore(board_dir=board_dir)
    store.add_card("Keep")
    assert store.archive_done() == 0


def test_list_cards_by_assignee(board_dir):
    store = KanbanStore(board_dir=board_dir)
    store.add_card("Mine", assignee="mana")
    store.add_card("Theirs")
    results = store.list_cards(assignee="mana")
    assert [c.title for c in results] == ["Mine"]
