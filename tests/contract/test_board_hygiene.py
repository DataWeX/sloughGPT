"""The board hygiene doctor must fix only what needs no judgement — and settle
the rest strictly from the notes.

``.kanban/board.jsonl`` is append-only and shared by every session, and it is a
*derived* view -- notes are the source of truth.  Three defect classes have
accumulated in it, and each has exactly one legitimate resolver:

* exact duplicates (same id, column, title, description) carry no information
  the surviving copy lacks, so they are safe to drop;
* a list-typed ``description`` is a schema violation that breaks every
  consumer doing ``description.strip()``;
* multi-column duplicates (one id in ``todo`` *and* ``done``) and
  divergent-body duplicates ask which copy is *true*.  ``settle`` answers that
  **from the notes** (note status for columns, superset/note-body for text) and
  must leave the id untouched and reported whenever the sources are silent or
  disagreeing -- ambiguity is never resolved by guessing.

The tests below pin that boundary -- ``repair`` may drop redundancy but never a
decision; ``settle`` may drop a decision but only with a note behind it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import check_board_hygiene as doctor  # noqa: E402


def _line(card_id: str, column: str, title: str, description: object) -> str:
    return json.dumps(
        {"id": card_id, "column": column, "title": title, "description": description},
        sort_keys=True,
    )


IDENTICAL_A = _line("aaaa-1111", "todo", "Same card", "body")
MULTI_TODO = _line("bbbb-2222", "todo", "Split card", "body")
MULTI_DONE = _line("bbbb-2222", "done", "Split card", "body")
DIVERGENT_1 = _line("cccc-3333", "todo", "Divergent card", "the long one")
DIVERGENT_2 = _line("cccc-3333", "todo", "Divergent card", "the short one")
LIST_DESC = _line("dddd-4444", "todo", "List desc", ["part one", "part two"])
HEADER = json.dumps({"schema": "planner/1", "name": "board", "columns": []})


def _lines(*cards: str) -> list[str]:
    return [HEADER, *cards]


def test_exact_duplicates_are_identified() -> None:
    classes = doctor.classify(_lines(IDENTICAL_A, IDENTICAL_A))

    assert len(classes["identical"]) == 1, classes["identical"]
    card_id, group = classes["identical"][0]
    assert card_id == "aaaa-1111" and len(group) == 2


def test_multi_column_duplicates_are_their_own_class() -> None:
    """A card in two columns is not a redundancy -- it is a contradiction."""
    classes = doctor.classify(_lines(MULTI_TODO, MULTI_DONE))

    assert classes["multi_column"], "a card in todo AND done must be flagged"
    assert not classes["identical"], "columns differ, so this is not an exact duplicate"


def test_divergent_bodies_are_their_own_class() -> None:
    classes = doctor.classify(_lines(DIVERGENT_1, DIVERGENT_2))

    assert len(classes["divergent"]) == 1, classes["divergent"]
    assert not classes["identical"], "different bodies must not count as duplicates"


def test_list_typed_descriptions_are_flagged() -> None:
    classes = doctor.classify(_lines(LIST_DESC))

    assert len(classes["bad_type"]) == 1, classes["bad_type"]
    assert classes["bad_type"][0]["id"] == "dddd-4444"


def test_unparseable_lines_are_reported_not_raised() -> None:
    classes = doctor.classify([HEADER, "{not json", IDENTICAL_A])

    assert len(classes["malformed"]) == 1, classes["malformed"]


def test_repair_drops_only_the_redundant_copy() -> None:
    lines = _lines(IDENTICAL_A, IDENTICAL_A, MULTI_TODO, MULTI_DONE)
    classes = doctor.classify(lines)

    rewritten, dropped, _fixed = doctor.repair(lines, classes)

    assert dropped == 1, "exactly one redundant line should go"
    assert rewritten.count(IDENTICAL_A) == 1, "the surviving copy must remain"
    assert MULTI_TODO in rewritten and MULTI_DONE in rewritten, (
        "repair must never choose between two columns -- that needs a decision"
    )


def test_repair_keeps_divergent_duplicates() -> None:
    lines = _lines(DIVERGENT_1, DIVERGENT_2)
    classes = doctor.classify(lines)

    rewritten, dropped, _fixed = doctor.repair(lines, classes)

    assert dropped == 0, "neither divergent copy may be dropped"
    assert DIVERGENT_1 in rewritten and DIVERGENT_2 in rewritten


def test_repair_normalizes_list_descriptions() -> None:
    lines = _lines(LIST_DESC)
    classes = doctor.classify(lines)

    rewritten, _dropped, fixed = doctor.repair(lines, classes)

    assert fixed == 1
    repaired = json.loads(rewritten[1])
    assert repaired["description"] == "part one\npart two", (
        f"a list description must become the string consumers expect, got "
        f"{repaired['description']!r}"
    )
    assert isinstance(repaired["description"], str)


def test_apply_writes_a_backup_and_leaves_a_parseable_board(tmp_path: Path) -> None:
    board = tmp_path / "board.jsonl"
    board.write_text("\n".join(_lines(IDENTICAL_A, IDENTICAL_A)) + "\n", encoding="utf-8")

    dropped, _fixed, _settled, _unresolved = doctor.apply_repairs(
        board, doctor.classify(doctor._read_lines(board))
    )

    backup = board.with_suffix(board.suffix + ".bak-hygiene")
    assert backup.is_file(), "the original must be preserved before any repair"
    assert json.loads(backup.read_text().splitlines()[1]) == json.loads(IDENTICAL_A), (
        "the backup must hold the pre-repair content"
    )
    assert dropped == 1
    kept = [json.loads(line) for line in board.read_text().splitlines() if line.strip()]
    assert len(kept) == 2, f"header + one card expected, got {len(kept)}"
    assert kept[-1]["id"] == "aaaa-1111"


def test_strict_exits_non_zero_while_any_defect_remains(tmp_path: Path) -> None:
    board = tmp_path / "board.jsonl"
    # A multi-column defect survives repair, so --strict must still fail.
    board.write_text(
        "\n".join(_lines(IDENTICAL_A, IDENTICAL_A, MULTI_TODO, MULTI_DONE)) + "\n",
        encoding="utf-8",
    )
    argv = ["prog", "--board", str(board), "--apply", "--strict"]

    import contextlib
    import io

    original = sys.argv
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(io.StringIO()):
            rc = doctor.main()
    finally:
        sys.argv = original

    assert rc == 1, (
        "--strict must fail while a decision is still outstanding, even after "
        "the mechanical repairs"
    )
    assert doctor.has_defects(doctor.classify(doctor._read_lines(board))), (
        "the multi-column defect must still be reported"
    )
    remaining = doctor.classify(doctor._read_lines(board))
    assert not remaining["identical"], (
        "the exact duplicates should already have been repaired by --apply"
    )


# ── settle: decision classes resolved FROM THE NOTES ────────────────────────


def test_settle_multi_column_keeps_note_status_column() -> None:
    """The note's status is the source of truth: keep its row, drop the stale ones."""
    truth = {"Split card": {"status": "done", "body": "body", "source": "journal"}}

    lines, settled, unresolved = doctor.settle(_lines(MULTI_TODO, MULTI_DONE), truth)

    assert not unresolved, unresolved
    assert len(settled) == 1, settled
    records = [json.loads(line) for line in lines[1:]]
    assert [r["column"] for r in records] == ["done"]


def test_settle_without_note_leaves_multi_column_alone() -> None:
    """No source of truth, no decision: report, never drop."""
    lines, settled, unresolved = doctor.settle(_lines(MULTI_TODO, MULTI_DONE), {})

    assert not settled
    assert len(unresolved) == 1 and "no note" in unresolved[0], unresolved
    assert len(lines) == 3, "nothing may be dropped without a source of truth"


def test_settle_divergent_prefers_the_superset() -> None:
    """An annotation appended after the note snapshot must never be destroyed:
    the copy containing the other copy wins, even with no note at all."""
    long = _line("cccc-3333", "todo", "Divergent card", "the short one\n[COORD] claimed")
    short = _line("cccc-3333", "todo", "Divergent card", "the short one")

    lines, settled, unresolved = doctor.settle(_lines(long, short), {})

    assert not unresolved, unresolved
    assert len(settled) == 1 and "superset" in settled[0], settled
    records = [json.loads(line) for line in lines[1:]]
    assert records[0]["description"] == "the short one\n[COORD] claimed"


def test_settle_divergent_falls_back_to_note_body() -> None:
    """Neither copy contains the other: the copy matching the note's body wins."""
    truth = {"Divergent card": {"status": "todo", "body": "the short one", "source": "journal"}}

    lines, settled, unresolved = doctor.settle(_lines(DIVERGENT_1, DIVERGENT_2), truth)

    assert not unresolved, unresolved
    assert len(settled) == 1 and "note-body" in settled[0], settled
    records = [json.loads(line) for line in lines[1:]]
    assert records[0]["description"] == "the short one"


def test_settle_marks_conflicting_sources_unresolved(tmp_path: Path) -> None:
    """journal vs markdown disagree = the open canonical-store question
    (card b3435a2f): the conflict is marked, never guessed."""
    journal = tmp_path / "notes.journal.jsonl"
    journal.write_text(json.dumps({"title": "Split card", "status": "todo", "body": "x"}) + "\n")
    md_dir = tmp_path / "md"
    md_dir.mkdir()
    (md_dir / "note.md").write_text("---\ntitle: Split card\nstatus: done\n---\n\nbody\n")

    truth = doctor.load_note_truth(journal, md_dir)

    assert truth["Split card"]["status"] is None, "disagreeing sources must not resolve"
    assert truth["Split card"]["source"].startswith("conflict:"), truth["Split card"]
    lines, settled, unresolved = doctor.settle(_lines(MULTI_TODO, MULTI_DONE), truth)
    assert not settled and len(unresolved) == 1, (settled, unresolved)
    assert len(lines) == 3, "an unresolved id must keep all its rows"


def test_load_note_truth_links_by_board_uuid_and_normalizes_titles(tmp_path: Path) -> None:
    md_dir = tmp_path / "md"
    md_dir.mkdir()
    (md_dir / "note.md").write_text(
        '---\nboard: bbbb-2222\ntitle: "Card 087 — Split card"\nstatus: done\n---\n\nbody\n'
    )

    truth = doctor.load_note_truth(None, md_dir)

    assert truth["bbbb-2222"]["status"] == "done", "the board uuid link must resolve directly"
    assert truth["Split card"]["status"] == "done", "the Card 0NN prefix must be normalized"
    assert truth["Split card"]["body"] == "body"


def test_strict_passes_once_settle_resolves_the_decisions(tmp_path: Path) -> None:
    board = tmp_path / "board.jsonl"
    board.write_text("\n".join(_lines(MULTI_TODO, MULTI_DONE)) + "\n", encoding="utf-8")
    journal = tmp_path / "notes.journal.jsonl"
    journal.write_text(json.dumps({"title": "Split card", "status": "done", "body": "body"}) + "\n")

    import contextlib
    import io

    argv = [
        "prog",
        "--board",
        str(board),
        "--settle",
        "--journal",
        str(journal),
        "--md-dir",
        str(tmp_path / "no-such-dir"),
        "--strict",
    ]
    original = sys.argv
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(io.StringIO()):
            rc = doctor.main()
    finally:
        sys.argv = original

    assert rc == 0, "--strict must pass once the notes have settled every decision"
    assert not doctor.has_defects(doctor.classify(doctor._read_lines(board)))
