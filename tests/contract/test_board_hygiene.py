"""The board hygiene doctor must fix only what needs no judgement.

``.kanban/board.jsonl`` is append-only and shared by every session, and it is a
*derived* view -- notes are the source of truth.  Three defect classes have
accumulated in it, and only two of them are mechanical:

* exact duplicates (same id, column, title, description) carry no information
  the surviving copy lacks, so they are safe to drop;
* a list-typed ``description`` is a schema violation that breaks every
  consumer doing ``description.strip()``;
* multi-column duplicates (one id in ``todo`` *and* ``done``) and
  divergent-body duplicates ask which copy is *true*, which is a judgement
  about the work.  Repairing those automatically would silently pick a winner.

The tests below pin that boundary -- ``repair`` may drop redundancy but must
never drop a decision.
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

    dropped, _fixed = doctor.apply_repairs(board, doctor.classify(doctor._read_lines(board)))

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
