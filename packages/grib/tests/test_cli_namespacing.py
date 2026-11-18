"""Namespace the CLI: store-touching commands are reachable under
``grib note ...`` / ``grib board ...``, not only as bare top-level verbs.

The top-level spellings (``grib sync``, ``grib verify``, ...) stay supported
on purpose — docs, ``scripts/kanban_sync.sh`` and other sessions' habits call
them. The nested form is a projection of the SAME handlers, never a
replacement, so both spellings must behave byte-identically.
"""

from __future__ import annotations

import contextlib
import io
import uuid

import pytest
from grib.cli import build_parser
from grib.cli import main as cli_main
from grib.store import reset_store


@pytest.fixture(autouse=True)
def _reset():
    reset_store()
    yield
    reset_store()


def _run(store_dir, *parts) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            code = cli_main(["--board-dir", str(store_dir), "--notes-dir", str(store_dir), *parts])
        except SystemExit as e:
            code = e.code or 1
    return code, buf.getvalue()


def _seed(store_dir, title: str = "Seeded") -> None:
    """Create a note WITHOUT a card.

    Via the CLI, `note new` auto-syncs (_auto_sync), so seeding that way
    would pre-create the card and make every sync assertion see a no-op.
    """
    from grib.store import PlannerStore

    PlannerStore(board_dir=store_dir, notes_dir=store_dir).create_note(title)


# ── Nested paths dispatch to the real handlers ───────────────────────────


def test_board_sync_creates_cards(tmp_path):
    _seed(tmp_path)
    code, out = _run(tmp_path, "board", "sync")
    assert code == 0, out
    assert "1 new card" in out


def test_kanban_alias_sync_dispatches_too(tmp_path):
    """`grib kanban sync` must work — the alias is part of the surface."""
    _seed(tmp_path)
    code, out = _run(tmp_path, "kanban", "sync")
    assert code == 0, out
    assert "1 new card" in out


def test_note_verify_dispatches_to_verify_handler(tmp_path):
    """Exit 1 + verify's own message, not argparse's exit 2 + usage.

    An unregistered nested command would surface as `invalid choice` (2);
    reaching _verify with no match is 1 with `verify: no note matching`.
    """
    ghost = f"definitely-no-note-{uuid.uuid4().hex}"
    code, out = _run(tmp_path, "note", "verify", ghost)
    assert code == 1, out
    assert "verify: no note matching" in out
    assert "invalid choice" not in out


def test_board_snapshot_dispatches(tmp_path):
    _seed(tmp_path)
    _run(tmp_path, "board", "sync")
    repo = tmp_path / "repo"
    repo.mkdir()
    code, out = _run(tmp_path, "board", "snapshot", "--repo", str(repo))
    assert code == 0, out


def test_board_export_dispatches(tmp_path):
    _seed(tmp_path)
    _run(tmp_path, "board", "sync")
    repo = tmp_path / "repo"
    repo.mkdir()
    code, out = _run(tmp_path, "board", "export", "--repo", str(repo))
    assert code == 0, out
    assert (repo / ".kanban" / "board.jsonl").exists()


# ── Both spellings must be equivalent ────────────────────────────────────


def test_sync_spellings_produce_identical_output(tmp_path):
    from grib.store import reset_store

    top = tmp_path / "top"
    nested = tmp_path / "nested"
    top.mkdir()
    nested.mkdir()
    for d in (top, nested):
        reset_store()  # get_store() is a singleton — rebind per directory
        _seed(d)

    reset_store()
    code_a, out_a = _run(top, "sync", "--quiet")
    reset_store()
    code_b, out_b = _run(nested, "board", "sync", "--quiet")

    assert (code_a, out_a) == (code_b, out_b)


def test_verify_spellings_produce_identical_output(tmp_path):
    ghost = f"definitely-no-note-{uuid.uuid4().hex}"
    code_a, out_a = _run(tmp_path, "verify", ghost)
    code_b, out_b = _run(tmp_path, "note", "verify", ghost)

    assert (code_a, out_a) == (code_b, out_b)


# ── Surface shape: both spellings parse, dispatch keys stay aligned ─────


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["sync"], ("sync", None)),
        (["board", "sync"], ("board", "sync")),
        (["kanban", "sync"], ("kanban", "sync")),
        (["verify", "x"], ("verify", None)),
        (["note", "verify", "x"], ("note", "verify")),
        (["notes", "verify", "x"], ("notes", "verify")),
        (["snapshot"], ("snapshot", None)),
        (["board", "snapshot"], ("board", "snapshot")),
        (["export"], ("export", None)),
        (["board", "export"], ("board", "export")),
        # Both-stores admin verbs stay top-level.
        (["migrate"], ("migrate", None)),
        (["gui"], ("gui", None)),
    ],
)
def test_spellings_parse_to_expected_dispatch_keys(argv, expected):
    args = build_parser().parse_args(argv)
    assert (args.command, getattr(args, "subcommand", None)) == expected


def test_nested_sync_exposes_same_flags_as_top_level():
    top = build_parser().parse_args(["sync", "--quiet", "--prune"])
    nested = build_parser().parse_args(["board", "sync", "--quiet", "--prune"])
    assert (top.quiet, top.prune) == (nested.quiet, nested.prune) == (True, True)


def test_nested_verify_exposes_same_flags_as_top_level():
    top = build_parser().parse_args(["verify", "x", "--repo", "/r", "--board-file", "/b"])
    nested = build_parser().parse_args(
        ["note", "verify", "x", "--repo", "/r", "--board-file", "/b"]
    )
    assert (top.note, top.repo, top.board_file) == (
        nested.note,
        nested.repo,
        nested.board_file,
    )
