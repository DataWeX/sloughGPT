"""Board hygiene doctor: detect, and optionally repair, `.kanban/board.jsonl` defects.

The board is append-only JSONL shared by every session and worktree on this
repo (~83 at time of writing), and it is a *derived* view -- notes are the
source of truth, ``sync`` regenerates cards from them.  Two classes of defect
have accumulated in it, both of which break the tools that read it:

* **Duplicate ids.** The same card uuid written more than once.  ``move_card``
  and ``sync`` address a card by id, so a duplicate means every move is issued
  twice and a "converged" board keeps reporting the same move run over run
  (see the positional-matching comment on ``PlannerStore.sync``).
* **Multi-column duplicates.** The same id present in *different* columns --
  historically ``todo`` + ``in_progress`` + ``done``.  Worse than plain
  duplication: the card is simultaneously open and finished, so no queue
  reader can tell which is true.

A third, smaller defect is a **schema violation**: ``description`` written as a
JSON list rather than a string, which breaks every consumer that does
``description.strip()``.

This script reports all three.  ``--apply`` repairs only what can be repaired
without a decision:

* exact duplicates (identical id, column, title and description) are reduced to
  one -- nothing is lost, the dropped lines are byte-identical to the one kept;
* list-typed descriptions are joined back into a string.

Multi-column and divergent-description duplicates are **never** auto-fixed:
which copy is correct is a judgement about the work, not a mechanical
question, so they are reported for a human to decide.

Safety, because the file is live and shared:

* a backup is written beside the board before anything is modified;
* the rewrite is atomic (tmp -> fsync -> ``os.replace``), never truncate in
  place -- a torn board is unreadable by every session;
* the file is re-hashed immediately before publishing, and the run aborts if
  another session wrote while we were working.  This is the same race that
  produced the repeated notes-journal wipes, and a blind read-modify-write
  here would reproduce it.

Exit status is 0 by default and 1 under ``--strict`` while any defect remains,
so it can gate CI the way ``check_feature_tags.py`` does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_BOARD = Path(".kanban/board.jsonl")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_lines(path: Path) -> list[str]:
    """Every non-blank line, in file order, verbatim."""
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _desc_key(card: dict[str, Any]) -> str:
    """Stable rendering of a description for equality, whatever its type."""
    return json.dumps(card.get("description"), sort_keys=True, ensure_ascii=False)


def _normalized_description(card: dict[str, Any]) -> Any:
    """A list-typed description joined into the string form consumers expect."""
    desc = card.get("description")
    if isinstance(desc, list):
        return "\n".join(str(part) for part in desc)
    return desc


def classify(lines: list[str]) -> dict[str, Any]:
    """Sort board lines into the defect classes this doctor knows about."""
    cards: list[dict[str, Any]] = []
    malformed: list[str] = []
    for raw in lines:
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            malformed.append(raw[:80])
            continue
        if isinstance(record, dict) and "id" in record:
            cards.append(record)

    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        by_id[str(card["id"])].append(card)

    identical: list[tuple[str, list[dict[str, Any]]]] = []
    multi_column: list[tuple[str, list[dict[str, Any]]]] = []
    divergent: list[tuple[str, list[dict[str, Any]]]] = []
    for card_id, group in by_id.items():
        if len(group) < 2:
            continue
        signatures = {(c.get("title"), _desc_key(c)) for c in group}
        columns = {c.get("column") for c in group}
        if len(columns) > 1:
            multi_column.append((card_id, group))
        elif len(signatures) == 1:
            identical.append((card_id, group))
        else:
            divergent.append((card_id, group))

    bad_type = [
        c
        for c in cards
        if c.get("description") is not None and not isinstance(c.get("description"), str)
    ]

    return {
        "cards": cards,
        "malformed": malformed,
        "identical": identical,
        "multi_column": multi_column,
        "divergent": divergent,
        "bad_type": bad_type,
    }


def repair(lines: list[str], classes: dict[str, Any]) -> tuple[list[str], int, int]:
    """Return rewritten lines, fixing exact duplicates and list descriptions.

    Nothing else is touched: multi-column and divergent duplicates need a
    decision, so they survive the repair and stay reported.
    """
    redundant_ids = {card_id for card_id, _ in classes["identical"]}
    fixed_type = len(classes["bad_type"])
    seen: dict[str, int] = defaultdict(int)
    out: list[str] = []
    dropped = 0
    for raw in lines:
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            out.append(raw)
            continue
        if not (isinstance(record, dict) and "id" in record):
            out.append(raw)
            continue
        card_id = str(record["id"])
        seen[card_id] += 1
        if card_id in redundant_ids and seen[card_id] > 1:
            dropped += 1
            continue
        if record.get("description") is not None and not isinstance(record.get("description"), str):
            record["description"] = _normalized_description(record)
        out.append(json.dumps(record, sort_keys=True, ensure_ascii=False))
    return out, dropped, fixed_type


def report(classes: dict[str, Any], out: Any = sys.stdout) -> None:
    write = lambda text: print(text, file=out)  # noqa: E731
    write(f"  cards: {len(classes['cards'])}")
    if classes["malformed"]:
        write(f"  !! unparseable lines: {len(classes['malformed'])}")
        for raw in classes["malformed"][:5]:
            write(f"       {raw}")
    write(
        f"  exact duplicates : {len(classes['identical'])} id(s)  "
        f"-> {sum(len(g) - 1 for _, g in classes['identical'])} redundant line(s)  [auto-fixable]"
    )
    for card_id, group in classes["identical"]:
        write(f"      {card_id[:8]} x{len(group)} column={group[0].get('column')!r}")
    write(f"  multi-column     : {len(classes['multi_column'])} id(s)  [needs a decision]")
    for card_id, group in classes["multi_column"]:
        cols = [c.get("column") for c in group]
        write(f"      {card_id[:8]} columns={cols}  {group[0].get('title', '')[:48]!r}")
    write(f"  divergent body   : {len(classes['divergent'])} id(s)  [needs a decision]")
    for card_id, group in classes["divergent"]:
        lens = [len(c.get("description") or "") for c in group]
        write(f"      {card_id[:8]} description lengths={lens}")
    write(f"  list-typed desc  : {len(classes['bad_type'])} card(s)  [auto-fixable]")


def has_defects(classes: dict[str, Any]) -> bool:
    return bool(
        classes["malformed"]
        or classes["identical"]
        or classes["multi_column"]
        or classes["divergent"]
        or classes["bad_type"]
    )


def apply_repairs(board: Path, classes: dict[str, Any]) -> tuple[int, int]:
    """Repair the safe classes on disk.  Backs up, writes atomically, and
    aborts if another session touched the board while we were working."""
    before = _digest(board)
    lines = _read_lines(board)
    rewritten, dropped, fixed = repair(lines, classes)

    backup = board.with_suffix(board.suffix + ".bak-hygiene")
    backup.write_text(board.read_text(encoding="utf-8"), encoding="utf-8")

    if _digest(board) != before:
        raise RuntimeError(
            "board changed while repairing -- aborting without writing "
            "(another session is working on it); re-run"
        )

    handle, tmp_name = tempfile.mkstemp(dir=str(board.parent), prefix=".board.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as tmp:
            tmp.write("\n".join(rewritten) + "\n")
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, board)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return dropped, fixed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD, help="board.jsonl to inspect")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="repair the auto-fixable defects (backs up first; aborts on concurrent writes)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 while any defect remains, including the ones needing a decision",
    )
    args = parser.parse_args()

    if not args.board.is_file():
        print(f"[ERR] no board at {args.board}", file=sys.stderr)
        return 2

    lines = _read_lines(args.board)
    classes = classify(lines)

    if args.apply:
        dropped, fixed = apply_repairs(args.board, classes)
        print(
            f"  repaired: dropped {dropped} duplicate line(s), "
            f"normalized {fixed} list-typed description(s)"
        )
        print(f"  backup:   {args.board.with_suffix(args.board.suffix + '.bak-hygiene')}")
        lines = _read_lines(args.board)
        classes = classify(lines)

    report(classes)
    if has_defects(classes) and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
