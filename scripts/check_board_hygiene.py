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

Multi-column and divergent-description duplicates ask which copy is *true*.
``--settle`` answers that **mechanically, from the notes** -- never by eye --
encoding the rule the board cards themselves cite ("the note's status is the
source of truth, so reconcile against it"):

* a card in several columns keeps the row whose column equals the note's
  status (notes are the source of truth; ``sync`` derives columns from them);
* divergent bodies: if one copy strictly contains the other, the superset is
  kept -- a superset-first rule never destroys an annotation appended after
  the note snapshot (e.g. a claiming session's COORD line);
* otherwise the copy matching the note's body is kept;
* if sources *disagree* (journal vs markdown -- the open canonical-store
  question, card ``b3435a2f``) or no note exists, the id is left alone and
  reported: ambiguity is never resolved by guessing.

The note index (``load_note_truth``) reads the live journal
(``.dev-notes/notes.journal.jsonl``) and the markdown notes
(``~/.config/dev-notes``), keyed by the explicit ``board:`` uuid link when a
note carries one, else by normalized title (markdown titles may be prefixed
``Card 0NN — ``).

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
import re
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


# ── settle: decision-class duplicates, resolved from the notes ───────────────


def _normalize_title(title: str) -> str:
    """Markdown note titles may carry a ``Card 0NN — `` prefix; strip it so a
    note matches the card title it was written for."""
    t = str(title).strip().strip('"').strip("'")
    return re.sub(r"^Card\s+\d+\s*[—\-]\s*", "", t)


def _front_matter(text: str) -> dict[str, str] | None:
    """Parse a markdown note's single-line YAML front matter (None if absent)."""
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end < 0:
        return None
    fm: dict[str, str] = {}
    for line in text[3:end].splitlines():
        if line.startswith((" ", "\t")) or ":" not in line:
            continue
        key, _, value = line.partition(":")
        fm[key.strip()] = value.strip().strip('"').strip("'")
    return fm


def _strip_front_matter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end >= 0:
            rest = text[end + 4 :]
            return rest[1:] if rest.startswith("\n") else rest
    return text


def load_note_truth(journal: Path | None, md_dir: Path | None) -> dict[str, dict[str, Any]]:
    """Build the card-identity → note-status index both note stores provide.

    Keyed twice: by the explicit ``board:`` uuid link (markdown front matter)
    and by normalized title (both stores).  When two candidates for one key
    disagree on status -- journal vs markdown, or two same-title journal notes
    -- the key resolves to ``status=None``: the canonical-store question is an
    open user decision (card ``b3435a2f``), so ambiguity is marked, never
    guessed.
    """
    raw: dict[str, list[tuple[str | None, str, str]]] = defaultdict(list)

    if journal is not None and journal.is_file():
        for line in journal.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                note = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(note, dict) or "title" not in note:
                continue
            key = _normalize_title(note.get("title") or "")
            if key:
                raw[key].append((note.get("status"), str(note.get("body") or ""), "journal"))

    if md_dir is not None and md_dir.is_dir():
        for path in sorted(md_dir.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            fm = _front_matter(text)
            if not fm:
                continue
            source = f"md:{path.name}"
            body = _strip_front_matter(text).strip()
            board_ref = str(fm.get("board") or "").strip()
            if board_ref:
                raw[board_ref].append((fm.get("status"), body, source))
            key = _normalize_title(fm.get("title") or "")
            if key:
                raw[key].append((fm.get("status"), body, source))

    truth: dict[str, dict[str, Any]] = {}
    for key, candidates in raw.items():
        statuses = {status for status, _body, _src in candidates}
        sources = "+".join(dict.fromkeys(src for _st, _b, src in candidates))
        if len(statuses) == 1:
            status = next(iter(statuses))
            truth[key] = {
                "status": status,
                "body": max((b for _s, b, _src in candidates), key=len),
                "source": sources,
            }
        else:
            truth[key] = {"status": None, "body": None, "source": f"conflict:{sources}"}
    return truth


def settle(
    lines: list[str], truth: dict[str, dict[str, Any]]
) -> tuple[list[str], list[str], list[str]]:
    """Reduce multi-column / divergent duplicates to one row per id, deciding
    **from the notes only**.  Returns ``(lines, settled, unresolved)``.

    Rules (never by eye, never guessing):
    * multi-column -> keep the row whose column equals the note's status; if
      the notes disagree or say nothing, the id is left alone and reported;
    * divergent bodies in one column -> a copy strictly containing every other
      is kept (superset-first: never destroy an annotation appended after the
      note snapshot, e.g. a claiming session's COORD line);
    * else the copy matching the note's body;
    * else left alone and reported.  Exact-identical copies are ``repair()``'s
      job, not this one's.
    """
    pairs: list[tuple[str, dict[str, Any] | None]] = []
    for raw in lines:
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            pairs.append((raw, None))
            continue
        pairs.append((raw, record if isinstance(record, dict) and "id" in record else None))

    groups: dict[str, list[int]] = defaultdict(list)
    for i, (_raw, record) in enumerate(pairs):
        if record is not None:
            groups[str(record["id"])].append(i)

    drop: set[int] = set()
    settled: list[str] = []
    unresolved: list[str] = []

    def lookup(record: dict[str, Any]) -> dict[str, Any] | None:
        return truth.get(str(record.get("id"))) or truth.get(
            _normalize_title(record.get("title") or "")
        )

    for card_id, idxs in groups.items():
        if len(idxs) < 2:
            continue
        records = [pairs[i][1] for i in idxs]
        columns = {r.get("column") for r in records}
        descs = [
            r.get("description") if isinstance(r.get("description"), str) else "" for r in records
        ]

        if len(columns) > 1:
            entry = lookup(records[0])
            status = entry.get("status") if entry else None
            keeper = next((k for k, r in enumerate(records) if r.get("column") == status), None)
            if status and keeper is not None:
                drop.update(idxs[k] for k in range(len(idxs)) if k != keeper)
                settled.append(
                    f"{card_id[:8]}: multi-column -> kept column={status!r} "
                    f"(note source: {entry['source']}), dropped {len(idxs) - 1} stale row(s)"
                )
            else:
                reason = "no note" if not entry else f"note status={status!r} has no row"
                unresolved.append(f"{card_id[:8]}: multi-column, {reason} -- left alone")
            continue

        if len(set(descs)) == 1:
            continue  # identical bodies: repair() drops the redundancy

        keeper = next(
            (
                k
                for k, d in enumerate(descs)
                if d.strip()
                and all(descs[j].strip() in d.strip() for j in range(len(descs)) if j != k)
            ),
            None,
        )
        rule = "superset"
        if keeper is None:
            entry = lookup(records[0])
            body = (entry or {}).get("body")
            if isinstance(body, str) and body.strip():
                body = body.strip()
                keeper = next(
                    (
                        k
                        for k, d in enumerate(descs)
                        if d.strip()
                        and (d.strip() == body or body in d.strip() or d.strip() in body)
                    ),
                    None,
                )
                rule = "note-body"
        if keeper is not None:
            drop.update(idxs[k] for k in range(len(idxs)) if k != keeper)
            settled.append(
                f"{card_id[:8]}: divergent body -> kept the {rule} copy "
                f"({len(descs[keeper])}ch), dropped {len(idxs) - 1}"
            )
        else:
            unresolved.append(
                f"{card_id[:8]}: divergent body, no superset and no note match -- left alone"
            )

    out = [raw for i, (raw, _record) in enumerate(pairs) if i not in drop]
    return out, settled, unresolved


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


def apply_repairs(
    board: Path, classes: dict[str, Any], truth: dict[str, dict[str, Any]] | None = None
) -> tuple[int, int, list[str], list[str]]:
    """Repair on disk.  Backs up, writes atomically, and aborts if another
    session touched the board while we were working.

    Mechanical fixes run first (dedupe identical rows, normalize list
    descriptions), then -- when a note-truth index is supplied -- ``settle``
    resolves the decision classes against the notes, all inside one backup and
    one OCC window.  Returns ``(dropped, fixed, settled, unresolved)``."""
    before = _digest(board)
    lines = _read_lines(board)
    rewritten, dropped, fixed = repair(lines, classes)
    settled: list[str] = []
    unresolved: list[str] = []
    if truth:
        rewritten, settled, unresolved = settle(rewritten, truth)

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
    return dropped, fixed, settled, unresolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD, help="board.jsonl to inspect")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="repair the auto-fixable defects (backs up first; aborts on concurrent writes)",
    )
    parser.add_argument(
        "--settle",
        action="store_true",
        help="also settle multi-column and divergent duplicates against the notes (implies --apply)",
    )
    parser.add_argument(
        "--journal",
        type=Path,
        default=None,
        help="notes journal (default: <repo>/.dev-notes/notes.journal.jsonl beside the board)",
    )
    parser.add_argument(
        "--md-dir",
        type=Path,
        default=None,
        help="markdown notes dir (default: ~/.config/dev-notes)",
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

    if args.apply or args.settle:
        truth = None
        if args.settle:
            journal = args.journal or (
                args.board.parent.parent / ".dev-notes" / "notes.journal.jsonl"
            )
            md_dir = args.md_dir or (Path.home() / ".config" / "dev-notes")
            truth = load_note_truth(journal, md_dir)
        dropped, fixed, settled, unresolved = apply_repairs(args.board, classes, truth)
        print(
            f"  repaired: dropped {dropped} duplicate line(s), "
            f"normalized {fixed} list-typed description(s)"
        )
        for action in settled:
            print(f"  settled:  {action}")
        for note in unresolved:
            print(f"  UNRESOLVED: {note}")
        print(f"  backup:   {args.board.with_suffix(args.board.suffix + '.bak-hygiene')}")
        lines = _read_lines(args.board)
        classes = classify(lines)

    report(classes)
    if has_defects(classes) and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
