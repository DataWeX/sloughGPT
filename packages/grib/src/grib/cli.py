"""
grib — unified notes + kanban CLI.

One entry point for both notes and board operations. Every note mutation
optionally syncs to the kanban board so the two views stay in step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from grib import config
from grib.ids import display_id
from grib.notes_controller import NotesController, NoteStoreError, NoteValidationError
from grib.store import PlannerStore, get_store

# ── Helpers ──────────────────────────────────────────────────────────────


def _get_store(args: argparse.Namespace) -> PlannerStore:
    board_dir = Path(args.board_dir) if hasattr(args, "board_dir") and args.board_dir else None
    notes_dir = Path(args.notes_dir) if hasattr(args, "notes_dir") and args.notes_dir else None
    return get_store(board_dir=board_dir, notes_dir=notes_dir)


def _auto_sync(store: PlannerStore) -> tuple[int, int, int] | None:
    """Best-effort sync after note mutations. Silently skips on error."""
    try:
        return store.sync()
    except Exception:  # noqa: BLE001
        return None


def _note_ctl(args: argparse.Namespace) -> NotesController:
    """Controller over the CLI's store — one validation/read-model contract."""
    return NotesController(_get_store(args))


# ── Note subcommands ─────────────────────────────────────────────────────


def _note_new(args: argparse.Namespace) -> int:
    ctl = _note_ctl(args)
    try:
        note = ctl.create(
            args.title,
            tags=[t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else [],
            status=args.status or "open",
            body=args.body or "",
            sprint=args.sprint or "",
            gh=args.gh or "",
            assignee=args.author or "",
        )
    except NoteValidationError as e:
        print(f"{e}", file=sys.stderr)
        return 2
    _auto_sync(ctl.store)
    print(f"Created note: {display_id(note.id)} — {note.title}")
    return 0


def _note_list(args: argparse.Namespace) -> int:
    ctl = _note_ctl(args)
    notes = ctl.list_notes(tag=args.tag, status=args.status, limit=args.limit)
    for note in notes:
        icon = "\u25cf" if note.status == "done" else "\u25cb"
        tags = ",".join(note.tags[:3]) if note.tags else ""
        print(f"  {icon} [{display_id(note.id)}] {note.title}" + (f"  ({tags})" if tags else ""))
    print(f"\n{len(notes)} note(s)")
    return 0


def _note_show(args: argparse.Namespace) -> int:
    ctl = _note_ctl(args)
    try:
        note = ctl.get(args.id)
    except NoteValidationError as e:
        print(f"{e}", file=sys.stderr)
        return 2
    if not note:
        print(f"Note not found: {args.id}", file=sys.stderr)
        return 1
    print(f"Title:   {note.title}")
    print(f"Status:  {note.status}")
    print(f"Tags:    {', '.join(note.tags)}")
    print(f"Sprint:  {note.sprint or '-'}")
    print(f"GitHub:  {note.gh or '-'}")
    print(f"Created: {note.created_at}")
    print(f"Updated: {note.updated_at}")
    if note.body:
        print(f"\n{note.body}")
    return 0


def _note_update(args: argparse.Namespace) -> int:
    ctl = _note_ctl(args)
    updates: dict = {}
    if args.status:
        updates["status"] = args.status
    if args.tags:
        updates["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
    if args.title:
        updates["title"] = args.title
    if args.body is not None:
        updates["body"] = args.body
    try:
        note = ctl.update(args.id, **updates)
    except NoteValidationError as e:
        print(f"{e}", file=sys.stderr)
        return 2
    if not note:
        print(f"Note not found: {args.id}", file=sys.stderr)
        return 1
    _auto_sync(ctl.store)
    print(f"Updated: {display_id(note.id)} — {note.title}")
    return 0


def _note_delete(args: argparse.Namespace) -> int:
    ctl = _note_ctl(args)
    try:
        gone = ctl.delete(args.id)
    except NoteStoreError as e:
        # The store claimed a delete that did not take: report it rather than
        # print a success that would be false.
        print(f"{e}", file=sys.stderr)
        return 1
    if gone:
        _auto_sync(ctl.store)
        print(f"Deleted: {display_id(args.id)}")
        return 0
    print(f"Note not found: {args.id}", file=sys.stderr)
    return 1


def _note_search(args: argparse.Namespace) -> int:
    ctl = _note_ctl(args)
    try:
        notes = ctl.search(args.query)
    except NoteValidationError as e:
        print(f"{e}", file=sys.stderr)
        return 2
    for note in notes:
        icon = "\u25cf" if note.status == "done" else "\u25cb"
        print(f"  {icon} [{display_id(note.id)}] {note.title}")
    print(f"\n{len(notes)} result(s)")
    return 0


# ── Board subcommands ────────────────────────────────────────────────────


def _board_show(args: argparse.Namespace) -> int:
    store = _get_store(args)
    card_id = getattr(args, "id", None)
    if card_id:
        card = store.get_card(card_id)
        if not card:
            print(f"Card not found: {card_id}", file=sys.stderr)
            return 1
        # The card id is the durable ticket id — the one that never changes.
        print(f"ID:          {card.id}")
        print(f"Title:       {card.title}")
        print(f"Column:      {card.column}")
        print(f"Priority:    {card.priority}")
        print(f"Tags:        {', '.join(card.tags) if card.tags else 'none'}")
        print(f"Assignee:    {card.assignee or '-'}")
        print(f"Due:         {card.due_date or '-'}")
        print(f"Type:        {card.card_type or '-'}")
        print(f"Blocked by:  {', '.join(card.blocked_by) if card.blocked_by else 'none'}")
        print(f"Description: {card.description or '-'}")
        print(f"Sprint:      {card.sprint or '-'}")
        print(f"GitHub:      {card.gh or '-'}")
        # Reverse lookup rides this command: one ticket, many notes, oldest
        # first so the card reads as its own history (no new subcommand).
        linked = sorted(
            (n for n in store.list_notes(limit=99999) if n.card_id == card.id),
            key=lambda n: (n.created_at or "", n.id or ""),
        )
        print(f"Notes:       {len(linked) if linked else 'none'}")
        for note in linked:
            print(f"  {note.id}  [{note.status}]")
        return 0
    board = store.load_board()
    col_widths: dict[str, int] = {}
    for col in board.columns:
        name = col["name"]
        cards = [c for c in board.cards if c.column == name]
        col_widths[name] = max(len(name), max((len(c.title) for c in cards), default=0))

    for col in board.columns:
        name = col["name"]
        cards = [c for c in board.cards if c.column == name]
        wip = col.get("wip_limit", 0)
        wip_str = f" (WIP {wip})" if wip else ""
        print(f"\n\u250c\u2500 {name.upper()}{wip_str} \u2500" + "\u2500" * 40)
        if not cards:
            print("\u2502  (empty)")
        for card in cards:
            pri = {"low": " ", "medium": "!", "high": "!!", "critical": "!!!"}.get(
                card.priority, " "
            )
            print(f"\u2502  [{pri}] {card.title}")
    print()
    return 0


def _board_add(args: argparse.Namespace) -> int:
    store = _get_store(args)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
    try:
        card = store.add_card(
            args.title,
            column=args.column or "todo",
            priority=args.priority or "medium",
            description=args.description or "",
            tags=tags,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Created: {display_id(card.id)} — {card.title}")
    return 0


def _board_move(args: argparse.Namespace) -> int:
    store = _get_store(args)
    try:
        moved = store.move_card(args.id, args.column)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if moved:
        print(f"Moved {display_id(args.id)} → {args.column}")
        return 0
    print(f"Card not found: {args.id}", file=sys.stderr)
    return 1


def _board_delete(args: argparse.Namespace) -> int:
    store = _get_store(args)
    if store.delete_card(args.id):
        print(f"Deleted: {display_id(args.id)}")
        return 0
    print(f"Card not found: {args.id}", file=sys.stderr)
    return 1


def _board_tags(args: argparse.Namespace) -> int:
    store = _get_store(args)
    tags = store.get_tags()
    for tag, count in sorted(tags.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")
    return 0


def _board_stats(args: argparse.Namespace) -> int:
    store = _get_store(args)
    stats = store.get_stats()
    print(f"Total cards: {stats['total']}")
    for col, count in stats["byColumn"].items():
        print(f"  {col}: {count}")
    return 0


def _board_index(args: argparse.Namespace) -> int:
    """Rebuild the read-only MogDB query index from board.jsonl."""
    from grib.board_index import BoardIndex

    board_dir = Path(args.board_dir) if getattr(args, "board_dir", None) else None
    index = BoardIndex(board_dir=board_dir)
    try:
        count = index.sync()
    except Exception as e:  # noqa: BLE001 — report, never crash the CLI
        print(f"Index unavailable (board.jsonl is still the source of truth): {e}", file=sys.stderr)
        return 1
    finally:
        index.close()
    print(f"Indexed {count} card(s) → {index.index_dir}")
    return 0


def _board_query(args: argparse.Namespace) -> int:
    """Filter cards via the MogDB index, falling back to a file read."""
    from grib.board_index import BoardIndex

    board_dir = Path(args.board_dir) if getattr(args, "board_dir", None) else None
    index = BoardIndex(board_dir=board_dir)
    try:
        cards = index.query(
            column=getattr(args, "column", None),
            priority=getattr(args, "priority", None),
            assignee=getattr(args, "assignee", None),
            tag=getattr(args, "tag", None),
            text=getattr(args, "text", None),
            limit=getattr(args, "limit", None),
        )
    finally:
        index.close()
    for card in cards:
        print(f"{card.get('id', '?'):<55} {card.get('column', '?'):<8} {card.get('title', '')}")
    print(f"{len(cards)} card(s)")
    return 0


# ── Sync ─────────────────────────────────────────────────────────────────


def _verify(args: argparse.Namespace) -> int:
    from grib.verify import find_md_by_title, format_report, parse_frontmatter, verify_meta

    store = _get_store(args)
    note_path = Path(args.note)
    note = None

    if note_path.exists():
        note = store.get_note(note_path.stem)
    else:
        # Resolve through the journal first: it is the live store. md-only
        # notes (pre-backend-flip) still fall through to the path scan below.
        q = args.note.lower()
        notes = store.list_notes(limit=99999)
        hits = [n for n in notes if n.id.lower() == q]
        if not hits:
            hits = [n for n in notes if n.id.lower().startswith(q)]
        if not hits:
            hits = [n for n in notes if q in (n.title or "").lower()]
        if len(hits) > 1:
            print(f"verify: '{args.note}' matches {len(hits)} notes — use a longer id:")
            for n in hits[:8]:
                print(f"  {n.id}  [{n.status}]  {(n.title or '')[:60]}")
            return 1
        if hits:
            note = hits[0]

    if note is not None:
        meta: dict[str, str] = {}
        # md still supplies what the Note model does not carry (landed:).
        # Join by title — md filenames (YYYY-MM-DD-slug) are not note ids.
        md = find_md_by_title(note.title, [config.default_notes_dir(), config.NOTES_FALLBACK])
        if md is not None:
            meta.update(parse_frontmatter(md))
        # Journal wins on everything it models — md frontmatter froze at the
        # backend flip, so trusting it here is the 20261119_105234 bug.
        meta["title"] = note.title
        meta["status"] = note.status
        meta["board"] = note.card_id
    else:
        # Legacy md-only note: use the given path, else resolve one by name.
        if not note_path.exists():
            search_dirs = [config.default_notes_dir()]
            if config.NOTES_FALLBACK not in search_dirs:
                search_dirs.append(config.NOTES_FALLBACK)
            matches = [
                p
                for d in search_dirs
                for p in sorted(Path(d).glob("*.md"))
                if args.note.lower() in p.stem.lower()
            ]
            matches = sorted(set(matches))
            if len(matches) == 1:
                note_path = matches[0]
            elif not matches:
                print(f"verify: no note matching '{args.note}'")
                return 1
            else:
                print(
                    f"verify: '{args.note}' matches {len(matches)} notes — give a path or more text"
                )
                return 1
        meta = parse_frontmatter(note_path)
        meta.setdefault("title", note_path.stem)

    board_file = (
        Path(args.board_file) if args.board_file else config.default_board_dir() / "board.jsonl"
    )
    findings = verify_meta(meta, board_file, Path(args.repo))
    print(format_report(findings))
    return 0 if all(f.ok for f in findings) else 1


def _sync(args: argparse.Namespace) -> int:
    store = _get_store(args)
    prune = getattr(args, "prune", False)
    # Read the drift before syncing: with --prune these are about to vanish.
    orphans = store.orphan_cards()
    added, updated, total = store.sync(prune=prune)
    if not args.quiet:
        board = store.load_board()
        for card in board.cards:
            icon = "\u2713" if card.column == "done" else "\u25cb"
            print(f"  {icon} [{card.column:12s}] {card.title}")
        print(f"\n{added} new card(s), {updated} moved, {total} total")
    else:
        print(f"{added} new card(s), {updated} moved, {total} total")
    if prune:
        print(f"{len(orphans)} orphan card(s) removed")
    elif orphans:
        # Advisory: sync is additive by default, so drift accumulates silently
        # (a deleted note leaves its card forever). Report, never delete.
        print(
            f"\n{len(orphans)} card(s) have no matching note "
            f"— run `grib sync --prune` to remove them:"
        )
        for card in orphans:
            print(f"    [{card.column}] {card.title}")
    return 0


def _migrate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from grib.migrate import MigrationError, migrate

    try:
        result = migrate(
            Path(args.source_root) if args.source_root else None,
            dry_run=args.dry_run,
            board_dir=Path(args.board_dir) if args.board_dir else None,
            notes_dir=Path(args.notes_dir) if args.notes_dir else None,
        )
    except MigrationError as exc:
        print(f"migration refused: {exc}")
        return 1
    verb = "would import" if result["dry_run"] else "imported"
    print(f"source: {result['source']}")
    print(
        f"board: {verb} {result['board']['imported']} card(s), "
        f"{result['board']['skipped']} already present"
    )
    print(
        f"notes: {verb} {result['notes']['imported']} note(s), "
        f"{result['notes']['skipped']} already present"
    )
    if result["board"].get("cycles"):
        print(f"warning: blocked_by cycles involving: {result['board']['cycles'][:10]}")
    for backup in result["backups"]:
        print(f"backup: {backup}")
    return 0


def _snapshot(args: argparse.Namespace) -> int:
    from pathlib import Path

    from grib.snapshot import SnapshotError, snapshot

    try:
        result = snapshot(
            Path(args.repo),
            board_dir=Path(args.board_dir) if args.board_dir else None,
        )
    except SnapshotError as exc:
        print(f"snapshot failed: {exc}")
        return 1
    print(f"snapshot: {result['snapshot']} ({result['bytes']} bytes)")
    print(
        f"suggested: git add {result['snapshot']} && git commit -m 'chore(kanban): board snapshot'"
    )
    return 0


def _export(args: argparse.Namespace) -> int:
    from pathlib import Path

    from grib.export import ExportError, export

    try:
        result = export(
            Path(args.repo),
            board_dir=Path(args.board_dir) if args.board_dir else None,
            notes_dir=Path(args.notes_dir) if args.notes_dir else None,
        )
    except ExportError as exc:
        print(f"export failed: {exc}")
        return 1
    print(
        "export: {card_count} cards, {note_count} notes "
        "({notes_removed} stale removed) -> {manifest}".format(**result)
    )
    return 0


# ── Ledger (docs/AGENT_SYNC.md — read-only; the doc stays the source) ────


def _ledger_doc(args: argparse.Namespace) -> Path | None:
    """``--doc`` override, else walk up from cwd. Prints the failure and
    returns None (callers map that to exit 2) when no ledger is in scope."""
    from grib.ledger import find_doc

    if args.doc:
        doc = Path(args.doc)
        if doc.is_file():
            return doc
        print(f"ledger: {doc} not found")
        return None
    doc = find_doc()
    if doc is None:
        print("ledger: docs/AGENT_SYNC.md not found (no ledger above the cwd)")
    return doc


def _claims_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "board_dir", None):
        return Path(args.board_dir)
    return config.default_board_dir()


def _claim_add(args: argparse.Namespace) -> int:
    from grib.claims import DEFAULT_TTL_MIN, append_event, default_session, detect_git

    session = args.session or default_session()
    # A claim is presence, not a board mutation: it must never fail because
    # the note->board sync hasn't created the card yet. An unknown id is
    # reported, not rejected (the assignee fill below is conditional anyway).
    store = _get_store(args) if args.card else None
    card = store.get_card(args.card) if (store and args.card) else None
    if args.card and card is None:
        print(f"claim: note — card not on board yet: {args.card}")
    worktree, branch = args.worktree, args.branch
    if worktree is None or branch is None:
        gw, gb = detect_git()
        worktree = worktree if worktree is not None else gw
        branch = branch if branch is not None else gb
    ev = append_event(
        _claims_dir(args),
        event="claim",
        session=session,
        card_id=args.card,
        worktree=worktree,
        branch=branch,
        ttl_min=args.ttl if args.ttl is not None else DEFAULT_TTL_MIN,
    )
    assignee_note = ""
    if card is not None:
        if not card.assignee:
            store.update_card(args.card, assignee=session)
            assignee_note = f" assignee={session}"
        else:
            assignee_note = f" assignee={card.assignee} (kept)"
    where = f"worktree={ev['worktree'] or '-'} branch={ev['branch'] or '-'}"
    print(
        f"claim: active session={session} card={args.card or '-'} {where} "
        f"ttl={ev['ttl_min']}m{assignee_note}"
    )
    return 0


def _claim_list(args: argparse.Namespace) -> int:
    from grib.claims import fold

    claims = fold(_claims_dir(args))
    if args.active:
        claims = [c for c in claims if c["status"] == "active"]
    if not claims:
        print("no claims")
        return 0
    for c in claims:
        print(
            f"[{c['status']}] {c['session']}  card={c['card_id'] or '-'}  "
            f"worktree={c['worktree'] or '-'}  branch={c['branch'] or '-'}  "
            f"age={c['age_min']:.0f}m/ttl={c['ttl_min']}m"
        )
    return 0


def _claim_release(args: argparse.Namespace) -> int:
    from grib.claims import append_event, default_session

    session = args.session or default_session()
    append_event(_claims_dir(args), event="release", session=session)
    print(f"claim: released {session}")
    return 0


def _ledger_entry(args: argparse.Namespace) -> int:
    from grib.ledger import prepend_entry

    doc = _ledger_doc(args)
    if doc is None:
        return 2
    line = prepend_entry(doc, args.text, day=args.date)
    print(f"ledger: sealed entry prepended -> {doc}")
    print(line)
    return 0


def _ledger_parse(args: argparse.Namespace) -> int:
    from grib.ledger import dumps, parse

    doc = _ledger_doc(args)
    if doc is None:
        return 2
    print(dumps(parse(doc)))
    return 0


def _ledger_query(args: argparse.Namespace) -> int:
    from grib.ledger import query

    doc = _ledger_doc(args)
    if doc is None:
        return 2
    hits = query(
        doc,
        card=args.card,
        sha=args.sha,
        lane=args.lane,
        since=args.since,
        grep=args.grep,
    )
    if not hits:
        print("ledger: no matching entries")
        return 0
    for h in hits:
        print(f"{h['line']:>5}  {h['date']}  {h['head'][:120]}")
    return 0


def _ledger_check(args: argparse.Namespace) -> int:
    import json

    from grib.ledger import check, fingerprint

    doc = _ledger_doc(args)
    if doc is None:
        return 2
    ok, violations = check(doc)

    if args.baseline:
        # Shrink-only ratchet (page-baseline precedent): known findings pass,
        # unknown ones redden, and a fixed finding whose baseline line survives
        # reddens too — the baseline may only shrink.
        try:
            with open(args.baseline, encoding="utf-8") as fh:
                baseline = list(json.load(fh).get("violations", []))
        except (OSError, ValueError, AttributeError) as exc:
            print(f"ledger: baseline unreadable: {exc}")
            return 2
        known = set(baseline)
        current = {fingerprint(v) for v in violations}
        new = [v for v in violations if fingerprint(v) not in known]
        stale = [b for b in baseline if b not in current]
        for v in new:
            print(f"ledger: NEW line {v['line']}: {v['kind']} — {v['msg']}")
        for b in stale:
            print(f"ledger: STALE baseline entry (no longer occurs — delete it): {b}")
        if new or stale:
            print(f"ledger: {len(new)} new, {len(stale)} stale (baseline must only shrink)")
            return 1
        print(f"ledger: ok — {len(violations)} known finding(s), baseline aligned: {doc}")
        return 0

    if ok:
        print(f"ledger: ok — {doc}")
        return 0
    for v in violations:
        print(f"ledger: line {v['line']}: {v['kind']} — {v['msg']}")
    print(f"ledger: {len(violations)} violation(s)")
    return 1


def _gui(args: argparse.Namespace) -> int:
    from grib.gui import main as gui_main

    gui_args = []
    if args.board_dir:
        gui_args.extend(["--board-dir", str(args.board_dir)])
    if args.notes_dir:
        gui_args.extend(["--notes-dir", str(args.notes_dir)])
    if args.host:
        gui_args.extend(["--host", args.host])
    if args.port:
        gui_args.extend(["--port", str(args.port)])
    if args.backend:
        gui_args.extend(["--backend", args.backend])
    return gui_main(gui_args)


# ── CLI Parser ───────────────────────────────────────────────────────────


def _add_sync_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--prune",
        action="store_true",
        help=(
            "Also delete cards whose note no longer exists. "
            "Without this flag orphaned cards are only reported, never removed."
        ),
    )


def _add_verify_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("note", help="Note file path or title substring")
    p.add_argument("--repo", default=".", help="Git repo to check landed: shas in (default: cwd)")
    p.add_argument("--board-file", default=None, help="board.jsonl path (default: config)")


def _add_snapshot_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo", default=".", help="Destination repo root (default: cwd)")


def _add_export_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo", default=".", help="Destination repo root (default: cwd)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grib",
        description="Unified notes + kanban planner.",
    )
    parser.add_argument("--notes-dir", default=None, help="Notes directory")
    parser.add_argument("--board-dir", default=None, help="Board directory")
    sub = parser.add_subparsers(dest="command")

    # Notes
    notes = sub.add_parser("note", aliases=["notes"], help="Note operations")
    note_sub = notes.add_subparsers(dest="subcommand")
    p = note_sub.add_parser("new", help="Create a note")
    p.add_argument("title")
    p.add_argument("--tags", default=None)
    p.add_argument("--status", default=None)
    p.add_argument("--body", default=None)
    p.add_argument("--sprint", default=None)
    p.add_argument("--gh", default=None)
    p.add_argument("--author", default=None)

    p = note_sub.add_parser("list", aliases=["ls"], help="List notes")
    p.add_argument("--tag", default=None)
    p.add_argument("--status", default=None)
    p.add_argument("--limit", type=int, default=50)

    p = note_sub.add_parser("show", help="Show a note")
    p.add_argument("id")

    p = note_sub.add_parser("update", help="Update a note")
    p.add_argument("id")
    p.add_argument("--title", default=None)
    p.add_argument("--status", default=None)
    p.add_argument("--tags", default=None)
    p.add_argument("--body", default=None)

    p = note_sub.add_parser("delete", aliases=["rm"], help="Delete a note")
    p.add_argument("id")

    p = note_sub.add_parser("search", help="Search notes")
    p.add_argument("query")

    # Board
    board = sub.add_parser("board", aliases=["kanban"], help="Board operations")
    board_sub = board.add_subparsers(dest="subcommand")
    p = board_sub.add_parser("show", aliases=["ls"], help="Show the board or a specific card")
    p.add_argument("id", nargs="?", default=None)

    p = board_sub.add_parser("add", help="Add a card")
    p.add_argument("title")
    p.add_argument("--column", default=None)
    p.add_argument("--priority", default=None)
    p.add_argument("--description", default=None)
    p.add_argument("--tags", default=None)

    p = board_sub.add_parser("move", help="Move a card")
    p.add_argument("id")
    p.add_argument("column")

    p = board_sub.add_parser("delete", aliases=["rm"], help="Delete a card")
    p.add_argument("id")

    board_sub.add_parser("tags", help="List tags")
    board_sub.add_parser("stats", help="Show stats")

    p = board_sub.add_parser("index", help="Rebuild the read-only MogDB query index")
    p = board_sub.add_parser("query", help="Filter cards via the MogDB index (file fallback)")
    p.add_argument("--column", default=None)
    p.add_argument("--priority", default=None)
    p.add_argument("--assignee", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--text", default=None)
    p.add_argument("--limit", type=int, default=None)

    # Sync — canonical path `grib board sync`; top-level kept as an alias.
    sync_p = sub.add_parser("sync", help="Sync notes to board (alias of `grib board sync`)")
    _add_sync_args(sync_p)
    p = board_sub.add_parser("sync", help="Sync notes to board")
    _add_sync_args(p)

    # Verify — canonical path `grib note verify`; top-level kept as an alias.
    verify_p = sub.add_parser(
        "verify",
        help="Diff a note card against its board card + landed git shas (alias of `grib note verify`)",
        description=(
            "Three-way consistency check: note frontmatter (status/board/landed) "
            "vs the board card vs actual git history. Exits 1 on any drift."
        ),
    )
    _add_verify_args(verify_p)
    p = note_sub.add_parser(
        "verify",
        help="Diff a note card against its board card + landed git shas",
        description=(
            "Three-way consistency check: note frontmatter (status/board/landed) "
            "vs the board card vs actual git history. Exits 1 on any drift."
        ),
    )
    _add_verify_args(p)

    # Migrate
    migrate_p = sub.add_parser(
        "migrate",
        help="Import repo-local .kanban/.dev-notes into the canonical store",
        description=(
            "One-time import of repo-local planner data into the canonical "
            "user-config store. Source chain is verified first; canonical "
            "rows win on id conflicts; existing canonical files are backed "
            "up before any write; source files are never modified."
        ),
    )
    migrate_p.add_argument(
        "--source-root",
        default=None,
        help="Repo root holding .kanban/.dev-notes (default: repository detection)",
    )
    migrate_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify + report what would be imported; write nothing",
    )

    # Snapshot
    snapshot_p = sub.add_parser(
        "snapshot",
        help="Copy the canonical board into <repo>/.kanban/snapshots/ (alias of `grib board snapshot`)",
        description=(
            "Export a dated, byte-exact copy of the canonical board into the "
            "repo for versioned history. Prints the suggested commit command; "
            "never runs git itself."
        ),
    )
    _add_snapshot_args(snapshot_p)
    p = board_sub.add_parser(
        "snapshot",
        help="Copy the canonical board into <repo>/.kanban/snapshots/",
        description=(
            "Export a dated, byte-exact copy of the canonical board into the "
            "repo for versioned history. Prints the suggested commit command; "
            "never runs git itself."
        ),
    )
    _add_snapshot_args(p)

    # Export (live mirror for the GitHub sync pipeline — see grib.export)
    export_p = sub.add_parser(
        "export",
        help="Mirror canonical board+notes into the repo (alias of `grib board export`)",
        description=(
            "Producer half of the kanban GitHub sync: byte-exact mirror of the "
            "canonical board and notes into the repo working copies plus "
            ".kanban/manifest.json for the CI freshness alarm. Never runs git — "
            "scripts/kanban_sync.sh commits and pushes."
        ),
    )
    _add_export_args(export_p)
    p = board_sub.add_parser(
        "export",
        help="Mirror canonical board+notes into <repo>/.kanban/board.jsonl + <repo>/.dev-notes/",
        description=(
            "Producer half of the kanban GitHub sync: byte-exact mirror of the "
            "canonical board and notes into the repo working copies plus "
            ".kanban/manifest.json for the CI freshness alarm. Never runs git — "
            "scripts/kanban_sync.sh commits and pushes."
        ),
    )
    _add_export_args(p)

    # Ledger (docs/AGENT_SYNC.md — the session sync log; doc stays the source)
    ledger = sub.add_parser(
        "ledger",
        help="Parse/query/check the session ledger (docs/AGENT_SYNC.md)",
        description=(
            "Read-only view of the session ledger. An entry is a bold-dated "
            "line (**YYYY-MM-DD — …**), newest first at line 1. check gates "
            "the prepend invariant: well-formed headers, dates non-increasing."
        ),
    )
    ledger_sub = ledger.add_subparsers(dest="subcommand")
    _doc_help = "Path to AGENT_SYNC.md (default: walk up from cwd)"
    p = ledger_sub.add_parser("parse", help="JSON index of every entry")
    p.add_argument("--doc", default=None, help=_doc_help)
    p = ledger_sub.add_parser("query", help="Filter entries (all filters AND)")
    p.add_argument("--doc", default=None, help=_doc_help)
    p.add_argument("--card", default=None, help="Card id substring (short or long form)")
    p.add_argument("--sha", default=None, help="Git sha substring (single or range side)")
    p.add_argument("--lane", default=None, help="Lane substring, case-insensitive")
    p.add_argument("--since", default=None, help="Only entries on/after this YYYY-MM-DD")
    p.add_argument("--grep", default=None, help="Substring anywhere in the entry")
    p = ledger_sub.add_parser(
        "check",
        help="Gate: well-formed headers + prepend order + seals (exit 1 on violation)",
    )
    p.add_argument("--doc", default=None, help=_doc_help)
    p.add_argument(
        "--baseline",
        default=None,
        help=(
            "JSON file pinning known findings (shrink-only): unknown "
            "violations fail, stale baseline entries fail until deleted"
        ),
    )
    p = ledger_sub.add_parser(
        "entry",
        help="Prepend a sealed entry at line 1 (grib computes the hash trail)",
    )
    p.add_argument("--doc", default=None, help=_doc_help)
    p.add_argument("--text", required=True, help="Entry prose after the date")
    p.add_argument("--date", default=None, help="Entry date YYYY-MM-DD (default: today)")

    # Claims (lane presence — which session owns which worktree/card right now)
    from grib.claims import DEFAULT_TTL_MIN

    claims = sub.add_parser(
        "claim",
        help="Lane claims: record/refresh which worktree+card a session owns",
        description=(
            "Append-only presence registry in the canonical store "
            "(claims.jsonl). A session claims its worktree/card at start and "
            "refreshes on queue reads, releases when done; stale heartbeats "
            "expire after the TTL. Notification is read-time visibility — "
            "there is no push bus."
        ),
    )
    claims_sub = claims.add_subparsers(dest="subcommand")
    p = claims_sub.add_parser(
        "add",
        help="Record or refresh this session's claim (sets card assignee when empty)",
    )
    p.add_argument("--card", default=None, help="Card id the session is working")
    p.add_argument("--session", default=None, help="Session id (default: $OPENCODE_SESSION_ID)")
    p.add_argument(
        "--worktree",
        default=None,
        help="Worktree path (default: git toplevel of cwd)",
    )
    p.add_argument("--branch", default=None, help="Branch (default: current branch)")
    p.add_argument(
        "--ttl",
        type=int,
        default=DEFAULT_TTL_MIN,
        help=f"Heartbeat expiry in minutes (default: {DEFAULT_TTL_MIN})",
    )
    p = claims_sub.add_parser("list", help="Show claims folded per session")
    p.add_argument("--active", action="store_true", help="Only non-released, non-expired claims")
    p = claims_sub.add_parser("release", help="Release this session's claim")
    p.add_argument("--session", default=None, help="Session id (default: $OPENCODE_SESSION_ID)")

    # GUI
    gui_p = sub.add_parser(
        "gui",
        help="Local web interface",
        description="Local web interface for notes + kanban board.",
    )
    gui_p.add_argument("--host", default="127.0.0.1")
    gui_p.add_argument("--port", type=int, default=8765)
    gui_p.add_argument("--backend", default=None, choices=["file", "mogdb"])

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    cmd = args.command
    sub = getattr(args, "subcommand", None)

    dispatch = {
        ("note", "new"): _note_new,
        ("note", "list"): _note_list,
        ("note", "ls"): _note_list,
        ("note", "show"): _note_show,
        ("note", "update"): _note_update,
        ("note", "delete"): _note_delete,
        ("note", "rm"): _note_delete,
        ("note", "search"): _note_search,
        ("notes", "new"): _note_new,
        ("notes", "list"): _note_list,
        ("notes", "ls"): _note_list,
        ("notes", "show"): _note_show,
        ("notes", "update"): _note_update,
        ("notes", "delete"): _note_delete,
        ("notes", "rm"): _note_delete,
        ("notes", "search"): _note_search,
        ("board", "show"): _board_show,
        ("board", "ls"): _board_show,
        ("board", "add"): _board_add,
        ("board", "move"): _board_move,
        ("board", "delete"): _board_delete,
        ("board", "rm"): _board_delete,
        ("board", "tags"): _board_tags,
        ("board", "stats"): _board_stats,
        ("board", "index"): _board_index,
        ("board", "query"): _board_query,
        ("kanban", "show"): _board_show,
        ("kanban", "ls"): _board_show,
        ("kanban", "add"): _board_add,
        ("kanban", "move"): _board_move,
        ("kanban", "delete"): _board_delete,
        ("kanban", "rm"): _board_delete,
        ("kanban", "tags"): _board_tags,
        ("kanban", "stats"): _board_stats,
        ("kanban", "index"): _board_index,
        ("kanban", "query"): _board_query,
        ("sync", None): _sync,
        ("board", "sync"): _sync,
        ("kanban", "sync"): _sync,
        ("verify", None): _verify,
        ("note", "verify"): _verify,
        ("notes", "verify"): _verify,
        ("migrate", None): _migrate,
        ("snapshot", None): _snapshot,
        ("board", "snapshot"): _snapshot,
        ("kanban", "snapshot"): _snapshot,
        ("export", None): _export,
        ("board", "export"): _export,
        ("kanban", "export"): _export,
        ("ledger", "parse"): _ledger_parse,
        ("ledger", "query"): _ledger_query,
        ("ledger", "check"): _ledger_check,
        ("ledger", "entry"): _ledger_entry,
        ("claim", "add"): _claim_add,
        ("claim", "list"): _claim_list,
        ("claim", "release"): _claim_release,
        ("gui", None): _gui,
    }

    handler = dispatch.get((cmd, sub))
    if handler:
        return handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())


# Backward compatibility
cli_main = main
