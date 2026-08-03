"""
planner gui — local web interface for notes and the kanban board.

Zero-dependency single-page app served over Python's stdlib ``http.server``.
No framework, no CDN assets, no install step beyond the planner package.

Usage::

    planner gui [--notes-dir DIR] [--board-dir DIR] [--backend file|mogdb]
                [--host HOST] [--port PORT] [--no-open] [--sync]

Opens ``http://127.0.0.1:8787`` in the default browser (unless ``--no-open``).
If the requested port is in use, the server steps to the next free port
(``--port 0`` asks the kernel for an ephemeral port).

The GUI reads and writes the same data as the ``notes`` and ``kanban`` CLI
commands. Board cards can be dragged between columns; moving a card that
matches a note also updates the note's status so notes and board stay in
sync. The Sync button creates board cards for notes that do not have one yet.

API (all JSON):
    GET  /                    SPA shell
    GET  /api/notes           list notes (?tag= &status= &query= &limit=)
    POST /api/notes           create note {title, tags[], status, sprint, gh, body}
    GET  /api/notes/{id}      one note
    PUT  /api/notes/{id}      update note (partial fields)
    DELETE /api/notes/{id}    delete note
    GET  /api/board           columns + cards
    POST /api/board/move      {id, column} — move card, sync note status
    POST /api/sync            notes -> board cards for unmatched notes
    GET  /api/tags            tag counts
    GET  /api/stats           counts by status / date, totals
"""

from __future__ import annotations

import argparse
import errno
import json
import logging
import re
import threading
import webbrowser
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import planner.core as core_module

from . import config
from .kanban import KanbanStore
from .sync import sync_notes_to_board

logger = logging.getLogger("planner.gui")

STATUS_TO_COLUMN = config.STATUS_TO_COLUMN
COLUMN_TO_STATUS = config.COLUMN_TO_STATUS
STATUSES = config.STATUSES


def _note_to_dict(note: Any) -> dict:
    return {
        "id": note.id,
        "title": note.title,
        "created_at": note.created_at,
        "updated_at": note.updated_at,
        "tags": list(note.tags),
        "status": note.status,
        "sprint": note.sprint,
        "gh": note.gh,
        "body": note.body,
    }


def _card_to_dict(card: Any) -> dict:
    return {
        "id": card.id,
        "title": card.title,
        "description": card.description,
        "column": card.column,
        "priority": card.priority,
        "tags": list(card.tags),
        "due_date": card.due_date,
        "assignee": card.assignee,
        "notes": [{"id": n.id, "text": n.text, "author": n.author} for n in card.notes],
    }


class GuiServer(ThreadingHTTPServer):
    """HTTP server that carries the note/board stores and a write lock."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple,
        handler,
        note_store,
        kanban_store: KanbanStore,
    ):
        super().__init__(address, handler)
        self.note_store = note_store
        self.kanban_store = kanban_store
        self.lock = threading.Lock()


class GuiHandler(BaseHTTPRequestHandler):
    server_version = "PlannerGUI/0.1"

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug(fmt, *args)

    def _send(self, code: int, body: Any, ctype: str = "application/json") -> None:
        if ctype == "application/json":
            data = json.dumps(body, default=str).encode("utf-8")
        else:
            data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc

    def _error(self, code: int, message: str) -> None:
        self._send(code, {"error": message})

    @property
    def stores(self) -> GuiServer:
        return self.server  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # GET
    # ------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            if path == "/":
                return self._send(200, GUI_HTML, "text/html; charset=utf-8")
            if path == "/api/notes":
                return self._handle_list_notes(query)
            if path.startswith("/api/notes/"):
                note_id = _unquote_path(path[len("/api/notes/"):])
                if not note_id:
                    return self._error(400, "missing note id")
                return self._handle_get_note(note_id)
            if path == "/api/board":
                return self._handle_board()
            if path == "/api/tags":
                return self._handle_tags()
            if path == "/api/stats":
                return self._handle_stats()
            return self._error(404, f"not found: {path}")
        except Exception as exc:  # noqa: BLE001 — fail with a clean 500
            logger.exception("GET %s failed", path)
            self._error(500, str(exc))

    def _handle_list_notes(self, query: dict) -> None:
        limit = _int_or(query.get("limit", ["500"])[0], 500)
        tag = query.get("tag", [None])[0]
        status = query.get("status", [None])[0]
        q = query.get("query", [None])[0]
        notes = self.stores.note_store.list_notes(limit=9999)
        if tag:
            notes = [n for n in notes if tag in n.tags]
        if status:
            notes = [n for n in notes if n.status == status]
        if q:
            ql = q.lower()
            notes = [
                n for n in notes
                if ql in n.title.lower() or ql in " ".join(n.tags).lower() or ql in n.body.lower()
            ]
        self._send(200, {"notes": [_note_to_dict(n) for n in notes[:limit]]})

    def _handle_get_note(self, note_id: str) -> None:
        note = self.stores.note_store.get(note_id)
        if note is None:
            return self._error(404, f"note not found: {note_id}")
        self._send(200, {"note": _note_to_dict(note)})

    def _handle_board(self) -> None:
        board = self.stores.kanban_store.load_board()
        columns = sorted(
            ({"name": c.name, "wip_limit": c.wip_limit, "order": c.order} for c in board.columns),
            key=lambda c: c["order"],
        )
        cards = [_card_to_dict(c) for c in board.cards]
        self._send(200, {"board": {"name": board.name, "columns": columns, "cards": cards}})

    def _handle_tags(self) -> None:
        counts: dict[str, int] = {}
        for note in self.stores.note_store.list_notes(limit=9999):
            for tag in note.tags:
                counts[tag] = counts.get(tag, 0) + 1
        tags = [{"name": name, "count": count} for name, count in sorted(counts.items())]
        self._send(200, {"tags": tags})

    def _handle_stats(self) -> None:
        notes = self.stores.note_store.list_notes(limit=9999)
        by_status: dict[str, int] = {}
        today_str = date.today().isoformat()
        today = 0
        for n in notes:
            by_status[n.status or "open"] = by_status.get(n.status or "open", 0) + 1
            if n.date_str == today_str:
                today += 1
        self._send(200, {
            "total": len(notes),
            "today": today,
            "by_status": by_status,
        })

    # ------------------------------------------------------------------
    # POST / PUT / DELETE
    # ------------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            body = self._read_json()
            if path == "/api/notes":
                return self._handle_create_note(body)
            if path == "/api/board/move":
                return self._handle_move(body)
            if path == "/api/sync":
                return self._handle_sync()
            return self._error(404, f"not found: {path}")
        except ValueError as exc:
            return self._error(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("POST %s failed", path)
            self._error(500, str(exc))

    def do_PUT(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if not path.startswith("/api/notes/"):
            return self._error(404, f"not found: {path}")
        note_id = _unquote_path(path[len("/api/notes/"):])
        try:
            body = self._read_json()
            return self._handle_update_note(note_id, body)
        except ValueError as exc:
            return self._error(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("PUT %s failed", path)
            self._error(500, str(exc))

    def do_DELETE(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if not path.startswith("/api/notes/"):
            return self._error(404, f"not found: {path}")
        note_id = _unquote_path(path[len("/api/notes/"):])
        try:
            with self.stores.lock:
                ok = self.stores.note_store.delete(note_id)
            if not ok:
                return self._error(404, f"note not found: {note_id}")
            return self._send(200, {"ok": True})
        except Exception as exc:  # noqa: BLE001
            logger.exception("DELETE %s failed", path)
            self._error(500, str(exc))

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_create_note(self, body: dict) -> None:
        title = (body.get("title") or "").strip()
        if not title:
            return self._error(400, "title is required")
        tags = _coerce_tags(body.get("tags"))
        status = body.get("status") or "open"
        if status not in STATUSES:
            return self._error(400, f"invalid status: {status}")
        with self.stores.lock:
            note = self.stores.note_store.create(
                title=title,
                tags=tags,
                status=status,
                sprint=body.get("sprint") or "",
                gh=body.get("gh") or "",
                body=body.get("body") or "",
            )
        self._send(200, {"note": _note_to_dict(note)})

    def _handle_update_note(self, note_id: str, body: dict) -> None:
        kwargs: dict[str, Any] = {}
        if "title" in body:
            title = (body.get("title") or "").strip()
            if not title:
                return self._error(400, "title cannot be empty")
            kwargs["title"] = title
        if "tags" in body:
            kwargs["tags"] = _coerce_tags(body.get("tags"))
        if "status" in body:
            status = body.get("status")
            if status not in STATUSES:
                return self._error(400, f"invalid status: {status}")
            kwargs["status"] = status
        if "sprint" in body:
            kwargs["sprint"] = body.get("sprint") or ""
        if "gh" in body:
            kwargs["gh"] = body.get("gh") or ""
        if "body" in body:
            kwargs["body"] = body.get("body") or ""
        if not kwargs:
            return self._error(400, "no fields to update")
        with self.stores.lock:
            note = self.stores.note_store.update(note_id, **kwargs)
        if note is None:
            return self._error(404, f"note not found: {note_id}")
        self._send(200, {"note": _note_to_dict(note)})

    def _handle_move(self, body: dict) -> None:
        card_id = (body.get("id") or "").strip()
        column = (body.get("column") or "").strip()
        if not card_id or not column:
            return self._error(400, "id and column are required")
        with self.stores.lock:
            card = self.stores.kanban_store.move_card(card_id, column)
            if card is None:
                return self._error(404, f"card not found or unknown column: {card_id}")
            new_status = COLUMN_TO_STATUS.get(column)
            if new_status:
                for note in self.stores.note_store.list_notes(limit=9999):
                    if note.title == card.title and note.status != new_status:
                        self.stores.note_store.update(note.id, status=new_status)
                        break
        self._send(200, {"card": _card_to_dict(card)})

    def _handle_sync(self) -> None:
        with self.stores.lock:
            added, updated, total = sync_notes_to_board(self.stores.note_store, self.stores.kanban_store)
        self._send(200, {"added": added, "updated": updated, "total": total})


def _bind_server(host: str, port: int, handler, note_store, kanban_store, attempts: int = 20) -> GuiServer:
    """Bind a GuiServer, stepping past ports already in use.

    Tries *port*, then ``port + 1`` ... ``port + attempts - 1`` when an
    ``EADDRINUSE`` error is raised. ``port == 0`` requests an ephemeral
    kernel-assigned port (never conflicts, so the first try wins). Raises the
    last ``OSError`` when no candidate binds.

    Side effects:
        - Binds a listening socket on the first free port.
    """
    last: OSError | None = None
    for offset in range(attempts):
        target = port + offset if port else 0
        try:
            return GuiServer((host, target), handler, note_store, kanban_store)
        except OSError as exc:
            last = exc
            if getattr(exc, "errno", None) == errno.EADDRINUSE and port:
                continue
            raise
    raise last


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="planner gui",
        description="Local web interface for notes + kanban board.",
    )
    parser.add_argument("--notes-dir", default=None, help="Notes directory (default: config/env)")
    parser.add_argument("--board-dir", default=None, help="Board directory (default: config/env)")
    parser.add_argument("--backend", default=None, choices=config.BACKENDS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser")
    parser.add_argument("--sync", action="store_true", help="Sync notes to board on start")
    args = parser.parse_args(argv)

    notes_dir = Path(args.notes_dir) if args.notes_dir else config.default_notes_dir()
    board_dir = Path(args.board_dir) if args.board_dir else config.default_board_dir()

    backend = args.backend or config.default_backend(notes_dir=notes_dir)
    note_store = core_module.NoteStore(notes_dir=notes_dir, backend=backend)
    kanban_store = KanbanStore(board_dir=board_dir)

    handler = GuiHandler
    server = _bind_server(args.host, args.port, handler, note_store, kanban_store)
    host, port = server.server_address[:2]

    if args.sync:
        with server.lock:
            added, updated, total = sync_notes_to_board(server.note_store, server.kanban_store)
        print(f"Synced notes to board: {added} new, {updated} moved, {total} total")

    if args.port and port != args.port:
        print(f"Planner GUI:  http://{host}:{port}   (port {args.port} in use, moved to {port})")
    else:
        print(f"Planner GUI:  http://{host}:{port}")
    print(f"  notes:  {note_store._dir}")
    print(f"  board:  {kanban_store._dir}")
    print(f"  backend: {backend}   (Ctrl+C to stop)")

    if not args.no_open:
        threading.Timer(0.3, lambda: _try_open(f"http://{host}:{port}")).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _try_open(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001 — headless environments have no browser
        pass


def _unquote_path(path: str) -> str:
    from urllib.parse import unquote

    return unquote(path)


def _coerce_tags(tags: Any) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    if isinstance(tags, (list, tuple)):
        return [str(t).strip() for t in tags if str(t).strip()]
    return []


def _int_or(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Embedded single-page app (no external assets — works fully offline)
# ---------------------------------------------------------------------------

GUI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Planner</title>
<style>
:root{
  --bg:#111016; --panel:#17151e; --card:#1c1926; --border:#342e48;
  --muted:#968cac; --muted2:#5c5470;
  --primary:#c0aaf4; --accent:#f0b082; --success:#48c08c;
  --warning:#f0c050; --destructive:#eb646e; --fg:#e8e3f5;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
a{color:var(--primary)}
header{display:flex;align-items:center;gap:14px;padding:12px 18px;
  border-bottom:1px solid var(--border);background:var(--panel);
  position:sticky;top:0;z-index:5;flex-wrap:wrap}
header h1{font-size:18px;margin:0;font-weight:600;letter-spacing:.3px}
header h1 span{color:var(--primary)}
.tabs{display:flex;gap:4px}
.tabs button{background:transparent;border:1px solid transparent;color:var(--muted);
  padding:6px 12px;border-radius:8px;cursor:pointer;font-size:13px}
.tabs button.active{background:var(--primary);color:#151220;font-weight:600}
.tabs button:hover:not(.active){color:var(--fg);border-color:var(--border)}
.spacer{flex:1}
input[type=text],select,textarea{background:var(--card);border:1px solid var(--border);
  color:var(--fg);border-radius:8px;padding:7px 10px;font-size:13px;outline:none}
input[type=text]:focus,select:focus,textarea:focus{border-color:var(--primary)}
.btn{background:var(--primary);color:#151220;border:none;border-radius:8px;
  padding:7px 14px;cursor:pointer;font-weight:600;font-size:13px}
.btn.ghost{background:transparent;color:var(--muted);border:1px solid var(--border)}
.btn:hover{filter:brightness(1.08)}
.btn.danger{background:var(--destructive);color:#fff}
.btn.sm{padding:4px 9px;font-size:12px}
main{padding:18px;max-width:1400px;margin:0 auto}
.hidden{display:none}

/* board */
.board{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
  gap:12px;align-items:start}
.col{background:var(--panel);border:1px solid var(--border);border-radius:12px;
  min-height:160px;padding:10px}
.col.drag-over{border-color:var(--primary);background:#1e1a2c}
.col h2{margin:0 0 8px;font-size:12px;text-transform:uppercase;letter-spacing:1px;
  color:var(--muted);display:flex;justify-content:space-between;align-items:center}
.col h2 .count{background:var(--card);border-radius:20px;padding:0 8px;font-size:11px}
.col .cards{display:flex;flex-direction:column;gap:8px;min-height:40px}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:9px 11px;cursor:grab;transition:box-shadow .12s}
.card:hover{box-shadow:0 4px 14px rgba(0,0,0,.35);border-color:var(--muted2)}
.card.dragging{opacity:.4}
.card .t{font-weight:600;font-size:13px;word-break:break-word}
.card .meta{display:flex;gap:6px;align-items:center;margin-top:6px;flex-wrap:wrap}
.chip{background:#241f36;color:var(--muted);border-radius:6px;padding:1px 7px;font-size:11px}
.pri{font-size:11px;color:var(--accent);font-weight:700;letter-spacing:1px}
.due{font-size:11px;color:var(--warning)}

/* notes list */
.note-row{display:flex;gap:12px;align-items:flex-start;background:var(--card);
  border:1px solid var(--border);border-radius:10px;padding:10px 14px;cursor:pointer;
  margin-bottom:8px}
.note-row:hover{border-color:var(--muted2)}
.note-row .st{font-size:16px;margin-top:2px}
.note-row .body{flex:1;min-width:0}
.note-row .title{font-weight:600}
.note-row .preview{color:var(--muted);font-size:12px;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;max-width:70ch}
.note-row .tags{margin-top:4px}
.note-row .when{color:var(--muted2);font-size:12px;white-space:nowrap}
.note-row .acts{display:flex;gap:6px;opacity:0;transition:opacity .12s}
.note-row:hover .acts{opacity:1}

/* stats */
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.stat{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}
.stat .n{font-size:30px;font-weight:700;color:var(--primary)}
.stat .l{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:1px}
.stat .bar{height:8px;background:var(--panel);border-radius:20px;overflow:hidden;margin-top:10px}
.stat .bar i{display:block;height:100%;background:var(--primary)}

/* modal */
.overlay{position:fixed;inset:0;background:rgba(8,7,12,.7);display:flex;align-items:flex-start;
  justify-content:center;padding:5vh 16px;z-index:20;overflow:auto}
.modal{background:var(--panel);border:1px solid var(--border);border-radius:14px;
  width:100%;max-width:640px;padding:20px}
.modal h2{margin:0 0 14px;font-size:16px}
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.field{display:flex;flex-direction:column;gap:5px}
.field.full{grid-column:1/-1}
.field label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}
.field textarea{resize:vertical;min-height:160px;font-family:ui-monospace,Menlo,Consolas,monospace}
.modal .acts{display:flex;gap:10px;justify-content:flex-end;margin-top:16px}

#toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);z-index:40;
  background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:9px 16px;font-size:13px;box-shadow:0 6px 24px rgba(0,0,0,.4);
  opacity:0;transition:opacity .15s;pointer-events:none}
#toast.show{opacity:1}
#toast.err{border-color:var(--destructive)}
.empty{color:var(--muted2);text-align:center;padding:40px 0}
@media (max-width:720px){.form-grid{grid-template-columns:1fr}.spacer{display:none}}
</style>
</head>
<body>
<header>
  <h1>Planner<span>.</span></h1>
  <nav class="tabs">
    <button data-tab="board" class="active">Board</button>
    <button data-tab="notes">Notes</button>
    <button data-tab="stats">Stats</button>
  </nav>
  <input type="text" id="search" placeholder="Search notes / cards..." style="flex:1;max-width:320px" />
  <div class="spacer"></div>
  <button class="btn ghost" id="syncBtn">Sync board</button>
  <button class="btn" id="newBtn">+ New note</button>
</header>

<main>
  <section id="tab-board">
    <div class="board" id="board"></div>
  </section>
  <section id="tab-notes" class="hidden"></section>
  <section id="tab-stats" class="hidden">
    <div class="stats-grid" id="stats"></div>
  </section>
</main>

<div id="toast"></div>

<div class="overlay hidden" id="editor">
  <div class="modal">
    <h2 id="editorTitle">New note</h2>
    <form id="noteForm">
      <div class="form-grid">
        <div class="field full">
          <label>Title</label>
          <input type="text" id="f_title" required maxlength="120" />
        </div>
        <div class="field">
          <label>Status</label>
          <select id="f_status">
            <option value="open">open</option>
            <option value="wip">wip</option>
            <option value="review">review</option>
            <option value="done">done</option>
            <option value="blocked">blocked</option>
          </select>
        </div>
        <div class="field">
          <label>Tags (comma separated)</label>
          <input type="text" id="f_tags" placeholder="core, bugfix, shell" />
        </div>
        <div class="field">
          <label>Sprint</label>
          <input type="text" id="f_sprint" placeholder="S1, 2026-Q3" />
        </div>
        <div class="field">
          <label>GitHub issue</label>
          <input type="text" id="f_gh" placeholder="owner/repo#123" />
        </div>
        <div class="field full">
          <label>Body</label>
          <textarea id="f_body" placeholder="Details..."></textarea>
        </div>
      </div>
      <div class="acts">
        <button type="button" class="btn ghost" id="cancelBtn">Cancel</button>
        <button type="button" class="btn danger hidden" id="deleteBtn">Delete</button>
        <button type="submit" class="btn">Save</button>
      </div>
    </form>
  </div>
</div>

<script>
"use strict";
const SICON = {open:"\u25cb", wip:"\u25d0", done:"\u25cf", blocked:"\u2715", review:"\u25c8"};
const SCOL  = {open:"#968cac", wip:"#f0c050", done:"#48c08c", blocked:"#eb646e", review:"#c0aaf4"};
const PICO  = {low:"", medium:"!", high:"!!", critical:"!!!"};
let state = { tab:"board", notes:[], board:null, tags:[], stats:null, editing:null };

const $ = id => document.getElementById(id);

async function api(path, opts={}) {
  const res = await fetch(path, {headers:{"Content-Type":"application/json"}, ...opts});
  const data = await res.json().catch(()=>({}));
  if (!res.ok) throw new Error(data.error || ("HTTP " + res.status));
  return data;
}

let toastTimer = null;
function toast(msg, isErr=false) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "show" + (isErr ? " err" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 2600);
}

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g,
    c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

function statusColor(s){ return SCOL[s] || "#968cac"; }

/* ---------------- loaders ---------------- */

async function refresh() {
  await Promise.all([loadBoard(), loadNotes()]);
  render();
}

async function loadBoard() {
  const data = await api("/api/board");
  state.board = data.board;
}

async function loadNotes() {
  const data = await api("/api/notes?limit=9999");
  state.notes = data.notes;
}

async function loadStats() {
  state.stats = await api("/api/stats");
  const t = await api("/api/tags");
  state.tags = t.tags;
}

/* ---------------- tabs ---------------- */

function render() {
  const q = ($("search").value || "").trim().toLowerCase();
  if (state.tab === "board") renderBoard(q);
  else if (state.tab === "notes") renderNotes(q);
  else if (state.tab === "stats") renderStats();
}

function renderBoard(q) {
  const b = state.board;
  if (!b) { $("board").innerHTML = '<div class="empty">Loading...</div>'; return; }
  const cards = b.cards.filter(c => !q ||
    c.title.toLowerCase().includes(q) ||
    c.tags.join(" ").toLowerCase().includes(q));
  $("board").innerHTML = b.columns.map(col => {
    const items = cards.filter(c => c.column === col.name);
    const wip = col.wip_limit ? `<span class="count">${items.length}/${col.wip_limit}</span>`
                              : `<span class="count">${items.length}</span>`;
    return `<div class="col" data-col="${esc(col.name)}" data-wip="${col.wip_limit}">
      <h2>${esc(col.name)} ${wip}</h2>
      <div class="cards">${
        items.map(c => cardHtml(c)).join("")
      }</div></div>`;
  }).join("");
  bindBoard();
}

function cardHtml(c) {
  const tags = c.tags.slice(0,3).map(t => `<span class="chip">${esc(t)}</span>`).join("");
  const due = c.due_date ? `<span class="due">&#128197; ${esc(c.due_date)}</span>` : "";
  const n = c.notes.length ? `<span class="chip">${c.notes.length} note${c.notes.length>1?"s":""}</span>` : "";
  return `<div class="card" draggable="true" data-id="${esc(c.id)}" title="${esc(c.description||"")}">
    <div class="t">${esc(c.title)}</div>
    <div class="meta">${PICO[c.priority] ? `<span class="pri">${PICO[c.priority]}</span>` : ""}${tags}${n}${due}</div>
  </div>`;
}

function bindBoard() {
  document.querySelectorAll(".card").forEach(card => {
    card.addEventListener("dragstart", e => {
      e.dataTransfer.setData("text/plain", card.dataset.id);
      card.classList.add("dragging");
    });
    card.addEventListener("dragend", () => card.classList.remove("dragging"));
  });
  document.querySelectorAll(".col").forEach(col => {
    col.addEventListener("dragover", e => { e.preventDefault(); col.classList.add("drag-over"); });
    col.addEventListener("dragleave", () => col.classList.remove("drag-over"));
    col.addEventListener("drop", async e => {
      e.preventDefault();
      col.classList.remove("drag-over");
      const id = e.dataTransfer.getData("text/plain");
      const target = col.dataset.col;
      try {
        const res = await api("/api/board/move", {method:"POST", body:JSON.stringify({id, column:target})});
        await refresh();
        toast(`Moved \u2192 ${target}`);
        if (res.card) {} 
      } catch (err) { toast(err.message, true); }
    });
  });
}

function renderNotes(q) {
  const notes = state.notes.filter(n => !q ||
    n.title.toLowerCase().includes(q) ||
    n.tags.join(" ").toLowerCase().includes(q) ||
    n.body.toLowerCase().includes(q));
  const box = $("tab-notes");
  if (!notes.length) { box.innerHTML = '<div class="empty">No notes found.</div>'; return; }
  box.innerHTML = notes.map(n => {
    const tags = n.tags.map(t => `<span class="chip">${esc(t)}</span>`).join("");
    const preview = (n.body||"").replace(/\s+/g," ").trim();
    const when = (n.updated_at||n.created_at||"").slice(0,10);
    return `<div class="note-row" data-id="${esc(n.id)}">
      <div class="st" style="color:${statusColor(n.status)}">${SICON[n.status]||"\u25cb"}</div>
      <div class="body">
        <div class="title">${esc(n.title)}</div>
        ${preview ? `<div class="preview">${esc(preview.slice(0,140))}</div>` : ""}
        <div class="tags">${tags}</div>
      </div>
      <div class="when">${when}</div>
      <div class="acts">
        <button class="btn ghost sm" data-act="edit">Edit</button>
        <button class="btn danger sm" data-act="del">Del</button>
      </div>
    </div>`;
  }).join("");
  box.querySelectorAll(".note-row").forEach(row => {
    const id = row.dataset.id;
    row.querySelector('[data-act="edit"]').addEventListener("click", e => {
      e.stopPropagation();
      const note = state.notes.find(n => n.id === id);
      if (note) openEditor(note);
    });
    row.querySelector('[data-act="del"]').addEventListener("click", async e => {
      e.stopPropagation();
      if (!confirm("Delete this note?")) return;
      try {
        await api("/api/notes/" + encodeURIComponent(id), {method:"DELETE"});
        await refresh();
        toast("Deleted");
      } catch (err) { toast(err.message, true); }
    });
    row.addEventListener("click", () => {
      const note = state.notes.find(n => n.id === id);
      if (note) openEditor(note);
    });
  });
}

async function renderStats() {
  if (!state.stats) await loadStats();
  const s = state.stats;
  const max = Math.max(1, ...Object.values(s.by_status||{}));
  const rows = Object.entries(s.by_status||{}).map(([k,v]) => `
    <div class="stat">
      <div class="n">${v}</div>
      <div class="l">${esc(k)}</div>
      <div class="bar"><i style="width:${Math.round(v/max*100)}%;background:${statusColor(k)}"></i></div>
    </div>`).join("");
  const tagBox = state.tags.length
    ? state.tags.map(t => `<div class="stat"><div class="n" style="font-size:22px">${t.count}</div><div class="l">${esc(t.name)}</div></div>`).join("")
    : '<div class="empty">No tags yet.</div>';
  $("stats").innerHTML = `
    <div class="stat"><div class="n">${s.total||0}</div><div class="l">Total notes</div></div>
    <div class="stat"><div class="n">${s.today||0}</div><div class="l">Created today</div></div>
    ${rows}
    <div class="stat" style="grid-column:1/-1">
      <div class="l" style="margin-bottom:10px">Tags</div>
      <div class="stats-grid">${tagBox}</div>
    </div>`;
}

/* ---------------- editor ---------------- */

function openEditor(note) {
  state.editing = note ? note.id : null;
  $("editorTitle").textContent = note ? "Edit note" : "New note";
  $("f_title").value = note ? note.title : "";
  $("f_status").value = note ? (note.status||"open") : "open";
  $("f_tags").value = note ? note.tags.join(", ") : "";
  $("f_sprint").value = note ? (note.sprint||"") : "";
  $("f_gh").value = note ? (note.gh||"") : "";
  $("f_body").value = note ? (note.body||"") : "";
  $("deleteBtn").classList.toggle("hidden", !note);
  $("editor").classList.remove("hidden");
  $("f_title").focus();
}

function closeEditor() {
  $("editor").classList.add("hidden");
  state.editing = null;
}

$("noteForm").addEventListener("submit", async e => {
  e.preventDefault();
  const payload = {
    title: $("f_title").value.trim(),
    status: $("f_status").value,
    tags: $("f_tags").value.split(",").map(s => s.trim()).filter(Boolean),
    sprint: $("f_sprint").value.trim(),
    gh: $("f_gh").value.trim(),
    body: $("f_body").value,
  };
  try {
    if (state.editing) {
      await api("/api/notes/" + encodeURIComponent(state.editing), {method:"PUT", body:JSON.stringify(payload)});
      toast("Note updated");
    } else {
      await api("/api/notes", {method:"POST", body:JSON.stringify(payload)});
      toast("Note created");
    }
    closeEditor();
    await refresh();
  } catch (err) { toast(err.message, true); }
});

$("deleteBtn").addEventListener("click", async () => {
  if (!state.editing || !confirm("Delete this note?")) return;
  try {
    await api("/api/notes/" + encodeURIComponent(state.editing), {method:"DELETE"});
    closeEditor();
    await refresh();
    toast("Deleted");
  } catch (err) { toast(err.message, true); }
});
$("cancelBtn").addEventListener("click", closeEditor);
$("editor").addEventListener("click", e => { if (e.target === $("editor")) closeEditor(); });

/* ---------------- top bar ---------------- */

document.querySelectorAll(".tabs button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.tab = btn.dataset.tab;
    $("tab-board").classList.toggle("hidden", state.tab !== "board");
    $("tab-notes").classList.toggle("hidden", state.tab !== "notes");
    $("tab-stats").classList.toggle("hidden", state.tab !== "stats");
    if (state.tab === "stats") renderStats();
    else render();
  });
});

let searchTimer = null;
$("search").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(render, 120);
});
$("newBtn").addEventListener("click", () => openEditor(null));

$("syncBtn").addEventListener("click", async () => {
  $("syncBtn").disabled = true;
  try {
    const r = await api("/api/sync", {method:"POST"});
    await refresh();
    const parts = [];
    if (r.added) parts.push(`${r.added} added`);
    if (r.updated) parts.push(`${r.updated} moved`);
    toast(parts.length ? `Synced — ${parts.join(", ")}` : "Board up to date");
  } catch (err) { toast(err.message, true); }
  finally { $("syncBtn").disabled = false; }
});

/* ---------------- boot ---------------- */
refresh().catch(err => toast(err.message, true));
</script>
</body>
</html>
"""
