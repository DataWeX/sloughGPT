"""Planner HTTP projection — the retired Next ``/api/planner`` handlers, ported.

Parity contract (card ``51cd0e65``): every response body matches what
``apps/web/app/api/planner`` served — ``http-client`` unwraps only the shared
``{"status": "success", "data": …}`` envelope (``apps/lib/request.ts``), so
handlers return the exact ``{board}`` / ``{card}`` / ``{success}`` payloads the
typed frontend clients expect. Success statuses match too: creates answer 201
(``RouteSpec.status_code``), deletes/moves answer ``{success: true}``, misses
answer 404 via ``raise_error(..., "E_NOT_FOUND", status_code=404)``.

The store is ``PlannerStore`` — OCC writes, chain-sealed board — never raw
file IO: a raw writer strips ``chain_index``/``chain_hash`` from every card on
the next rewrite (observed when a chain-less planner CLI rewrote the board).

Data-source notes vs the Next handlers:
  * notes come from the active journal ``.dev-notes/notes.journal.jsonl``
    (PlannerStore default) instead of the stale ``.dev-notes/store/`` copy
    (untouched since 2026-09-29);
  * ``POST /api/planner/sync`` stays a no-op returning zeros — exactly what
    the Next handler returned. ``BuffetEngine`` calls it during UI flows, and
    a real notes→board reconciliation there would move cards under the user;
    reconciliation stays the CLI's job (``planner sync``);
  * ``x-workspace-id`` isolates into ``.kanban/<ws>`` / ``.dev-notes/<ws>``
    (the Next notes layout ``.dev-notes/<ws>/store`` was never exercised —
    no frontend caller sends the header).
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from app_planner import config as planner_config
from app_planner.store import PlannerStore, get_store
from fastapi import APIRouter, Request

# Boot tripwire: a chain-less app_planner (resolved from a checkout that
# predates the hash chain) would strip chains from every board write. Fail
# the projection loudly instead of silently corrupting the shared board.
if not hasattr(PlannerStore, "compute_chains"):
    raise RuntimeError(
        "app_planner resolved to a chain-less copy (PlannerStore.compute_chains missing) — "
        "point PYTHONPATH at <main checkout>/packages/app-planner/src"
    )

from infrastructure.contract import RouteSpec, create_router  # noqa: E402
from schemas.common import raise_error  # noqa: E402

from domain.agents import ToolParam, ToolSpec  # noqa: E402

_WS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ALL = 9999  # the Next handlers always returned every note


def _store(request: Request) -> tuple[PlannerStore, Path]:
    """Store for this request: default project board, or an isolated workspace."""
    ws = request.headers.get("x-workspace-id", "")
    if not ws or not _WS_RE.match(ws):
        return get_store(), planner_config.default_board_dir()
    root = planner_config.find_project_root() or Path.cwd()
    board_dir = root / ".kanban" / ws
    return PlannerStore(board_dir=board_dir, notes_dir=root / ".dev-notes" / ws), board_dir


def _overlay_feed(payload: dict[str, Any], board_dir: Path) -> None:
    """Merge unknown cards from ``<board_dir>/app-planner-feed/board.json`` (Next parity)."""
    feed_file = board_dir / "app-planner-feed" / "board.json"
    if not feed_file.is_file():
        return
    try:
        parsed = json.loads(feed_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    cards = parsed.get("cards") if isinstance(parsed, dict) else None
    if not isinstance(cards, list):
        return
    known = {c.get("id") for c in payload["cards"]}
    for card in cards:
        if not isinstance(card, dict) or not card.get("id") or not card.get("title"):
            continue
        if card["id"] in known:
            continue
        known.add(card["id"])
        payload["cards"].append(card)


def _board_payload(store: PlannerStore, board_dir: Path) -> dict[str, Any]:
    board = store.load_board()
    payload: dict[str, Any] = {
        "name": board.name,
        "columns": board.columns,
        "cards": [card.to_dict() for card in board.cards],
    }
    _overlay_feed(payload, board_dir)
    return payload


# ── Board ───────────────────────────────────────────────────────────────────


async def read_board(request: Request) -> dict[str, Any]:
    def work() -> dict[str, Any]:
        store, board_dir = _store(request)
        return {"board": _board_payload(store, board_dir)}

    return await asyncio.to_thread(work)


async def create_card(
    request: Request,
    title: str,
    description: str | None = None,
    column: str | None = None,
    priority: str | None = None,
    tags: list[str] | None = None,
    due_date: str | None = None,
    assignee: str | None = None,
    sprint: str | None = None,
    gh: str | None = None,
) -> dict[str, Any]:
    def work() -> dict[str, Any]:
        store, _ = _store(request)
        extra = {
            k: v
            for k, v in dict(
                description=description,
                column=column,
                priority=priority,
                tags=tags,
                due_date=due_date,
                assignee=assignee,
                sprint=sprint,
                gh=gh,
            ).items()
            if v is not None
        }
        card = store.add_card(title=title, **extra)
        return {"card": card.to_dict()}

    return await asyncio.to_thread(work)


async def update_card(
    request: Request,
    title: str | None = None,
    description: str | None = None,
    priority: str | None = None,
    tags: list[str] | None = None,
    due_date: str | None = None,
    assignee: str | None = None,
    column: str | None = None,
    sprint: str | None = None,
    gh: str | None = None,
) -> dict[str, Any]:
    card_id = request.path_params["id"]

    def work() -> dict[str, Any]:
        store, _ = _store(request)
        fields = {
            k: v
            for k, v in dict(
                title=title,
                description=description,
                priority=priority,
                tags=tags,
                due_date=due_date,
                assignee=assignee,
                column=column,
                sprint=sprint,
                gh=gh,
            ).items()
            if v is not None
        }
        card = store.update_card(card_id, **fields)
        if card is None:
            raise_error("Card not found", "E_NOT_FOUND", status_code=404)
        return {"card": card.to_dict()}

    return await asyncio.to_thread(work)


async def delete_card(request: Request) -> dict[str, Any]:
    card_id = request.path_params["id"]

    def work() -> dict[str, Any]:
        store, _ = _store(request)
        if not store.delete_card(card_id):
            raise_error("Card not found", "E_NOT_FOUND", status_code=404)
        return {"success": True}

    return await asyncio.to_thread(work)


async def move_card(request: Request, card_id: str, column: str) -> dict[str, Any]:
    def work() -> dict[str, Any]:
        store, _ = _store(request)
        if not store.move_card(card_id, column):
            raise_error("Card not found", "E_NOT_FOUND", status_code=404)
        return {"success": True}

    return await asyncio.to_thread(work)


# ── Notes ───────────────────────────────────────────────────────────────────


async def read_notes(request: Request) -> dict[str, Any]:
    def work() -> dict[str, Any]:
        store, _ = _store(request)
        return {"notes": [note.to_dict() for note in store.list_notes(limit=_ALL)]}

    return await asyncio.to_thread(work)


async def create_note(
    request: Request,
    title: str,
    body: str | None = None,
    status: str | None = None,
    tags: list[str] | None = None,
    sprint: str | None = None,
    gh: str | None = None,
) -> dict[str, Any]:
    def work() -> dict[str, Any]:
        store, _ = _store(request)
        extra = {
            k: v
            for k, v in dict(body=body, status=status, tags=tags, sprint=sprint, gh=gh).items()
            if v is not None
        }
        note = store.create_note(title=title, **extra)
        return {"note": note.to_dict()}

    return await asyncio.to_thread(work)


async def update_note(
    request: Request,
    title: str | None = None,
    body: str | None = None,
    status: str | None = None,
    tags: list[str] | None = None,
    sprint: str | None = None,
    gh: str | None = None,
) -> dict[str, Any]:
    note_id = request.path_params["id"]

    def work() -> dict[str, Any]:
        store, _ = _store(request)
        fields = {
            k: v
            for k, v in dict(title=title, body=body, status=status, tags=tags, sprint=sprint, gh=gh).items()
            if v is not None
        }
        note = store.update_note(note_id, **fields)
        if note is None:
            raise_error("Note not found", "E_NOT_FOUND", status_code=404)
        return {"note": note.to_dict()}

    return await asyncio.to_thread(work)


async def delete_note(request: Request) -> dict[str, Any]:
    note_id = request.path_params["id"]

    def work() -> dict[str, Any]:
        store, _ = _store(request)
        if not store.delete_note(note_id):
            raise_error("Note not found", "E_NOT_FOUND", status_code=404)
        return {"success": True}

    return await asyncio.to_thread(work)


# ── Stats, tags, sync ───────────────────────────────────────────────────────


async def read_stats(request: Request) -> dict[str, Any]:
    def work() -> dict[str, Any]:
        store, board_dir = _store(request)
        payload = _board_payload(store, board_dir)
        by_column: dict[str, int] = {}
        for card in payload["cards"]:
            col = card.get("column", "")
            by_column[col] = by_column.get(col, 0) + 1
        notes = store.list_notes(limit=_ALL)
        # Exact Next shape (helpers.getStats) — deliberately not store.get_stats.
        stats = {
            "total_cards": len(payload["cards"]),
            "byColumn": by_column,
            "columns": len(payload["columns"]),
            "total_notes": len(notes),
        }
        return {"stats": stats}

    return await asyncio.to_thread(work)


async def read_tags(request: Request) -> dict[str, Any]:
    def work() -> dict[str, Any]:
        store, board_dir = _store(request)
        payload = _board_payload(store, board_dir)
        counts: dict[str, int] = {}
        for card in payload["cards"]:
            for tag in card.get("tags") or []:
                counts[tag] = counts.get(tag, 0) + 1
        # Descending by count; ties keep board order (stable sort — Next used Map order).
        tags = [{"name": name, "count": n} for name, n in sorted(counts.items(), key=lambda kv: -kv[1])]
        return {"tags": tags}

    return await asyncio.to_thread(work)


async def sync_notes() -> dict[str, Any]:
    # Next parity: intentionally a no-op. Reconciliation is the CLI's job.
    return {"added": 0, "updated": 0, "total": 0}


# ── Descriptors + projection ────────────────────────────────────────────────

_TAGS = ("planner",)
_PREFIX = "/api/planner"
_OBJ = {"type": "object"}
_SUCC = {"type": "object", "properties": {"success": {"type": "boolean"}}}


# Descriptor literals stay inline: scripts/check_contract.py validates them by
# AST (no imports) — name/description/auth_scope/version must be literals and
# `parameters` a literal list of ToolParam(...) calls. Helpers would hide them.
_PROJECTIONS: tuple[tuple[ToolSpec, RouteSpec, Any], ...] = (
    (
        ToolSpec(
            name="planner.board.read",
            description="Read the kanban board (name, columns, cards) as the planner UI sees it.",
            version="1",
            parameters=[],
            execute=read_board,
            auth_scope="authenticated",
            idempotent=True,
            result={"type": "object", "properties": {"board": _OBJ}},
        ),
        RouteSpec(path="/board", method="GET", summary="Read the board"),
        read_board,
    ),
    (
        ToolSpec(
            name="planner.card.create",
            description="Create a card on the board.",
            version="1",
            parameters=[
                ToolParam("title", "string", "Card title.", required=True),
                ToolParam("column", "string", "Board column (todo/wip/review/done)."),
                ToolParam("description", "string", "Card description."),
                ToolParam("priority", "string", "Card priority (low/medium/high)."),
                ToolParam("tags", "array", "Tag labels."),
                ToolParam("due_date", "string", "Due date (ISO)."),
                ToolParam("assignee", "string", "Assignee."),
                ToolParam("sprint", "string", "Sprint label."),
                ToolParam("gh", "string", "GitHub reference."),
            ],
            execute=create_card,
            auth_scope="authenticated",
            idempotent=False,
            result={"type": "object", "properties": {"card": _OBJ}},
        ),
        RouteSpec(path="/board/cards", method="POST", summary="Create a card", status_code=201),
        create_card,
    ),
    (
        ToolSpec(
            name="planner.card.update",
            description="Update fields of an existing card.",
            version="1",
            parameters=[
                ToolParam("title", "string", "Card title."),
                ToolParam("column", "string", "Board column (todo/wip/review/done)."),
                ToolParam("description", "string", "Card description."),
                ToolParam("priority", "string", "Card priority (low/medium/high)."),
                ToolParam("tags", "array", "Tag labels."),
                ToolParam("due_date", "string", "Due date (ISO)."),
                ToolParam("assignee", "string", "Assignee."),
                ToolParam("sprint", "string", "Sprint label."),
                ToolParam("gh", "string", "GitHub reference."),
            ],
            execute=update_card,
            auth_scope="authenticated",
            idempotent=True,
            result={"type": "object", "properties": {"card": _OBJ}},
        ),
        RouteSpec(path="/board/cards/{id}", method="PUT", summary="Update a card"),
        update_card,
    ),
    (
        ToolSpec(
            name="planner.card.delete",
            description="Delete a card by id (identity carried in the path, no body).",
            version="1",
            parameters=[],
            execute=delete_card,
            auth_scope="authenticated",
            idempotent=True,
            result=_SUCC,
        ),
        RouteSpec(path="/board/cards/{id}", method="DELETE", summary="Delete a card"),
        delete_card,
    ),
    (
        ToolSpec(
            name="planner.card.move",
            description="Move a card to another column.",
            version="1",
            parameters=[
                ToolParam("card_id", "string", "Card to move.", required=True),
                ToolParam("column", "string", "Destination column.", required=True),
            ],
            execute=move_card,
            auth_scope="authenticated",
            idempotent=False,
            result=_SUCC,
        ),
        RouteSpec(path="/board/move", method="POST", summary="Move a card"),
        move_card,
    ),
    (
        ToolSpec(
            name="planner.notes.read",
            description="List dev-journal notes.",
            version="1",
            parameters=[],
            execute=read_notes,
            auth_scope="authenticated",
            idempotent=True,
            result={"type": "object", "properties": {"notes": {"type": "array", "items": _OBJ}}},
        ),
        RouteSpec(path="/notes", method="GET", summary="List notes"),
        read_notes,
    ),
    (
        ToolSpec(
            name="planner.note.create",
            description="Create a note.",
            version="1",
            parameters=[
                ToolParam("title", "string", "Note title.", required=True),
                ToolParam("body", "string", "Note body."),
                ToolParam("status", "string", "Note status."),
                ToolParam("tags", "array", "Tag labels."),
                ToolParam("sprint", "string", "Sprint label."),
                ToolParam("gh", "string", "GitHub reference."),
            ],
            execute=create_note,
            auth_scope="authenticated",
            idempotent=False,
            result={"type": "object", "properties": {"note": _OBJ}},
        ),
        RouteSpec(path="/notes", method="POST", summary="Create a note", status_code=201),
        create_note,
    ),
    (
        ToolSpec(
            name="planner.note.update",
            description="Update fields of an existing note.",
            version="1",
            parameters=[
                ToolParam("title", "string", "Note title."),
                ToolParam("body", "string", "Note body."),
                ToolParam("status", "string", "Note status."),
                ToolParam("tags", "array", "Tag labels."),
                ToolParam("sprint", "string", "Sprint label."),
                ToolParam("gh", "string", "GitHub reference."),
            ],
            execute=update_note,
            auth_scope="authenticated",
            idempotent=True,
            result={"type": "object", "properties": {"note": _OBJ}},
        ),
        RouteSpec(path="/notes/{id}", method="PUT", summary="Update a note"),
        update_note,
    ),
    (
        ToolSpec(
            name="planner.note.delete",
            description="Delete a note by id (identity carried in the path, no body).",
            version="1",
            parameters=[],
            execute=delete_note,
            auth_scope="authenticated",
            idempotent=True,
            result=_SUCC,
        ),
        RouteSpec(path="/notes/{id}", method="DELETE", summary="Delete a note"),
        delete_note,
    ),
    (
        ToolSpec(
            name="planner.stats.read",
            description="Board/note counters for the planner header.",
            version="1",
            parameters=[],
            execute=read_stats,
            auth_scope="authenticated",
            idempotent=True,
            result={"type": "object", "properties": {"stats": _OBJ}},
        ),
        RouteSpec(path="/stats", method="GET", summary="Read stats"),
        read_stats,
    ),
    (
        ToolSpec(
            name="planner.tags.read",
            description="Tag counts across cards, descending.",
            version="1",
            parameters=[],
            execute=read_tags,
            auth_scope="authenticated",
            idempotent=True,
            result={"type": "object", "properties": {"tags": {"type": "array", "items": _OBJ}}},
        ),
        RouteSpec(path="/tags", method="GET", summary="List tags"),
        read_tags,
    ),
    (
        ToolSpec(
            name="planner.notes.sync",
            description="Notes/board sync tick (parity no-op — reconciliation runs in the CLI).",
            version="1",
            parameters=[],
            execute=sync_notes,
            auth_scope="authenticated",
            idempotent=False,
            result={
                "type": "object",
                "properties": {
                    "added": {"type": "integer"},
                    "updated": {"type": "integer"},
                    "total": {"type": "integer"},
                },
            },
        ),
        RouteSpec(path="/sync", method="POST", summary="Sync tick"),
        sync_notes,
    ),
)


router = APIRouter()
for _spec, _route, _handler in _PROJECTIONS:
    router.include_router(
        create_router(_spec, _route, _handler, prefix=_PREFIX, tags=_TAGS)
    )
