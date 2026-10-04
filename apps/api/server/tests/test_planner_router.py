"""Planner router — exact request/response parity with the retired Next handlers.

Every test documents the contract the shipped frontend depends on
(``apps/web/lib/planner-client.ts`` unwraps only the shared success envelope
via ``apps/lib/request.ts``, so the inner ``{board}`` / ``{card}`` /
``{success}`` shapes must match the Next ``app/api/planner`` handlers byte
for byte). Divergences from Next are named where they exist and why
(contract-derived 422 vs hand-rolled 400; store-validated columns).

Descriptor names exercised here (contract gate: a descriptor without a test
is a wish — ``scripts/check_contract.py``):
    planner.board.read, planner.card.create, planner.card.update,
    planner.card.delete, planner.card.move, planner.notes.read,
    planner.note.create, planner.note.update, planner.note.delete,
    planner.stats.read, planner.tags.read, planner.notes.sync.
"""

from __future__ import annotations

import json

import pytest
from app_planner import config as planner_config
from app_planner.store import PlannerStore, reset_store
from infrastructure.contract import REGISTRY
from test_support import _data, get_test_client

DESCRIPTORS = (
    "planner.board.read",
    "planner.card.create",
    "planner.card.update",
    "planner.card.delete",
    "planner.card.move",
    "planner.notes.read",
    "planner.note.create",
    "planner.note.update",
    "planner.note.delete",
    "planner.stats.read",
    "planner.tags.read",
    "planner.notes.sync",
)


@pytest.fixture(autouse=True)
def planner_tmp(tmp_path, monkeypatch):
    """Every request must resolve to a throwaway board/notes tree — never the repo's."""
    monkeypatch.setenv("APP_PLANNER_BOARD_DIR", str(tmp_path / "kanban"))
    monkeypatch.setenv("APP_PLANNER_NOTES_DIR", str(tmp_path / "notes"))
    monkeypatch.delenv("APP_PLANNER_CALENDAR_DIR", raising=False)
    # workspace isolation resolves through find_project_root — pin it to tmp too
    monkeypatch.setattr(planner_config, "find_project_root", lambda *a, **k: tmp_path)
    reset_store()
    yield tmp_path
    reset_store()


def _seed_card(title: str = "Seed", column: str = "todo", **kw):
    return get_store().add_card(title=title, column=column, **kw)


def get_store():
    from app_planner.store import get_store as _gs

    return _gs()


# ── GET /api/planner/board — planner.board.read ────────────────────────────


def test_board_read_is_the_next_shape_under_the_shared_envelope():
    resp = get_test_client().get("/api/planner/board")
    assert resp.status_code == 200
    body = resp.json()
    # what http-client sees before unwrapping .data
    assert body["status"] == "success"
    board = _data(resp)["board"]
    assert set(board) == {"name", "columns", "cards"}
    assert isinstance(board["columns"], list) and isinstance(board["cards"], list)


def test_board_read_exposes_the_card_fields_the_frontend_types_require():
    _seed_card("First card", tags=["api"])
    board = _data(get_test_client().get("/api/planner/board"))["board"]
    card = next(c for c in board["cards"] if c["title"] == "First card")
    for field in (
        "id", "title", "description", "column", "priority", "tags",
        "due_date", "assignee", "sprint", "gh", "notes", "created_at", "updated_at",
    ):
        assert field in card, f"TS BoardCard requires {field}"
    assert card["column"] == "todo"
    assert card["priority"] == "medium"


def test_board_read_merges_app_planner_feed_cards():
    """Next merged <board>/app-planner-feed/board.json unknown cards — so do we."""
    feed_dir = planner_config.default_board_dir() / "app-planner-feed"
    feed_dir.mkdir(parents=True, exist_ok=True)
    (feed_dir / "board.json").write_text(
        json.dumps(
            {
                "cards": [
                    {"id": "feed-1", "title": "From feed"},
                    {"id": "feed-1", "title": "duplicate ignored"},
                    {"id": "", "title": "no id ignored"},
                    {"title": "no title ignored"},
                ]
            }
        ),
        encoding="utf-8",
    )
    board = _data(get_test_client().get("/api/planner/board"))["board"]
    feed_cards = [c for c in board["cards"] if c.get("id") == "feed-1"]
    assert len(feed_cards) == 1
    assert feed_cards[0]["title"] == "From feed"


def test_workspace_header_isolates_the_board(tmp_path):
    ws_store = PlannerStore(
        board_dir=tmp_path / ".kanban" / "ws1",
        notes_dir=tmp_path / ".dev-notes" / "ws1",
    )
    ws_store.add_card("Workspace card")
    resp = get_test_client().get("/api/planner/board", headers={"x-workspace-id": "ws1"})
    titles = [c["title"] for c in _data(resp)["board"]["cards"]]
    assert "Workspace card" in titles
    # and the default board is untouched
    default_titles = [
        c["title"] for c in _data(get_test_client().get("/api/planner/board"))["board"]["cards"]
    ]
    assert "Workspace card" not in default_titles


# ── POST /api/planner/board/cards — planner.card.create ─────────────────────


def test_card_create_answers_201_and_returns_the_card():
    resp = get_test_client().post(
        "/api/planner/board/cards", json={"title": "New card", "tags": ["x"]}
    )
    assert resp.status_code == 201  # Next parity — not the projection default 200
    card = _data(resp)["card"]
    assert card["title"] == "New card"
    assert card["column"] == "todo"
    assert card["priority"] == "medium"
    assert card["tags"] == ["x"]
    assert card["id"] and card["created_at"]


def test_card_create_with_missing_title_is_a_contract_violation():
    """Next answered a hand-rolled 400 here; the projection answers 422.

    Same rejection, contract-derived: the descriptor marks title required, so
    validation is generated, not re-decided. No frontend path sends a titleless
    card (the create form requires one).
    """
    resp = get_test_client().post("/api/planner/board/cards", json={"description": "no title"})
    assert resp.status_code == 422


# ── PUT /api/planner/board/cards/{id} — planner.card.update ─────────────────


def test_card_update_patches_fields_and_preserves_the_rest():
    card = _seed_card("Original")
    resp = get_test_client().put(
        f"/api/planner/board/cards/{card.id}", json={"title": "Renamed", "priority": "high"}
    )
    assert resp.status_code == 200
    updated = _data(resp)["card"]
    assert updated["title"] == "Renamed"
    assert updated["priority"] == "high"
    assert updated["description"] == card.description  # untouched


def test_card_update_unknown_id_is_404():
    resp = get_test_client().put("/api/planner/board/cards/nope", json={"title": "x"})
    assert resp.status_code == 404
    assert resp.json()["error"]


# ── DELETE /api/planner/board/cards/{id} — planner.card.delete ──────────────


def test_card_delete_sends_no_body_and_returns_success_true():
    """apiDelete() sends NO body — asserted in http-client.test.ts. Must still work."""
    card = _seed_card("Doomed")
    resp = get_test_client().delete(f"/api/planner/board/cards/{card.id}")  # no json=
    assert resp.status_code == 200
    assert _data(resp) == {"success": True}
    titles = [c["title"] for c in _data(get_test_client().get("/api/planner/board"))["board"]["cards"]]
    assert "Doomed" not in titles


def test_card_delete_unknown_id_is_404():
    assert get_test_client().delete("/api/planner/board/cards/nope").status_code == 404


# ── POST /api/planner/board/move — planner.card.move ────────────────────────


def test_card_move_returns_success_true_and_moves_the_card():
    card = _seed_card("Mover")
    resp = get_test_client().post(
        "/api/planner/board/move", json={"card_id": card.id, "column": "wip"}
    )
    assert resp.status_code == 200
    assert _data(resp) == {"success": True}
    moved = [c for c in _data(get_test_client().get("/api/planner/board"))["board"]["cards"]
             if c["id"] == card.id][0]
    assert moved["column"] == "wip"


def test_card_move_unknown_id_is_404():
    resp = get_test_client().post(
        "/api/planner/board/move", json={"card_id": "nope", "column": "done"}
    )
    assert resp.status_code == 404


def test_card_move_missing_fields_is_a_contract_violation():
    """Next answered 400 {error}; the projection answers 422 (same rejection)."""
    assert get_test_client().post("/api/planner/board/move", json={}).status_code == 422


def test_card_update_to_a_retired_column_is_rejected_by_the_store():
    """Store invariant (card 46f0fac4) wins over Next permissiveness: the
    column vocabulary lives in the store, so a retired spelling cannot enter
    the board through the API either."""
    card = _seed_card("Guarded")
    resp = get_test_client().put(
        f"/api/planner/board/cards/{card.id}", json={"column": "in_progress"}
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "E_BAD_REQUEST"


# ── /api/planner/notes — planner.notes.read / planner.note.* ────────────────


def test_notes_read_returns_every_note():
    store = get_store()
    for i in range(3):
        store.create_note(title=f"Note {i}")
    notes = _data(get_test_client().get("/api/planner/notes"))["notes"]
    assert len(notes) == 3  # limit=9999 — Next returned the whole journal
    assert {"id", "title", "body", "status", "tags", "created_at"} <= set(notes[0])


def test_note_create_answers_201_and_returns_the_note():
    resp = get_test_client().post("/api/planner/notes", json={"title": "Hello", "body": "world"})
    assert resp.status_code == 201
    note = _data(resp)["note"]
    assert note["title"] == "Hello"
    assert note["body"] == "world"
    assert note["status"] == "open"
    assert note["id"]


def test_note_update_patches_and_404s():
    client = get_test_client()
    note = _data(client.post("/api/planner/notes", json={"title": "T"}))["note"]
    updated = _data(client.put(f"/api/planner/notes/{note['id']}", json={"status": "wip"}))
    assert updated["note"]["status"] == "wip"
    assert updated["note"]["title"] == "T"
    assert client.put("/api/planner/notes/nope", json={"status": "wip"}).status_code == 404


def test_note_delete_bodyless_and_404s():
    client = get_test_client()
    note = _data(client.post("/api/planner/notes", json={"title": "Gone"}))["note"]
    resp = client.delete(f"/api/planner/notes/{note['id']}")  # no body — apiDelete parity
    assert resp.status_code == 200
    assert _data(resp) == {"success": True}
    assert client.delete("/api/planner/notes/nope").status_code == 404


# ── planner.stats.read / planner.tags.read / planner.notes.sync ─────────────


def test_stats_return_the_exact_next_shape_not_store_get_stats():
    _seed_card("A", column="todo", tags=["api"])
    _seed_card("B", column="wip", tags=["api", "web"])
    get_store().create_note(title="N1")
    stats = _data(get_test_client().get("/api/planner/stats"))["stats"]
    # helpers.getStats shape — NOT app_planner.get_stats() keys (total/byPriority)
    assert set(stats) == {"total_cards", "byColumn", "columns", "total_notes"}
    assert stats["total_cards"] == 2
    assert stats["byColumn"] == {"todo": 1, "wip": 1}
    assert stats["columns"] == 4
    assert stats["total_notes"] == 1


def test_tags_are_counted_descending_as_name_count_pairs():
    _seed_card("A", tags=["api", "web"])
    _seed_card("B", tags=["api"])
    tags = _data(get_test_client().get("/api/planner/tags"))["tags"]
    assert tags == [
        {"name": "api", "count": 2},
        {"name": "web", "count": 1},
    ]


def test_sync_stays_a_noop_returning_zeros():
    """Next's sync was deliberately a no-op ({added:0,updated:0,total:0}).

    BuffetEngine fires it during UI flows; making it reconcile would move
    cards under the user. Reconciliation stays the CLI's job (planner sync).
    """
    resp = get_test_client().post("/api/planner/sync")
    assert resp.status_code == 200
    assert _data(resp) == {"added": 0, "updated": 0, "total": 0}


# ── registration + auth ─────────────────────────────────────────────────────


def test_every_descriptor_is_projected_and_served():
    served = {(e["method"], e["path"]) for e in REGISTRY.routes()}
    for name in DESCRIPTORS:
        assert any(name == e.get("name") or name in e.get("operation_id", "") for e in REGISTRY.routes()), name
    for expected in (
        ("GET", "/api/planner/board"),
        ("POST", "/api/planner/board/cards"),
        ("PUT", "/api/planner/board/cards/{id}"),
        ("DELETE", "/api/planner/board/cards/{id}"),
        ("POST", "/api/planner/board/move"),
        ("GET", "/api/planner/notes"),
        ("POST", "/api/planner/notes"),
        ("PUT", "/api/planner/notes/{id}"),
        ("DELETE", "/api/planner/notes/{id}"),
        ("GET", "/api/planner/stats"),
        ("GET", "/api/planner/tags"),
        ("POST", "/api/planner/sync"),
    ):
        assert expected in served, expected


def test_anonymous_access_is_allowed_while_auth_is_disabled(monkeypatch):
    monkeypatch.delenv("SLO_AUTH_REQUIRED", raising=False)
    assert get_test_client().get("/api/planner/board").status_code == 200


def test_anonymous_access_is_rejected_when_auth_is_required(monkeypatch):
    monkeypatch.setenv("SLO_AUTH_REQUIRED", "true")
    assert get_test_client().get("/api/planner/board").status_code == 401
