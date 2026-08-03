"""
Tests for the planner GUI server (stdlib HTTP server + embedded SPA).

Spins up a real ``GuiServer`` on an ephemeral port with temp data dirs and
exercises the REST API over HTTP with httpx. Runs against both the ``file``
and ``mogdb`` note backends.
"""

import subprocess
import sys
import threading

import httpx
import pytest

from planner.gui import GuiHandler, GuiServer, GUI_HTML, _bind_server
from planner.core import NoteStore, cli_main
from planner.kanban import KanbanStore

BACKENDS = ["file", "mogdb"]


class _Server:
    def __init__(self, notes_dir, board_dir, backend):
        self._server = GuiServer(
            ("127.0.0.1", 0),
            GuiHandler,
            NoteStore(notes_dir=notes_dir, backend=backend),
            KanbanStore(board_dir=board_dir),
        )
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture(params=BACKENDS)
def server(tmp_path, request):
    srv = _Server(tmp_path / "notes", tmp_path / "kanban", request.param).start()
    yield srv
    srv.stop()


def _client():
    return httpx.Client(timeout=5.0)


def test_index_serves_spa(server):
    with _client() as c:
        r = c.get(server.base_url + "/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert '<div class="board" id="board">' in r.text
    assert "Planner" in r.text


def test_card_html_note_chip_has_no_tdz_self_reference():
    """Regression: cardHtml used `const n = ... ${n} ...`, a temporal
    dead zone error that crashed board rendering for any card with notes."""
    assert "${c.notes.length} note" in GUI_HTML
    assert "${n} note${n>1" not in GUI_HTML


def test_create_list_and_get_note(server):
    with _client() as c:
        r = c.post(server.base_url + "/api/notes", json={
            "title": "GUI test note", "tags": ["gui", "test"], "status": "wip",
            "body": "Some body text",
        })
        assert r.status_code == 200
        created = r.json()["note"]
        assert created["title"] == "GUI test note"
        assert created["tags"] == ["gui", "test"]
        assert created["status"] == "wip"
        assert created["body"] == "Some body text"

        lst = c.get(server.base_url + "/api/notes").json()["notes"]
        assert any(n["id"] == created["id"] for n in lst)

        got = c.get(server.base_url + "/api/notes/" + created["id"])
        assert got.status_code == 200
        assert got.json()["note"]["title"] == "GUI test note"


def test_create_note_requires_title(server):
    with _client() as c:
        r = c.post(server.base_url + "/api/notes", json={"title": "   "})
    assert r.status_code == 400


def test_create_note_rejects_bad_status(server):
    with _client() as c:
        r = c.post(server.base_url + "/api/notes", json={
            "title": "Bad status", "status": "nonsense",
        })
    assert r.status_code == 400


def test_update_note(server):
    with _client() as c:
        created = c.post(server.base_url + "/api/notes", json={
            "title": "Before", "tags": ["a"], "status": "open",
        }).json()["note"]
        r = c.put(server.base_url + "/api/notes/" + created["id"], json={
            "title": "After", "status": "done", "tags": ["a", "b"], "body": "updated",
        })
        assert r.status_code == 200
        note = r.json()["note"]
        assert note["title"] == "After"
        assert note["status"] == "done"
        assert note["tags"] == ["a", "b"]
        assert note["body"] == "updated"


def test_update_unknown_note_404(server):
    with _client() as c:
        r = c.put(server.base_url + "/api/notes/does_not_exist", json={"title": "x"})
    assert r.status_code == 404


def test_delete_note(server):
    with _client() as c:
        created = c.post(server.base_url + "/api/notes", json={"title": "To delete"}).json()["note"]
        r = c.delete(server.base_url + "/api/notes/" + created["id"])
        assert r.status_code == 200
        assert r.json()["ok"] is True
        got = c.get(server.base_url + "/api/notes/" + created["id"])
        assert got.status_code == 404
        r2 = c.delete(server.base_url + "/api/notes/" + created["id"])
        assert r2.status_code == 404


def test_list_filters(server):
    with _client() as c:
        c.post(server.base_url + "/api/notes", json={
            "title": "alpha", "tags": ["core"], "status": "wip", "body": "zebra stripes",
        })
        c.post(server.base_url + "/api/notes", json={
            "title": "beta", "tags": ["ui"], "status": "done", "body": "buttons",
        })
        by_tag = c.get(server.base_url + "/api/notes", params={"tag": "core"}).json()["notes"]
        assert [n["title"] for n in by_tag] == ["alpha"]
        by_status = c.get(server.base_url + "/api/notes", params={"status": "done"}).json()["notes"]
        assert [n["title"] for n in by_status] == ["beta"]
        by_query = c.get(server.base_url + "/api/notes", params={"query": "buttons"}).json()["notes"]
        assert [n["title"] for n in by_query] == ["beta"]
        all_notes = c.get(server.base_url + "/api/notes").json()["notes"]
        assert len(all_notes) == 2


def test_board_returns_default_columns(server):
    with _client() as c:
        data = c.get(server.base_url + "/api/board").json()["board"]
    names = [col["name"] for col in data["columns"]]
    assert names == ["todo", "in_progress", "review", "done"]
    assert data["cards"] == []


def test_sync_creates_cards_for_notes(server):
    with _client() as c:
        c.post(server.base_url + "/api/notes", json={
            "title": "Cardable note", "tags": ["x"], "status": "open",
        })
        r = c.post(server.base_url + "/api/sync")
        assert r.status_code == 200
        body = r.json()
        assert body["added"] == 1
        assert body["total"] == 1
        board = c.get(server.base_url + "/api/board").json()["board"]
        assert board["cards"][0]["title"] == "Cardable note"
        assert board["cards"][0]["column"] == "todo"


def test_sync_is_idempotent(server):
    with _client() as c:
        c.post(server.base_url + "/api/notes", json={"title": "One card"})
        first = c.post(server.base_url + "/api/sync").json()
        second = c.post(server.base_url + "/api/sync").json()
    assert first["added"] == 1
    assert second["added"] == 0
    assert first["total"] == second["total"] == 1


def test_move_card_syncs_note_status(server):
    with _client() as c:
        created = c.post(server.base_url + "/api/notes", json={
            "title": "Drag me to done", "status": "wip",
        }).json()["note"]
        c.post(server.base_url + "/api/sync")
        card = c.get(server.base_url + "/api/board").json()["board"]["cards"][0]
        assert card["title"] == created["title"]

        r = c.post(server.base_url + "/api/board/move", json={"id": card["id"], "column": "done"})
        assert r.status_code == 200
        assert r.json()["card"]["column"] == "done"

        note = c.get(server.base_url + "/api/notes").json()["notes"][0]
        assert note["id"] == created["id"]
        assert note["status"] == "done"


def test_move_card_bad_column_404(server):
    with _client() as c:
        c.post(server.base_url + "/api/notes", json={"title": "Bounce"})
        c.post(server.base_url + "/api/sync")
        card = c.get(server.base_url + "/api/board").json()["board"]["cards"][0]
        r = c.post(server.base_url + "/api/board/move", json={"id": card["id"], "column": "nope"})
    assert r.status_code == 404


def test_tags_endpoint(server):
    with _client() as c:
        c.post(server.base_url + "/api/notes", json={"title": "t1", "tags": ["core", "gui"]})
        c.post(server.base_url + "/api/notes", json={"title": "t2", "tags": ["core"]})
        tags = c.get(server.base_url + "/api/tags").json()["tags"]
    counts = {t["name"]: t["count"] for t in tags}
    assert counts == {"core": 2, "gui": 1}


def test_stats_endpoint(server):
    with _client() as c:
        c.post(server.base_url + "/api/notes", json={"title": "s1", "status": "open"})
        c.post(server.base_url + "/api/notes", json={"title": "s2", "status": "done"})
        stats = c.get(server.base_url + "/api/stats").json()
    assert stats["total"] == 2
    assert stats["by_status"]["open"] == 1
    assert stats["by_status"]["done"] == 1
    assert stats["today"] == 2


def test_unknown_route_404(server):
    with _client() as c:
        r = c.get(server.base_url + "/api/nope")
    assert r.status_code == 404


def test_bad_json_body_400(server):
    with _client() as c:
        r = c.post(
            server.base_url + "/api/notes",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
    assert r.status_code == 400


def test_cli_gui_subcommand_dispatches():
    with pytest.raises(SystemExit) as exc:
        cli_main(["gui", "--help"])
    assert exc.value.code == 0


def test_python_m_dispatch():
    proc = subprocess.run(
        [sys.executable, "-m", "planner", "gui", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    assert "Local web interface" in proc.stdout


def test_bind_server_steps_past_occupied_port(tmp_path):
    import socket
    note_store = NoteStore(notes_dir=tmp_path / "notes", backend="file")
    kanban_store = KanbanStore(board_dir=tmp_path / "kanban")
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    taken = blocker.getsockname()[1]
    try:
        srv = _bind_server("127.0.0.1", taken, GuiHandler, note_store, kanban_store)
        try:
            assert srv.server_address[1] == taken + 1
        finally:
            srv.server_close()
    finally:
        blocker.close()


def test_bind_server_ephemeral_port(tmp_path):
    note_store = NoteStore(notes_dir=tmp_path / "notes", backend="file")
    kanban_store = KanbanStore(board_dir=tmp_path / "kanban")
    srv = _bind_server("127.0.0.1", 0, GuiHandler, note_store, kanban_store)
    try:
        assert srv.server_address[1] != 0
    finally:
        srv.server_close()
