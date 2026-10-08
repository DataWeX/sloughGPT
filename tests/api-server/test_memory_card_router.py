"""Router tests for the loadable memory-card endpoints (card f74b9e75).

- GET    /memory/cards            list saved cards (validity flags)
- POST   /memory/cards/save       snapshot current memory to a card
- POST   /memory/cards/load       restore a card (replace|merge)
- DELETE /memory/cards/{name}     remove a card

Plus the model-load wiring: POST /models/load must trigger
``ensure_active_card()`` so the model wakes up remembering.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_support import get_test_client


def _d(resp):
    j = resp.json()
    return j.get("data", j)


def _cleanup_store():
    from domain.learner._internal.knowledge import get_knowledge_memory

    get_knowledge_memory().clear_all()


def test_cards_crud_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("SLO_MEMORY_CARDS_DIR", str(tmp_path / "cards"))
    monkeypatch.delenv("SLO_MEMORY_ACTIVE_CARD", raising=False)
    _cleanup_store()
    client = get_test_client()

    # seed one fact
    assert (
        client.post(
            "/memory/store", json={"content": "router card fact", "topic": "t", "source": "api"}
        ).status_code
        == 200
    )

    # save
    save = client.post("/memory/cards/save", json={"name": "router-card"})
    assert save.status_code == 200, save.text
    body = _d(save)
    assert body["ok"] is True
    assert body["name"] == "router-card"
    assert body["facts_count"] >= 1

    # duplicate name -> 409 without overwrite, 200 with overwrite
    conflict = client.post("/memory/cards/save", json={"name": "router-card"})
    assert conflict.status_code == 409
    overwritten = client.post("/memory/cards/save", json={"name": "router-card", "overwrite": True})
    assert overwritten.status_code == 200

    # list
    listed = client.get("/memory/cards")
    assert listed.status_code == 200
    cards = {c["name"]: c for c in _d(listed)["cards"]}
    assert "router-card" in cards
    assert cards["router-card"]["valid"] is True

    # load merge
    load = client.post("/memory/cards/load", json={"name": "router-card", "mode": "merge"})
    assert load.status_code == 200, load.text
    assert _d(load)["ok"] is True

    # missing card -> 404
    missing = client.post("/memory/cards/load", json={"name": "never-existed"})
    assert missing.status_code == 404

    # invalid name -> 400
    bad_name = client.post("/memory/cards/save", json={"name": "../escape"})
    assert bad_name.status_code == 400

    # delete: 200 then 404
    deleted = client.delete("/memory/cards/router-card")
    assert deleted.status_code == 200
    assert _d(deleted)["deleted"] is True
    gone = client.delete("/memory/cards/router-card")
    assert gone.status_code == 404

    _cleanup_store()


def test_cards_list_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SLO_MEMORY_CARDS_DIR", str(tmp_path / "no-cards-yet"))
    client = get_test_client()
    resp = client.get("/memory/cards")
    assert resp.status_code == 200
    assert _d(resp)["cards"] == []


def _models_client():
    from apps.api.server.infrastructure.exception_handlers import register_all_handlers
    from apps.api.server.routers.models import ModelsRouter

    app = FastAPI()
    register_all_handlers(app)
    app.include_router(ModelsRouter().router)
    return TestClient(app, raise_server_exceptions=False)


@patch("apps.api.server.routers.models.get_models_controller")
def test_model_load_triggers_active_card(mock_ctrl, monkeypatch):
    mock_ctrl.return_value.load_model.return_value = {"status": "loaded", "device": "cpu"}
    monkeypatch.setenv("SLO_MEMORY_ACTIVE_CARD", "my-card")
    client = _models_client()

    with patch("domain.memory.get_memory_service") as get_svc:
        svc = MagicMock()
        get_svc.return_value = svc
        resp = client.post("/models/load", json={"model_id": "gpt2", "device": "cpu"})

    assert resp.status_code == 200, resp.text
    svc.ensure_active_card.assert_called_once()


@patch("apps.api.server.routers.models.get_models_controller")
def test_model_load_error_does_not_touch_cards(mock_ctrl, monkeypatch):
    mock_ctrl.return_value.load_model.return_value = {"status": "error", "error": "boom"}
    monkeypatch.setenv("SLO_MEMORY_ACTIVE_CARD", "my-card")
    client = _models_client()

    with patch("domain.memory.get_memory_service") as get_svc:
        svc = MagicMock()
        get_svc.return_value = svc
        resp = client.post("/models/load", json={"model_id": "gpt2", "device": "cpu"})

    assert resp.status_code == 200, resp.text  # error status still 200, no card hook
    svc.ensure_active_card.assert_not_called()
