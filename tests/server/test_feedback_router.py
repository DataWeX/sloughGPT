"""
Tests for the feedback router — feedback CRUD, stats, conversations.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.feedback import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


FEEDBACK_RESPONSE = {
    "status": "ok",
    "feedback_id": "fb-1",
    "message_id": "msg-1",
    "rating": "thumbs_up",
    "timestamp": "2026-07-29T12:00:00",
}

STATS_RESPONSE = {
    "thumbs_up": 10,
    "thumbs_down": 2,
    "total": 12,
    "up_ratio": 0.833,
}

CONV_RESPONSE = {
    "id": "conv-1",
    "name": "Test",
    "session_id": "sess-1",
    "created_at": "2026-07-29T12:00:00",
    "updated_at": "2026-07-29T12:00:00",
    "pinned": False,
    "starred": False,
    "message_count": 0,
}


@patch("apps.api.server.routers.feedback.get_feedback_controller")
class TestRecordFeedback:
    """POST /feedback"""

    def test_records_feedback(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.record_feedback.return_value = dict(FEEDBACK_RESPONSE)
        mock_get_ctrl.return_value = ctrl
        resp = client.post("/feedback", json={
            "message_id": "msg-1",
            "rating": "thumbs_up",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["feedback_id"] == "fb-1"

    def test_rejects_invalid_rating(self, mock_get_ctrl, client):
        resp = client.post("/feedback", json={
            "message_id": "msg-1",
            "rating": "invalid",
        })
        assert resp.status_code == 422

    def test_rejects_missing_message_id(self, mock_get_ctrl, client):
        resp = client.post("/feedback", json={"rating": "thumbs_up"})
        assert resp.status_code == 422


@patch("apps.api.server.routers.feedback.get_feedback_controller")
class TestRecordFeedbackWorkflow:
    """POST /feedback/workflow-record"""

    def test_records_workflow_feedback(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.record_feedback.return_value = {"feedback_id": "fb-wf-1"}
        mock_get_ctrl.return_value = ctrl
        resp = client.post("/feedback/workflow-record", json={
            "conversation_id": "conv-1",
            "rating": "thumbs_up",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"]["workflow_active"] is True

    def test_rejects_invalid_workflow_rating(self, mock_get_ctrl, client):
        resp = client.post("/feedback/workflow-record", json={
            "conversation_id": "conv-1",
            "rating": "bad",
        })
        assert resp.status_code == 422

    def test_rejects_missing_conversation_id(self, mock_get_ctrl, client):
        resp = client.post("/feedback/workflow-record", json={
            "rating": "thumbs_up",
        })
        assert resp.status_code == 422


@patch("apps.api.server.routers.feedback.get_feedback_controller")
class TestFeedbackStats:
    """GET /feedback/stats/summary"""

    def test_returns_stats(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_stats.return_value = dict(STATS_RESPONSE)
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/feedback/stats/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["thumbs_up"] == 10
        assert body["total"] == 12


@patch("apps.api.server.routers.feedback.get_feedback_controller")
class TestConversations:
    """CRUD /feedback/conversations"""

    def test_create_conversation(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.create_conversation.return_value = dict(CONV_RESPONSE)
        mock_get_ctrl.return_value = ctrl
        resp = client.post("/feedback/conversations", json={"name": "Test"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Test"

    def test_list_conversations(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.list_conversations.return_value = [dict(CONV_RESPONSE)]
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/feedback/conversations")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_get_conversation_found(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_conversation.return_value = dict(CONV_RESPONSE)
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/feedback/conversations/conv-1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "conv-1"

    def test_get_conversation_not_found(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_conversation.return_value = None
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/feedback/conversations/nonexistent")
        assert resp.status_code == 404

    def test_update_conversation(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        updated = {**CONV_RESPONSE, "name": "Updated"}
        ctrl.update_conversation.return_value = updated
        mock_get_ctrl.return_value = ctrl
        resp = client.patch("/feedback/conversations/conv-1", json={"name": "Updated"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated"

    def test_update_conversation_not_found(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.update_conversation.return_value = None
        mock_get_ctrl.return_value = ctrl
        resp = client.patch("/feedback/conversations/nonexistent", json={"name": "Nope"})
        assert resp.status_code == 404

    def test_delete_conversation(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        mock_get_ctrl.return_value = ctrl
        resp = client.delete("/feedback/conversations/conv-1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"


@patch("apps.api.server.routers.feedback.get_feedback_controller")
class TestGetFeedback:
    """GET /feedback/{message_id}"""

    def test_get_feedback_found(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_feedback.return_value = dict(FEEDBACK_RESPONSE)
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/feedback/msg-1")
        assert resp.status_code == 200
        assert resp.json()["feedback_id"] == "fb-1"

    def test_get_feedback_not_found(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_feedback.return_value = None
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/feedback/nonexistent")
        assert resp.status_code == 404
