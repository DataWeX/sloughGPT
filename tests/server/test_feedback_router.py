"""
Tests for the feedback router — feedback CRUD, stats, conversations.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from infrastructure.exception_handlers import register_all_handlers
from apps.api.server.routers.feedback import router


@pytest.fixture
def app():
    _app = FastAPI()
    register_all_handlers(_app)
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

    def test_rejects_neutral_rating(self, mock_get_ctrl, client):
        resp = client.post("/feedback", json={
            "message_id": "msg-1",
            "rating": "neutral",
        })
        assert resp.status_code == 422

    def test_records_feedback_with_session_id(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.record_feedback.return_value = dict(FEEDBACK_RESPONSE)
        mock_get_ctrl.return_value = ctrl
        resp = client.post("/feedback", json={
            "message_id": "msg-1",
            "rating": "thumbs_down",
            "session_id": "sess-1",
        })
        assert resp.status_code == 200
        ctrl.record_feedback.assert_called_once()
        call_kwargs = ctrl.record_feedback.call_args
        assert call_kwargs.kwargs["session_id"] == "sess-1"

    def test_records_feedback_with_content(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.record_feedback.return_value = dict(FEEDBACK_RESPONSE)
        mock_get_ctrl.return_value = ctrl
        resp = client.post("/feedback", json={
            "message_id": "msg-1",
            "rating": "thumbs_up",
            "message_content": "Great response!",
            "user_message": "Tell me about AI",
            "assistant_response": "AI is artificial intelligence.",
        })
        assert resp.status_code == 200


@patch("controllers.feedback.get_feedback_controller")
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

    def test_workflow_with_long_messages(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.record_feedback.return_value = {"feedback_id": "fb-wf-2"}
        mock_get_ctrl.return_value = ctrl
        resp = client.post("/feedback/workflow-record", json={
            "conversation_id": "conv-1",
            "rating": "thumbs_down",
            "assistant_response": "x" * 5000,
            "user_message": "y" * 5000,
        })
        assert resp.status_code == 200
        assert resp.json()["data"]["workflow_active"] is True

    def test_workflow_rejects_thumbs_down_rating(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.record_feedback.return_value = {"feedback_id": "fb-wf-3"}
        mock_get_ctrl.return_value = ctrl
        resp = client.post("/feedback/workflow-record", json={
            "conversation_id": "conv-1",
            "rating": "thumbs_down",
        })
        assert resp.status_code == 200

    def test_workflow_passes_mapped_fields(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.record_feedback.return_value = {"feedback_id": "fb-wf-4"}
        mock_get_ctrl.return_value = ctrl
        resp = client.post("/feedback/workflow-record", json={
            "conversation_id": "conv-9",
            "rating": "thumbs_up",
            "assistant_response": "ai text",
            "user_message": "user text",
        })
        assert resp.status_code == 200
        kwargs = ctrl.record_feedback.call_args.kwargs
        assert kwargs["message_id"] == "conv-9"
        assert kwargs["session_id"] == "conv-9"
        assert kwargs["message_content"] == "ai text"
        assert kwargs["assistant_response"] == "ai text"
        assert kwargs["user_message"] == "user text"

    def test_workflow_feedback_id_empty_fallback(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.record_feedback.return_value = {}
        mock_get_ctrl.return_value = ctrl
        resp = client.post("/feedback/workflow-record", json={
            "conversation_id": "conv-1",
            "rating": "thumbs_up",
        })
        assert resp.json()["data"]["feedback_id"] == ""

    def test_workflow_overlong_assistant_response_422(self, mock_get_ctrl, client):
        resp = client.post("/feedback/workflow-record", json={
            "conversation_id": "conv-1",
            "rating": "thumbs_up",
            "assistant_response": "x" * 10001,
        })
        assert resp.status_code == 422

    def test_workflow_overlong_conversation_id_422(self, mock_get_ctrl, client):
        resp = client.post("/feedback/workflow-record", json={
            "conversation_id": "c" * 257,
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

    def test_returns_empty_stats(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_stats.return_value = {"thumbs_up": 0, "thumbs_down": 0, "total": 0, "up_ratio": 0.0}
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/feedback/stats/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["up_ratio"] == 0.0


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

    def test_list_conversations_custom_limit(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.list_conversations.return_value = [dict(CONV_RESPONSE)] * 3
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/feedback/conversations?limit=10")
        assert resp.status_code == 200
        ctrl.list_conversations.assert_called_once_with(limit=10)

    def test_list_conversations_default_limit(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.list_conversations.return_value = []
        mock_get_ctrl.return_value = ctrl
        client.get("/feedback/conversations")
        ctrl.list_conversations.assert_called_once_with(limit=50)

    def test_list_conversations_zero_limit_rejected(self, mock_get_ctrl, client):
        resp = client.get("/feedback/conversations?limit=0")
        assert resp.status_code == 422

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

    def test_update_conversation_partial_pinned(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        updated = {**CONV_RESPONSE, "pinned": True}
        ctrl.update_conversation.return_value = updated
        mock_get_ctrl.return_value = ctrl
        resp = client.patch("/feedback/conversations/conv-1", json={"pinned": True})
        assert resp.status_code == 200
        assert resp.json()["pinned"] is True

    def test_update_conversation_exclude_unset(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.update_conversation.return_value = dict(CONV_RESPONSE)
        mock_get_ctrl.return_value = ctrl
        client.patch("/feedback/conversations/conv-1", json={"name": "Renamed"})
        args = ctrl.update_conversation.call_args.args
        assert args[0] == "conv-1"
        assert args[1] == {"name": "Renamed"}

    def test_delete_conversation_calls_controller(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        mock_get_ctrl.return_value = ctrl
        resp = client.delete("/feedback/conversations/conv-7")
        assert resp.status_code == 200
        ctrl.delete_conversation.assert_called_once_with("conv-7")
        assert resp.json()["id"] == "conv-7"

    def test_delete_conversation(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        mock_get_ctrl.return_value = ctrl
        resp = client.delete("/feedback/conversations/conv-1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_create_conversation_with_session_id(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        conv = {**CONV_RESPONSE, "session_id": "sess-42"}
        ctrl.create_conversation.return_value = conv
        mock_get_ctrl.return_value = ctrl
        resp = client.post("/feedback/conversations", json={
            "name": "My Chat",
            "session_id": "sess-42",
        })
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "sess-42"


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

    def test_get_feedback_different_message(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        fb = {**FEEDBACK_RESPONSE, "message_id": "msg-99", "rating": "thumbs_down"}
        ctrl.get_feedback.return_value = fb
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/feedback/msg-99")
        assert resp.status_code == 200
        assert resp.json()["message_id"] == "msg-99"
        assert resp.json()["rating"] == "thumbs_down"


@patch("apps.api.server.routers.feedback.get_feedback_controller")
class TestFeedbackMethodRestrictions:
    """Wrong-method rejection across feedback routes."""

    def test_put_root_405(self, mock_get_ctrl, client):
        assert client.put("/feedback", json={}).status_code == 405

    def test_post_stats_405(self, mock_get_ctrl, client):
        assert client.post("/feedback/stats/summary").status_code == 405

    def test_put_conversation_list_405(self, mock_get_ctrl, client):
        assert client.put("/feedback/conversations", json={}).status_code == 405

    def test_post_conversation_detail_405(self, mock_get_ctrl, client):
        assert client.post("/feedback/conversations/conv-1").status_code == 405

    def test_get_message_feedback_wrong_method_405(self, mock_get_ctrl, client):
        assert client.post("/feedback/msg-1", json={}).status_code == 405
