"""Tests for the feedback API router (routers/feedback.py).

Covers: record_feedback_workflow, get_feedback_stats, conversations CRUD, get_feedback.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.feedback import FeedbackRouter  # noqa: E402


def _mock_ctrl(**overrides) -> MagicMock:
    ctrl = MagicMock()
    ctrl.record_feedback.return_value = {
        "feedback_id": "fb-1",
        "message_id": "msg-1",
        "rating": "thumbs_up",
        **overrides,
    }
    ctrl.get_stats.return_value = {
        "total": 10,
        "thumbs_up": 7,
        "thumbs_down": 3,
    }
    ctrl.create_conversation.return_value = {"id": "conv-1", "name": "Test", "session_id": "s1", "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00"}
    ctrl.list_conversations.return_value = [{"id": "conv-1", "name": "Test", "session_id": "s1", "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00"}]
    ctrl.get_conversation.return_value = {"id": "conv-1", "name": "Test", "session_id": "s1", "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00"}
    ctrl.update_conversation.return_value = {"id": "conv-1", "name": "Updated", "session_id": "s1", "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00"}
    ctrl.delete_conversation.return_value = True
    ctrl.get_feedback.return_value = {"message_id": "msg-1", "rating": "thumbs_up"}
    return ctrl


def _app(fr: FeedbackRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(fr.router)
    return app


class TestWorkflowFeedback:
    @patch("routers.feedback.get_feedback_controller")
    def test_record_workflow_feedback(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        fr = FeedbackRouter()
        client = TestClient(_app(fr))
        resp = client.post("/feedback/workflow-record", json={
            "conversation_id": "conv-1",
            "rating": "thumbs_up",
            "assistant_response": "Hello!",
            "user_message": "Hi",
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["workflow_active"] is True


class TestFeedbackStats:
    @patch("routers.feedback.get_feedback_controller")
    def test_get_stats(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        fr = FeedbackRouter()
        client = TestClient(_app(fr))
        resp = client.get("/feedback/stats/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 10


class TestConversations:
    @patch("routers.feedback.get_feedback_controller")
    def test_create_conversation(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        fr = FeedbackRouter()
        client = TestClient(_app(fr))
        resp = client.post("/feedback/conversations", json={"name": "Test", "session_id": "s1"})
        assert resp.status_code == 200
        assert resp.json()["id"] == "conv-1"

    @patch("routers.feedback.get_feedback_controller")
    def test_list_conversations(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        fr = FeedbackRouter()
        client = TestClient(_app(fr))
        resp = client.get("/feedback/conversations")
        assert resp.status_code == 200

    @patch("routers.feedback.get_feedback_controller")
    def test_get_conversation_found(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        fr = FeedbackRouter()
        client = TestClient(_app(fr))
        resp = client.get("/feedback/conversations/conv-1")
        assert resp.status_code == 200

    @patch("routers.feedback.get_feedback_controller")
    def test_get_conversation_not_found(self, mock_get):
        ctrl = _mock_ctrl()
        ctrl.get_conversation.return_value = None
        mock_get.return_value = ctrl
        fr = FeedbackRouter()
        client = TestClient(_app(fr))
        resp = client.get("/feedback/conversations/nonexistent")
        assert resp.status_code == 404

    @patch("routers.feedback.get_feedback_controller")
    def test_update_conversation(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        fr = FeedbackRouter()
        client = TestClient(_app(fr))
        resp = client.patch("/feedback/conversations/conv-1", json={"name": "Updated"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated"

    @patch("routers.feedback.get_feedback_controller")
    def test_update_conversation_not_found(self, mock_get):
        ctrl = _mock_ctrl()
        ctrl.update_conversation.return_value = None
        mock_get.return_value = ctrl
        fr = FeedbackRouter()
        client = TestClient(_app(fr))
        resp = client.patch("/feedback/conversations/nonexistent", json={"name": "X"})
        assert resp.status_code == 404

    @patch("routers.feedback.get_feedback_controller")
    def test_delete_conversation(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        fr = FeedbackRouter()
        client = TestClient(_app(fr))
        resp = client.delete("/feedback/conversations/conv-1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"


class TestGetFeedback:
    @patch("routers.feedback.get_feedback_controller")
    def test_get_feedback_found(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        fr = FeedbackRouter()
        client = TestClient(_app(fr))
        resp = client.get("/feedback/msg-1")
        assert resp.status_code == 200

    @patch("routers.feedback.get_feedback_controller")
    def test_get_feedback_not_found(self, mock_get):
        ctrl = _mock_ctrl()
        ctrl.get_feedback.return_value = None
        mock_get.return_value = ctrl
        fr = FeedbackRouter()
        client = TestClient(_app(fr))
        resp = client.get("/feedback/nonexistent")
        assert resp.status_code == 404
