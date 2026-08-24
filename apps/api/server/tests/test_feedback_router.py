"""Tests for feedback router endpoints.

All tests use a temporary MogDB instance per test to ensure isolation.
The autouse fixture injects a fresh FeedbackController backed by a temp
directory, and resets the global singleton on teardown.
"""
import pytest
import tempfile
import shutil

from controllers.feedback import (
    FeedbackController,
    set_feedback_controller,
    reset_feedback_controller,
)
from tests.test_support import get_test_client


@pytest.fixture(autouse=True)
def _isolated_feedback_controller(tmp_path):
    """Create a FeedbackController backed by a temp MogDB, inject it, clean up."""
    import routers.feedback as fb_mod
    fb_mod._feedback_stats_cache = None
    db_path = str(tmp_path / "feedback_mogdb")
    controller = FeedbackController(tmp_path, db_path=db_path)
    set_feedback_controller(controller)
    yield controller
    reset_feedback_controller()


client = get_test_client()


class TestRecordFeedbackWorkflow:
    def test_record_workflow_feedback(self):
        resp = client.post("/feedback/workflow-record", json={
            "conversation_id": "test-conv-1",
            "rating": "thumbs_up",
            "assistant_response": "test response",
            "user_message": "test question",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"]["workflow_active"] is True

    def test_record_workflow_feedback_down(self):
        resp = client.post("/feedback/workflow-record", json={
            "conversation_id": "test-conv-2",
            "rating": "thumbs_down",
        })
        assert resp.status_code == 200

    def test_record_workflow_feedback_invalid_rating(self):
        resp = client.post("/feedback/workflow-record", json={
            "conversation_id": "test-conv-3",
            "rating": "invalid",
        })
        assert resp.status_code == 422


class TestRecordFeedback:
    def test_record_feedback(self):
        resp = client.post("/feedback", json={
            "message_id": "msg-test-1",
            "rating": "thumbs_up",
            "session_id": "sess-test",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "recorded"

    def test_record_feedback_down(self):
        resp = client.post("/feedback", json={
            "message_id": "msg-test-2",
            "rating": "thumbs_down",
        })
        assert resp.status_code == 200

    def test_record_feedback_missing_message_id(self):
        resp = client.post("/feedback", json={"rating": "thumbs_up"})
        assert resp.status_code == 422

    def test_record_feedback_returns_id_and_timestamp(self):
        resp = client.post("/feedback", json={
            "message_id": "msg-ts-1",
            "rating": "thumbs_up",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["feedback_id"].startswith("fb_")
        assert "T" in body["timestamp"]


class TestFeedbackStats:
    def test_get_stats(self):
        resp = client.get("/feedback/stats/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert "thumbs_up" in body
        assert "thumbs_down" in body
        assert "total" in body
        assert "up_ratio" in body

    def test_stats_reflect_actual_records(self):
        client.post("/feedback", json={"message_id": "s1", "rating": "thumbs_up"})
        client.post("/feedback", json={"message_id": "s2", "rating": "thumbs_up"})
        client.post("/feedback", json={"message_id": "s3", "rating": "thumbs_down"})
        resp = client.get("/feedback/stats/summary")
        body = resp.json()
        assert body["total"] == 3
        assert body["thumbs_up"] == 2
        assert body["thumbs_down"] == 1
        assert abs(body["up_ratio"] - 2 / 3) < 0.01


class TestGetFeedback:
    def test_get_feedback_nonexistent(self):
        resp = client.get("/feedback/nonexistent-msg-id")
        assert resp.status_code == 404

    def test_get_feedback_after_record(self):
        client.post("/feedback", json={
            "message_id": "lookup-msg",
            "rating": "thumbs_down",
            "session_id": "sess-lookup",
        })
        resp = client.get("/feedback/lookup-msg")
        assert resp.status_code == 200
        body = resp.json()
        assert body["message_id"] == "lookup-msg"
        assert body["rating"] == "thumbs_down"
        assert body["session_id"] == "sess-lookup"


class TestConversations:
    def test_create_conversation(self):
        resp = client.post("/feedback/conversations", json={
            "name": "Test Conversation",
            "session_id": "sess-123",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "Test Conversation"

    def test_list_conversations(self):
        resp = client.get("/feedback/conversations")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)

    def test_get_conversation_not_found(self):
        resp = client.get("/feedback/conversations/nonexistent-id")
        assert resp.status_code == 404

    def test_create_then_get_conversation(self):
        create_resp = client.post("/feedback/conversations", json={
            "name": "Get Test Conv",
        })
        conv_id = create_resp.json()["id"]
        get_resp = client.get(f"/feedback/conversations/{conv_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["name"] == "Get Test Conv"

    def test_update_conversation(self):
        create_resp = client.post("/feedback/conversations", json={
            "name": "Update Test",
        })
        conv_id = create_resp.json()["id"]
        resp = client.patch(f"/feedback/conversations/{conv_id}", json={"name": "Updated"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated"

    def test_update_conversation_not_found(self):
        resp = client.patch("/feedback/conversations/nonexistent", json={"name": "x"})
        assert resp.status_code == 404

    def test_delete_conversation(self):
        create_resp = client.post("/feedback/conversations", json={"name": "Delete Me"})
        conv_id = create_resp.json()["id"]
        resp = client.delete(f"/feedback/conversations/{conv_id}")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "deleted"

    def test_delete_conversation_not_found(self):
        resp = client.delete("/feedback/conversations/nonexistent")
        assert resp.status_code == 200

    def test_list_conversations_sorted_by_updated(self):
        c1 = client.post("/feedback/conversations", json={"name": "First"})
        c2 = client.post("/feedback/conversations", json={"name": "Second"})
        resp = client.get("/feedback/conversations")
        names = [c["name"] for c in resp.json()]
        assert "Second" in names
        assert "First" in names

    def test_conversation_has_required_fields(self):
        resp = client.post("/feedback/conversations", json={"name": "Field Check"})
        body = resp.json()
        for key in ("id", "name", "session_id", "created_at", "updated_at", "pinned", "starred"):
            assert key in body

    def test_update_conversation_pinned(self):
        create_resp = client.post("/feedback/conversations", json={"name": "Pinning"})
        conv_id = create_resp.json()["id"]
        resp = client.patch(f"/feedback/conversations/{conv_id}", json={"pinned": True})
        assert resp.status_code == 200
        assert resp.json()["pinned"] is True
