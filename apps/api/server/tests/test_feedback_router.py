"""Tests for feedback router endpoints."""
import pytest

from tests.test_support import get_test_client

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


class TestFeedbackStats:
    def test_get_stats(self):
        resp = client.get("/feedback/stats/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert "thumbs_up" in body
        assert "thumbs_down" in body
        assert "total" in body
        assert "up_ratio" in body


class TestGetFeedback:
    def test_get_feedback_nonexistent(self):
        resp = client.get("/feedback/nonexistent-msg-id")
        assert resp.status_code == 404


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
        assert resp.json()["status"] == "deleted"

    def test_delete_conversation_not_found(self):
        resp = client.delete("/feedback/conversations/nonexistent")
        assert resp.status_code == 200
