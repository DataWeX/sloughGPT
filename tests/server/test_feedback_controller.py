"""Tests for FeedbackController."""
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'api', 'server'))

from controllers.feedback import FeedbackController, get_feedback_controller


@pytest.fixture
def ctrl(tmp_path):
    return FeedbackController(tmp_path)


class TestInit:
    def test_creates_dirs(self, tmp_path):
        ctrl = FeedbackController(tmp_path)
        assert ctrl.feedback_dir.exists()
        assert ctrl.conversations_dir.exists()

    def test_workflow_initially_none(self, ctrl):
        assert ctrl._workflow is None

    def test_lora_updater_initially_none(self, ctrl):
        assert ctrl._lora_updater is None


class TestConversations:
    def test_list_conversations_empty(self, ctrl):
        result = ctrl.list_conversations()
        assert result == []

    def test_create_conversation(self, ctrl):
        result = ctrl.create_conversation("Test Chat")
        assert result["name"] == "Test Chat"
        assert "id" in result

    def test_get_conversation(self, ctrl):
        conv = ctrl.create_conversation("Chat")
        result = ctrl.get_conversation(conv["id"])
        assert result is not None
        assert result["name"] == "Chat"

    def test_get_nonexistent(self, ctrl):
        assert ctrl.get_conversation("no-such-id") is None

    def test_delete_conversation(self, ctrl):
        conv = ctrl.create_conversation("Delete Me")
        ctrl.delete_conversation(conv["id"])
        assert ctrl.get_conversation(conv["id"]) is None

    def test_update_conversation(self, ctrl):
        conv = ctrl.create_conversation("Update Me")
        result = ctrl.update_conversation(conv["id"], {"pinned": True})
        assert result["pinned"] is True

    def test_update_star(self, ctrl):
        conv = ctrl.create_conversation("Star Me")
        result = ctrl.update_conversation(conv["id"], {"starred": True})
        assert result["starred"] is True

    def test_list_after_create(self, ctrl):
        ctrl.create_conversation("A")
        ctrl.create_conversation("B")
        result = ctrl.list_conversations()
        assert len(result) == 2

    def test_update_nonexistent_returns_none(self, ctrl):
        result = ctrl.update_conversation("nonexistent", {"name": "x"})
        assert result is None

    def test_list_sorted_by_updated_at(self, ctrl):
        c1 = ctrl.create_conversation("First")
        c2 = ctrl.create_conversation("Second")
        # Update c1 to make it newer
        ctrl.update_conversation(c1["id"], {"name": "Updated"})
        result = ctrl.list_conversations()
        assert result[0]["id"] == c1["id"]

    def test_list_with_limit(self, ctrl):
        for i in range(5):
            ctrl.create_conversation(f"Chat {i}")
        result = ctrl.list_conversations(limit=3)
        assert len(result) == 3

    def test_create_conversation_with_session_id(self, ctrl):
        result = ctrl.create_conversation("Chat", session_id="sess-123")
        assert result["session_id"] == "sess-123"

    def test_delete_nonexistent_is_noop(self, ctrl):
        ctrl.delete_conversation("nonexistent")

    def test_conversation_has_timestamps(self, ctrl):
        result = ctrl.create_conversation("Timestamped")
        assert "created_at" in result
        assert "updated_at" in result


class TestFeedbackStats:
    def test_empty_stats(self, ctrl):
        stats = ctrl.get_stats()
        assert stats["thumbs_up"] == 0
        assert stats["thumbs_down"] == 0
        assert stats["total"] == 0

    def test_stats_after_record(self, ctrl):
        # Write feedback directly to test stats
        fb_file = ctrl.feedback_dir / "feedback.jsonl"
        with open(fb_file, "w") as f:
            f.write(json.dumps({"rating": "thumbs_up", "message_id": "m1"}) + "\n")
            f.write(json.dumps({"rating": "thumbs_down", "message_id": "m2"}) + "\n")
            f.write(json.dumps({"rating": "thumbs_up", "message_id": "m3"}) + "\n")
        stats = ctrl.get_stats()
        assert stats["thumbs_up"] == 2
        assert stats["thumbs_down"] == 1
        assert stats["total"] == 3
        assert stats["up_ratio"] == pytest.approx(2 / 3)

    def test_get_feedback_not_found(self, ctrl):
        result = ctrl.get_feedback("nonexistent")
        assert result is None

    def test_get_feedback_found(self, ctrl):
        fb_file = ctrl.feedback_dir / "feedback.jsonl"
        with open(fb_file, "w") as f:
            f.write(json.dumps({"message_id": "m1", "rating": "thumbs_up"}) + "\n")
        result = ctrl.get_feedback("m1")
        assert result is not None
        assert result["rating"] == "thumbs_up"


class TestSingleton:
    def test_returns_same_instance(self):
        a = get_feedback_controller()
        b = get_feedback_controller()
        assert a is b


class TestRecordFeedback:
    @patch("domains.training.executor.get_training_executor")
    def test_record_feedback_basic(self, mock_executor, ctrl):
        mock_exec = MagicMock()
        mock_executor.return_value = mock_exec
        result = ctrl.record_feedback(
            message_id="m1",
            rating="thumbs_up",
            user_message="hello",
            assistant_response="hi",
        )
        assert result["status"] == "recorded"
        assert result["message_id"] == "m1"
        assert result["rating"] == "thumbs_up"

    @patch("domains.training.executor.get_training_executor")
    def test_record_feedback_writes_to_file(self, mock_executor, ctrl):
        mock_exec = MagicMock()
        mock_executor.return_value = mock_exec
        ctrl.record_feedback(message_id="m2", rating="thumbs_down")
        fb = ctrl.get_feedback("m2")
        assert fb is not None
        assert fb["rating"] == "thumbs_down"
