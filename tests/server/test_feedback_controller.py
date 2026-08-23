"""Tests for FeedbackController."""
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'api', 'server'))

from controllers.feedback import FeedbackController, get_feedback_controller
import controllers.feedback as feedback_module


@pytest.fixture
def ctrl(tmp_path):
    return FeedbackController(tmp_path)


class TestInit:
    def test_creates_dirs(self, tmp_path):
        ctrl = FeedbackController(tmp_path)
        assert ctrl._db is not None

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
        ctrl._feedback.insert_one({"rating": "thumbs_up", "message_id": "m1"})
        ctrl._feedback.insert_one({"rating": "thumbs_down", "message_id": "m2"})
        ctrl._feedback.insert_one({"rating": "thumbs_up", "message_id": "m3"})
        stats = ctrl.get_stats()
        assert stats["thumbs_up"] == 2
        assert stats["thumbs_down"] == 1
        assert stats["total"] == 3
        assert stats["up_ratio"] == pytest.approx(2 / 3)

    def test_get_feedback_not_found(self, ctrl):
        result = ctrl.get_feedback("nonexistent")
        assert result is None

    def test_get_feedback_found(self, ctrl):
        ctrl._feedback.insert_one({"message_id": "m1", "rating": "thumbs_up"})
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


class TestRecordFeedbackPipeline:
    @patch("domains.training.executor.get_training_executor")
    def test_workflow_and_lora_called(self, mock_executor, ctrl):
        mock_exec = MagicMock()
        mock_executor.return_value = mock_exec
        workflow = MagicMock()
        lora = MagicMock()
        with patch.object(ctrl, "_get_workflow", return_value=workflow), \
             patch.object(ctrl, "_get_lora_updater", return_value=lora):
            ctrl.record_feedback(
                message_id="m3", rating="thumbs_up", session_id="s1",
                user_message="hello", assistant_response="hi",
            )
        workflow.record_feedback.assert_called_once()
        kwargs = workflow.record_feedback.call_args.kwargs
        assert kwargs["user_message"] == "hello"
        assert kwargs["assistant_response"] == "hi"
        assert kwargs["rating"] == "thumbs_up"
        assert kwargs["conversation_id"] == "s1"
        lora.add_feedback.assert_called_once()

    @patch("domains.training.executor.get_training_executor")
    def test_no_pipeline_components_no_crash(self, mock_executor, ctrl):
        mock_exec = MagicMock()
        mock_executor.return_value = mock_exec
        with patch.object(ctrl, "_get_workflow", return_value=None), \
             patch.object(ctrl, "_get_lora_updater", return_value=None):
            result = ctrl.record_feedback(
                message_id="m4", rating="thumbs_up", user_message="a", assistant_response="b",
            )
        assert result["status"] == "recorded"

    @patch("domains.training.executor.get_training_executor")
    def test_no_conversation_text_skips_pipeline(self, mock_executor, ctrl):
        mock_exec = MagicMock()
        mock_executor.return_value = mock_exec
        with patch.object(ctrl, "_get_workflow") as mock_wf, \
             patch.object(ctrl, "_get_lora_updater") as mock_lora:
            ctrl.record_feedback(message_id="m5", rating="thumbs_up")
        mock_wf.assert_not_called()
        mock_lora.assert_not_called()

    @patch("domains.training.executor.get_training_executor")
    def test_thumbs_down_submits_dpo(self, mock_executor, ctrl):
        mock_exec = MagicMock()
        mock_executor.return_value = mock_exec
        ctrl.record_feedback(message_id="m6", rating="thumbs_down")
        assert mock_exec.submit.call_count == 1
        fn, job_id = mock_exec.submit.call_args.args
        assert fn.__name__ == "_trigger_hf_dpo"
        assert job_id.startswith("dpo_")

    @patch("domains.training.executor.get_training_executor")
    def test_thumbs_up_no_dpo(self, mock_executor, ctrl):
        mock_exec = MagicMock()
        mock_executor.return_value = mock_exec
        ctrl.record_feedback(message_id="m7", rating="thumbs_up")
        mock_exec.submit.assert_not_called()

    def test_get_workflow_failure_returns_none(self, ctrl):
        with patch("domains.feedback.workflow.get_feedback_workflow", side_effect=RuntimeError("down")):
            assert ctrl._get_workflow() is None

    def test_get_lora_failure_returns_none(self, ctrl):
        with patch("domains.feedback.online_train.get_online_lora_updater", side_effect=RuntimeError("down")):
            assert ctrl._get_lora_updater() is None

    @patch("domains.feedback.workflow.get_feedback_workflow")
    def test_workflow_wired_with_model(self, mock_get_wf, ctrl):
        workflow = MagicMock()
        mock_get_wf.return_value = workflow
        student = object()
        tokenizer = object()
        with patch("routers.auto_train.state.student_net", student, create=True), \
             patch("routers.auto_train.state.student_tokenizer", tokenizer, create=True):
            wf = ctrl._get_workflow()
        assert wf is workflow
        workflow.set_model.assert_called_once_with(student, tokenizer)


class TestTriggerHFDpo:
    @patch("domains.feedback.hf_dpo.HFDPOTrainer")
    def test_no_model_returns(self, mock_trainer, ctrl):
        with patch("state.model", None), patch("state.tokenizer", None):
            feedback_module._trigger_hf_dpo()  # should not raise

    @patch("domains.feedback.hf_dpo.HFDPOTrainer")
    def test_fewer_than_two_pairs_skips_train(self, mock_trainer, ctrl):
        trainer = MagicMock()
        trainer.prepare_dpo_pairs.return_value = [{"a": 1}]
        mock_trainer.return_value = trainer
        with patch("state.model", object()), patch("state.tokenizer", object()):
            feedback_module._trigger_hf_dpo()
        trainer.train.assert_not_called()

    @patch("domains.feedback.hf_dpo.HFDPOTrainer")
    def test_full_dpo_run(self, mock_trainer, ctrl):
        trainer = MagicMock()
        trainer.prepare_dpo_pairs.return_value = [{"a": 1}, {"b": 2}]
        trainer.train.return_value = {"status": "ok"}
        mock_trainer.return_value = trainer
        with patch("state.model", object()), patch("state.tokenizer", object()):
            feedback_module._trigger_hf_dpo()
        trainer.train.assert_called_once()

    @patch("domains.feedback.hf_dpo.HFDPOTrainer")
    def test_exception_suppressed(self, mock_trainer, ctrl):
        trainer = MagicMock()
        trainer.prepare_dpo_pairs.side_effect = RuntimeError("boom")
        mock_trainer.return_value = trainer
        with patch("state.model", object()), patch("state.tokenizer", object()):
            feedback_module._trigger_hf_dpo()  # should not raise


class TestFeedbackEdgeCases:
    def test_get_feedback_returns_first_match(self, ctrl):
        ctrl._feedback.insert_one({"message_id": "m1", "rating": "thumbs_up"})
        ctrl._feedback.insert_one({"message_id": "m1", "rating": "thumbs_down"})
        result = ctrl.get_feedback("m1")
        assert result["rating"] == "thumbs_up"

    def test_stats_unknown_rating_counts_total(self, ctrl):
        ctrl._feedback.insert_one({"message_id": "m1", "rating": "neutral"})
        stats = ctrl.get_stats()
        assert stats["total"] == 1
        assert stats["thumbs_up"] == 0
        assert stats["thumbs_down"] == 0

    def test_record_feedback_has_iso_timestamp(self, ctrl):
        from datetime import datetime
        result = ctrl.record_feedback(message_id="m8", rating="thumbs_up")
        datetime.fromisoformat(result["timestamp"])  # raises if not parseable

    def test_record_feedback_has_uuid_suffix(self, ctrl):
        result = ctrl.record_feedback(message_id="m9", rating="thumbs_down")
        assert len(result["feedback_id"]) > 10
