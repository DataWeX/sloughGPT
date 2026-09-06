"""Tests for feedback.training — TrainingExample, DPOPair, FeedbackTrainer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from domains.feedback.training import (
    TrainingExample, DPOPair, FeedbackTrainer, create_training_pipeline,
)


def _make_message(msg_id, role, content, conv_id="conv1", created_at="2024-01-01T00:00:00"):
    return {"_id": msg_id, "id": msg_id, "role": role, "content": content, "conversation_id": conv_id, "created_at": created_at}


def _make_feedback(msg_id, rating, quality_score=None, created_at="2024-01-01T00:00:01"):
    return {"message_id": msg_id, "rating": rating, "quality_score": quality_score, "created_at": created_at}


# ── TrainingExample / DPOPair ──────────────────────────────────────────────


class TestTrainingExample:

    def test_defaults(self):
        ex = TrainingExample(prompt="p", response="r", rating="thumbs_up")
        assert ex.quality_score is None

    def test_custom_quality(self):
        ex = TrainingExample(prompt="p", response="r", rating="thumbs_up", quality_score=0.8)
        assert ex.quality_score == 0.8


class TestDPOPair:

    def test_init(self):
        pair = DPOPair(chosen="a", rejected="b", prompt="p")
        assert pair.chosen == "a"
        assert pair.rejected == "b"
        assert pair.prompt == "p"


# ── FeedbackTrainer ────────────────────────────────────────────────────────


class TestFeedbackTrainer:

    def setup_method(self):
        self.mock_mog = MagicMock()
        self.mock_msgs = MagicMock()
        self.mock_fb = MagicMock()
        self.mock_mog.collection.side_effect = lambda name: {
            "messages": self.mock_msgs,
            "feedback": self.mock_fb,
        }[name]

    @patch("domains.feedback.training.MogDB")
    def test_init(self, MockMogDB):
        MockMogDB.return_value = self.mock_mog
        trainer = FeedbackTrainer(db_path="/tmp/test.db")
        assert trainer._messages is self.mock_msgs
        assert trainer._feedback is self.mock_fb

    @patch("domains.feedback.training.MogDB")
    def test_message_by_id(self, MockMogDB):
        MockMogDB.return_value = self.mock_mog
        self.mock_msgs.find_one.return_value = {"_id": "m1", "content": "hi"}
        trainer = FeedbackTrainer()
        result = trainer._message_by_id("m1")
        assert result["content"] == "hi"

    @patch("domains.feedback.training.MogDB")
    def test_message_by_id_not_found(self, MockMogDB):
        MockMogDB.return_value = self.mock_mog
        self.mock_msgs.find_one.return_value = None
        trainer = FeedbackTrainer()
        assert trainer._message_by_id("nope") is None

    @patch("domains.feedback.training.MogDB")
    def test_prompt_for(self, MockMogDB):
        MockMogDB.return_value = self.mock_mog
        msg = _make_message("m1", "assistant", "response", created_at="2024-01-02")
        self.mock_msgs.find.return_value = [{"content": "user question"}]
        trainer = FeedbackTrainer()
        assert trainer._prompt_for(msg) == "user question"

    @patch("domains.feedback.training.MogDB")
    def test_prompt_for_empty(self, MockMogDB):
        MockMogDB.return_value = self.mock_mog
        msg = _make_message("m1", "assistant", "response")
        self.mock_msgs.find.return_value = []
        trainer = FeedbackTrainer()
        assert trainer._prompt_for(msg) == ""

    @patch("domains.feedback.training.MogDB")
    def test_get_training_examples_basic(self, MockMogDB):
        MockMogDB.return_value = self.mock_mog
        fb = _make_feedback("m1", "thumbs_up", 0.9)
        msg = _make_message("m1", "assistant", "response")
        self.mock_fb.find.return_value = [fb]
        self.mock_msgs.find_one.return_value = msg
        self.mock_msgs.find.return_value = [{"content": "prompt"}]

        trainer = FeedbackTrainer()
        examples = trainer.get_training_examples()
        assert len(examples) == 1
        assert examples[0].response == "response"
        assert examples[0].rating == "thumbs_up"

    @patch("domains.feedback.training.MogDB")
    def test_get_training_examples_filters_none_rating(self, MockMogDB):
        MockMogDB.return_value = self.mock_mog
        self.mock_fb.find.return_value = [{"rating": None}]
        trainer = FeedbackTrainer()
        assert trainer.get_training_examples() == []

    @patch("domains.feedback.training.MogDB")
    def test_get_training_examples_filters_low_quality(self, MockMogDB):
        MockMogDB.return_value = self.mock_mog
        fb = _make_feedback("m1", "thumbs_up", 0.3)
        self.mock_fb.find.return_value = [fb]
        self.mock_msgs.find_one.return_value = _make_message("m1", "assistant", "r")
        self.mock_msgs.find.return_value = [{"content": "p"}]

        trainer = FeedbackTrainer()
        assert trainer.get_training_examples(min_quality=0.5) == []

    @patch("domains.feedback.training.MogDB")
    def test_get_training_examples_filters_non_assistant(self, MockMogDB):
        MockMogDB.return_value = self.mock_mog
        self.mock_fb.find.return_value = [_make_feedback("m1", "thumbs_up")]
        self.mock_msgs.find_one.return_value = _make_message("m1", "user", "text")
        trainer = FeedbackTrainer()
        assert trainer.get_training_examples() == []

    @patch("domains.feedback.training.MogDB")
    def test_get_training_examples_limit(self, MockMogDB):
        MockMogDB.return_value = self.mock_mog
        fbs = [_make_feedback(f"m{i}", "thumbs_up") for i in range(5)]
        msgs = [_make_message(f"m{i}", "assistant", f"resp{i}", created_at=f"2024-01-0{i}") for i in range(5)]
        self.mock_fb.find.return_value = fbs
        self.mock_msgs.find_one.side_effect = lambda q: next((m for m in msgs if m["_id"] == q.get("_id")), None)
        self.mock_msgs.find.return_value = [{"content": "p"}]

        trainer = FeedbackTrainer()
        assert len(trainer.get_training_examples(limit=3)) == 3

    @patch("domains.feedback.training.MogDB")
    def test_prepare_dpo_pairs(self, MockMogDB):
        MockMogDB.return_value = self.mock_mog
        msg_up = _make_message("m1", "assistant", "good response", conv_id="c1")
        msg_down = _make_message("m2", "assistant", "bad response", conv_id="c1")
        self.mock_msgs.find.return_value = [msg_up, msg_down]
        self.mock_fb.find.return_value = [
            _make_feedback("m1", "thumbs_up"),
            _make_feedback("m2", "thumbs_down"),
        ]
        self.mock_msgs.find_one.side_effect = lambda q: msg_up if q.get("_id") == "m1" else msg_down

        trainer = FeedbackTrainer()
        pairs = trainer.prepare_dpo_pairs()
        assert len(pairs) == 1
        assert pairs[0].chosen == "good response"
        assert pairs[0].rejected == "bad response"

    @patch("domains.feedback.training.MogDB")
    def test_prepare_dpo_pairs_no_pairs(self, MockMogDB):
        MockMogDB.return_value = self.mock_mog
        msg = _make_message("m1", "assistant", "r", conv_id="c1")
        self.mock_msgs.find.return_value = [msg]
        self.mock_fb.find.return_value = [_make_feedback("m1", "thumbs_up")]
        trainer = FeedbackTrainer()
        assert trainer.prepare_dpo_pairs() == []

    @patch("domains.feedback.training.MogDB")
    def test_prepare_sft_data(self, MockMogDB):
        MockMogDB.return_value = self.mock_mog
        fb = _make_feedback("m1", "thumbs_up")
        msg = _make_message("m1", "assistant", "response")
        self.mock_fb.find.return_value = [fb]
        self.mock_msgs.find_one.return_value = msg
        self.mock_msgs.find.return_value = [{"content": "prompt"}]

        trainer = FeedbackTrainer()
        sft = trainer.prepare_sft_data()
        assert len(sft) == 1
        assert sft[0]["prompt"] == "prompt"
        assert sft[0]["response"] == "response"

    @patch("domains.feedback.training.MogDB")
    def test_prepare_sft_data_filters_non_positive(self, MockMogDB):
        MockMogDB.return_value = self.mock_mog
        fb = _make_feedback("m1", "thumbs_down")
        msg = _make_message("m1", "assistant", "response")
        self.mock_fb.find.return_value = [fb]
        self.mock_msgs.find_one.return_value = msg
        self.mock_msgs.find.return_value = [{"content": "prompt"}]
        trainer = FeedbackTrainer()
        assert trainer.prepare_sft_data() == []

    @patch("domains.feedback.training.MogDB")
    def test_export_for_alignment_dpo(self, MockMogDB, tmp_path):
        MockMogDB.return_value = self.mock_mog
        msg_up = _make_message("m1", "assistant", "good", conv_id="c1")
        msg_down = _make_message("m2", "assistant", "bad", conv_id="c1")
        self.mock_msgs.find.return_value = [msg_up, msg_down]
        self.mock_fb.find.return_value = [
            _make_feedback("m1", "thumbs_up"),
            _make_feedback("m2", "thumbs_down"),
        ]
        self.mock_msgs.find_one.side_effect = lambda q: msg_up if q.get("_id") == "m1" else msg_down

        trainer = FeedbackTrainer()
        results = trainer.export_for_alignment(output_dir=str(tmp_path), formats=["dpo"])
        assert "dpo" in results
        assert Path(results["dpo"]).exists()

    @patch("domains.feedback.training.MogDB")
    def test_export_for_alignment_sft(self, MockMogDB, tmp_path):
        MockMogDB.return_value = self.mock_mog
        fb = _make_feedback("m1", "thumbs_up")
        msg = _make_message("m1", "assistant", "response")
        self.mock_fb.find.return_value = [fb]
        self.mock_msgs.find_one.return_value = msg
        self.mock_msgs.find.return_value = [{"content": "prompt"}]

        trainer = FeedbackTrainer()
        results = trainer.export_for_alignment(output_dir=str(tmp_path), formats=["sft"])
        assert "sft" in results
        assert Path(results["sft"]).exists()

    @patch("domains.feedback.training.MogDB")
    def test_export_for_alignment_reward(self, MockMogDB, tmp_path):
        MockMogDB.return_value = self.mock_mog
        fb = _make_feedback("m1", "thumbs_up")
        msg = _make_message("m1", "assistant", "response")
        self.mock_fb.find.return_value = [fb]
        self.mock_msgs.find_one.return_value = msg
        self.mock_msgs.find.return_value = [{"content": "prompt"}]

        trainer = FeedbackTrainer()
        results = trainer.export_for_alignment(output_dir=str(tmp_path), formats=["reward"])
        assert "reward" in results
        with open(results["reward"]) as f:
            data = json.loads(f.readline())
        assert data["reward"] == 1.0

    @patch("domains.feedback.training.MogDB")
    def test_get_training_stats(self, MockMogDB):
        MockMogDB.return_value = self.mock_mog
        msg_up = _make_message("m1", "assistant", "good", conv_id="c1")
        msg_down = _make_message("m2", "assistant", "bad", conv_id="c1")
        self.mock_msgs.find.return_value = [msg_up, msg_down]
        self.mock_fb.find.return_value = [
            _make_feedback("m1", "thumbs_up"),
            _make_feedback("m2", "thumbs_down"),
        ]
        self.mock_msgs.find_one.side_effect = lambda q: msg_up if q.get("_id") == "m1" else msg_down

        trainer = FeedbackTrainer()
        stats = trainer.get_training_stats()
        assert stats["total_conversations"] == 1
        assert stats["total_responses"] == 2
        assert stats["thumbs_up"] == 1
        assert stats["thumbs_down"] == 1

    @patch("domains.feedback.training.MogDB")
    def test_export_dpo(self, MockMogDB, tmp_path):
        MockMogDB.return_value = self.mock_mog
        self.mock_msgs.find.return_value = []
        self.mock_fb.find.return_value = []
        trainer = FeedbackTrainer()
        path = str(tmp_path / "dpo.jsonl")
        count = trainer.export_dpo(path)
        assert count == 0
        assert Path(path).exists()

    @patch("domains.feedback.training.MogDB")
    def test_export_sft(self, MockMogDB, tmp_path):
        MockMogDB.return_value = self.mock_mog
        self.mock_fb.find.return_value = []
        self.mock_msgs.find.return_value = []
        trainer = FeedbackTrainer()
        path = str(tmp_path / "sft.jsonl")
        count = trainer.export_sft(path)
        assert count == 0
        assert Path(path).exists()


# ── Factory ────────────────────────────────────────────────────────────────


class TestFactory:

    @patch("domains.feedback.training.MogDB")
    def test_create_training_pipeline(self, MockMogDB):
        trainer = create_training_pipeline()
        assert isinstance(trainer, FeedbackTrainer)
