"""Tests for domains.feedback.training — TrainingExample, DPOPair, FeedbackTrainer."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from domains.feedback.training import (
    TrainingExample,
    DPOPair,
    FeedbackTrainer,
    create_training_pipeline,
)


# ---------------------------------------------------------------------------
# TrainingExample
# ---------------------------------------------------------------------------

class TestTrainingExample:
    def test_basic_fields(self):
        te = TrainingExample(prompt="hello", response="hi", rating="positive")
        assert te.prompt == "hello"
        assert te.response == "hi"
        assert te.rating == "positive"
        assert te.quality_score is None

    def test_with_quality_score(self):
        te = TrainingExample(prompt="q", response="a", rating="good", quality_score=0.9)
        assert te.quality_score == 0.9

    def test_quality_score_zero(self):
        te = TrainingExample(prompt="q", response="a", rating="r", quality_score=0.0)
        assert te.quality_score == 0.0

    def test_quality_score_one(self):
        te = TrainingExample(prompt="q", response="a", rating="r", quality_score=1.0)
        assert te.quality_score == 1.0

    def test_quality_score_negative(self):
        te = TrainingExample(prompt="q", response="a", rating="r", quality_score=-0.5)
        assert te.quality_score == -0.5

    def test_empty_strings(self):
        te = TrainingExample(prompt="", response="", rating="")
        assert te.prompt == ""
        assert te.response == ""
        assert te.rating == ""

    def test_unicode_fields(self):
        te = TrainingExample(prompt="你好", response="世界", rating="good")
        assert te.prompt == "你好"
        assert te.response == "世界"

    def test_long_fields(self):
        long_text = "x" * 10000
        te = TrainingExample(prompt=long_text, response=long_text, rating="r")
        assert len(te.prompt) == 10000

    def test_equality(self):
        a = TrainingExample(prompt="p", response="r", rating="good")
        b = TrainingExample(prompt="p", response="r", rating="good")
        assert a == b

    def test_inequality(self):
        a = TrainingExample(prompt="p", response="r", rating="good")
        b = TrainingExample(prompt="p", response="r", rating="bad")
        assert a != b

    def test_dataclass_fields(self):
        fields = {f.name for f in TrainingExample.__dataclass_fields__.values()}
        assert fields == {"prompt", "response", "rating", "quality_score"}

    def test_quality_score_large(self):
        te = TrainingExample(prompt="q", response="a", rating="r", quality_score=100.0)
        assert te.quality_score == 100.0

    def test_quality_score_float_precision(self):
        te = TrainingExample(prompt="q", response="a", rating="r", quality_score=0.123456789)
        assert abs(te.quality_score - 0.123456789) < 1e-9

    def test_rating_variations(self):
        for rating in ["thumbs_up", "thumbs_down", "neutral", "5 stars", "terrible"]:
            te = TrainingExample(prompt="q", response="a", rating=rating)
            assert te.rating == rating

    def test_quality_score_none_default(self):
        te = TrainingExample(prompt="q", response="a", rating="r")
        assert te.quality_score is None

    def test_inequality_different_prompt(self):
        a = TrainingExample(prompt="p1", response="r", rating="good")
        b = TrainingExample(prompt="p2", response="r", rating="good")
        assert a != b

    def test_inequality_different_response(self):
        a = TrainingExample(prompt="p", response="r1", rating="good")
        b = TrainingExample(prompt="p", response="r2", rating="good")
        assert a != b


# ---------------------------------------------------------------------------
# DPOPair
# ---------------------------------------------------------------------------

class TestDPOPair:
    def test_basic_fields(self):
        dp = DPOPair(chosen="good answer", rejected="bad answer", prompt="question")
        assert dp.chosen == "good answer"
        assert dp.rejected == "bad answer"
        assert dp.prompt == "question"

    def test_empty_strings(self):
        dp = DPOPair(chosen="", rejected="", prompt="")
        assert dp.chosen == ""
        assert dp.rejected == ""
        assert dp.prompt == ""

    def test_unicode(self):
        dp = DPOPair(chosen="好", rejected="坏", prompt="问")
        assert dp.chosen == "好"
        assert dp.rejected == "坏"
        assert dp.prompt == "问"

    def test_long_text(self):
        text = "y" * 5000
        dp = DPOPair(chosen=text, rejected=text, prompt=text)
        assert len(dp.chosen) == 5000

    def test_equality(self):
        a = DPOPair(chosen="a", rejected="b", prompt="c")
        b = DPOPair(chosen="a", rejected="b", prompt="c")
        assert a == b

    def test_inequality(self):
        a = DPOPair(chosen="a", rejected="b", prompt="c")
        b = DPOPair(chosen="x", rejected="b", prompt="c")
        assert a != b

    def test_dataclass_fields(self):
        fields = {f.name for f in DPOPair.__dataclass_fields__.values()}
        assert fields == {"chosen", "rejected", "prompt"}

    def test_inequality_different_rejected(self):
        a = DPOPair(chosen="a", rejected="b", prompt="c")
        b = DPOPair(chosen="a", rejected="x", prompt="c")
        assert a != b

    def test_inequality_different_prompt(self):
        a = DPOPair(chosen="a", rejected="b", prompt="c1")
        b = DPOPair(chosen="a", rejected="b", prompt="c2")
        assert a != b

    def test_long_chosen(self):
        dp = DPOPair(chosen="x" * 20000, rejected="b", prompt="c")
        assert len(dp.chosen) == 20000

    def test_special_characters(self):
        dp = DPOPair(chosen="line\nbreak", rejected="tab\there", prompt="<html>")
        assert "\n" in dp.chosen
        assert "\t" in dp.rejected

    def test_same_chosen_rejected(self):
        dp = DPOPair(chosen="same", rejected="same", prompt="q")
        assert dp.chosen == dp.rejected


# ---------------------------------------------------------------------------
# create_training_pipeline factory
# ---------------------------------------------------------------------------

class TestCreateTrainingPipeline:
    def test_returns_feedback_trainer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_feedback.db")
            trainer = create_training_pipeline(db_path)
            assert isinstance(trainer, FeedbackTrainer)

    def test_default_db_path(self):
        trainer = create_training_pipeline()
        assert isinstance(trainer, FeedbackTrainer)


# ---------------------------------------------------------------------------
# FeedbackTrainer — initialization
# ---------------------------------------------------------------------------

class TestFeedbackTrainerInit:
    def test_creates_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "fb.db")
            trainer = FeedbackTrainer(db_path=db_path)
            assert trainer._db is not None

    def test_collections_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "fb.db")
            trainer = FeedbackTrainer(db_path=db_path)
            assert trainer._messages is not None
            assert trainer._feedback is not None

    def test_stores_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "fb.db")
            trainer = FeedbackTrainer(db_path=db_path)
            assert trainer.db_path == db_path

    def test_unique_instances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            t1 = FeedbackTrainer(db_path=os.path.join(tmpdir, "a.db"))
            t2 = FeedbackTrainer(db_path=os.path.join(tmpdir, "b.db"))
            assert t1 is not t2

    def test_db_is_mogdb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            from mogdb import MogDB
            assert isinstance(trainer._db, MogDB)


# ---------------------------------------------------------------------------
# FeedbackTrainer — empty database operations
# ---------------------------------------------------------------------------

class TestFeedbackTrainerEmpty:
    def test_get_training_examples_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            examples = trainer.get_training_examples()
            assert examples == []

    def test_prepare_dpo_pairs_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            pairs = trainer.prepare_dpo_pairs()
            assert pairs == []

    def test_prepare_sft_data_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            data = trainer.prepare_sft_data()
            assert data == []

    def test_get_training_stats_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            stats = trainer.get_training_stats()
            assert stats["total_conversations"] == 0
            assert stats["total_responses"] == 0
            assert stats["thumbs_up"] == 0
            assert stats["thumbs_down"] == 0
            assert stats["available_dpo_pairs"] == 0
            assert stats["available_sft_examples"] == 0

    def test_export_for_alignment_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            out_dir = os.path.join(tmpdir, "export")
            results = trainer.export_for_alignment(output_dir=out_dir, formats=["dpo", "sft"])
            assert "dpo" in results
            assert "sft" in results
            assert Path(results["dpo"]).exists()
            assert Path(results["sft"]).exists()

    def test_export_dpo_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            out = os.path.join(tmpdir, "dpo.jsonl")
            count = trainer.export_dpo(out)
            assert count == 0

    def test_export_sft_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            out = os.path.join(tmpdir, "sft.jsonl")
            count = trainer.export_sft(out)
            assert count == 0

    def test_export_reward_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            out_dir = os.path.join(tmpdir, "export")
            results = trainer.export_for_alignment(output_dir=out_dir, formats=["reward"])
            assert Path(results["reward"]).exists()


# ---------------------------------------------------------------------------
# FeedbackTrainer — with data
# ---------------------------------------------------------------------------

_seed_counter = 0


def _seed_data(trainer: FeedbackTrainer, conv_id: str = "conv1"):
    """Insert minimal message + feedback documents into the trainer's DB."""
    global _seed_counter
    _seed_counter += 1
    prefix = f"s{_seed_counter}"
    msg_id = f"{prefix}_msg_user"
    assistant_msg_id = f"{prefix}_msg_asst"
    trainer._messages.insert_one({
        "_id": msg_id,
        "id": msg_id,
        "role": "user",
        "content": "What is 2+2?",
        "conversation_id": conv_id,
        "created_at": "2024-01-01T00:00:00Z",
    })
    trainer._messages.insert_one({
        "_id": assistant_msg_id,
        "id": assistant_msg_id,
        "role": "assistant",
        "content": "4",
        "conversation_id": conv_id,
        "created_at": "2024-01-01T00:00:01Z",
    })
    trainer._feedback.insert_one({
        "_id": f"{prefix}_fb",
        "message_id": assistant_msg_id,
        "rating": "thumbs_up",
        "quality_score": 0.95,
        "created_at": "2024-01-01T00:00:02Z",
    })


def _seed_dpo_pair(trainer: FeedbackTrainer, conv_id: str = "conv_dpo"):
    """Insert a conversation with both thumbs_up and thumbs_down for DPO."""
    user_id = "msg_u1"
    trainer._messages.insert_one({
        "_id": user_id,
        "id": user_id,
        "role": "user",
        "content": "Explain X",
        "conversation_id": conv_id,
        "created_at": "2024-01-01T00:00:00Z",
    })
    chosen_id = "msg_c1"
    trainer._messages.insert_one({
        "_id": chosen_id,
        "id": chosen_id,
        "role": "assistant",
        "content": "Good explanation",
        "conversation_id": conv_id,
        "created_at": "2024-01-01T00:00:01Z",
    })
    trainer._feedback.insert_one({
        "_id": "fb_c1",
        "message_id": chosen_id,
        "rating": "thumbs_up",
        "quality_score": 0.9,
        "created_at": "2024-01-01T00:00:02Z",
    })
    rejected_id = "msg_r1"
    trainer._messages.insert_one({
        "_id": rejected_id,
        "id": rejected_id,
        "role": "assistant",
        "content": "Bad explanation",
        "conversation_id": conv_id,
        "created_at": "2024-01-01T00:00:03Z",
    })
    trainer._feedback.insert_one({
        "_id": "fb_r1",
        "message_id": rejected_id,
        "rating": "thumbs_down",
        "quality_score": 0.2,
        "created_at": "2024-01-01T00:00:04Z",
    })


class TestFeedbackTrainerWithData:
    def test_get_training_examples_returns_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            _seed_data(trainer)
            examples = trainer.get_training_examples()
            assert len(examples) >= 1
            assert examples[0].response == "4"

    def test_get_training_examples_min_quality_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            _seed_data(trainer)
            examples_high = trainer.get_training_examples(min_quality=0.5)
            assert len(examples_high) == 1
            examples_low = trainer.get_training_examples(min_quality=1.0)
            assert len(examples_low) == 0

    def test_get_training_examples_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            _seed_data(trainer, conv_id="c1")
            _seed_data(trainer, conv_id="c2")
            examples = trainer.get_training_examples(limit=1)
            assert len(examples) == 1

    def test_prepare_dpo_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            _seed_dpo_pair(trainer)
            pairs = trainer.prepare_dpo_pairs()
            assert len(pairs) == 1
            assert pairs[0].chosen == "Good explanation"
            assert pairs[0].rejected == "Bad explanation"

    def test_prepare_dpo_pairs_min_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            _seed_dpo_pair(trainer)
            pairs = trainer.prepare_dpo_pairs(min_pairs=10)
            assert len(pairs) == 1

    def test_prepare_sft_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            _seed_data(trainer)
            data = trainer.prepare_sft_data()
            assert len(data) >= 1
            assert data[0]["response"] == "4"
            assert "prompt" in data[0]
            assert "quality_score" in data[0]

    def test_get_training_stats_with_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            _seed_data(trainer)
            stats = trainer.get_training_stats()
            assert stats["total_responses"] >= 1
            assert stats["thumbs_up"] >= 1

    def test_export_for_alignment_with_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            _seed_data(trainer)
            out_dir = os.path.join(tmpdir, "export")
            results = trainer.export_for_alignment(output_dir=out_dir)
            assert Path(results["dpo"]).exists()
            assert Path(results["sft"]).exists()

    def test_export_dpo_with_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            _seed_dpo_pair(trainer)
            out = os.path.join(tmpdir, "dpo.jsonl")
            count = trainer.export_dpo(out)
            assert count == 1
            with open(out) as f:
                line = json.loads(f.readline())
            assert "chosen" in line
            assert "rejected" in line

    def test_export_sft_with_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            _seed_data(trainer)
            out = os.path.join(tmpdir, "sft.jsonl")
            count = trainer.export_sft(out)
            assert count >= 1
            with open(out) as f:
                line = json.loads(f.readline())
            assert "prompt" in line
            assert "response" in line

    def test_export_reward_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            _seed_data(trainer)
            out_dir = os.path.join(tmpdir, "export")
            results = trainer.export_for_alignment(output_dir=out_dir, formats=["reward"])
            assert "reward" in results
            assert Path(results["reward"]).exists()
            with open(results["reward"]) as f:
                line = json.loads(f.readline())
            assert "reward" in line
            assert line["reward"] == 1.0

    def test_export_reward_rejects_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            _seed_dpo_pair(trainer)
            out_dir = os.path.join(tmpdir, "export")
            results = trainer.export_for_alignment(output_dir=out_dir, formats=["reward"])
            with open(results["reward"]) as f:
                lines = [json.loads(l) for l in f]
            rewards = [l["reward"] for l in lines]
            assert 1.0 in rewards
            assert 0.0 in rewards

    def test_multiple_conversations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            _seed_data(trainer, conv_id="c1")
            _seed_data(trainer, conv_id="c2")
            _seed_data(trainer, conv_id="c3")
            stats = trainer.get_training_stats()
            assert stats["total_conversations"] == 3

    def test_sft_quality_score_default_one(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            trainer._messages.insert_one({
                "_id": "m0", "id": "m0", "role": "user", "content": "q",
                "conversation_id": "c", "created_at": "2024-01-01T00:00:00Z",
            })
            trainer._messages.insert_one({
                "_id": "m1", "id": "m1", "role": "assistant", "content": "a",
                "conversation_id": "c", "created_at": "2024-01-01T00:00:01Z",
            })
            trainer._feedback.insert_one({
                "_id": "f1", "message_id": "m1", "rating": "thumbs_up",
                "created_at": "2024-01-01T00:00:02Z",
            })
            data = trainer.prepare_sft_data()
            assert len(data) == 1
            assert data[0]["quality_score"] == 1.0


# ---------------------------------------------------------------------------
# FeedbackTrainer — edge cases
# ---------------------------------------------------------------------------

class TestFeedbackTrainerEdgeCases:
    def test_missing_rating_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            trainer._messages.insert_one({
                "_id": "m1", "id": "m1", "role": "assistant", "content": "hi",
                "conversation_id": "c", "created_at": "2024-01-01T00:00:00Z",
            })
            trainer._feedback.insert_one({
                "_id": "f1", "message_id": "m1", "rating": None,
                "created_at": "2024-01-01T00:00:01Z",
            })
            examples = trainer.get_training_examples()
            assert examples == []

    def test_non_assistant_role_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            trainer._messages.insert_one({
                "_id": "m1", "id": "m1", "role": "user", "content": "hi",
                "conversation_id": "c", "created_at": "2024-01-01T00:00:00Z",
            })
            trainer._feedback.insert_one({
                "_id": "f1", "message_id": "m1", "rating": "thumbs_up",
                "created_at": "2024-01-01T00:00:01Z",
            })
            examples = trainer.get_training_examples()
            assert examples == []

    def test_dpo_single_rating_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            trainer._messages.insert_one({
                "_id": "m1", "id": "m1", "role": "user", "content": "q",
                "conversation_id": "c", "created_at": "2024-01-01T00:00:00Z",
            })
            trainer._messages.insert_one({
                "_id": "m2", "id": "m2", "role": "assistant", "content": "a",
                "conversation_id": "c", "created_at": "2024-01-01T00:00:01Z",
            })
            trainer._feedback.insert_one({
                "_id": "f1", "message_id": "m2", "rating": "thumbs_up",
                "created_at": "2024-01-01T00:00:02Z",
            })
            pairs = trainer.prepare_dpo_pairs()
            assert pairs == []

    def test_sft_filters_non_thumbs_up(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            trainer._messages.insert_one({
                "_id": "m1", "id": "m1", "role": "assistant", "content": "a",
                "conversation_id": "c", "created_at": "2024-01-01T00:00:00Z",
            })
            trainer._messages.insert_one({
                "_id": "m0", "id": "m0", "role": "user", "content": "q",
                "conversation_id": "c", "created_at": "2024-01-01T00:00:01Z",
            })
            trainer._feedback.insert_one({
                "_id": "f1", "message_id": "m1", "rating": "thumbs_down",
                "quality_score": 0.9,
                "created_at": "2024-01-01T00:00:02Z",
            })
            data = trainer.prepare_sft_data()
            assert data == []

    def test_empty_prompt_in_sft_excluded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            trainer._messages.insert_one({
                "_id": "m1", "id": "m1", "role": "assistant", "content": "a",
                "conversation_id": "c", "created_at": "2024-01-01T00:00:00Z",
            })
            trainer._feedback.insert_one({
                "_id": "f1", "message_id": "m1", "rating": "thumbs_up",
                "created_at": "2024-01-01T00:00:01Z",
            })
            data = trainer.prepare_sft_data()
            assert data == []

    def test_feedback_for_nonexistent_message_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            trainer._feedback.insert_one({
                "_id": "f1", "message_id": "missing_msg", "rating": "thumbs_up",
                "created_at": "2024-01-01T00:00:00Z",
            })
            examples = trainer.get_training_examples()
            assert examples == []

    def test_training_stats_excludes_orphan_feedback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            trainer._feedback.insert_one({
                "_id": "f1", "message_id": "orphan", "rating": "thumbs_up",
                "created_at": "2024-01-01T00:00:00Z",
            })
            stats = trainer.get_training_stats()
            assert stats["thumbs_up"] == 0

    def test_export_jsonl_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            _seed_data(trainer)
            out = os.path.join(tmpdir, "sft.jsonl")
            trainer.export_sft(out)
            with open(out) as f:
                for line in f:
                    json.loads(line)

    def test_dpo_pair_requires_both_ratings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            conv_id = "partial"
            trainer._messages.insert_one({
                "_id": "u", "id": "u", "role": "user", "content": "q",
                "conversation_id": conv_id, "created_at": "2024-01-01T00:00:00Z",
            })
            trainer._messages.insert_one({
                "_id": "a1", "id": "a1", "role": "assistant", "content": "a1",
                "conversation_id": conv_id, "created_at": "2024-01-01T00:00:01Z",
            })
            trainer._messages.insert_one({
                "_id": "a2", "id": "a2", "role": "assistant", "content": "a2",
                "conversation_id": conv_id, "created_at": "2024-01-01T00:00:02Z",
            })
            trainer._feedback.insert_one({
                "_id": "f1", "message_id": "a1", "rating": "thumbs_up",
                "created_at": "2024-01-01T00:00:03Z",
            })
            trainer._feedback.insert_one({
                "_id": "f2", "message_id": "a2", "rating": "thumbs_up",
                "created_at": "2024-01-01T00:00:04Z",
            })
            pairs = trainer.prepare_dpo_pairs()
            assert pairs == []

    def test_get_training_examples_ascending_quality(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FeedbackTrainer(db_path=os.path.join(tmpdir, "fb.db"))
            trainer._messages.insert_one({
                "_id": "u0", "id": "u0", "role": "user", "content": "q",
                "conversation_id": "c", "created_at": "2023-12-31T23:59:59Z",
            })
            for i in range(5):
                mid = f"m{i}"
                trainer._messages.insert_one({
                    "_id": mid, "id": mid, "role": "assistant", "content": f"resp{i}",
                    "conversation_id": "c", "created_at": f"2024-01-01T00:00:{i:02d}Z",
                })
                trainer._feedback.insert_one({
                    "_id": f"f{i}", "message_id": mid, "rating": "thumbs_up",
                    "quality_score": i * 0.2,
                    "created_at": f"2024-01-01T00:00:{i:02d}Z",
                })
            examples = trainer.get_training_examples(min_quality=0.5)
            assert len(examples) >= 1
