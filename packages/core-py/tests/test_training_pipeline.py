"""Tests for domains.infrastructure.training_pipeline — TrainingDataPipeline.

Covers: dataclasses, validation, quality scoring, conversation CRUD, training
pairs, training runs, export (jsonl/json), stats, backup, singleton, migration.
Uses tmp_path to avoid filesystem side effects.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.infrastructure.training_pipeline import (
    Conversation,
    TrainingPair,
    TrainingRun,
    TrainingDataPipeline,
    get_pipeline,
    FEEDBACK_UP,
    FEEDBACK_DOWN,
    FEEDBACK_NEUTRAL,
    NEUTRAL_QUALITY,
    GOOD_QUALITY,
    BAD_QUALITY,
)


@pytest.fixture
def pipeline(tmp_path):
    """Create a fresh pipeline in a temp directory for each test."""
    return TrainingDataPipeline(str(tmp_path / "data"))


class TestConversationDataclass:
    def test_fields(self):
        c = Conversation(
            id="c1", session_id="s1", user_message="hi",
            assistant_message="hello", model="m1", timestamp="2024-01-01",
        )
        assert c.id == "c1"
        assert c.feedback is None
        assert c.metadata == {}

    def test_with_feedback(self):
        c = Conversation(
            id="c1", session_id="s1", user_message="hi",
            assistant_message="hello", model="m1", timestamp="2024-01-01",
            feedback=FEEDBACK_UP, tokens=10,
        )
        assert c.feedback == FEEDBACK_UP
        assert c.tokens == 10


class TestTrainingPairDataclass:
    def test_fields(self):
        p = TrainingPair(
            id="p1", conversation_id="c1", prompt="hi",
            response="hello", quality_score=0.8, feedback=FEEDBACK_UP,
            created_at="2024-01-01",
        )
        assert p.quality_score == 0.8
        assert p.used_in_training is False


class TestTrainingRunDataclass:
    def test_fields(self):
        r = TrainingRun(
            id="r1", created_at="2024-01-01", dataset_version="v1",
            pairs_count=10, model_used="gpt2", status="pending",
        )
        assert r.status == "pending"
        assert r.metrics == {}


class TestValidateFeedback:
    def test_valid_feedback(self):
        TrainingDataPipeline._validate_feedback(FEEDBACK_UP)
        TrainingDataPipeline._validate_feedback(FEEDBACK_DOWN)
        TrainingDataPipeline._validate_feedback(FEEDBACK_NEUTRAL)
        TrainingDataPipeline._validate_feedback(None)

    def test_invalid_feedback(self):
        with pytest.raises(ValueError, match="Invalid feedback"):
            TrainingDataPipeline._validate_feedback("bad_rating")


class TestQualityForFeedback:
    def test_thumbs_up(self):
        assert TrainingDataPipeline._quality_for_feedback(FEEDBACK_UP) == 1.0

    def test_thumbs_down(self):
        assert TrainingDataPipeline._quality_for_feedback(FEEDBACK_DOWN) == 0.0

    def test_neutral(self):
        assert TrainingDataPipeline._quality_for_feedback(FEEDBACK_NEUTRAL) == NEUTRAL_QUALITY

    def test_none(self):
        assert TrainingDataPipeline._quality_for_feedback(None) == NEUTRAL_QUALITY


class TestScoreQuality:
    def test_empty_response(self):
        assert TrainingDataPipeline._score_quality(FEEDBACK_UP, "") == 0.0
        assert TrainingDataPipeline._score_quality(FEEDBACK_UP, None) == 0.0
        assert TrainingDataPipeline._score_quality(FEEDBACK_UP, "   ") == 0.0

    def test_with_response(self):
        assert TrainingDataPipeline._score_quality(FEEDBACK_UP, "hello") == 1.0
        assert TrainingDataPipeline._score_quality(FEEDBACK_DOWN, "hello") == 0.0
        assert TrainingDataPipeline._score_quality(None, "hello") == NEUTRAL_QUALITY


class TestToModel:
    def test_conversation_from_doc(self):
        doc = {"id": "c1", "session_id": "s1", "user_message": "hi",
               "assistant_message": "hello", "model": "m1", "timestamp": "t"}
        c = TrainingDataPipeline._to_model(Conversation, doc)
        assert c.id == "c1"
        assert c.feedback is None

    def test_conversation_missing_required(self):
        doc = {"id": "c1"}
        c = TrainingDataPipeline._to_model(Conversation, doc)
        assert c.session_id is None

    def test_pair_from_doc(self):
        doc = {"id": "p1", "conversation_id": "c1", "prompt": "hi",
               "response": "hello", "quality_score": 0.8, "feedback": None,
               "created_at": "t"}
        p = TrainingDataPipeline._to_model(TrainingPair, doc)
        assert p.id == "p1"


class TestAddConversation:
    def test_add_conversation(self, pipeline):
        conv = pipeline.add_conversation(
            session_id="s1", user_message="hi",
            assistant_message="hello", model="gpt2",
        )
        assert conv.session_id == "s1"
        assert conv.user_message == "hi"
        assert conv.assistant_message == "hello"

    def test_creates_training_pair(self, pipeline):
        pipeline.add_conversation(
            session_id="s1", user_message="hi",
            assistant_message="hello", model="gpt2",
        )
        pairs = pipeline.get_training_pairs()
        assert len(pairs) == 1
        assert pairs[0].prompt == "hi"
        assert pairs[0].response == "hello"

    def test_with_feedback(self, pipeline):
        conv = pipeline.add_conversation(
            session_id="s1", user_message="hi",
            assistant_message="hello", model="gpt2",
            feedback=FEEDBACK_UP,
        )
        assert conv.feedback == FEEDBACK_UP
        pairs = pipeline.get_training_pairs()
        assert pairs[0].quality_score == 1.0

    def test_with_metadata(self, pipeline):
        conv = pipeline.add_conversation(
            session_id="s1", user_message="hi",
            assistant_message="hello", model="gpt2",
            metadata={"key": "val"},
        )
        assert conv.metadata == {"key": "val"}

    def test_invalid_metadata(self, pipeline):
        with pytest.raises(TypeError, match="metadata must be a dict"):
            pipeline.add_conversation(
                session_id="s1", user_message="hi",
                assistant_message="hello", model="gpt2",
                metadata="not a dict",
            )

    def test_invalid_feedback(self, pipeline):
        with pytest.raises(ValueError, match="Invalid feedback"):
            pipeline.add_conversation(
                session_id="s1", user_message="hi",
                assistant_message="hello", model="gpt2",
                feedback="bad",
            )


class TestAddFeedback:
    def test_add_feedback(self, pipeline):
        conv = pipeline.add_conversation(
            session_id="s1", user_message="hi",
            assistant_message="hello", model="gpt2",
        )
        result = pipeline.add_feedback(conv.id, FEEDBACK_UP)
        assert result is True
        convs = pipeline.get_conversations()
        assert convs[0].feedback == FEEDBACK_UP

    def test_feedback_updates_quality(self, pipeline):
        conv = pipeline.add_conversation(
            session_id="s1", user_message="hi",
            assistant_message="hello", model="gpt2",
        )
        pipeline.add_feedback(conv.id, FEEDBACK_DOWN)
        pairs = pipeline.get_training_pairs()
        assert pairs[0].quality_score == 0.0

    def test_feedback_nonexistent(self, pipeline):
        result = pipeline.add_feedback("nonexistent", FEEDBACK_UP)
        assert result is False


class TestGetConversations:
    def test_empty(self, pipeline):
        assert pipeline.get_conversations() == []

    def test_with_data(self, pipeline):
        pipeline.add_conversation("s1", "a", "b", "m1")
        pipeline.add_conversation("s1", "c", "d", "m1")
        convs = pipeline.get_conversations()
        assert len(convs) == 2

    def test_filter_by_session(self, pipeline):
        pipeline.add_conversation("s1", "a", "b", "m1")
        pipeline.add_conversation("s2", "c", "d", "m1")
        convs = pipeline.get_conversations(session_id="s1")
        assert len(convs) == 1

    def test_filter_by_feedback(self, pipeline):
        pipeline.add_conversation("s1", "a", "b", "m1", feedback=FEEDBACK_UP)
        pipeline.add_conversation("s1", "c", "d", "m1", feedback=FEEDBACK_DOWN)
        convs = pipeline.get_conversations(feedback=FEEDBACK_UP)
        assert len(convs) == 1

    def test_limit(self, pipeline):
        for i in range(5):
            pipeline.add_conversation("s1", f"q{i}", f"a{i}", "m1")
        convs = pipeline.get_conversations(limit=3)
        assert len(convs) == 3


class TestGetTrainingPairs:
    def test_empty(self, pipeline):
        assert pipeline.get_training_pairs() == []

    def test_with_data(self, pipeline):
        pipeline.add_conversation("s1", "a", "b", "m1", feedback=FEEDBACK_UP)
        pairs = pipeline.get_training_pairs()
        assert len(pairs) == 1

    def test_filter_min_quality(self, pipeline):
        pipeline.add_conversation("s1", "a", "b", "m1", feedback=FEEDBACK_UP)
        pipeline.add_conversation("s1", "c", "d", "m1", feedback=FEEDBACK_DOWN)
        pairs = pipeline.get_training_pairs(min_quality=0.5)
        assert len(pairs) == 1

    def test_exclude_used(self, pipeline):
        conv = pipeline.add_conversation("s1", "a", "b", "m1", feedback=FEEDBACK_UP)
        pairs = pipeline.get_training_pairs()
        pipeline.mark_pairs_used([pairs[0].id], "run_1")
        pairs = pipeline.get_training_pairs(include_used=False)
        assert len(pairs) == 0

    def test_limit(self, pipeline):
        for i in range(5):
            pipeline.add_conversation("s1", f"q{i}", f"a{i}", "m1")
        pairs = pipeline.get_training_pairs(limit=2)
        assert len(pairs) == 2


class TestMarkPairsUsed:
    def test_mark_used(self, pipeline):
        conv = pipeline.add_conversation("s1", "a", "b", "m1")
        pairs = pipeline.get_training_pairs()
        pipeline.mark_pairs_used([pairs[0].id], "run_1")
        updated = pipeline.get_training_pairs()
        assert updated[0].used_in_training is True
        assert updated[0].training_run_id == "run_1"


class TestTrainingRuns:
    def test_create_run(self, pipeline):
        run = pipeline.create_training_run("v1", 10, "gpt2")
        assert run.status == "pending"
        assert run.pairs_count == 10

    def test_update_run(self, pipeline):
        run = pipeline.create_training_run("v1", 10, "gpt2")
        pipeline.update_training_run(run.id, "running")
        runs = pipeline.get_training_runs()
        assert runs[0].status == "running"

    def test_update_run_metrics(self, pipeline):
        run = pipeline.create_training_run("v1", 10, "gpt2")
        pipeline.update_training_run(run.id, "completed", {"loss": 0.5})
        pipeline.update_training_run(run.id, "completed", {"acc": 0.9})
        runs = pipeline.get_training_runs()
        assert runs[0].metrics == {"loss": 0.5, "acc": 0.9}

    def test_update_nonexistent(self, pipeline):
        pipeline.update_training_run("nonexistent", "running")  # should not raise

    def test_get_runs_limit(self, pipeline):
        for i in range(5):
            pipeline.create_training_run(f"v{i}", i, "m1")
        runs = pipeline.get_training_runs(limit=3)
        assert len(runs) == 3


class TestExportTrainingData:
    def test_export_jsonl(self, pipeline, tmp_path):
        pipeline.add_conversation("s1", "hi", "hello", "m1", feedback=FEEDBACK_UP)
        filepath = pipeline.export_training_data(version="test_v1")
        assert Path(filepath).exists()
        with open(filepath) as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["prompt"] == "hi"
        assert data["response"] == "hello"

    def test_export_json(self, pipeline, tmp_path):
        pipeline.add_conversation("s1", "hi", "hello", "m1", feedback=FEEDBACK_UP)
        filepath = pipeline.export_training_data(format="json", version="test_v1")
        assert Path(filepath).exists()
        with open(filepath) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["prompt"] == "hi"

    def test_export_creates_latest(self, pipeline):
        pipeline.add_conversation("s1", "hi", "hello", "m1", feedback=FEEDBACK_UP)
        filepath = pipeline.export_training_data(version="test_v1")
        latest = Path(filepath).parent / "latest.jsonl"
        assert latest.exists()

    def test_export_marks_pairs_used(self, pipeline):
        conv = pipeline.add_conversation("s1", "hi", "hello", "m1", feedback=FEEDBACK_UP)
        pipeline.export_training_data(version="v1")
        pairs = pipeline.get_training_pairs(include_used=False)
        assert len(pairs) == 0

    def test_export_empty_raises(self, pipeline):
        with pytest.raises(ValueError, match="No training pairs"):
            pipeline.export_training_data()

    def test_export_unknown_format(self, pipeline):
        pipeline.add_conversation("s1", "hi", "hello", "m1", feedback=FEEDBACK_UP)
        with pytest.raises(ValueError, match="Unknown format"):
            pipeline.export_training_data(format="xml")


class TestGetStats:
    def test_empty(self, pipeline):
        stats = pipeline.get_stats()
        assert stats["conversations_total"] == 0
        assert stats["training_pairs_total"] == 0

    def test_with_data(self, pipeline):
        pipeline.add_conversation("s1", "a", "b", "m1", feedback=FEEDBACK_UP)
        pipeline.add_conversation("s1", "c", "d", "m1", feedback=FEEDBACK_DOWN)
        stats = pipeline.get_stats()
        assert stats["conversations_total"] == 2
        assert stats["conversations_with_feedback"] == 2
        assert stats["training_pairs_total"] == 2
        assert stats["training_pairs_good"] == 1
        assert stats["training_pairs_bad"] == 1


class TestCreateBackup:
    def test_backup(self, pipeline):
        pipeline.add_conversation("s1", "a", "b", "m1")
        backup_path = pipeline.create_backup()
        assert Path(backup_path).exists()
        assert Path(backup_path).is_dir()


class TestSingleton:
    def test_get_pipeline_singleton(self, tmp_path):
        import domains.infrastructure.training_pipeline as mod
        old = mod._pipeline
        try:
            mod._pipeline = None
            p1 = get_pipeline(str(tmp_path / "data"))
            p2 = get_pipeline(str(tmp_path / "data"))
            assert p1 is p2
        finally:
            mod._pipeline = old

    def test_get_pipeline_wrong_dir(self, tmp_path):
        import domains.infrastructure.training_pipeline as mod
        old = mod._pipeline
        try:
            mod._pipeline = None
            get_pipeline(str(tmp_path / "data1"))
            with pytest.raises(ValueError, match="already bound"):
                get_pipeline(str(tmp_path / "data2"))
        finally:
            mod._pipeline = old


class TestMigration:
    def test_migrate_conversations(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        legacy = {
            "records": [
                {"id": "c1", "session_id": "s1", "user_message": "hi",
                 "assistant_message": "hello", "model": "m1", "timestamp": "t"},
            ]
        }
        (data_dir / "conversations.db").write_text(json.dumps(legacy))
        pipeline = TrainingDataPipeline(str(data_dir))
        convs = pipeline.get_conversations()
        assert len(convs) == 1
        assert convs[0].id == "c1"
        assert not (data_dir / "conversations.db").exists()

    def test_migrate_skips_existing(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        legacy = {
            "records": [
                {"id": "c1", "session_id": "s1", "user_message": "hi",
                 "assistant_message": "hello", "model": "m1", "timestamp": "t"},
            ]
        }
        (data_dir / "conversations.db").write_text(json.dumps(legacy))
        pipeline = TrainingDataPipeline(str(data_dir))
        # Run migration again — should not duplicate
        pipeline._migrate_from_json()
        convs = pipeline.get_conversations()
        assert len(convs) == 1

    def test_migrate_bad_json(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "conversations.db").write_text("not json {{{")
        pipeline = TrainingDataPipeline(str(data_dir))
        assert pipeline.get_conversations() == []
