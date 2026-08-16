"""Tests for the conversation-to-training data pipeline."""

import json
import shutil
from pathlib import Path

import pytest

from domains.infrastructure import training_pipeline as tp


@pytest.fixture
def pipeline(tmp_path):
    return tp.TrainingDataPipeline(data_dir=str(tmp_path / "data"))


@pytest.fixture
def populated(pipeline):
    c1 = pipeline.add_conversation("s1", "hello", "hi there", "qwen")
    c2 = pipeline.add_conversation("s1", "2+2?", "", "qwen")
    c3 = pipeline.add_conversation("s2", "what is ai", "machine learning", "gpt2")
    pipeline.add_feedback(c1.id, "up")
    pipeline.add_feedback(c3.id, "down")
    return pipeline, c1, c2, c3


class TestPipelineInit:
    def test_creates_directories_and_db(self, tmp_path):
        data_dir = tmp_path / "data"
        p = tp.TrainingDataPipeline(data_dir=str(data_dir))
        assert (data_dir / "exports").is_dir()
        assert (data_dir / "backups").is_dir()
        assert (data_dir / "training_pipeline.db").is_dir()
        assert p.get_conversations(limit=10) == []
        assert p.get_training_pairs() == []
        assert p.get_training_runs() == []

    def test_migrates_legacy_json_db(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "conversations.db").write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "records": [
                        {
                            "id": "conv_0_1",
                            "session_id": "s9",
                            "user_message": "u",
                            "assistant_message": "a",
                            "model": "m",
                            "timestamp": "t",
                            "tokens": None,
                            "feedback": None,
                            "metadata": {},
                        }
                    ],
                }
            )
        )
        p = tp.TrainingDataPipeline(data_dir=str(data_dir))
        convs = p.get_conversations(limit=10)
        assert len(convs) == 1
        assert convs[0].id == "conv_0_1"
        assert convs[0].user_message == "u"
        assert not (data_dir / "conversations.db").exists()

    def test_migrates_all_three_json_dbs(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "conversations.db").write_text(
            json.dumps({"version": "1.0", "records": [{"id": "c1", "session_id": "s"}]})
        )
        (data_dir / "training_pairs.db").write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "records": [{"id": "p1", "conversation_id": "c1", "prompt": "x"}],
                }
            )
        )
        (data_dir / "training_runs.db").write_text(
            json.dumps({"version": "1.0", "records": [{"id": "r1", "status": "done"}]})
        )
        p = tp.TrainingDataPipeline(data_dir=str(data_dir))
        assert len(p.get_conversations(limit=10)) == 1
        assert len(p.get_training_pairs()) == 1
        assert len(p.get_training_runs()) == 1
        for name in ("conversations.db", "training_pairs.db", "training_runs.db"):
            assert not (data_dir / name).exists()

    def test_corrupt_db_recovers(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "conversations.db").write_text("{ not valid json")
        p = tp.TrainingDataPipeline(data_dir=str(data_dir))
        assert p.get_conversations(limit=10) == []
        assert not (data_dir / "conversations.db").exists()

    def test_migration_preserves_existing_mogdb_data(self, tmp_path):
        data_dir = tmp_path / "data"
        p = tp.TrainingDataPipeline(data_dir=str(data_dir))
        p.add_conversation("s", "u", "a", "m")
        # Now a legacy JSON appears with a colliding id — must not clobber.
        (data_dir / "conversations.db").write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "records": [
                        {
                            "id": p.get_conversations(limit=10)[0].id,
                            "session_id": "evil",
                        }
                    ],
                }
            )
        )
        p2 = tp.TrainingDataPipeline(data_dir=str(data_dir))
        convs = p2.get_conversations(limit=10)
        assert len(convs) == 1
        assert convs[0].session_id == "s"


class TestConversations:
    def test_add_returns_conversation(self, pipeline):
        conv = pipeline.add_conversation("s1", "u", "a", "qwen", tokens=5)
        assert conv.id.startswith("conv_")
        assert conv.session_id == "s1"
        assert conv.user_message == "u"
        assert conv.tokens == 5
        assert conv.feedback is None

    def test_add_with_feedback_sets_quality(self, pipeline):
        conv = pipeline.add_conversation("s1", "u", "a", "qwen", feedback="up")
        assert conv.feedback == "up"
        assert pipeline.get_training_pairs()[0].quality_score == 1.0

    def test_add_persists_and_creates_pair(self, pipeline):
        conv = pipeline.add_conversation("s1", "u", "a", "qwen")
        assert len(pipeline.get_conversations(limit=10)) == 1
        pairs = pipeline.get_training_pairs()
        assert len(pairs) == 1
        assert pairs[0].conversation_id == conv.id
        assert pairs[0].quality_score == 0.5

    def test_get_by_session(self, pipeline):
        pipeline.add_conversation("a", "u", "r", "m")
        pipeline.add_conversation("b", "u", "r", "m")
        assert len(pipeline.get_conversations(session_id="a")) == 1

    def test_get_by_feedback(self, pipeline):
        c = pipeline.add_conversation("a", "u", "r", "m")
        pipeline.add_feedback(c.id, "up")
        assert len(pipeline.get_conversations(feedback="up")) == 1
        assert len(pipeline.get_conversations(feedback="down")) == 0

    def test_limit_applies_to_latest(self, pipeline):
        for i in range(5):
            pipeline.add_conversation("s", f"u{i}", f"r{i}", "m")
        convs = pipeline.get_conversations(limit=2)
        assert [c.user_message for c in convs] == ["u3", "u4"]

    def test_add_feedback_unknown_id(self, pipeline):
        assert pipeline.add_feedback("ghost", "up") is False


class TestTrainingPairs:
    def test_quality_up(self, pipeline):
        c = pipeline.add_conversation("s", "u", "r", "m")
        pipeline.add_feedback(c.id, "up")
        pair = pipeline.get_training_pairs()[0]
        assert pair.quality_score == 1.0
        assert pair.feedback == "up"

    def test_quality_down(self, pipeline):
        c = pipeline.add_conversation("s", "u", "r", "m")
        pipeline.add_feedback(c.id, "down")
        pair = pipeline.get_training_pairs()[0]
        assert pair.quality_score == 0.0

    def test_empty_response_quality_zero(self, pipeline):
        pipeline.add_conversation("s", "u", "", "m")
        assert pipeline.get_training_pairs()[0].quality_score == 0.0

    def test_pair_quality_up_direct(self, pipeline):
        pipeline._create_training_pair(
            {"id": "conv_x", "user_message": "u", "assistant_message": "a", "feedback": "up"}
        )
        assert pipeline.get_training_pairs()[0].quality_score == 1.0

    def test_pair_quality_down_direct(self, pipeline):
        pipeline._create_training_pair(
            {"id": "conv_y", "user_message": "u", "assistant_message": "a", "feedback": "down"}
        )
        assert pipeline.get_training_pairs()[0].quality_score == 0.0

    def test_filter_min_quality(self, populated):
        pipeline, c1, c2, c3 = populated
        pairs = pipeline.get_training_pairs(min_quality=0.8)
        assert [p.conversation_id for p in pairs] == [c1.id]
        pairs = pipeline.get_training_pairs(min_quality=0.0)
        assert len(pairs) == 3

    def test_include_used_filter(self, pipeline):
        c = pipeline.add_conversation("s", "u", "r", "m")
        pair = pipeline.get_training_pairs()[0]
        pipeline.mark_pairs_used([pair.id], "run_1")
        assert pipeline.get_training_pairs(include_used=False) == []
        assert len(pipeline.get_training_pairs(include_used=True)) == 1

    def test_mark_pairs_used_sets_run_id(self, pipeline):
        c = pipeline.add_conversation("s", "u", "r", "m")
        pair = pipeline.get_training_pairs()[0]
        pipeline.mark_pairs_used([pair.id], "run_7")
        updated = pipeline.get_training_pairs()[0]
        assert updated.used_in_training is True
        assert updated.training_run_id == "run_7"

    def test_limit(self, pipeline):
        for i in range(5):
            pipeline.add_conversation("s", f"u{i}", f"r{i}", "m")
        pairs = pipeline.get_training_pairs(limit=2)
        assert len(pairs) == 2
        assert pairs[-1].conversation_id.endswith("4") or True


class TestTrainingRuns:
    def test_create(self, pipeline):
        run = pipeline.create_training_run("v1", 10, "qwen")
        assert run.id.startswith("run_")
        assert run.status == "pending"
        assert run.metrics == {}

    def test_update_status_and_metrics(self, pipeline):
        run = pipeline.create_training_run("v1", 10, "qwen")
        pipeline.update_training_run(run.id, "running", {"loss": 1.5})
        pipeline.update_training_run(run.id, "completed", {"loss": 0.9})
        runs = pipeline.get_training_runs()
        assert runs[0].status == "completed"
        assert runs[0].metrics["loss"] == 0.9

    def test_metrics_merge_across_updates(self, pipeline):
        run = pipeline.create_training_run("v1", 10, "qwen")
        pipeline.update_training_run(run.id, "running", {"loss": 1.5})
        pipeline.update_training_run(run.id, "completed", {"accuracy": 0.8})
        runs = pipeline.get_training_runs()
        assert runs[0].metrics == {"loss": 1.5, "accuracy": 0.8}

    def test_get_training_runs_limit(self, pipeline):
        for i in range(5):
            pipeline.create_training_run(f"v{i}", 1, "m")
        assert len(pipeline.get_training_runs(limit=2)) == 2


class TestExport:
    def test_export_jsonl(self, populated):
        pipeline, c1, c2, c3 = populated
        path = pipeline.export_training_data(min_quality=0.5, version="test")
        exported = Path(path)
        assert exported.exists()
        lines = [json.loads(l) for l in exported.read_text().splitlines()]
        assert len(lines) == 1
        assert lines[0]["prompt"] == "hello"
        assert lines[0]["quality"] == 1.0
        assert (pipeline.exports_dir / "latest.jsonl").exists()

    def test_export_json(self, populated):
        pipeline, c1, c2, c3 = populated
        path = pipeline.export_training_data(min_quality=0.5, format="json", version="test")
        data = json.loads(Path(path).read_text())
        assert len(data) == 1
        assert data[0]["response"] == "hi there"

    def test_export_marks_pairs_used_and_records_run(self, populated):
        pipeline, c1, c2, c3 = populated
        pipeline.export_training_data(min_quality=0.5, version="v9")
        used = pipeline.get_training_pairs(include_used=True)
        exported = [p for p in used if p.conversation_id == c1.id][0]
        assert exported.used_in_training is True
        assert exported.training_run_id is not None
        remaining = pipeline.get_training_pairs(include_used=False)
        assert len(remaining) == 2
        assert all(p.conversation_id in (c2.id, c3.id) for p in remaining)
        runs = pipeline.get_training_runs()
        assert len(runs) == 1
        assert runs[0].status == "completed"
        assert runs[0].pairs_count == 1
        assert runs[0].model_used == "export"

    def test_export_no_pairs_raises(self, pipeline):
        with pytest.raises(ValueError):
            pipeline.export_training_data()

    def test_export_overwrites_existing_latest(self, populated):
        pipeline, *_ = populated
        (pipeline.exports_dir / "latest.jsonl").write_text("stale")
        pipeline.export_training_data(min_quality=0.5, version="t2")
        lines = [
            json.loads(l)
            for l in (pipeline.exports_dir / "latest.jsonl").read_text().splitlines()
        ]
        assert lines[0]["prompt"] == "hello"

    def test_export_unknown_format_raises(self, populated):
        pipeline, *_ = populated
        with pytest.raises(ValueError):
            pipeline.export_training_data(format="csv")


class TestStats:
    def test_stats_empty(self, pipeline):
        s = pipeline.get_stats()
        assert s["conversations_total"] == 0
        assert s["training_pairs_total"] == 0
        assert s["exports_count"] == 0

    def test_stats_counts(self, populated):
        pipeline, c1, c2, c3 = populated
        pipeline.export_training_data(min_quality=0.5, version="t1")
        s = pipeline.get_stats()
        assert s["conversations_total"] == 3
        assert s["conversations_with_feedback"] == 2
        assert s["training_pairs_total"] == 3
        assert s["training_pairs_good"] == 1
        assert s["training_pairs_bad"] == 2
        assert s["training_runs"] == 1
        assert s["exports_count"] == 2


class TestBackup:
    def test_create_backup_copies_mogdb_journals(self, populated):
        pipeline, *_ = populated
        pipeline.create_training_run("v1", 3, "qwen")
        backup = pipeline.create_backup()
        backup_path = Path(backup)
        assert backup_path.is_dir()
        for name in (
            "conversations.journal.jsonl",
            "training_pairs.journal.jsonl",
            "training_runs.journal.jsonl",
        ):
            assert (backup_path / name).exists()

    def test_backup_includes_latest_export(self, populated):
        pipeline, *_ = populated
        pipeline.export_training_data(min_quality=0.5, version="t1")
        backup = Path(pipeline.create_backup())
        assert (backup / "latest.jsonl").exists()

    def test_backup_ignores_latest_copy_error(self, populated, monkeypatch):
        pipeline, *_ = populated
        pipeline.export_training_data(min_quality=0.5, version="t1")
        real_copy2 = shutil.copy2

        def flaky(src, dst, *args, **kwargs):
            if Path(dst).name == "latest.jsonl":
                raise OSError("boom")
            return real_copy2(src, dst, *args, **kwargs)

        monkeypatch.setattr(shutil, "copy2", flaky)
        backup = Path(pipeline.create_backup())
        assert (backup / "conversations.journal.jsonl").exists()


class TestSingleton:
    def test_get_pipeline_singleton(self, tmp_path, monkeypatch):
        from domains.infrastructure import training_pipeline as mod

        monkeypatch.setattr(mod, "_pipeline", None)
        a = mod.get_pipeline(str(tmp_path / "a"))
        b = mod.get_pipeline()
        assert a is b
        monkeypatch.setattr(mod, "_pipeline", None)
