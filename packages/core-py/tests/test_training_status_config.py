"""Tests for domains.training.status — TrainingStage, CompletionStatus, StageStatus, TrainingCompletionReport; domains.training.auto_config — DatasetAnalysis, TrainingConfig; domains.infrastructure.download_manager — DownloadStatus, DownloadProgress."""

from domains.training.status import (
    TrainingStage, CompletionStatus, StageStatus, TrainingCompletionReport,
)
from domains.training.auto_config import DatasetAnalysis, TrainingConfig
from domains.infrastructure.download_manager import DownloadStatus, DownloadProgress


class TestTrainingStage:
    def test_all_members(self):
        assert len(TrainingStage) == 6
    def test_values(self):
        assert TrainingStage.NOT_STARTED.value == "not_started"
        assert TrainingStage.PRETRAINING.value == "pretraining"
        assert TrainingStage.COMPLETE.value == "complete"
        assert TrainingStage.FAILED.value == "failed"


class TestCompletionStatus:
    def test_all_members(self):
        assert len(CompletionStatus) == 5
    def test_values(self):
        assert CompletionStatus.IN_PROGRESS.value == "in_progress"
        assert CompletionStatus.COMPLETED.value == "completed"
        assert CompletionStatus.FAILED.value == "failed"


class TestStageStatus:
    def test_defaults(self):
        ss = StageStatus(name="pretraining")
        assert ss.name == "pretraining"
        assert ss.status == CompletionStatus.NOT_STARTED
        assert ss.epochs_completed == 0
        assert ss.best_loss == 0.0


class TestTrainingCompletionReport:
    def test_defaults(self):
        tcr = TrainingCompletionReport(model_name="test", created_at="2024-01-01")
        assert tcr.model_name == "test"
        assert tcr.completion_status == CompletionStatus.NOT_STARTED
        assert tcr.pretraining is None
        assert tcr.federated is None


class TestDatasetAnalysis:
    def test_defaults(self):
        da = DatasetAnalysis(path="/tmp/data.txt", format="text")
        assert da.path == "/tmp/data.txt"
        assert da.format == "text"
        assert da.sample_count == 0

    def test_is_dialogue(self):
        da = DatasetAnalysis(path="/tmp/data.txt", format="text", has_dialogue_markers=True)
        assert da.is_dialogue is True

    def test_not_dialogue(self):
        da = DatasetAnalysis(path="/tmp/data.txt", format="text")
        assert da.is_dialogue is False

    def test_is_messages_format(self):
        da = DatasetAnalysis(path="/tmp/data.txt", format="messages")
        assert da.is_messages_format is True

    def test_size_category_tiny(self):
        da = DatasetAnalysis(path="", format="text", word_count=100)
        assert da.size_category == "tiny"

    def test_size_category_small(self):
        da = DatasetAnalysis(path="", format="text", word_count=5000)
        assert da.size_category == "small"

    def test_size_category_medium(self):
        da = DatasetAnalysis(path="", format="text", word_count=50000)
        assert da.size_category == "medium"

    def test_size_category_large(self):
        da = DatasetAnalysis(path="", format="text", word_count=200000)
        assert da.size_category == "large"


class TestTrainingConfig:
    def test_defaults(self):
        tc = TrainingConfig()
        assert tc.model == "gpt2"
        assert tc.method == "finetune"
        assert tc.epochs == 3
        assert tc.use_lora is True
        assert tc.lora_rank == 8

    def test_custom(self):
        tc = TrainingConfig(model="llama", epochs=10)
        assert tc.model == "llama"
        assert tc.epochs == 10


class TestDownloadStatus:
    def test_all_members(self):
        assert len(DownloadStatus) == 5
    def test_values(self):
        assert DownloadStatus.QUEUED.value == "queued"
        assert DownloadStatus.DOWNLOADING.value == "downloading"
        assert DownloadStatus.COMPLETE.value == "complete"


class TestDownloadProgress:
    def test_defaults(self):
        dp = DownloadProgress(model_id="gpt2", status=DownloadStatus.QUEUED)
        assert dp.model_id == "gpt2"
        assert dp.status == DownloadStatus.QUEUED
        assert dp.bytes_downloaded == 0
        assert dp.percentage == 0.0

    def test_to_dict(self):
        dp = DownloadProgress(model_id="gpt2", status=DownloadStatus.COMPLETE, percentage=100.0)
        d = dp.to_dict()
        assert isinstance(d, dict)
        assert d["model_id"] == "gpt2"
        assert d["percentage"] == 100.0
        assert d["status"] == "complete"

    def test_custom(self):
        dp = DownloadProgress(
            model_id="llama", status=DownloadStatus.DOWNLOADING,
            bytes_downloaded=500, total_bytes=1000, percentage=50.0,
        )
        assert dp.bytes_downloaded == 500
        assert dp.total_bytes == 1000
