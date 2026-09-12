"""
Training Checkpoint and Export Tests — checkpoint operations, export formats, model metadata.

Tests checkpoint loading, listing, export formats, and model metadata handling.

Usage:
    .venv/bin/python -m pytest tests/test_training_checkpoint_export.py -x -v
"""
import tempfile

DATA_TEXT = "The quick brown fox jumps over the lazy dog. " * 50

FAST_CONFIG = {
    "method": "sft",
    "data_quality_threshold": 0.0,
    "epochs": 1,
    "batch_size": 8,
    "block_size": 32,
    "max_steps": 2,
    "n_embed": 32,
    "n_layer": 2,
    "n_head": 2,
}


class TestCheckpointOperations:
    """Tests for checkpoint loading and listing."""

    def test_find_checkpoint_nonexistent(self):
        from domains.training.checkpoints import find_checkpoint

        result = find_checkpoint("nonexistent_checkpoint_xyz")
        assert result is None

    def test_load_soul_nonexistent(self):
        from domains.training.checkpoints import load_soul

        result = load_soul("nonexistent_soul_xyz")
        assert result is None

    def test_scan_all_checkpoints(self):
        from domains.training.checkpoints import _scan_all_checkpoints

        checkpoints = _scan_all_checkpoints()
        assert isinstance(checkpoints, list)

    def test_checkpoint_dirs_exist(self):
        from domains.training.state import CHECKPOINTS_DIR, LORA_DIR, TURBO_DIR

        assert CHECKPOINTS_DIR.exists()
        assert LORA_DIR.exists()
        assert TURBO_DIR.exists()


class TestExportFormats:
    """Tests for model export format handling."""

    def test_list_export_formats(self):
        from domains.training.export import list_export_formats

        formats = list_export_formats()
        assert isinstance(formats, dict)
        assert len(formats) > 0

    def test_export_formats_have_descriptions(self):
        from domains.training.export import list_export_formats

        formats = list_export_formats()
        for name, desc in formats.items():
            assert isinstance(name, str)
            assert isinstance(desc, str)
            assert len(desc) > 0


class TestModelMetadata:
    """Tests for model metadata handling."""

    def test_create_model_metadata(self):
        from domains.training.export import ModelMetadata

        metadata = ModelMetadata(
            name="test_model",
            model_type="slnet",
            architecture="transformer",
        )
        assert metadata is not None
        assert metadata.name == "test_model"
        assert metadata.model_type == "slnet"

    def test_metadata_has_timestamp(self):
        from domains.training.export import ModelMetadata

        metadata = ModelMetadata(
            name="test_model",
            model_type="slnet",
            architecture="transformer",
        )
        assert metadata.created_at is not None

    def test_metadata_to_dict(self):
        from domains.training.export import ModelMetadata

        metadata = ModelMetadata(
            name="test_model",
            model_type="slnet",
            architecture="transformer",
        )
        assert hasattr(metadata, "to_dict")
        d = metadata.to_dict()
        assert isinstance(d, dict)
        assert "name" in d


class TestExportConfig:
    """Tests for export configuration."""

    def test_export_config_defaults(self):
        from domains.training.export import ExportConfig

        config = ExportConfig()
        assert config.output_path is not None
        assert config.format is not None

    def test_export_config_custom(self):
        from domains.training.export import ExportConfig

        config = ExportConfig(
            output_path="test_model",
            format="gguf",
        )
        assert config.output_path == "test_model"
        assert config.format == "gguf"


class TestTrainingStateConstants:
    """Tests for training state constants."""

    def test_valid_checkpoint_name_regex(self):
        from domains.training.state import VALID_CKPT_NAME

        assert VALID_CKPT_NAME.match("my-checkpoint_v1")
        assert VALID_CKPT_NAME.match("checkpoint.001")
        assert not VALID_CKPT_NAME.match("checkpoint with spaces")
        assert not VALID_CKPT_NAME.match("checkpoint/slash")

    def test_max_checkpoint_disk_mb(self):
        from domains.training.state import MAX_CHECKPOINT_DISK_MB

        assert MAX_CHECKPOINT_DISK_MB > 0
        assert isinstance(MAX_CHECKPOINT_DISK_MB, int)


class TestTrainingSequenceExtended:
    """Extended tests for training sequence."""

    def test_sequence_ordered_phases_count(self):
        from domains.training.sequence import TrainingSequence

        phases = TrainingSequence.ordered_phases()
        assert len(phases) == 9

    def test_phase_result_to_dict(self):
        from domains.training.sequence import PhaseResult, TrainingSequence

        pr = PhaseResult(
            phase=TrainingSequence.TRAIN,
            status="success",
            metrics={"loss": 0.5},
        )
        d = pr.to_dict()
        assert d["phase"] == "train"
        assert d["status"] == "success"
        assert d["metrics"]["loss"] == 0.5

    def test_training_run_config_defaults(self):
        from domains.training.sequence import TrainingRunConfig

        config = TrainingRunConfig()
        assert config.skip_generate is False
        assert config.max_epochs == 10
        assert config.early_stop_patience == 3

    def test_training_sequence_state_transitions(self):
        from domains.training.sequence import TrainingSequence, TrainingSequenceState

        state = TrainingSequenceState()
        assert state.current_phase == TrainingSequence.IDLE

        state.start_phase(TrainingSequence.GENERATE_DATA)
        assert state.current_phase == TrainingSequence.GENERATE_DATA
        assert state.is_running

        state.complete_phase(TrainingSequence.GENERATE_DATA, metrics={"samples": 100})
        completed = [pr for pr in state.phase_results if pr.status == "success"]
        assert len(completed) > 0


class TestComprehensiveTrainerExtended:
    """Extended tests for ComprehensiveTrainer."""

    def test_trainer_multiple_runs(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            for _ in range(3):
                trainer = ComprehensiveTrainer()
                result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
                assert result.success

    def test_trainer_result_performance(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            perf = result.performance
            assert "total_duration_s" in perf
            assert "phase_durations" in perf
            assert perf["total_duration_s"] > 0

    def test_trainer_phases_recorded(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert len(result.phases) >= 4
            for phase in result.phases:
                assert phase.duration_s >= 0
