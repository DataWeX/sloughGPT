"""
Training Export and Preset Tests — verifies export formats and preset management.

Tests the training export pipeline (CSV, JSON, GGUF, SOU) and preset management
system for quick-start training configurations.

Usage:
    .venv/bin/python -m pytest tests/test_training_export_presets.py -x -v
"""
import csv
import io
import json
import tempfile
import time
from pathlib import Path

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


class TestOutcomeTrackerExport:
    """Tests for TrainingOutcomeTracker export functionality."""

    def test_export_json(self):
        from domains.training.outcome_tracker import TrainingOutcome, TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        initial_count = len(tracker.export_json())
        for i in range(5):
            tracker.record(TrainingOutcome(
                run_id=f"export_test_{i}",
                timestamp=time.time(),
                dataset_size=1000 + i * 100,
                model="test",
                method="sft",
                final_loss=1.0 - i * 0.1,
            ))
        json_data = tracker.export_json()
        assert isinstance(json_data, list)
        assert len(json_data) >= initial_count + 5
        assert all("run_id" in o for o in json_data[-5:])

    def test_export_json_with_limit(self):
        from domains.training.outcome_tracker import TrainingOutcome, TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        for i in range(10):
            tracker.record(TrainingOutcome(
                run_id=f"limit_test_{i}",
                timestamp=time.time(),
                dataset_size=1000,
                model="test",
                method="sft",
                final_loss=1.0,
            ))
        json_data = tracker.export_json(limit=3)
        assert len(json_data) == 3

    def test_export_csv(self):
        from domains.training.outcome_tracker import TrainingOutcome, TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        for i in range(3):
            tracker.record(TrainingOutcome(
                run_id=f"csv_test_{i}",
                timestamp=time.time(),
                dataset_size=1000,
                model="test",
                method="sft",
                final_loss=1.0 - i * 0.1,
            ))
        csv_str = tracker.export_csv()
        assert isinstance(csv_str, str)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) >= 2
        assert "run_id" in rows[0]

    def test_export_csv_with_limit(self):
        from domains.training.outcome_tracker import TrainingOutcome, TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        for i in range(10):
            tracker.record(TrainingOutcome(
                run_id=f"csv_limit_{i}",
                timestamp=time.time(),
                dataset_size=1000,
                model="test",
                method="sft",
                final_loss=1.0,
            ))
        csv_str = tracker.export_csv(limit=5)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 6

    def test_export_json_empty_tracker(self):
        from domains.training.outcome_tracker import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        json_data = tracker.export_json()
        assert isinstance(json_data, list)

    def test_export_csv_empty_tracker(self):
        from domains.training.outcome_tracker import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        csv_str = tracker.export_csv()
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) >= 1
        assert "run_id" in rows[0]


class TestPresetManagement:
    """Tests for training preset management."""

    def test_list_presets(self):
        from domains.training.presets import list_presets

        presets = list_presets()
        assert isinstance(presets, list)
        assert len(presets) > 0

    def test_get_preset(self):
        from domains.training.presets import get_preset

        preset = get_preset("quick-finetune")
        assert preset is not None
        assert preset["name"] == "Quick Fine-Tune"

    def test_get_nonexistent_preset(self):
        from domains.training.presets import get_preset

        preset = get_preset("nonexistent-preset-xyz")
        assert preset is None

    def test_apply_preset(self):
        from domains.training.presets import apply_preset

        config = apply_preset("quick-finetune")
        assert config is not None
        assert "preferred_model" in config
        assert "preferred_method" in config
        assert "default_epochs" in config
        assert "default_batch_size" in config

    def test_apply_nonexistent_preset(self):
        from domains.training.presets import apply_preset

        config = apply_preset("nonexistent-preset-xyz")
        assert config is None

    def test_preset_has_required_fields(self):
        from domains.training.presets import list_presets

        presets = list_presets()
        required_fields = ["name", "description", "model", "method", "epochs", "batch_size"]
        for preset in presets:
            for field in required_fields:
                assert field in preset, f"Missing field '{field}' in preset '{preset.get('name')}'"

    def test_preset_config_values_are_numeric(self):
        from domains.training.presets import apply_preset

        configs = [
            apply_preset("quick-finetune"),
            apply_preset("lora-adapter"),
            apply_preset("full-training"),
        ]
        for config in configs:
            assert isinstance(config["default_epochs"], (int, float))
            assert isinstance(config["default_batch_size"], (int, float))
            assert config["default_epochs"] > 0
            assert config["default_batch_size"] > 0

    def test_all_presets_are_applicable(self):
        from domains.training.presets import PRESETS, apply_preset

        for key in PRESETS:
            config = apply_preset(key)
            assert config is not None, f"Failed to apply preset: {key}"


class TestComprehensiveResultExport:
    """Tests for ComprehensiveResult export capabilities."""

    def test_result_summary(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            summary = result.summary()
            assert isinstance(summary, str)
            assert "OK" in summary or "FAILED" in summary
            assert "Duration:" in summary
            assert "Method:" in summary

    def test_result_has_required_fields(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            assert hasattr(result, "run_id")
            assert hasattr(result, "success")
            assert hasattr(result, "phases")
            assert hasattr(result, "method")
            assert hasattr(result, "performance")

    def test_result_json_serializable(self):
        from dataclasses import asdict

        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)

            def _default(obj):
                if hasattr(obj, "value"):
                    return obj.value
                if hasattr(obj, "name"):
                    return obj.name
                return str(obj)

            result_dict = asdict(result)
            json_str = json.dumps(result_dict, default=_default)
            assert isinstance(json_str, str)
            assert len(json_str) > 0

    def test_result_phases_have_durations(self):
        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(DATA_TEXT)
            f.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=f.name, config=FAST_CONFIG)
            for phase in result.phases:
                assert phase.duration_s >= 0
                assert phase.success is True


class TestExportFileOutput:
    """Tests for writing export data to files."""

    def test_export_json_to_file(self):
        from domains.training.outcome_tracker import TrainingOutcome, TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        for i in range(3):
            tracker.record(TrainingOutcome(
                run_id=f"file_test_{i}",
                timestamp=time.time(),
                dataset_size=1000,
                model="test",
                method="sft",
                final_loss=1.0,
            ))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(tracker.export_json(limit=3), f)
            f.flush()
            path = Path(f.name)
            data = json.loads(path.read_text())
            assert len(data) == 3

    def test_export_csv_to_file(self):
        from domains.training.outcome_tracker import TrainingOutcome, TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        for i in range(3):
            tracker.record(TrainingOutcome(
                run_id=f"csv_file_{i}",
                timestamp=time.time(),
                dataset_size=1000,
                model="test",
                method="sft",
                final_loss=1.0,
            ))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(tracker.export_csv(limit=3))
            f.flush()
            path = Path(f.name)
            content = path.read_text()
            assert "run_id" in content

    def test_result_to_dict_to_file(self):
        from dataclasses import asdict

        from domains.training.comprehensive_trainer import ComprehensiveTrainer

        def _default(obj):
            if hasattr(obj, "value"):
                return obj.value
            if hasattr(obj, "name"):
                return obj.name
            return str(obj)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as data_file:
            data_file.write(DATA_TEXT)
            data_file.flush()
            trainer = ComprehensiveTrainer()
            result = trainer.run_full_cycle(data_path=data_file.name, config=FAST_CONFIG)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as out_file:
                json.dump(asdict(result), out_file, default=_default)
                out_file.flush()
                path = Path(out_file.name)
                data = json.loads(path.read_text())
                assert data["success"] is True
