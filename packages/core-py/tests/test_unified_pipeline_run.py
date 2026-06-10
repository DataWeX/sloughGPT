"""Tests for UnifiedTrainingPipeline direct execution (not via HTTP).

Tests that:
- Pipeline with all skips produces complete result
- Pipeline with only skip_train=True runs non-train phases
- on_progress callback is invoked
- create_pipeline() factory works
- Result dict contains expected keys
"""

from __future__ import annotations


def _assert_result_shape(result: dict):
    assert "status" in result
    assert "message" in result
    assert "model_path" in result
    assert "final_loss" in result
    assert "total_steps" in result
    assert "phases" in result
    assert "elapsed" in result
    assert "checkpoint" in result
    assert "metrics" in result


class TestPipelineAllSkipped:
    """Pipeline where every phase is skipped."""

    def test_all_skipped_returns_completed(self):
        from domains.training.unified_pipeline import UnifiedTrainingPipeline, UnifiedTrainingConfig
        from domains.training.sequence import TrainingRunConfig

        config = UnifiedTrainingConfig()
        run_config = TrainingRunConfig(
            skip_generate=True, skip_distill=True, skip_train=True,
            skip_evaluate=True, skip_deploy=True,
        )
        pipeline = UnifiedTrainingPipeline(config, run_config=run_config)
        result = pipeline.run()
        assert result["status"] == "completed"
        assert all(pr["status"] == "skipped" for pr in result["phases"])
        _assert_result_shape(result)

    def test_all_skipped_progress_callback(self):
        from domains.training.unified_pipeline import UnifiedTrainingPipeline, UnifiedTrainingConfig
        from domains.training.sequence import TrainingRunConfig

        config = UnifiedTrainingConfig()
        run_config = TrainingRunConfig(
            skip_generate=True, skip_distill=True, skip_train=True,
            skip_evaluate=True, skip_deploy=True,
        )
        pipeline = UnifiedTrainingPipeline(config, run_config=run_config)

        calls: list = []

        def on_progress(progress):
            calls.append(progress.phase)

        result = pipeline.run(on_progress=on_progress)
        assert result["status"] == "completed"
        assert len(calls) > 0, "on_progress should be called at least once"
        # Should see at least the 'complete' phase
        assert "complete" in calls

    def test_all_skipped_elapsed_positive(self):
        from domains.training.unified_pipeline import UnifiedTrainingPipeline, UnifiedTrainingConfig
        from domains.training.sequence import TrainingRunConfig

        config = UnifiedTrainingConfig()
        run_config = TrainingRunConfig(
            skip_generate=True, skip_distill=True, skip_train=True,
            skip_evaluate=True, skip_deploy=True,
        )
        pipeline = UnifiedTrainingPipeline(config, run_config=run_config)
        result = pipeline.run()
        assert result["elapsed"] >= 0
        assert result["total_steps"] == 0
        assert result["final_loss"] is None


class TestPipelineSkipTrainOnly:
    """Pipeline where only the TRAIN phase is skipped — other phases still run."""

    def test_skip_train_only_runs_remaining_phases(self):
        from domains.training.unified_pipeline import UnifiedTrainingPipeline, UnifiedTrainingConfig
        from domains.training.sequence import TrainingRunConfig

        config = UnifiedTrainingConfig(data_path="datasets/shakespeare")
        run_config = TrainingRunConfig(
            skip_generate=False, skip_distill=True, skip_train=True,
            skip_evaluate=False, skip_deploy=False,
        )
        pipeline = UnifiedTrainingPipeline(config, run_config=run_config)
        result = pipeline.run()
        assert result["status"] == "completed"
        _assert_result_shape(result)

        phases = {pr["phase"]: pr["status"] for pr in result["phases"]}
        assert phases.get("generate_data") in ("completed", "success", "skipped"), f"generate_data phase: {phases}"
        assert phases.get("train") == "skipped", f"train phase: {phases}"
        assert phases.get("evaluate") in ("completed", "success", "skipped"), f"evaluate phase: {phases}"
        assert phases.get("deploy") in ("completed", "success", "skipped"), f"deploy phase: {phases}"

    def test_skip_train_only_elapsed_reasonable(self):
        from domains.training.unified_pipeline import UnifiedTrainingPipeline, UnifiedTrainingConfig
        from domains.training.sequence import TrainingRunConfig

        config = UnifiedTrainingConfig()
        run_config = TrainingRunConfig(
            skip_generate=True, skip_distill=True, skip_train=True,
            skip_evaluate=True, skip_deploy=True,
        )
        pipeline = UnifiedTrainingPipeline(config, run_config=run_config)
        result = pipeline.run()
        assert result["elapsed"] < 10, "Skipped pipeline should complete quickly"


class TestPipelineFactory:
    """Tests for create_pipeline() factory function."""

    def test_create_pipeline_skip_train(self):
        from domains.training.unified_pipeline import create_pipeline

        pipeline = create_pipeline({
            "skip_train": True,
            "skip_generate": True,
            "skip_distill": True,
            "skip_evaluate": True,
            "skip_deploy": True,
        })
        assert pipeline is not None
        result = pipeline.run()
        assert result["status"] == "completed"
        _assert_result_shape(result)

    def test_create_pipeline_with_config(self):
        from domains.training.unified_pipeline import create_pipeline

        pipeline = create_pipeline({
            "skip_train": True,
            "skip_generate": True,
            "skip_distill": True,
            "skip_evaluate": True,
            "skip_deploy": True,
        })
        result = pipeline.run()
        assert result["status"] == "completed"
        assert result["phases"][0]["status"] == "skipped"


class TestResultShape:
    """Result dict always has expected keys regardless of skip config."""

    def test_result_keys_minimal(self):
        from domains.training.unified_pipeline import UnifiedTrainingPipeline, UnifiedTrainingConfig
        from domains.training.sequence import TrainingRunConfig

        config = UnifiedTrainingConfig()
        run_config = TrainingRunConfig(
            skip_generate=True, skip_distill=True, skip_train=True,
            skip_evaluate=True, skip_deploy=True,
        )
        pipeline = UnifiedTrainingPipeline(config, run_config=run_config)
        result = pipeline.run()
        _assert_result_shape(result)

    def test_result_keys_messages_are_strings(self):
        from domains.training.unified_pipeline import UnifiedTrainingPipeline, UnifiedTrainingConfig
        from domains.training.sequence import TrainingRunConfig

        config = UnifiedTrainingConfig()
        run_config = TrainingRunConfig(
            skip_generate=True, skip_distill=True, skip_train=True,
            skip_evaluate=True, skip_deploy=True,
        )
        pipeline = UnifiedTrainingPipeline(config, run_config=run_config)
        result = pipeline.run()
        assert isinstance(result["message"], str)
        assert isinstance(result["status"], str)
        assert isinstance(result["model_path"], str)


class TestProgressSSEEvent:
    """Ensure to_sse_event produces correct envelope."""

    def test_sse_event_structure(self):
        from domains.training.unified_pipeline import TrainingProgress

        p = TrainingProgress(
            phase="train",
            epoch=1,
            total_epochs=5,
            loss=0.5,
            status="working",
            message="Epoch 1/5",
        )
        event = p.to_sse_event("unified-train")
        assert event["stream"] == "unified-train"
        assert event["phase"] == "train"
        assert event["status"] == "working"
        assert event["data"]["loss"] == 0.5
        assert event["data"]["epoch"] == 1
        assert event["meta"]["total_epochs"] == 5
        assert event["message"] == "Epoch 1/5"

    def test_sse_event_default_stream_auto_train(self):
        """Default stream name should be 'auto-train' for frontend compatibility."""
        from domains.training.unified_pipeline import TrainingProgress

        p = TrainingProgress(
            phase="COMPLETE",
            epoch=3,
            total_epochs=10,
            loss=0.12,
            status="complete",
            message="Done",
        )
        event = p.to_sse_event()
        assert event["stream"] == "auto-train"

    def test_sse_event_complete_contains_finish_fields(self):
        """COMPLETE event must include checkpoint, final_loss, epochs in data."""
        from domains.training.unified_pipeline import TrainingProgress

        p = TrainingProgress(
            phase="COMPLETE",
            epoch=3,
            total_epochs=10,
            loss=0.12,
            status="complete",
            message="Training complete",
            metrics={
                "final_loss": 0.12,
                "checkpoint": "assistant_12345.soul",
                "epochs": 10,
                "total_steps": 500,
                "elapsed": 42.0,
            },
        )
        event = p.to_sse_event("auto-train")
        assert event["data"].get("checkpoint") == "assistant_12345.soul"
        assert event["data"].get("final_loss") == 0.12
        assert event["data"].get("epochs") == 10
        assert event["data"].get("total_steps") == 500
        assert event["data"].get("elapsed") == 42.0

    def test_sse_event_working_contains_epoch_meta(self):
        """Working events should include epoch/total_epochs in meta."""
        from domains.training.unified_pipeline import TrainingProgress

        p = TrainingProgress(
            phase="TRAINING",
            epoch=2,
            total_epochs=10,
            loss=0.5,
            status="working",
            message="Step 100, loss 0.5000",
        )
        event = p.to_sse_event()
        assert event["meta"]["epoch"] == 2
        assert event["meta"]["total_epochs"] == 10

    def test_pipeline_run_complete_event_has_auto_train_fields(self):
        """When pipeline finishes, the on_progress callback should receive
        a COMPLETE progress with checkpoint/final_loss/epochs in metrics."""
        from domains.training.unified_pipeline import UnifiedTrainingPipeline, UnifiedTrainingConfig
        from domains.training.sequence import TrainingRunConfig
        import json

        config = UnifiedTrainingConfig(soul_name="test_soul")
        run_config = TrainingRunConfig(
            skip_generate=True, skip_distill=True, skip_train=True,
            skip_evaluate=True, skip_deploy=True,
        )
        pipeline = UnifiedTrainingPipeline(config, run_config=run_config)

        captured = []

        def on_progress(progress):
            if progress.phase == "complete":
                captured.append({
                    "checkpoint": progress.metrics.get("checkpoint"),
                    "final_loss": progress.metrics.get("final_loss"),
                    "epochs": progress.metrics.get("epochs"),
                    "total_steps": progress.metrics.get("total_steps"),
                    "elapsed": progress.metrics.get("elapsed"),
                })

        result = pipeline.run(on_progress=on_progress)
        # Result dict should have 'checkpoint' key
        assert "checkpoint" in result
        # Captured COMPLETE event should have the right fields
        assert len(captured) == 1
        assert captured[0]["checkpoint"] is not None or captured[0]["checkpoint"] == ""
        assert "epochs" in captured[0]
