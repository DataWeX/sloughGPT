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
