"""Tests for domains.training — PipelineConfig, PipelineStageType, TrainingPipeline."""

import asyncio
import pytest
from domains.training import (
    PipelineConfig, PipelineStageType, TrainingPipeline,
)


class TestPipelineStageType:
    def test_all_members(self):
        assert len(PipelineStageType) == 4
    def test_values(self):
        assert PipelineStageType.TRAIN.value == "train"
        assert PipelineStageType.VALIDATE.value == "validate"


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig(name="test")
        assert cfg.batch_size == 32
        assert cfg.epochs == 3
        assert cfg.learning_rate == 1e-4

    def test_custom(self):
        cfg = PipelineConfig(name="test", batch_size=64, epochs=10, learning_rate=0.001)
        assert cfg.batch_size == 64
        assert cfg.epochs == 10


class TestTrainingPipeline:
    def test_init(self):
        cfg = PipelineConfig(name="test")
        pipe = TrainingPipeline(cfg)
        assert pipe.config.name == "test"
        assert pipe.stages == []

    def test_add_stage(self):
        cfg = PipelineConfig(name="test")
        pipe = TrainingPipeline(cfg)
        result = pipe.add_stage("preprocess", PipelineStageType.PREPROCESS, lambda x: x)
        assert result is pipe
        assert len(pipe.stages) == 1

    def test_chaining(self):
        cfg = PipelineConfig(name="test")
        pipe = TrainingPipeline(cfg)
        pipe.add_stage("a", PipelineStageType.PREPROCESS, None)
        pipe.add_stage("b", PipelineStageType.TRAIN, None)
        assert len(pipe.stages) == 2

    def test_run(self):
        cfg = PipelineConfig(name="test", epochs=2)
        pipe = TrainingPipeline(cfg)
        pipe.add_stage("step1", PipelineStageType.TRAIN, None)
        result = asyncio.run(pipe.run(iter([])))
        assert result["epochs"] == 2
        assert len(result["stages"]) == 2
