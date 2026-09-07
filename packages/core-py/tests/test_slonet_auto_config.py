"""Tests for training.auto_config — auto configuration engine."""

from __future__ import annotations

import pytest
from domains.training.auto_config import (
    DatasetAnalysis,
    TrainingConfig,
)


# ── DatasetAnalysis ────────────────────────────────────────────────────────


class TestDatasetAnalysis:

    def test_init(self):
        analysis = DatasetAnalysis(path="test.txt", format="text")
        assert analysis.path == "test.txt"
        assert analysis.format == "text"

    def test_properties(self):
        analysis = DatasetAnalysis(path="test.txt", format="text")
        assert analysis.is_dialogue is False
        assert analysis.is_messages_format is False

    def test_size_category(self):
        analysis = DatasetAnalysis(path="test.txt", format="text", word_count=500)
        assert analysis.size_category == "tiny"

    def test_size_category_small(self):
        analysis = DatasetAnalysis(path="test.txt", format="text", word_count=5000)
        assert analysis.size_category == "small"

    def test_size_category_medium(self):
        analysis = DatasetAnalysis(path="test.txt", format="text", word_count=50000)
        assert analysis.size_category == "medium"

    def test_size_category_large(self):
        analysis = DatasetAnalysis(path="test.txt", format="text", word_count=500000)
        assert analysis.size_category == "large"

    def test_dialogue_markers(self):
        analysis = DatasetAnalysis(path="test.txt", format="text", has_dialogue_markers=True)
        assert analysis.is_dialogue is True

    def test_role_fields(self):
        analysis = DatasetAnalysis(path="test.txt", format="messages", has_role_fields=True)
        assert analysis.is_dialogue is True
        assert analysis.is_messages_format is True


# ── TrainingConfig ─────────────────────────────────────────────────────────


class TestTrainingConfig:

    def test_default(self):
        config = TrainingConfig()
        assert config.model == "gpt2"
        assert config.method == "finetune"
        assert config.use_lora is True

    def test_to_dict(self):
        config = TrainingConfig()
        d = config.to_dict()
        assert "model" in d
        assert "epochs" in d
        assert "learning_rate" in d
