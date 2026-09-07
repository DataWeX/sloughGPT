"""Tests for training.auto_trainer — AutoTrainer class."""

from __future__ import annotations

import pytest
import time
import tempfile
import os
from domains.training.auto_trainer import AutoTrainer


# ── AutoTrainer ─────────────────────────────────────────────────────────────


class TestAutoTrainer:

    def test_init(self):
        trainer = AutoTrainer()
        assert trainer.threshold == 10
        assert trainer.interval_s == 300

    def test_init_custom(self):
        trainer = AutoTrainer(threshold=5, interval_s=60)
        assert trainer.threshold == 5
        assert trainer.interval_s == 60

    def test_properties(self):
        trainer = AutoTrainer()
        assert trainer._last_train_ts == 0
        assert trainer._last_train_loss == 0
        assert trainer._conversation_count == 0

    def test_start_stop(self):
        trainer = AutoTrainer()
        trainer.start()
        assert trainer._thread is not None
        assert trainer._thread.is_alive()
        trainer.stop()
        assert not trainer._thread.is_alive()

    def test_stop_without_start(self):
        trainer = AutoTrainer()
        trainer.stop()

    def test_stats(self):
        trainer = AutoTrainer()
        assert hasattr(trainer, 'threshold')
        assert hasattr(trainer, '_total_trains')
