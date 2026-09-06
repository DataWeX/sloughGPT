"""Tests for sessions — from-sessions training business logic."""

from __future__ import annotations

import time

import pytest

from domains.training.sessions import start_from_sessions_training
from domains.training.state import TrainingState


@pytest.fixture
def fresh_state():
    return TrainingState(running=False, config={})


# ── start_from_sessions_training ───────────────────────────────────────────


class TestStartFromSessionsTraining:

    def test_sets_running_to_true(self, fresh_state):
        start_from_sessions_training(fresh_state, {})
        assert fresh_state.running is True

    def test_returns_config_dict(self, fresh_state):
        result = start_from_sessions_training(fresh_state, {})
        assert isinstance(result, dict)
        assert result["method"] == "from-sessions"

    def test_applies_defaults(self, fresh_state):
        result = start_from_sessions_training(fresh_state, {})
        assert result["epochs"] == 5
        assert result["learning_rate"] == 3e-4
        assert result["batch_size"] == 8
        assert result["n_embed"] == 128
        assert result["n_layer"] == 4
        assert result["n_head"] == 4
        assert result["block_size"] == 128
        assert result["dropout"] == 0.1
        assert result["soul_name"] == "chat-trained"
        assert result["min_pair_quality"] == 2.0
        assert result["max_pairs"] == 500

    def test_overrides_from_config(self, fresh_state):
        cfg = {
            "epochs": 10,
            "learning_rate": 1e-3,
            "batch_size": 16,
            "n_embed": 256,
            "n_layer": 8,
            "n_head": 8,
            "block_size": 256,
            "dropout": 0.2,
            "soul_name": "custom",
            "min_pair_quality": 3.0,
            "max_pairs": 1000,
        }
        result = start_from_sessions_training(fresh_state, cfg)
        assert result["epochs"] == 10
        assert result["learning_rate"] == 1e-3
        assert result["batch_size"] == 16
        assert result["n_embed"] == 256
        assert result["n_layer"] == 8
        assert result["n_head"] == 8
        assert result["block_size"] == 256
        assert result["dropout"] == 0.2
        assert result["soul_name"] == "custom"
        assert result["min_pair_quality"] == 3.0
        assert result["max_pairs"] == 1000

    def test_passes_through_optional_fields(self, fresh_state):
        cfg = {
            "checkpoint_name": "my-checkpoint",
            "session_ids": ["s1", "s2"],
            "experiment_id": "exp-42",
        }
        result = start_from_sessions_training(fresh_state, cfg)
        assert result["checkpoint_name"] == "my-checkpoint"
        assert result["session_ids"] == ["s1", "s2"]
        assert result["experiment_id"] == "exp-42"

    def test_started_at_is_recent(self, fresh_state):
        before = time.time()
        result = start_from_sessions_training(fresh_state, {})
        after = time.time()
        assert before <= result["started_at"] <= after

    def test_raises_when_already_running(self, fresh_state):
        fresh_state.running = True
        with pytest.raises(RuntimeError, match="already in progress"):
            start_from_sessions_training(fresh_state, {})

    def test_stores_config_on_state(self, fresh_state):
        result = start_from_sessions_training(fresh_state, {"epochs": 3})
        assert fresh_state.config == result
