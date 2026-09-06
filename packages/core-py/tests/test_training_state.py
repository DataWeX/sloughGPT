"""Tests for training.state — training state management singletons."""

from __future__ import annotations

import threading

import pytest

from domains.training.state import (
    TrainingState,
    get_state,
    get_turbo_state,
    get_turbo_lock,
    get_turbo_pause_event,
    get_turbo_cancel_event,
    get_cancel_event,
    set_cancel_event,
    get_pause_event,
    set_pause_event,
    get_pgq,
    VALID_CKPT_NAME,
    SOU_MAGIC,
    MAX_CHECKPOINT_DISK_MB,
)


# ── TrainingState dataclass ───────────────────────────────────────────────


class TestTrainingState:

    def test_defaults(self):
        s = TrainingState()
        assert s.running is False
        assert s.config == {}
        assert s.student_net is None
        assert s.student_tokenizer is None
        assert s.complete_enqueued is False

    def test_custom_values(self):
        s = TrainingState(running=True, config={"epochs": 5})
        assert s.running is True
        assert s.config == {"epochs": 5}

    def test_config_not_shared_between_instances(self):
        s1 = TrainingState()
        s2 = TrainingState()
        s1.config["key"] = "value"
        assert "key" not in s2.config


# ── Singleton accessors ───────────────────────────────────────────────────


class TestStateAccessors:

    def test_get_state_returns_same_instance(self):
        assert get_state() is get_state()

    def test_get_state_is_training_state(self):
        assert isinstance(get_state(), TrainingState)

    def test_get_turbo_state_returns_dict(self):
        ts = get_turbo_state()
        assert isinstance(ts, dict)
        assert "status" in ts
        assert "job_id" in ts

    def test_get_turbo_state_has_expected_keys(self):
        ts = get_turbo_state()
        expected = {
            "status", "job_id", "global_step", "total_steps", "progress",
            "loss", "learning_rate", "steps_per_sec", "eta_s", "elapsed_s",
            "avg_quality", "result", "error", "paused", "last_heartbeat",
        }
        assert expected == set(ts.keys())

    def test_get_turbo_lock_returns_lock(self):
        lock = get_turbo_lock()
        assert isinstance(lock, type(threading.Lock()))

    def test_get_turbo_lock_returns_same_instance(self):
        assert get_turbo_lock() is get_turbo_lock()

    def test_get_turbo_pause_event_returns_event(self):
        ev = get_turbo_pause_event()
        assert isinstance(ev, threading.Event)

    def test_get_turbo_pause_event_returns_same_instance(self):
        assert get_turbo_pause_event() is get_turbo_pause_event()

    def test_get_turbo_cancel_event_returns_event(self):
        ev = get_turbo_cancel_event()
        assert isinstance(ev, threading.Event)

    def test_get_turbo_cancel_event_returns_same_instance(self):
        assert get_turbo_cancel_event() is get_turbo_cancel_event()


# ── Cancel / pause events ────────────────────────────────────────────────


class TestCancelPauseEvents:

    def setup_method(self):
        set_cancel_event(None)
        set_pause_event(None)

    def teardown_method(self):
        set_cancel_event(None)
        set_pause_event(None)

    def test_get_cancel_event_default_none(self):
        assert get_cancel_event() is None

    def test_set_cancel_event(self):
        ev = threading.Event()
        set_cancel_event(ev)
        assert get_cancel_event() is ev

    def test_get_pause_event_default_none(self):
        assert get_pause_event() is None

    def test_set_pause_event(self):
        ev = threading.Event()
        set_pause_event(ev)
        assert get_pause_event() is ev


# ── Constants ─────────────────────────────────────────────────────────────


class TestConstants:

    def test_sou_magic(self):
        assert SOU_MAGIC == b"SOUL"

    def test_max_checkpoint_disk_mb(self):
        assert MAX_CHECKPOINT_DISK_MB == 500

    def test_valid_ckpt_name_regex(self):
        assert VALID_CKPT_NAME.match("my-checkpoint")
        assert VALID_CKPT_NAME.match("model_v1.soul")
        assert VALID_CKPT_NAME.match("test123")
        assert not VALID_CKPT_NAME.match("has spaces")
        assert not VALID_CKPT_NAME.match("has/slash")
        assert not VALID_CKPT_NAME.match("has@special!")


# ── PGQ singleton ─────────────────────────────────────────────────────────


class TestPGQ:

    def test_get_pgq_returns_something_or_none(self):
        result = get_pgq()
        assert result is None or hasattr(result, "name")
