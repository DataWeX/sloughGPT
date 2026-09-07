"""Tests for training.state — training state management."""

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
    set_cancel_event,
    get_cancel_event,
)


# ── TrainingState ───────────────────────────────────────────────────────────


class TestTrainingState:

    def test_default(self):
        state = TrainingState()
        assert state.running is False
        assert state.config == {}
        assert state.student_net is None

    def test_custom(self):
        state = TrainingState(running=True, config={"lr": 0.001})
        assert state.running is True
        assert state.config["lr"] == 0.001


# ── get_state ───────────────────────────────────────────────────────────────


class TestGetState:

    def test_returns_state(self):
        state = get_state()
        assert isinstance(state, TrainingState)

    def test_singleton(self):
        state1 = get_state()
        state2 = get_state()
        assert state1 is state2


# ── get_turbo_state ────────────────────────────────────────────────────────


class TestGetTurboState:

    def test_returns_dict(self):
        state = get_turbo_state()
        assert isinstance(state, dict)
        assert "status" in state
        assert "job_id" in state


# ── get_turbo_lock ─────────────────────────────────────────────────────────


class TestGetTurboLock:

    def test_returns_lock(self):
        lock = get_turbo_lock()
        assert hasattr(lock, 'acquire')
        assert hasattr(lock, 'release')


# ── get_turbo_pause_event ──────────────────────────────────────────────────


class TestGetTurboPauseEvent:

    def test_returns_event(self):
        event = get_turbo_pause_event()
        assert isinstance(event, threading.Event)


# ── get_turbo_cancel_event ─────────────────────────────────────────────────


class TestGetTurboCancelEvent:

    def test_returns_event(self):
        event = get_turbo_cancel_event()
        assert isinstance(event, threading.Event)


# ── set_cancel_event / get_cancel_event ────────────────────────────────────


class TestCancelEvent:

    def test_set_get(self):
        event = threading.Event()
        set_cancel_event(event)
        result = get_cancel_event()
        assert result is event
