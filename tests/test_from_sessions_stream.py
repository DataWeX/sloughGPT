"""Tests for /auto-train/from-sessions SSE endpoints."""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "core-py"))
sys.path.insert(0, str(Path(__file__).parent.parent / "apps" / "api" / "server"))


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def _mock_train_result():
    """Return a minimal (model, metadata) tuple from train_chat_model."""
    model = MagicMock()
    meta = {
        "checkpoint": "/tmp/test.soul",
        "final_loss": 2.5,
        "num_pairs": 8,
        "total_pairs": 10,
        "epochs_completed": 1,
        "vocab_size": 20,
        "train_losses": [3.0, 2.5],
        "val_losses": [3.1],
        "perplexity": 12.5,
        "samples": [{"prompt": "User: Hello", "response": "Hi there!"}],
        "avg_response_len": 3.0,
    }
    return model, meta


def _mock_train_fn(config=None, on_step=None, cancel_event=None):
    """Mock train_from_sessions that calls on_step before returning."""
    if on_step:
        on_step(1, 3.0, 0)
        on_step(2, 2.5, 0)
    return _mock_train_result()


class TestFromSessionsStart:
    """Test the POST /auto-train/from-sessions/start endpoint."""

    def test_start_sets_state(self):
        from routers.auto_train import start_from_sessions, FromSessionsRequest, state
        state.running = False
        state.config = {}
        req = FromSessionsRequest(epochs=3, soul_name="test-train")
        resp = asyncio.run(start_from_sessions(req))
        assert state.running is True
        assert state.config["method"] == "from-sessions"
        assert state.config["epochs"] == 3
        assert state.config["soul_name"] == "test-train"
        state.running = False
        state.config = {}

    def test_start_rejects_if_already_running(self):
        from routers.auto_train import start_from_sessions, FromSessionsRequest, state
        state.running = True
        state.config = {"method": "from-sessions"}
        req = FromSessionsRequest()
        resp = asyncio.run(start_from_sessions(req))
        state.running = False
        state.config = {}


class TestFromSessionsStream:
    """Test the GET /auto-train/from-sessions/stream SSE endpoint."""

    @classmethod
    def _get_phases(cls):
        if hasattr(cls, "_cached"):
            return cls._cached
        cls._cached = asyncio.run(cls._run_stream())
        return cls._cached

    @classmethod
    async def _run_stream(cls):
        from routers.auto_train import state as at_state, stream_from_sessions
        from unittest.mock import AsyncMock, patch

        from domains.infrastructure.training_queue import register_training_handlers
        register_training_handlers()

        at_state.running = True
        at_state.config = {
            "method": "from-sessions",
            "epochs": 1,
            "soul_name": "stream-test",
            "n_embed": 16,
            "n_layer": 1,
            "n_head": 2,
            "block_size": 16,
            "batch_size": 2,
            "min_pair_quality": 0.0,
            "max_pairs": 10,
        }

        mock_request = MagicMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        with patch("domains.training.chat_trainer.train_from_sessions", side_effect=_mock_train_fn):
            response = await stream_from_sessions(mock_request)
            phases = []
            async for chunk in response.body_iterator:
                text = chunk if isinstance(chunk, str) else chunk.decode()
                for ev in _parse_sse(text):
                    phases.append(ev)
                    if ev.get("status") in ("complete", "error"):
                        return phases
        return phases

    def test_pairs_phase(self):
        phases = self._get_phases()
        assert any(p["phase"] == "PAIRS" for p in phases), f"Phases: {[p['phase'] for p in phases]}"

    def test_train_phase(self):
        phases = self._get_phases()
        assert any(p["phase"] == "TRAIN" for p in phases), f"Phases: {[p['phase'] for p in phases]}"

    def test_complete_phase(self):
        phases = self._get_phases()
        assert any(p["phase"] == "COMPLETE" for p in phases), f"Phases: {[p['phase'] for p in phases]}"

    def test_phase_order(self):
        phases = self._get_phases()
        phase_names = [p["phase"] for p in phases]
        expected = ["PAIRS", "TRAIN", "COMPLETE"]
        filtered = [n for n in phase_names if n in expected]
        indices = [expected.index(n) for n in filtered]
        assert indices == sorted(indices), f"Phases out of order: {filtered}"

    def test_events_have_required_fields(self):
        phases = self._get_phases()
        for p in phases:
            assert "stream" in p
            assert p["stream"] == "auto-train"
            assert "phase" in p
            assert "status" in p

    def test_loss_values_are_finite(self):
        phases = self._get_phases()
        for p in phases:
            if p.get("status") in ("error",):
                continue
            d = p.get("data", {})
            loss = d.get("loss")
            if loss is not None:
                assert isinstance(loss, (int, float))
                assert loss >= 0

    def test_without_start_returns_error(self):
        from routers.auto_train import state as at_state, stream_from_sessions

        at_state.config = {}

        async def _get_error():
            mock_request = MagicMock()
            mock_request.is_disconnected = AsyncMock(return_value=False)
            response = await stream_from_sessions(mock_request)
            events = []
            async for chunk in response.body_iterator:
                text = chunk if isinstance(chunk, str) else chunk.decode()
                for ev in _parse_sse(text):
                    events.append(ev)
                    return events
            return events

        events = asyncio.run(_get_error())
        assert len(events) == 1
        assert events[0]["status"] == "error"


class TestCancelEndpoint:
    def test_cancel_sets_event(self):
        from routers.auto_train import cancel_from_sessions, _auto_train_cancel_event
        resp = asyncio.run(cancel_from_sessions())
        assert resp is not None


class TestFromSessionsRequestSchema:
    def test_defaults(self):
        from routers.auto_train import FromSessionsRequest
        req = FromSessionsRequest()
        assert req.epochs == 5
        assert req.soul_name == "chat-trained"
        assert req.min_pair_quality == 2.0

    def test_custom(self):
        from routers.auto_train import FromSessionsRequest
        req = FromSessionsRequest(epochs=20, n_embed=256, soul_name="custom")
        assert req.epochs == 20
        assert req.n_embed == 256
        assert req.soul_name == "custom"
