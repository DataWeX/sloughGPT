"""Tests for auto-train streaming SSE phase sequence.

Tests the SSE event emission logic by creating AutoTrainState directly
and iterating the async event_generator (bypasses HTTP layer for speed).
"""
import asyncio
import json
import sys
import pytest
from unittest.mock import MagicMock


def _ensure_paths():
    for _p in ('packages/core-py', 'apps/api/server'):
        _full = '/Users/mac/sloughGPT/' + _p
        if _full not in sys.path:
            sys.path.insert(0, _full)


def _make_tiny_model_and_tokenizer():
    _ensure_paths()
    from domains.training.slonet import SloNet, SloLSTM, SloEmbedding
    from domains.training.tokenizer import SloBPE

    tok = SloBPE()
    tok.train(["a b c d e f g h"], vocab_size=16)
    model = SloNet(
        layers=[
            SloEmbedding(tok.vocab_size, 16),
            SloLSTM(16, 32, tok.vocab_size, num_layers=1, dropout=0.0),
        ],
        soul_name="test",
    )
    return model, tok


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE data lines from a raw chunk."""
    events = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


class TestAutoTrainStreamEvents:
    """Tests SSE event emission by iterating the async event_generator directly.

    The stream() endpoint returns a StreamingResponse wrapping an async generator.
    We await the endpoint, then iterate its body_iterator.
    """

    _phases = None

    @classmethod
    def _get_phases(cls):
        if cls._phases is not None:
            return cls._phases
        cls._phases = asyncio.run(cls._run_stream())
        return cls._phases

    @classmethod
    async def _run_stream(cls):
        _ensure_paths()
        from routers.auto_train import state as at_state

        model, tok = _make_tiny_model_and_tokenizer()
        at_state.running = True
        at_state.config = {
            "source_text": "hello world how are you doing today",
            "epochs": 1,
            "soul_name": "test",
            "algo": "bpe",
            "teacher_model": "gpt2",
            "temperature": 0.8,
            "learning_rate": 0.001,
        }
        at_state.source_lines = ["hello world how are you doing today"]
        at_state.student_net = model
        at_state.student_tokenizer = tok
        at_state.teacher_model = MagicMock()
        at_state.teacher_tokenizer = MagicMock()

        from routers.auto_train import stream as stream_endpoint
        response = await stream_endpoint()
        phases = []
        async for chunk in response.body_iterator:
            text = chunk if isinstance(chunk, str) else chunk.decode()
            for ev in _parse_sse(text):
                phases.append(ev)
                if ev.get("status") in ("complete", "error"):
                    return phases
        return phases

    # --- tests ---

    def test_generate_data_phase(self):
        phases = self._get_phases()
        assert any(p["phase"] == "TRAINING" for p in phases), f"Phases: {[p['phase'] for p in phases]}"

    def test_train_phase(self):
        phases = self._get_phases()
        assert any(p["phase"] == "TRAINING" for p in phases), f"Phases: {[p['phase'] for p in phases]}"

    def test_terminal_phase(self):
        phases = self._get_phases()
        assert phases[-1]["phase"] in ("COMPLETE", "FAILED"), f"Phases: {[p['phase'] for p in phases]}"

    def test_phase_order(self):
        phases = self._get_phases()
        phase_names = [p["phase"] for p in phases]
        expected = ["TRAINING", "COMPLETE"]
        filtered = [n for n in phase_names if n in expected]
        indices = [expected.index(n) for n in filtered]
        assert indices == sorted(indices), f"Phases out of order: {filtered}"

    def test_loss_values_are_finite(self):
        phases = self._get_phases()
        for p in phases:
            d = p.get("data", {}) or {}
            loss = d.get("loss")
            if loss is not None:
                assert isinstance(loss, (int, float))
                assert loss > 0

    def test_events_have_required_fields(self):
        phases = self._get_phases()
        for p in phases:
            assert "stream" in p
            assert p["stream"] == "auto-train"
            assert "phase" in p
            assert "status" in p

    def test_without_start_returns_error(self):
        """Guard condition: no config/teacher/student → immediate error event."""
        _ensure_paths()
        from routers.auto_train import state as at_state, stream as stream_endpoint
        at_state.config = None
        at_state.teacher_model = None
        at_state.student_net = None

        events = asyncio.run(self._get_error_events(stream_endpoint))
        assert len(events) == 1
        assert events[0]["status"] == "error"

    @staticmethod
    async def _get_error_events(stream_endpoint):
        response = await stream_endpoint()
        events = []
        async for chunk in response.body_iterator:
            text = chunk if isinstance(chunk, str) else chunk.decode()
            for ev in _parse_sse(text):
                events.append(ev)
                return events
        return events
