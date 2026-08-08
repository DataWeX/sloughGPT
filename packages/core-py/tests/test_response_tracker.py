"""Tests for domains/feedback/response_tracker.py — response logging for benchmarking."""

import json
import tempfile
import pytest
from pathlib import Path
from domains.feedback.response_tracker import ResponseTracker, ResponseLog, get_response_tracker


class TestResponseLog:
    def test_dataclass_fields(self):
        log = ResponseLog(
            timestamp="2024-01-01T00:00:00",
            user_message="hi",
            assistant_response="hello",
            model="gpt2",
            temperature=0.8,
            max_tokens=256,
            session_id="s1",
            user_id="u1",
            tokens_generated=10,
            duration_ms=100.0,
        )
        assert log.user_message == "hi"
        assert log.tokens_generated == 10
        assert log.has_images is False
        assert log.context_tokens == 0
        assert log.eval_scores is None

    def test_optional_fields(self):
        log = ResponseLog(
            timestamp="t", user_message="u", assistant_response="a",
            model="m", temperature=0.5, max_tokens=128,
            session_id="s", user_id="u", tokens_generated=5,
            duration_ms=50.0, has_images=True, context_tokens=100,
            eval_scores={"bleu": 0.8},
        )
        assert log.has_images is True
        assert log.context_tokens == 100
        assert log.eval_scores == {"bleu": 0.8}


class TestResponseTracker:
    def _make_tracker(self, tmp_path):
        return ResponseTracker(log_dir=str(tmp_path / "logs"))

    def test_log_returns_entry(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        entry = tracker.log(
            user_message="Q",
            assistant_response="A",
            model="gpt2",
            config={"temperature": 0.8, "max_tokens": 256},
            session_id="s1",
            user_id="u1",
            tokens_generated=5,
            duration_ms=100.0,
        )
        assert isinstance(entry, ResponseLog)
        assert entry.user_message == "Q"
        assert entry.model == "gpt2"
        assert entry.temperature == 0.8

    def test_log_flushes_to_file(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.log(
            user_message="Q1", assistant_response="A1", model="gpt2",
            config={}, session_id="s1", user_id="u1",
            tokens_generated=1, duration_ms=10.0,
        )
        assert tracker.current_file.exists()
        lines = tracker.current_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["user_message"] == "Q1"

    def test_get_responses(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.log(
            user_message="Q1", assistant_response="A1", model="gpt2",
            config={}, session_id="s1", user_id="u1",
            tokens_generated=1, duration_ms=10.0,
        )
        tracker.log(
            user_message="Q2", assistant_response="A2", model="qwen",
            config={}, session_id="s2", user_id="u2",
            tokens_generated=2, duration_ms=20.0,
        )
        responses = tracker.get_responses()
        assert len(responses) == 2

    def test_get_responses_filter_by_model(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.log(
            user_message="Q1", assistant_response="A1", model="gpt2",
            config={}, session_id="s1", user_id="u1",
            tokens_generated=1, duration_ms=10.0,
        )
        tracker.log(
            user_message="Q2", assistant_response="A2", model="qwen",
            config={}, session_id="s2", user_id="u2",
            tokens_generated=1, duration_ms=10.0,
        )
        gpt2 = tracker.get_responses(model="gpt2")
        assert len(gpt2) == 1
        assert gpt2[0].model == "gpt2"

    def test_get_responses_limit(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        for i in range(5):
            tracker.log(
                user_message=f"Q{i}", assistant_response=f"A{i}", model="gpt2",
                config={}, session_id="s1", user_id="u1",
                tokens_generated=1, duration_ms=10.0,
            )
        responses = tracker.get_responses(limit=2)
        assert len(responses) == 2

    def test_get_responses_since(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.log(
            user_message="Q1", assistant_response="A1", model="gpt2",
            config={}, session_id="s1", user_id="u1",
            tokens_generated=1, duration_ms=10.0,
        )
        tracker.log(
            user_message="Q2", assistant_response="A2", model="gpt2",
            config={}, session_id="s2", user_id="u2",
            tokens_generated=1, duration_ms=10.0,
        )
        all_resps = tracker.get_responses()
        cutoff = all_resps[1].timestamp
        filtered = tracker.get_responses(since=cutoff)
        assert len(filtered) == 1
        assert filtered[0].user_message == "Q2"

    def test_get_responses_empty_file(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        assert tracker.get_responses() == []

    def test_get_responses_skips_corrupt_lines(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.current_file.write_text("not json\n{\"user_message\":\"ok\",\"assistant_response\":\"a\",\"model\":\"m\",\"temperature\":0.8,\"max_tokens\":256,\"session_id\":\"s\",\"user_id\":\"u\",\"tokens_generated\":1,\"duration_ms\":10.0,\"timestamp\":\"t\"}\n")
        responses = tracker.get_responses()
        assert len(responses) == 1

    def test_log_has_images_and_context(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        entry = tracker.log(
            user_message="Q", assistant_response="A", model="gpt2",
            config={}, session_id="s", user_id="u",
            tokens_generated=1, duration_ms=10.0,
            has_images=True, context_tokens=200,
        )
        assert entry.has_images is True
        assert entry.context_tokens == 200

    def test_export_jsonl(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.log(
            user_message="Q1", assistant_response="A1", model="gpt2",
            config={}, session_id="s1", user_id="u1",
            tokens_generated=1, duration_ms=10.0,
        )
        export_path = str(tmp_path / "export.jsonl")
        result = tracker.export_jsonl(export_path)
        assert result == export_path
        assert Path(export_path).exists()
        lines = Path(export_path).read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["session_id"] == "s1"

    def test_export_jsonl_multiple(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        for i in range(3):
            tracker.log(
                user_message=f"Q{i}", assistant_response=f"A{i}", model="gpt2",
                config={}, session_id=f"s{i}", user_id="u1",
                tokens_generated=i, duration_ms=10.0,
            )
        export_path = str(tmp_path / "export.jsonl")
        tracker.export_jsonl(export_path)
        lines = Path(export_path).read_text().strip().split("\n")
        assert len(lines) == 3

    def test_get_stats_empty(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        stats = tracker.get_stats()
        assert stats["total"] == 0

    def test_get_stats_with_data(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.log(
            user_message="Q", assistant_response="A", model="gpt2",
            config={}, session_id="s1", user_id="u1",
            tokens_generated=50, duration_ms=200.0,
        )
        stats = tracker.get_stats()
        assert stats["total"] == 1
        assert stats["avg_tokens"] == 50.0
        assert stats["avg_duration_ms"] == 200.0
        assert "gpt2" in stats["unique_models"]

    def test_get_stats_multiple_models(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.log(
            user_message="Q1", assistant_response="A1", model="gpt2",
            config={}, session_id="s1", user_id="u1",
            tokens_generated=10, duration_ms=100.0,
        )
        tracker.log(
            user_message="Q2", assistant_response="A2", model="qwen",
            config={}, session_id="s2", user_id="u2",
            tokens_generated=20, duration_ms=200.0,
        )
        stats = tracker.get_stats()
        assert stats["total"] == 2
        assert stats["avg_tokens"] == 15.0
        assert set(stats["unique_models"]) == {"gpt2", "qwen"}

    def test_default_config_values(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        entry = tracker.log(
            user_message="Q", assistant_response="A", model="gpt2",
            config={}, session_id="s1", user_id="u1",
            tokens_generated=1, duration_ms=10.0,
        )
        assert entry.temperature == 0.8
        assert entry.max_tokens == 256

    def test_custom_config_values(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        entry = tracker.log(
            user_message="Q", assistant_response="A", model="gpt2",
            config={"temperature": 0.2, "max_tokens": 512},
            session_id="s1", user_id="u1",
            tokens_generated=1, duration_ms=10.0,
        )
        assert entry.temperature == 0.2
        assert entry.max_tokens == 512

    def test_flush_empty_buffer(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker._flush()


class TestSingleton:
    def test_same_instance(self):
        a = get_response_tracker()
        b = get_response_tracker()
        assert a is b

    def test_singleton_type(self):
        t = get_response_tracker()
        assert isinstance(t, ResponseTracker)
