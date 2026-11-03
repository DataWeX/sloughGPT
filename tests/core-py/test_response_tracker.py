"""Tests for domain.feedback._internal.response_tracker.py — response logging for benchmarking."""

import json
from pathlib import Path

from domain.feedback._internal.response_tracker import (
    ResponseLog,
    ResponseTracker,
    _canonical_log_dir,
    get_response_tracker,
    read_response_docs,
)


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
            timestamp="t",
            user_message="u",
            assistant_response="a",
            model="m",
            temperature=0.5,
            max_tokens=128,
            session_id="s",
            user_id="u",
            tokens_generated=5,
            duration_ms=50.0,
            has_images=True,
            context_tokens=100,
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
            user_message="Q1",
            assistant_response="A1",
            model="gpt2",
            config={},
            session_id="s1",
            user_id="u1",
            tokens_generated=1,
            duration_ms=10.0,
        )
        assert tracker.current_file.exists()
        lines = tracker.current_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["user_message"] == "Q1"

    def test_get_responses(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.log(
            user_message="Q1",
            assistant_response="A1",
            model="gpt2",
            config={},
            session_id="s1",
            user_id="u1",
            tokens_generated=1,
            duration_ms=10.0,
        )
        tracker.log(
            user_message="Q2",
            assistant_response="A2",
            model="qwen",
            config={},
            session_id="s2",
            user_id="u2",
            tokens_generated=2,
            duration_ms=20.0,
        )
        responses = tracker.get_responses()
        assert len(responses) == 2

    def test_get_responses_filter_by_model(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.log(
            user_message="Q1",
            assistant_response="A1",
            model="gpt2",
            config={},
            session_id="s1",
            user_id="u1",
            tokens_generated=1,
            duration_ms=10.0,
        )
        tracker.log(
            user_message="Q2",
            assistant_response="A2",
            model="qwen",
            config={},
            session_id="s2",
            user_id="u2",
            tokens_generated=1,
            duration_ms=10.0,
        )
        gpt2 = tracker.get_responses(model="gpt2")
        assert len(gpt2) == 1
        assert gpt2[0].model == "gpt2"

    def test_get_responses_limit(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        for i in range(5):
            tracker.log(
                user_message=f"Q{i}",
                assistant_response=f"A{i}",
                model="gpt2",
                config={},
                session_id="s1",
                user_id="u1",
                tokens_generated=1,
                duration_ms=10.0,
            )
        responses = tracker.get_responses(limit=2)
        assert len(responses) == 2

    def test_get_responses_since(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.log(
            user_message="Q1",
            assistant_response="A1",
            model="gpt2",
            config={},
            session_id="s1",
            user_id="u1",
            tokens_generated=1,
            duration_ms=10.0,
        )
        tracker.log(
            user_message="Q2",
            assistant_response="A2",
            model="gpt2",
            config={},
            session_id="s2",
            user_id="u2",
            tokens_generated=1,
            duration_ms=10.0,
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
        tracker.current_file.write_text(
            'not json\n{"user_message":"ok","assistant_response":"a","model":"m","temperature":0.8,"max_tokens":256,"session_id":"s","user_id":"u","tokens_generated":1,"duration_ms":10.0,"timestamp":"t"}\n'
        )
        responses = tracker.get_responses()
        assert len(responses) == 1

    def test_log_has_images_and_context(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        entry = tracker.log(
            user_message="Q",
            assistant_response="A",
            model="gpt2",
            config={},
            session_id="s",
            user_id="u",
            tokens_generated=1,
            duration_ms=10.0,
            has_images=True,
            context_tokens=200,
        )
        assert entry.has_images is True
        assert entry.context_tokens == 200

    def test_export_jsonl(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.log(
            user_message="Q1",
            assistant_response="A1",
            model="gpt2",
            config={},
            session_id="s1",
            user_id="u1",
            tokens_generated=1,
            duration_ms=10.0,
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
                user_message=f"Q{i}",
                assistant_response=f"A{i}",
                model="gpt2",
                config={},
                session_id=f"s{i}",
                user_id="u1",
                tokens_generated=i,
                duration_ms=10.0,
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
            user_message="Q",
            assistant_response="A",
            model="gpt2",
            config={},
            session_id="s1",
            user_id="u1",
            tokens_generated=50,
            duration_ms=200.0,
        )
        stats = tracker.get_stats()
        assert stats["total"] == 1
        assert stats["avg_tokens"] == 50.0
        assert stats["avg_duration_ms"] == 200.0
        assert "gpt2" in stats["unique_models"]

    def test_get_stats_multiple_models(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.log(
            user_message="Q1",
            assistant_response="A1",
            model="gpt2",
            config={},
            session_id="s1",
            user_id="u1",
            tokens_generated=10,
            duration_ms=100.0,
        )
        tracker.log(
            user_message="Q2",
            assistant_response="A2",
            model="qwen",
            config={},
            session_id="s2",
            user_id="u2",
            tokens_generated=20,
            duration_ms=200.0,
        )
        stats = tracker.get_stats()
        assert stats["total"] == 2
        assert stats["avg_tokens"] == 15.0
        assert set(stats["unique_models"]) == {"gpt2", "qwen"}

    def test_default_config_values(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        entry = tracker.log(
            user_message="Q",
            assistant_response="A",
            model="gpt2",
            config={},
            session_id="s1",
            user_id="u1",
            tokens_generated=1,
            duration_ms=10.0,
        )
        assert entry.temperature == 0.8
        assert entry.max_tokens == 256

    def test_custom_config_values(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        entry = tracker.log(
            user_message="Q",
            assistant_response="A",
            model="gpt2",
            config={"temperature": 0.2, "max_tokens": 512},
            session_id="s1",
            user_id="u1",
            tokens_generated=1,
            duration_ms=10.0,
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


class TestMogDBIsolation:
    def test_custom_log_dir_gets_sibling_journal(self, tmp_path):
        """A non-canonical log dir journals to <name>_mogdb, never the canonical store."""
        tracker = ResponseTracker(log_dir=str(tmp_path / "logs"))
        assert tracker._mogdb is not None
        # The journal lives beside the log dir, not under the repo's data/.
        sibling = tmp_path / "logs_mogdb"
        assert sibling.exists(), f"expected sibling journal at {sibling}"

    def test_custom_tracker_does_not_touch_canonical_store(self, tmp_path):
        """Logging through a custom-dir tracker leaves the canonical journal empty."""
        before = get_response_tracker()._coll.count() if get_response_tracker()._coll else 0
        tracker = ResponseTracker(log_dir=str(tmp_path / "logs"))
        tracker.log(
            user_message="iso-Q",
            user_id="u1",
            assistant_response="iso-A",
            model="gpt2",
            config={},
            session_id="s",
            tokens_generated=1,
            duration_ms=1.0,
        )
        after = get_response_tracker()._coll.count() if get_response_tracker()._coll else 0
        assert after == before

    def test_custom_tracker_get_responses_roundtrip(self, tmp_path):
        """Custom-dir tracker: MogDB-first read returns its own journal, not files only."""
        tracker = ResponseTracker(log_dir=str(tmp_path / "logs"))
        tracker.log(
            user_message="Q",
            assistant_response="A",
            model="gpt2",
            config={},
            session_id="s1",
            user_id="u1",
            tokens_generated=1,
            duration_ms=10.0,
        )
        # Journal has the doc, and get_responses finds it via MogDB-first.
        assert tracker._coll.count() == 1
        responses = tracker.get_responses()
        assert len(responses) == 1
        assert responses[0].user_message == "Q"
        # Strips MogDB bookkeeping keys.
        assert not hasattr(responses[0], "_id")
        assert not hasattr(responses[0], "expires_at")


class TestReadResponseDocs:
    def test_non_canonical_dir_returns_none(self, tmp_path):
        """Custom dirs never read the canonical store — the JSONL fallback owns them."""
        assert read_response_docs(tmp_path) is None

    def test_canonical_string_path_resolves(self):
        """The canonical relative path resolves (gate passes) without error."""
        result = read_response_docs("data/response_logs")
        assert result is None or isinstance(result, list)

    def test_canonical_dir_with_empty_journal_returns_none(self, tmp_path, monkeypatch):
        """Canonical dir but empty MogDB → None, so callers fall back to JSONL."""
        monkeypatch.setattr(
            "domain.feedback._internal.response_tracker._canonical_log_dir",
            lambda: tmp_path / "response_logs",
        )
        (tmp_path / "response_logs").mkdir()
        # read_response_docs resolves the process-global tracker, which another
        # test file may already have populated with its own docs; patch it to an
        # empty collection so "empty journal" means empty, not "whatever ran first".
        monkeypatch.setattr(
            "domain.feedback._internal.response_tracker._response_tracker",
            ResponseTracker(log_dir=str(tmp_path / "response_logs")),
        )
        assert read_response_docs(tmp_path / "response_logs") is None

    def test_canonical_dir_returns_docs(self, tmp_path, monkeypatch):
        """Canonical dir with a populated journal returns chronological docs."""
        monkeypatch.setattr(
            "domain.feedback._internal.response_tracker._canonical_log_dir",
            lambda: tmp_path / "response_logs",
        )
        tracker = ResponseTracker(log_dir=str(tmp_path / "response_logs"))
        monkeypatch.setattr("domain.feedback._internal.response_tracker._response_tracker", tracker)
        for i in range(2):
            tracker.log(
                user_message=f"Q{i}",
                assistant_response=f"A{i}",
                model="gpt2",
                config={},
                session_id=f"s{i}",
                user_id="u",
                tokens_generated=1,
                duration_ms=1.0,
            )
        docs = read_response_docs(tmp_path / "response_logs")
        assert docs is not None
        assert len(docs) == 2
        assert [d["user_message"] for d in docs] == ["Q0", "Q1"]


class TestCanonicalLogDir:
    def test_is_data_response_logs(self):
        assert _canonical_log_dir().name == "response_logs"
        assert _canonical_log_dir().parent.name == "data"
