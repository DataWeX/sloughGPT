"""Tests for pair_extractor (extract training pairs from server logs)."""

import json
import time
from pathlib import Path

import pytest

from domains.training.pair_extractor import (
    extract_pairs_from_sessions,
    extract_pairs_from_logs,
    extract_pairs_from_corpus,
    write_training_text,
    count_pairs_in_sessions,
    count_pairs_in_logs,
    count_pairs_in_corpus,
)


def _write_session(d: Path, sid: str, messages: list) -> Path:
    """Write a session JSON file."""
    p = d / f"{sid}.json"
    p.write_text(json.dumps({"id": sid, "messages": messages}))
    return p


def _write_log(d: Path, entries: list) -> Path:
    """Write a response log JSONL file."""
    d.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d")
    p = d / f"responses_{ts}.jsonl"
    with open(p, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return p


def _write_corpus(d: Path, entries: list) -> Path:
    """Write a captured-corpus JSONL file (messages format)."""
    d.mkdir(parents=True, exist_ok=True)
    p = d / "corpus.jsonl"
    with open(p, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return p


class TestExtractPairsFromSessions:
    def test_basic_extraction(self, tmp_path, monkeypatch):
        """Extracts user→assistant pairs from session files."""
        monkeypatch.setattr("domains.training.pair_extractor._SESSIONS_DIR", tmp_path)
        _write_session(tmp_path, "s1", [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ])
        pairs = extract_pairs_from_sessions(limit=10, min_length=3)
        assert len(pairs) == 2
        assert pairs[0]["user_msg"] == "Hello there"
        assert pairs[0]["assistant_msg"] == "Hi! How can I help?"
        assert pairs[0]["session_id"] == "s1"

    def test_min_length_filter(self, tmp_path, monkeypatch):
        """Short messages are filtered out."""
        monkeypatch.setattr("domains.training.pair_extractor._SESSIONS_DIR", tmp_path)
        _write_session(tmp_path, "s1", [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hey"},
            {"role": "user", "content": "What is machine learning really about today?"},
            {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence."},
        ])
        pairs = extract_pairs_from_sessions(limit=10, min_length=5)
        assert len(pairs) == 1
        assert "machine learning" in pairs[0]["user_msg"]

    def test_deduplication(self, tmp_path, monkeypatch):
        """Duplicate pairs are deduplicated by content hash."""
        monkeypatch.setattr("domains.training.pair_extractor._SESSIONS_DIR", tmp_path)
        msgs = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi! How can I help?"},
        ]
        _write_session(tmp_path, "s1", msgs)
        _write_session(tmp_path, "s2", msgs)
        pairs = extract_pairs_from_sessions(limit=10, min_length=3)
        assert len(pairs) == 1

    def test_limit(self, tmp_path, monkeypatch):
        """Respects limit parameter."""
        monkeypatch.setattr("domains.training.pair_extractor._SESSIONS_DIR", tmp_path)
        for i in range(5):
            _write_session(tmp_path, f"s{i}", [
                {"role": "user", "content": f"Question {i} about something interesting"},
                {"role": "assistant", "content": f"Answer {i} with detailed information"},
            ])
        pairs = extract_pairs_from_sessions(limit=3, min_length=3)
        assert len(pairs) == 3

    def test_session_ids_filter(self, tmp_path, monkeypatch):
        """Only extracts from specified session IDs."""
        monkeypatch.setattr("domains.training.pair_extractor._SESSIONS_DIR", tmp_path)
        _write_session(tmp_path, "s1", [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there friend"},
        ])
        _write_session(tmp_path, "s2", [
            {"role": "user", "content": "Goodbye"},
            {"role": "assistant", "content": "See you later today"},
        ])
        pairs = extract_pairs_from_sessions(limit=10, min_length=3, session_ids=["s2"])
        assert len(pairs) == 1
        assert pairs[0]["session_id"] == "s2"

    def test_empty_sessions_dir(self, tmp_path, monkeypatch):
        """Returns empty list when directory doesn't exist."""
        monkeypatch.setattr("domains.training.pair_extractor._SESSIONS_DIR", tmp_path / "nonexistent")
        pairs = extract_pairs_from_sessions(limit=10, min_length=3)
        assert pairs == []

    def test_skip_non_user_assistant(self, tmp_path, monkeypatch):
        """Only counts user→assistant consecutive pairs."""
        monkeypatch.setattr("domains.training.pair_extractor._SESSIONS_DIR", tmp_path)
        _write_session(tmp_path, "s1", [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello there friend"},
            {"role": "user", "content": "Still me"},
            {"role": "assistant", "content": "Hi there friend"},
        ])
        pairs = extract_pairs_from_sessions(limit=10, min_length=3)
        assert len(pairs) == 1

    def test_malformed_json_skipped(self, tmp_path, monkeypatch):
        """Malformed JSON files are skipped without error."""
        monkeypatch.setattr("domains.training.pair_extractor._SESSIONS_DIR", tmp_path)
        (tmp_path / "bad.json").write_text("not json {{{")
        _write_session(tmp_path, "s1", [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi there friend"},
        ])
        pairs = extract_pairs_from_sessions(limit=10, min_length=3)
        assert len(pairs) == 1

    def test_newest_first(self, tmp_path, monkeypatch):
        """Newest sessions appear first."""
        monkeypatch.setattr("domains.training.pair_extractor._SESSIONS_DIR", tmp_path)
        _write_session(tmp_path, "old", [
            {"role": "user", "content": "Old question about something"},
            {"role": "assistant", "content": "Old answer with enough text"},
        ])
        time.sleep(0.01)
        _write_session(tmp_path, "new", [
            {"role": "user", "content": "New question about something"},
            {"role": "assistant", "content": "New answer with enough text"},
        ])
        pairs = extract_pairs_from_sessions(limit=10, min_length=3)
        assert pairs[0]["session_id"] == "new"

    def test_skip_short_session(self, tmp_path, monkeypatch):
        """Sessions with fewer than two messages are skipped."""
        monkeypatch.setattr("domains.training.pair_extractor._SESSIONS_DIR", tmp_path)
        _write_session(tmp_path, "s1", [{"role": "user", "content": "Only one message"}])
        _write_session(tmp_path, "s2", [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi there friend"},
        ])
        pairs = extract_pairs_from_sessions(limit=10, min_length=3)
        assert len(pairs) == 1

    def test_inner_limit_break(self, tmp_path, monkeypatch):
        """Stops scanning a session once the limit is reached."""
        monkeypatch.setattr("domains.training.pair_extractor._SESSIONS_DIR", tmp_path)
        _write_session(tmp_path, "s1", [
            {"role": "user", "content": "Question one about something"},
            {"role": "assistant", "content": "Answer one with detail"},
            {"role": "user", "content": "Question two about something"},
            {"role": "assistant", "content": "Answer two with detail"},
        ])
        pairs = extract_pairs_from_sessions(limit=1, min_length=3)
        assert len(pairs) == 1


class TestExtractPairsFromLogs:
    def test_basic_extraction(self, tmp_path, monkeypatch):
        """Extracts pairs from JSONL response logs."""
        monkeypatch.setattr("domains.training.pair_extractor._RESPONSE_LOGS_DIR", tmp_path)
        _write_log(tmp_path, [
            {"user_message": "Hello", "assistant_response": "Hi there!", "model": "gpt2", "session_id": "s1"},
            {"user_message": "Bye", "assistant_response": "Goodbye!", "model": "gpt2", "session_id": "s1"},
        ])
        pairs = extract_pairs_from_logs(limit=10, min_length=3)
        assert len(pairs) == 2
        assert pairs[0]["user_msg"] == "Hello"
        assert pairs[0]["model"] == "gpt2"

    def test_model_filter(self, tmp_path, monkeypatch):
        """Filters by model name."""
        monkeypatch.setattr("domains.training.pair_extractor._RESPONSE_LOGS_DIR", tmp_path)
        _write_log(tmp_path, [
            {"user_message": "Hello", "assistant_response": "Hi there!", "model": "gpt2", "session_id": "s1"},
            {"user_message": "Hello", "assistant_response": "Hey!", "model": "qwen", "session_id": "s2"},
        ])
        pairs = extract_pairs_from_logs(limit=10, min_length=3, model="gpt2")
        assert len(pairs) == 1
        assert pairs[0]["model"] == "gpt2"

    def test_deduplication(self, tmp_path, monkeypatch):
        """Duplicate entries are deduplicated."""
        monkeypatch.setattr("domains.training.pair_extractor._RESPONSE_LOGS_DIR", tmp_path)
        _write_log(tmp_path, [
            {"user_message": "Hello", "assistant_response": "Hi there!", "model": "gpt2", "session_id": "s1"},
            {"user_message": "Hello", "assistant_response": "Hi there!", "model": "gpt2", "session_id": "s1"},
        ])
        pairs = extract_pairs_from_logs(limit=10, min_length=3)
        assert len(pairs) == 1

    def test_empty_logs_dir(self, tmp_path, monkeypatch):
        """Returns empty list when directory doesn't exist."""
        monkeypatch.setattr("domains.training.pair_extractor._RESPONSE_LOGS_DIR", tmp_path / "nonexistent")
        pairs = extract_pairs_from_logs(limit=10, min_length=3)
        assert pairs == []

    def test_malformed_lines_skipped(self, tmp_path, monkeypatch):
        """Malformed JSON lines are skipped."""
        monkeypatch.setattr("domains.training.pair_extractor._RESPONSE_LOGS_DIR", tmp_path)
        ts = time.strftime("%Y%m%d")
        p = tmp_path / f"responses_{ts}.jsonl"
        p.write_text("not json\n{\"user_message\": \"Hello there\", \"assistant_response\": \"Hey there!\", \"model\": \"gpt2\", \"session_id\": \"s1\"}\n")
        pairs = extract_pairs_from_logs(limit=10, min_length=3)
        assert len(pairs) == 1

    def test_limit_per_file(self, tmp_path, monkeypatch):
        """Limit applies across files."""
        monkeypatch.setattr("domains.training.pair_extractor._RESPONSE_LOGS_DIR", tmp_path)
        for day in range(3):
            ts = f"202607{10 + day:02d}"
            p = tmp_path / f"responses_{ts}.jsonl"
            with open(p, "w") as f:
                f.write(json.dumps({"user_message": f"Question {day} about topic", "assistant_response": f"Answer {day} with detailed text", "model": "gpt2", "session_id": "s1"}) + "\n")
        pairs = extract_pairs_from_logs(limit=2, min_length=3)
        assert len(pairs) == 2

    def test_log_inner_limit_break(self, tmp_path, monkeypatch):
        """Stops reading a file once the limit is reached."""
        monkeypatch.setattr("domains.training.pair_extractor._RESPONSE_LOGS_DIR", tmp_path)
        _write_log(tmp_path, [
            {"user_message": "Hello there", "assistant_response": "Hi there!", "model": "gpt2"},
            {"user_message": "Bye now", "assistant_response": "Goodbye!", "model": "gpt2"},
        ])
        pairs = extract_pairs_from_logs(limit=1, min_length=3)
        assert len(pairs) == 1

    def test_blank_lines_skipped(self, tmp_path, monkeypatch):
        """Blank lines in a JSONL log are ignored."""
        monkeypatch.setattr("domains.training.pair_extractor._RESPONSE_LOGS_DIR", tmp_path)
        ts = time.strftime("%Y%m%d")
        p = tmp_path / f"responses_{ts}.jsonl"
        p.write_text(
            "\n\n"
            '{"user_message": "Hello there", "assistant_response": "Hey there!", "model": "gpt2"}\n'
            "\n"
        )
        pairs = extract_pairs_from_logs(limit=10, min_length=3)
        assert len(pairs) == 1

    def test_log_min_length_filter(self, tmp_path, monkeypatch):
        """Short messages in logs are filtered out."""
        monkeypatch.setattr("domains.training.pair_extractor._RESPONSE_LOGS_DIR", tmp_path)
        _write_log(tmp_path, [
            {"user_message": "Hi", "assistant_response": "Hey", "model": "gpt2"},
            {"user_message": "Hello there", "assistant_response": "Hi there friend!", "model": "gpt2"},
        ])
        pairs = extract_pairs_from_logs(limit=10, min_length=5)
        assert len(pairs) == 1
        assert pairs[0]["user_msg"] == "Hello there"

    def test_unreadable_log_skipped(self, tmp_path, monkeypatch):
        """Unreadable log files (e.g. directories) are skipped."""
        monkeypatch.setattr("domains.training.pair_extractor._RESPONSE_LOGS_DIR", tmp_path)
        ts = time.strftime("%Y%m%d")
        (tmp_path / f"zz_{ts}.jsonl").mkdir(parents=True)
        _write_log(tmp_path, [
            {"user_message": "Hello there", "assistant_response": "Hi there!", "model": "gpt2"},
        ])
        pairs = extract_pairs_from_logs(limit=10, min_length=3)
        assert len(pairs) == 1


class TestExtractPairsFromCorpus:
    def test_basic_extraction(self, tmp_path, monkeypatch):
        """Extracts user→assistant pairs from captured corpus JSONL."""
        monkeypatch.setattr("domains.training.pair_extractor._CAPTURED_DIR", tmp_path)
        _write_corpus(tmp_path, [
            {"messages": [{"role": "user", "content": "Hello there"}, {"role": "assistant", "content": "Hi! How can I help?"}], "meta": {"model": "qwen", "session_id": "s1"}},
        ])
        pairs = extract_pairs_from_corpus(limit=10, min_length=3)
        assert len(pairs) == 1
        assert pairs[0]["user_msg"] == "Hello there"
        assert pairs[0]["assistant_msg"] == "Hi! How can I help?"
        assert pairs[0]["session_id"] == "s1"
        assert pairs[0]["model"] == "qwen"

    def test_model_filter(self, tmp_path, monkeypatch):
        """Filters corpus entries by meta.model."""
        monkeypatch.setattr("domains.training.pair_extractor._CAPTURED_DIR", tmp_path)
        _write_corpus(tmp_path, [
            {"messages": [{"role": "user", "content": "Hello there"}, {"role": "assistant", "content": "Hi there!"}], "meta": {"model": "qwen"}},
            {"messages": [{"role": "user", "content": "Good day"}, {"role": "assistant", "content": "And to you!"}], "meta": {"model": "gpt2"}},
        ])
        pairs = extract_pairs_from_corpus(limit=10, min_length=3, model="qwen")
        assert len(pairs) == 1
        assert pairs[0]["model"] == "qwen"

    def test_deduplication(self, tmp_path, monkeypatch):
        """Duplicate corpus entries are deduplicated by content hash."""
        monkeypatch.setattr("domains.training.pair_extractor._CAPTURED_DIR", tmp_path)
        row = {"messages": [{"role": "user", "content": "Hello there"}, {"role": "assistant", "content": "Hi there friend!"}]}
        _write_corpus(tmp_path, [row, row])
        pairs = extract_pairs_from_corpus(limit=10, min_length=3)
        assert len(pairs) == 1

    def test_limit(self, tmp_path, monkeypatch):
        """Respects the limit parameter."""
        monkeypatch.setattr("domains.training.pair_extractor._CAPTURED_DIR", tmp_path)
        _write_corpus(tmp_path, [
            {"messages": [{"role": "user", "content": f"Question {i} about things"}, {"role": "assistant", "content": f"Answer {i} with detail"}]}
            for i in range(5)
        ])
        pairs = extract_pairs_from_corpus(limit=3, min_length=3)
        assert len(pairs) == 3

    def test_min_length_filter(self, tmp_path, monkeypatch):
        """Short messages are filtered out."""
        monkeypatch.setattr("domains.training.pair_extractor._CAPTURED_DIR", tmp_path)
        _write_corpus(tmp_path, [
            {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hey"}]},
            {"messages": [{"role": "user", "content": "What is machine learning today?"}, {"role": "assistant", "content": "It is a broad and deep field."}]},
        ])
        pairs = extract_pairs_from_corpus(limit=10, min_length=5)
        assert len(pairs) == 1
        assert "machine learning" in pairs[0]["user_msg"]

    def test_missing_corpus(self, tmp_path, monkeypatch):
        """Returns empty list when corpus file doesn't exist."""
        monkeypatch.setattr("domains.training.pair_extractor._CAPTURED_DIR", tmp_path / "nonexistent")
        pairs = extract_pairs_from_corpus(limit=10, min_length=3)
        assert pairs == []

    def test_malformed_lines_skipped(self, tmp_path, monkeypatch):
        """Malformed JSON lines are skipped."""
        monkeypatch.setattr("domains.training.pair_extractor._CAPTURED_DIR", tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "corpus.jsonl").write_text(
            "not json\n"
            '{"messages": [{"role": "user", "content": "Hello there"}, {"role": "assistant", "content": "Hi there friend!"}]}\n'
        )
        pairs = extract_pairs_from_corpus(limit=10, min_length=3)
        assert len(pairs) == 1

    def test_blank_lines_skipped(self, tmp_path, monkeypatch):
        """Blank lines in the corpus are ignored."""
        monkeypatch.setattr("domains.training.pair_extractor._CAPTURED_DIR", tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "corpus.jsonl").write_text(
            "\n\n"
            '{"messages": [{"role": "user", "content": "Hello there"}, {"role": "assistant", "content": "Hey there!"}]}\n'
            "\n"
        )
        pairs = extract_pairs_from_corpus(limit=10, min_length=3)
        assert len(pairs) == 1

    def test_unreadable_corpus_returns_empty(self, tmp_path, monkeypatch):
        """An OSError reading the corpus yields an empty result."""
        monkeypatch.setattr("domains.training.pair_extractor._CAPTURED_DIR", tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "corpus.jsonl").mkdir()
        pairs = extract_pairs_from_corpus(limit=10, min_length=3)
        assert pairs == []


class TestWriteTrainingText:
    def test_writes_text_file(self, tmp_path):
        """Writes pairs in User:/Assistant: format."""
        pairs = [
            {"user_msg": "Hello", "assistant_msg": "Hi there!"},
            {"user_msg": "Bye", "assistant_msg": "Goodbye!"},
        ]
        path = write_training_text(pairs, tmp_path)
        assert path.exists()
        content = path.read_text()
        assert "User: Hello\nAssistant: Hi there!\n\n" in content
        assert "User: Bye\nAssistant: Goodbye!\n\n" in content

    def test_empty_pairs(self, tmp_path):
        """Empty pair list still creates file."""
        path = write_training_text([], tmp_path)
        assert path.exists()
        assert path.read_text() == ""

    def test_default_output_dir(self):
        """Default output is data/mobile_training/."""
        path = write_training_text([{"user_msg": "Hi", "assistant_msg": "Hey!"}])
        assert "mobile_training" in str(path)
        assert path.exists()
        path.unlink()


class TestCountPairs:
    def test_count_in_sessions(self, tmp_path, monkeypatch):
        """Counts total pairs across session files."""
        monkeypatch.setattr("domains.training.pair_extractor._SESSIONS_DIR", tmp_path)
        _write_session(tmp_path, "s1", [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
            {"role": "assistant", "content": "Goodbye"},
        ])
        _write_session(tmp_path, "s2", [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Result"},
        ])
        assert count_pairs_in_sessions() == 3

    def test_count_in_logs(self, tmp_path, monkeypatch):
        """Counts total entries in log files."""
        monkeypatch.setattr("domains.training.pair_extractor._RESPONSE_LOGS_DIR", tmp_path)
        _write_log(tmp_path, [
            {"user_message": "a", "assistant_response": "b"},
            {"user_message": "c", "assistant_response": "d"},
        ])
        assert count_pairs_in_logs() == 2

    def test_count_empty_dir(self, tmp_path, monkeypatch):
        """Returns 0 for nonexistent directory."""
        monkeypatch.setattr("domains.training.pair_extractor._SESSIONS_DIR", tmp_path / "nope")
        assert count_pairs_in_sessions() == 0

    def test_count_sessions_skips_malformed(self, tmp_path, monkeypatch):
        """Malformed session files are skipped while counting."""
        monkeypatch.setattr("domains.training.pair_extractor._SESSIONS_DIR", tmp_path)
        (tmp_path / "bad.json").write_text("not json {{{")
        _write_session(tmp_path, "s1", [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ])
        assert count_pairs_in_sessions() == 1

    def test_count_logs_missing_dir(self, tmp_path, monkeypatch):
        """Returns 0 when the logs directory doesn't exist."""
        monkeypatch.setattr("domains.training.pair_extractor._RESPONSE_LOGS_DIR", tmp_path / "nope")
        assert count_pairs_in_logs() == 0

    def test_count_logs_skips_unreadable(self, tmp_path, monkeypatch):
        """Unreadable log files are skipped while counting."""
        monkeypatch.setattr("domains.training.pair_extractor._RESPONSE_LOGS_DIR", tmp_path)
        ts = time.strftime("%Y%m%d")
        (tmp_path / f"zz_{ts}.jsonl").mkdir(parents=True)
        _write_log(tmp_path, [
            {"user_message": "a", "assistant_response": "b"},
            {"user_message": "c", "assistant_response": "d"},
        ])
        assert count_pairs_in_logs() == 2

    def test_count_in_corpus(self, tmp_path, monkeypatch):
        """Counts total exchanges in the captured corpus."""
        monkeypatch.setattr("domains.training.pair_extractor._CAPTURED_DIR", tmp_path)
        _write_corpus(tmp_path, [
            {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
            {"messages": [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]},
        ])
        assert count_pairs_in_corpus() == 2

    def test_count_corpus_missing(self, tmp_path, monkeypatch):
        """Returns 0 when corpus file doesn't exist."""
        monkeypatch.setattr("domains.training.pair_extractor._CAPTURED_DIR", tmp_path / "nope")
        assert count_pairs_in_corpus() == 0

    def test_count_corpus_skips_blank(self, tmp_path, monkeypatch):
        """Blank lines are not counted."""
        monkeypatch.setattr("domains.training.pair_extractor._CAPTURED_DIR", tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "corpus.jsonl").write_text(
            "\n"
            '{"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}\n'
            "\n"
        )
        assert count_pairs_in_corpus() == 1

    def test_count_corpus_unreadable_returns_zero(self, tmp_path, monkeypatch):
        """An OSError counting the corpus yields zero."""
        monkeypatch.setattr("domains.training.pair_extractor._CAPTURED_DIR", tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "corpus.jsonl").mkdir()
        assert count_pairs_in_corpus() == 0
