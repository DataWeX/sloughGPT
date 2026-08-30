"""Tests for domains.benchmark.domain — pure logic, no external mocks."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from domains.benchmark.domain import (
    BenchmarkDomain,
    BenchmarkResult,
    _RESPONSES_DIR,
    get_benchmark_domain,
    reset_benchmark_domain,
)


# ---------------------------------------------------------------------------
# BenchmarkResult dataclass
# ---------------------------------------------------------------------------

class TestBenchmarkResult:
    def test_fields(self):
        r = BenchmarkResult(
            timestamp="2026-01-01",
            model="m1",
            num_responses=10,
            avg_length=50.0,
            length_std=5.0,
            repetition_rate=0.1,
            repetition_bigrams=2,
            avg_log_prob=-1.5,
            unique_bigrams=80.0,
            unique_trigrams=90.0,
        )
        assert r.model == "m1"
        assert r.num_responses == 10
        assert r.avg_length == 50.0

    def test_equality(self):
        a = BenchmarkResult("t", "m", 1, 1.0, 1.0, 1.0, 1, 1.0, 1.0, 1.0)
        b = BenchmarkResult("t", "m", 1, 1.0, 1.0, 1.0, 1, 1.0, 1.0, 1.0)
        assert a == b

    def test_inequality(self):
        a = BenchmarkResult("t", "m", 1, 1.0, 1.0, 1.0, 1, 1.0, 1.0, 1.0)
        b = BenchmarkResult("t", "m", 2, 1.0, 1.0, 1.0, 1, 1.0, 1.0, 1.0)
        assert a != b


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_returns_same_instance(self):
        reset_benchmark_domain()
        a = get_benchmark_domain()
        b = get_benchmark_domain()
        assert a is b

    def test_reset_breaks_singleton(self):
        a = get_benchmark_domain()
        reset_benchmark_domain()
        b = get_benchmark_domain()
        assert a is not b


# ---------------------------------------------------------------------------
# BenchmarkDomain — responses dir creation
# ---------------------------------------------------------------------------

class TestInit:
    def test_responses_dir_created(self, tmp_path):
        d = BenchmarkDomain()
        # The constructor creates _RESPONSES_DIR; verify it exists on disk
        assert d._responses_dir.exists()

    def test_responses_dir_idempotent(self):
        # Calling twice does not raise
        BenchmarkDomain()
        BenchmarkDomain()


# ---------------------------------------------------------------------------
# _load_responses
# ---------------------------------------------------------------------------

class TestLoadResponses:
    def test_empty_dir(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        assert d._load_responses() == []

    def test_nonexistent_dir(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path / "does_not_exist"
        assert d._load_responses() == []

    def test_single_json_object(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        (tmp_path / "r1.json").write_text(json.dumps({"text": "hello", "model": "m1"}))
        result = d._load_responses()
        assert len(result) == 1
        assert result[0]["text"] == "hello"

    def test_json_array(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        data = [{"text": "a"}, {"text": "b"}]
        (tmp_path / "r1.json").write_text(json.dumps(data))
        result = d._load_responses()
        assert len(result) == 2

    def test_multiple_files_sorted(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        (tmp_path / "02.json").write_text(json.dumps({"text": "second"}))
        (tmp_path / "01.json").write_text(json.dumps({"text": "first"}))
        result = d._load_responses()
        assert [r["text"] for r in result] == ["first", "second"]

    def test_ignores_non_json_files(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        (tmp_path / "notes.txt").write_text("not json")
        (tmp_path / "data.json").write_text(json.dumps({"text": "ok"}))
        result = d._load_responses()
        assert len(result) == 1

    def test_corrupt_json_skipped(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        (tmp_path / "bad.json").write_text("{invalid json")
        (tmp_path / "good.json").write_text(json.dumps({"text": "ok"}))
        result = d._load_responses()
        assert len(result) == 1

    def test_non_dict_items_filtered_by_get_stats(self, tmp_path):
        """Non-dict items are loaded but filtered out by stats/evaluate."""
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        (tmp_path / "r1.json").write_text(json.dumps("not a dict"))
        result = d._load_responses()
        assert result == ["not a dict"]


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_empty(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        stats = d.get_stats()
        assert stats["total_responses"] == 0
        assert stats["models"] == []
        assert stats["avg_length"] == 0

    def test_single_response(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        (tmp_path / "r1.json").write_text(json.dumps({"text": "hello world", "model": "gpt"}))
        stats = d.get_stats()
        assert stats["total_responses"] == 1
        assert stats["models"] == ["gpt"]
        assert stats["avg_length"] == 11.0

    def test_multiple_models(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        (tmp_path / "r1.json").write_text(json.dumps({"text": "aaa", "model": "m1"}))
        (tmp_path / "r2.json").write_text(json.dumps({"text": "bbbbb", "model": "m2"}))
        stats = d.get_stats()
        assert stats["total_responses"] == 2
        assert set(stats["models"]) == {"m1", "m2"}
        assert stats["avg_length"] == 4.0

    def test_missing_text_defaults_to_empty(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        (tmp_path / "r1.json").write_text(json.dumps({"model": "m1"}))
        stats = d.get_stats()
        assert stats["avg_length"] == 0

    def test_missing_model_defaults_to_unknown(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        (tmp_path / "r1.json").write_text(json.dumps({"text": "hi"}))
        stats = d.get_stats()
        assert stats["models"] == ["unknown"]

    def test_avg_length_rounded(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        (tmp_path / "r1.json").write_text(json.dumps({"text": "abc", "model": "m"}))
        (tmp_path / "r2.json").write_text(json.dumps({"text": "abcdef", "model": "m"}))
        stats = d.get_stats()
        # lengths: 3, 6 → avg = 4.5
        assert stats["avg_length"] == 4.5


# ---------------------------------------------------------------------------
# evaluate_latest
# ---------------------------------------------------------------------------

class TestEvaluateLatest:
    def test_empty(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        result = d.evaluate_latest()
        assert result["responses_analyzed"] == 0
        assert result["metrics"] == {}

    def test_basic_metrics(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        (tmp_path / "r1.json").write_text(json.dumps({"text": "hello world"}))
        (tmp_path / "r2.json").write_text(json.dumps({"text": "foo bar baz"}))
        result = d.evaluate_latest()
        assert result["responses_analyzed"] == 2
        m = result["metrics"]
        assert "avg_length" in m
        assert "length_std" in m
        assert "repetition_rate" in m
        assert "unique_bigram_ratio" in m

    def test_limit_parameter(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        for i in range(10):
            (tmp_path / f"r{i:02d}.json").write_text(json.dumps({"text": f"word{i} text{i}"}))
        result = d.evaluate_latest(limit=3)
        assert result["responses_analyzed"] == 3

    def test_limit_larger_than_data(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        (tmp_path / "r1.json").write_text(json.dumps({"text": "a b"}))
        result = d.evaluate_latest(limit=100)
        assert result["responses_analyzed"] == 1

    def test_repetition_rate_all_same_bigrams(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        # Same words repeated → high repetition rate
        text = "hello world " * 10
        (tmp_path / "r1.json").write_text(json.dumps({"text": text}))
        result = d.evaluate_latest()
        m = result["metrics"]
        assert m["repetition_rate"] > 0.5

    def test_no_repetition_unique_text(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        # Every bigram is unique
        words = [f"w{i}" for i in range(20)]
        text = " ".join(words)
        (tmp_path / "r1.json").write_text(json.dumps({"text": text}))
        result = d.evaluate_latest()
        m = result["metrics"]
        assert m["repetition_rate"] == 0
        assert m["unique_bigram_ratio"] == 1.0

    def test_unique_bigram_ratio(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        # Two texts share one bigram
        (tmp_path / "r1.json").write_text(json.dumps({"text": "a b c"}))
        (tmp_path / "r2.json").write_text(json.dumps({"text": "b c d"}))
        result = d.evaluate_latest()
        m = result["metrics"]
        # bigrams: (a,b), (b,c), (b,c), (c,d)
        # unique: (a,b), (b,c), (c,d) → 3 unique / 4 total
        assert m["unique_bigram_ratio"] == pytest.approx(0.75, abs=0.01)
        assert m["repetition_rate"] == pytest.approx(0.25, abs=0.01)

    def test_length_std_zero_single_response(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        (tmp_path / "r1.json").write_text(json.dumps({"text": "hello"}))
        result = d.evaluate_latest()
        assert result["metrics"]["length_std"] == 0.0

    def test_length_std_calculation(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        # lengths: 5, 5, 10, 10 → avg=7.5, std=2.5
        (tmp_path / "r1.json").write_text(json.dumps({"text": "12345"}))
        (tmp_path / "r2.json").write_text(json.dumps({"text": "12345"}))
        (tmp_path / "r3.json").write_text(json.dumps({"text": "1234567890"}))
        (tmp_path / "r4.json").write_text(json.dumps({"text": "1234567890"}))
        result = d.evaluate_latest()
        std = result["metrics"]["length_std"]
        assert std == pytest.approx(2.5, abs=0.1)

    def test_case_insensitive_bigrams(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        (tmp_path / "r1.json").write_text(json.dumps({"text": "Hello World"}))
        (tmp_path / "r2.json").write_text(json.dumps({"text": "hello world"}))
        result = d.evaluate_latest()
        # Both produce the same bigram (hello, world)
        assert result["metrics"]["repetition_rate"] > 0

    def test_empty_text_entries_skipped(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        (tmp_path / "r1.json").write_text(json.dumps({"text": ""}))
        (tmp_path / "r2.json").write_text(json.dumps({"text": "a b c"}))
        result = d.evaluate_latest()
        assert result["responses_analyzed"] == 2
        # Only one text has bigrams
        assert result["metrics"]["repetition_rate"] == 0

    def test_single_word_no_bigrams(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        (tmp_path / "r1.json").write_text(json.dumps({"text": "hello"}))
        result = d.evaluate_latest()
        assert result["metrics"]["repetition_rate"] == 0
        assert result["metrics"]["unique_bigram_ratio"] == 0


# ---------------------------------------------------------------------------
# clear_history
# ---------------------------------------------------------------------------

class TestClearHistory:
    def test_clear_removes_files(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        (tmp_path / "r1.json").write_text(json.dumps({"text": "a"}))
        (tmp_path / "r2.json").write_text(json.dumps({"text": "b"}))
        d.clear_history()
        assert list(tmp_path.iterdir()) == []

    def test_clear_recreates_dir(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path
        d.clear_history()
        assert tmp_path.exists()
        assert tmp_path.is_dir()

    def test_clear_on_nonexistent_dir_no_crash(self, tmp_path):
        d = BenchmarkDomain()
        d._responses_dir = tmp_path / "nope"
        d.clear_history()
        assert not d._responses_dir.exists()


# ---------------------------------------------------------------------------
# _RESPONSES_DIR constant
# ---------------------------------------------------------------------------

class TestResponsesDir:
    def test_is_path(self):
        assert isinstance(_RESPONSES_DIR, Path)

    def test_ends_with_logged_responses(self):
        assert _RESPONSES_DIR.name == "logged_responses"
