"""Tests for benchmark.domain — response quality tracking and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from domains.benchmark.domain import (
    BenchmarkDomain,
    BenchmarkResult,
    get_benchmark_domain,
    reset_benchmark_domain,
)


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """Create a BenchmarkDomain using a temp directory."""
    monkeypatch.setattr("domains.benchmark.domain._RESPONSES_DIR", tmp_path)
    reset_benchmark_domain()
    yield BenchmarkDomain()
    reset_benchmark_domain()


def _write_response(bench, filename, data):
    bench._responses_dir.mkdir(parents=True, exist_ok=True)
    (bench._responses_dir / filename).write_text(json.dumps(data))


# ── Singleton ─────────────────────────────────────────────────────────────


class TestBenchmarkDomainSingleton:

    def setup_method(self):
        reset_benchmark_domain()

    def teardown_method(self):
        reset_benchmark_domain()

    def test_get_returns_same_instance(self):
        b1 = get_benchmark_domain()
        b2 = get_benchmark_domain()
        assert b1 is b2

    def test_reset_creates_new_instance(self):
        b1 = get_benchmark_domain()
        reset_benchmark_domain()
        b2 = get_benchmark_domain()
        assert b1 is not b2


# ── BenchmarkResult ───────────────────────────────────────────────────────


class TestBenchmarkResult:

    def test_dataclass_fields(self):
        r = BenchmarkResult(
            timestamp="2026-01-01T00:00:00Z",
            model="gpt2",
            num_responses=10,
            avg_length=50.0,
            length_std=10.0,
            repetition_rate=0.1,
            repetition_bigrams=0.05,
            avg_log_prob=-0.5,
            unique_bigrams=0.9,
            unique_trigrams=0.8,
        )
        assert r.model == "gpt2"
        assert r.num_responses == 10


# ── get_stats ─────────────────────────────────────────────────────────────


class TestGetStats:

    def test_empty_returns_zero(self, bench):
        stats = bench.get_stats()
        assert stats["total_responses"] == 0
        assert stats["models"] == []
        assert stats["avg_length"] == 0

    def test_single_response(self, bench):
        _write_response(bench, "r1.json", [{"model": "gpt2", "text": "hello world"}])
        stats = bench.get_stats()
        assert stats["total_responses"] == 1
        assert "gpt2" in stats["models"]
        assert stats["avg_length"] == 11.0

    def test_multiple_responses(self, bench):
        _write_response(bench, "r1.json", [
            {"model": "gpt2", "text": "hello"},
            {"model": "llama", "text": "hello world"},
        ])
        stats = bench.get_stats()
        assert stats["total_responses"] == 2
        assert len(stats["models"]) == 2
        assert stats["avg_length"] == 8.0

    def test_handles_non_dict_entries(self, bench):
        _write_response(bench, "r1.json", ["not a dict", {"model": "gpt2", "text": "ok"}])
        stats = bench.get_stats()
        assert stats["total_responses"] == 2


# ── evaluate_latest ───────────────────────────────────────────────────────


class TestEvaluateLatest:

    def test_empty_returns_zero(self, bench):
        result = bench.evaluate_latest()
        assert result["responses_analyzed"] == 0
        assert result["metrics"] == {}

    def test_analyzes_single_response(self, bench):
        _write_response(bench, "r1.json", [{"text": "hello world"}])
        result = bench.evaluate_latest()
        assert result["responses_analyzed"] == 1
        assert result["metrics"]["avg_length"] == 11.0
        assert result["metrics"]["repetition_rate"] == 0.0

    def test_detects_repetition(self, bench):
        text = "the cat sat on the cat sat on the cat"
        _write_response(bench, "r1.json", [{"text": text}])
        result = bench.evaluate_latest()
        assert result["metrics"]["repetition_rate"] > 0

    def test_unique_bigram_ratio(self, bench):
        text = "a b c d e f g h i j"
        _write_response(bench, "r1.json", [{"text": text}])
        result = bench.evaluate_latest()
        assert result["metrics"]["unique_bigram_ratio"] == 1.0

    def test_respects_limit(self, bench):
        responses = [{"text": f"response {i}"} for i in range(100)]
        _write_response(bench, "r1.json", responses)
        result = bench.evaluate_latest(limit=10)
        assert result["responses_analyzed"] == 10

    def test_length_std_calculation(self, bench):
        _write_response(bench, "r1.json", [
            {"text": "short"},
            {"text": "a longer response here"},
        ])
        result = bench.evaluate_latest()
        assert result["metrics"]["length_std"] > 0


# ── clear_history ─────────────────────────────────────────────────────────


class TestClearHistory:

    def test_clears_all_files(self, bench):
        _write_response(bench, "r1.json", [{"text": "data"}])
        _write_response(bench, "r2.json", [{"text": "data2"}])
        bench.clear_history()
        files = list(bench._responses_dir.glob("*.json"))
        assert len(files) == 0

    def test_directory_still_exists(self, bench):
        _write_response(bench, "r1.json", [{"text": "data"}])
        bench.clear_history()
        assert bench._responses_dir.exists()
