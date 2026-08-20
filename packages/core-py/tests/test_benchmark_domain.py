"""Tests for domains.benchmark.domain — BenchmarkDomain, BenchmarkResult."""

import json
import pytest
from domains.benchmark.domain import (
    BenchmarkDomain, BenchmarkResult,
    get_benchmark_domain, reset_benchmark_domain,
)


class TestBenchmarkResult:
    def test_fields(self):
        r = BenchmarkResult(
            timestamp="2025-01-01",
            model="gpt2",
            num_responses=10,
            avg_length=50.0,
            length_std=10.0,
            repetition_rate=0.1,
            repetition_bigrams=5,
            avg_log_prob=-1.5,
            unique_bigrams=0.9,
            unique_trigrams=0.8,
        )
        assert r.model == "gpt2"
        assert r.num_responses == 10


class TestBenchmarkDomain:
    def test_singleton(self):
        reset_benchmark_domain()
        a = get_benchmark_domain()
        b = get_benchmark_domain()
        assert a is b
        reset_benchmark_domain()

    def test_empty_stats(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        stats = bd.get_stats()
        assert stats["total_responses"] == 0

    def test_stats_with_data(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([
            {"text": "hello world", "model": "gpt2"},
            {"text": "foo bar baz", "model": "gpt2"},
        ]))
        stats = bd.get_stats()
        assert stats["total_responses"] == 2
        assert "gpt2" in stats["models"]

    def test_evaluate_latest_empty(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        result = bd.evaluate_latest()
        assert result["responses_analyzed"] == 0

    def test_evaluate_latest_repetition(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([
            {"text": "the cat sat on the mat the cat sat", "model": "gpt2"},
        ]))
        result = bd.evaluate_latest()
        assert result["responses_analyzed"] == 1
        assert result["metrics"]["repetition_rate"] > 0

    def test_clear_history(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([{"text": "hello"}]))
        bd.clear_history()
        assert list(tmp_path.iterdir()) == []
