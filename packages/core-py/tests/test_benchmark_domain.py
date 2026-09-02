"""Tests for domains.benchmark.domain — BenchmarkDomain."""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from domains.benchmark.domain import BenchmarkDomain, BenchmarkResult, get_benchmark_domain, reset_benchmark_domain


class TestBenchmarkDomain:
    def setup_method(self):
        reset_benchmark_domain()
        self.bd = BenchmarkDomain()

    def test_get_stats_empty(self):
        stats = self.bd.get_stats()
        assert stats["total_responses"] == 0
        assert stats["models"] == []
        assert stats["avg_length"] == 0

    def test_evaluate_latest_empty(self):
        result = self.bd.evaluate_latest()
        assert result["responses_analyzed"] == 0
        assert result["metrics"] == {}

    def test_clear_history(self):
        self.bd.clear_history()
        assert self.bd._responses_dir.exists()


class TestBenchmarkResult:
    def test_dataclass(self):
        br = BenchmarkResult(
            timestamp="2025-01-01",
            model="test",
            num_responses=10,
            avg_length=100.0,
            length_std=10.0,
            repetition_rate=0.1,
            repetition_bigrams=0.05,
            avg_log_prob=-0.5,
            unique_bigrams=0.95,
            unique_trigrams=0.90,
        )
        assert br.model == "test"
        assert br.num_responses == 10


class TestSingleton:
    def test_get_singleton(self):
        bd1 = get_benchmark_domain()
        bd2 = get_benchmark_domain()
        assert bd1 is bd2

    def test_reset_singleton(self):
        bd1 = get_benchmark_domain()
        reset_benchmark_domain()
        bd2 = get_benchmark_domain()
        assert bd1 is not bd2
