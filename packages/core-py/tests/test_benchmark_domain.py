"""Tests for domains.benchmark.domain: response quality tracking and evaluation."""

import json

import pytest

import domains.benchmark.domain as bd


@pytest.fixture
def benchmark(tmp_path, monkeypatch):
    monkeypatch.setattr(bd, "_RESPONSES_DIR", tmp_path / "logged_responses")
    return bd.BenchmarkDomain()


def write_response(benchmark, name, payload):
    (benchmark._responses_dir / name).write_text(json.dumps(payload))
    return name


class TestLoadResponses:
    def test_empty_dir(self, benchmark):
        assert benchmark._load_responses() == []

    def test_loads_list_payloads(self, benchmark):
        write_response(benchmark, "a.json", [{"text": "one"}, {"text": "two"}])
        assert benchmark._load_responses() == [{"text": "one"}, {"text": "two"}]

    def test_loads_object_payloads(self, benchmark):
        write_response(benchmark, "b.json", {"text": "solo"})
        assert benchmark._load_responses() == [{"text": "solo"}]

    def test_skips_non_json(self, benchmark):
        (benchmark._responses_dir / "notes.txt").write_text("not json")
        assert benchmark._load_responses() == []

    def test_skips_corrupt_json(self, benchmark):
        write_response(benchmark, "bad.json", {"text": "x"})
        (benchmark._responses_dir / "broken.json").write_text("{not valid json")
        assert benchmark._load_responses() == [{"text": "x"}]

    def test_sorted_by_filename(self, benchmark):
        write_response(benchmark, "2.json", {"text": "two"})
        write_response(benchmark, "1.json", {"text": "one"})
        assert benchmark._load_responses() == [{"text": "one"}, {"text": "two"}]


class TestGetStats:
    def test_empty(self, benchmark):
        assert benchmark.get_stats() == {"total_responses": 0, "models": [], "avg_length": 0}

    def test_aggregates(self, benchmark):
        write_response(benchmark, "a.json", {"model": "gpt2", "text": "hello world"})
        write_response(benchmark, "b.json", {"model": "gpt2", "text": "hi"})
        stats = benchmark.get_stats()
        assert stats["total_responses"] == 2
        assert stats["models"] == ["gpt2"]
        assert stats["avg_length"] == 6.5

    def test_multiple_models(self, benchmark):
        write_response(benchmark, "a.json", {"model": "a", "text": "x"})
        write_response(benchmark, "b.json", {"model": "b", "text": "yy"})
        stats = benchmark.get_stats()
        assert set(stats["models"]) == {"a", "b"}

    def test_ignores_non_dict_entries(self, benchmark):
        write_response(benchmark, "a.json", [{"text": "hello"}, "not-a-dict"])
        stats = benchmark.get_stats()
        assert stats["total_responses"] == 2
        assert stats["avg_length"] == 5.0


class TestEvaluateLatest:
    def test_empty(self, benchmark):
        assert benchmark.evaluate_latest() == {"responses_analyzed": 0, "metrics": {}}

    def test_analyzes_limited_recent(self, benchmark):
        for i in range(5):
            write_response(benchmark, f"r{i}.json", {"text": "word word word"})
        result = benchmark.evaluate_latest(limit=3)
        assert result["responses_analyzed"] == 3

    def test_avg_length_and_std(self, benchmark):
        write_response(benchmark, "a.json", {"text": "abcd"})
        write_response(benchmark, "b.json", {"text": "abcdef"})
        result = benchmark.evaluate_latest()
        metrics = result["metrics"]
        assert metrics["avg_length"] == 5.0
        assert metrics["length_std"] == 1.0

    def test_repetition_rate_zero_for_unique_bigrams(self, benchmark):
        write_response(benchmark, "a.json", {"text": "alpha beta gamma"})
        metrics = benchmark.evaluate_latest()["metrics"]
        assert metrics["repetition_rate"] == 0.0
        assert metrics["unique_bigram_ratio"] == 1.0

    def test_repetition_rate_detects_duplicates(self, benchmark):
        write_response(benchmark, "a.json", {"text": "go go go go"})
        metrics = benchmark.evaluate_latest()["metrics"]
        assert metrics["repetition_rate"] > 0.0
        assert metrics["unique_bigram_ratio"] < 1.0

    def test_single_word_text(self, benchmark):
        write_response(benchmark, "a.json", {"text": "lonely"})
        result = benchmark.evaluate_latest()
        assert result["metrics"]["repetition_rate"] == 0.0
        assert result["metrics"]["unique_bigram_ratio"] == 0.0


class TestClearHistory:
    def test_clears_files(self, benchmark):
        write_response(benchmark, "a.json", {"text": "x"})
        assert benchmark._responses_dir.exists()
        benchmark.clear_history()
        assert benchmark._load_responses() == []
        assert benchmark._responses_dir.exists()


class TestSingleton:
    def test_get_and_reset(self):
        bd.reset_benchmark_domain()
        first = bd.get_benchmark_domain()
        assert bd.get_benchmark_domain() is first
        bd.reset_benchmark_domain()
        assert bd.get_benchmark_domain() is not first
        bd.reset_benchmark_domain()
