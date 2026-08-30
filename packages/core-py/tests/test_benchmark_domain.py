"""Tests for domains.benchmark.domain — BenchmarkDomain, BenchmarkResult."""

import json
import pytest
from domains.benchmark.domain import (
    BenchmarkDomain, BenchmarkResult,
    get_benchmark_domain, reset_benchmark_domain,
)


# ── BenchmarkResult ──────────────────────────────────────────────────

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

    def test_all_fields(self):
        r = BenchmarkResult(
            timestamp="2025-01-01",
            model="m",
            num_responses=1,
            avg_length=1.0,
            length_std=0.0,
            repetition_rate=0.0,
            repetition_bigrams=0,
            avg_log_prob=0.0,
            unique_bigrams=1.0,
            unique_trigrams=1.0,
        )
        assert r.timestamp == "2025-01-01"
        assert r.avg_length == 1.0
        assert r.unique_trigrams == 1.0

    def test_equality(self):
        r1 = BenchmarkResult(
            timestamp="t", model="m", num_responses=1, avg_length=1.0,
            length_std=0.0, repetition_rate=0.0, repetition_bigrams=0,
            avg_log_prob=0.0, unique_bigrams=1.0, unique_trigrams=1.0,
        )
        r2 = BenchmarkResult(
            timestamp="t", model="m", num_responses=1, avg_length=1.0,
            length_std=0.0, repetition_rate=0.0, repetition_bigrams=0,
            avg_log_prob=0.0, unique_bigrams=1.0, unique_trigrams=1.0,
        )
        assert r1 == r2


# ── Singleton ────────────────────────────────────────────────────────

class TestSingleton:
    def test_singleton(self):
        reset_benchmark_domain()
        a = get_benchmark_domain()
        b = get_benchmark_domain()
        assert a is b
        reset_benchmark_domain()

    def test_reset_creates_new(self):
        reset_benchmark_domain()
        a = get_benchmark_domain()
        reset_benchmark_domain()
        b = get_benchmark_domain()
        assert a is not b
        reset_benchmark_domain()


# ── BenchmarkDomain ──────────────────────────────────────────────────

class TestBenchmarkDomain:
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


# ── get_stats detailed ──────────────────────────────────────────────

class TestGetStats:
    def test_stats_multiple_models(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([
            {"text": "a", "model": "gpt2"},
            {"text": "b", "model": "llama"},
            {"text": "c", "model": "gpt2"},
        ]))
        stats = bd.get_stats()
        assert stats["total_responses"] == 3
        assert set(stats["models"]) == {"gpt2", "llama"}

    def test_stats_avg_length(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([
            {"text": "ab", "model": "m"},
            {"text": "abcd", "model": "m"},
        ]))
        stats = bd.get_stats()
        assert stats["avg_length"] == 3.0

    def test_stats_no_dir(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path / "nonexistent"
        stats = bd.get_stats()
        assert stats["total_responses"] == 0

    def test_stats_handles_malformed_json(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "bad.json"
        f.write_text("not json")
        stats = bd.get_stats()
        assert stats["total_responses"] == 0

    def test_stats_handles_non_list_json(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "single.json"
        f.write_text(json.dumps({"text": "hello", "model": "m"}))
        stats = bd.get_stats()
        assert stats["total_responses"] == 1


# ── evaluate_latest detailed ────────────────────────────────────────

class TestEvaluateLatest:
    def test_evaluate_unique_bigrams(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([
            {"text": "the cat sat on the mat", "model": "m"},
        ]))
        result = bd.evaluate_latest()
        assert result["metrics"]["unique_bigram_ratio"] > 0
        assert result["metrics"]["unique_bigram_ratio"] <= 1.0

    def test_evaluate_length_std(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([
            {"text": "short", "model": "m"},
            {"text": "a much longer response here", "model": "m"},
        ]))
        result = bd.evaluate_latest()
        assert result["metrics"]["length_std"] > 0

    def test_evaluate_limit(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        data = [{"text": f"response {i}", "model": "m"} for i in range(100)]
        f.write_text(json.dumps(data))
        result = bd.evaluate_latest(limit=10)
        assert result["responses_analyzed"] == 10

    def test_evaluate_single_response_no_repetition(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([
            {"text": "unique words here only once", "model": "m"},
        ]))
        result = bd.evaluate_latest()
        assert result["responses_analyzed"] == 1
        assert result["metrics"]["repetition_rate"] == 0.0

    def test_evaluate_empty_text(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([
            {"text": "", "model": "m"},
        ]))
        result = bd.evaluate_latest()
        assert result["responses_analyzed"] == 1
        assert result["metrics"]["avg_length"] == 0

    def test_evaluate_multiple_files(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f1 = tmp_path / "a.json"
        f1.write_text(json.dumps([{"text": "hello", "model": "m"}]))
        f2 = tmp_path / "b.json"
        f2.write_text(json.dumps([{"text": "world", "model": "m"}]))
        result = bd.evaluate_latest()
        assert result["responses_analyzed"] == 2


# ── clear_history detailed ──────────────────────────────────────────

class TestClearHistory:
    def test_clear_preserves_dir(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([{"text": "x"}]))
        bd.clear_history()
        assert tmp_path.exists()
        assert tmp_path.is_dir()

    def test_clear_multiple_files(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        for i in range(5):
            (tmp_path / f"f{i}.json").write_text(json.dumps([{"text": str(i)}]))
        bd.clear_history()
        assert list(tmp_path.iterdir()) == []

    def test_clear_empty_dir(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        bd.clear_history()  # no-op, should not raise
        assert tmp_path.exists()


# ── Multiple responses per file ──────────────────────────────────────

class TestMultipleResponsesPerFile:
    def test_file_with_list_of_responses(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "multi.json"
        f.write_text(json.dumps([
            {"text": "first", "model": "m"},
            {"text": "second", "model": "m"},
            {"text": "third", "model": "m"},
        ]))
        stats = bd.get_stats()
        assert stats["total_responses"] == 3

    def test_mixed_files(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f1 = tmp_path / "list.json"
        f1.write_text(json.dumps([{"text": "a"}, {"text": "b"}]))
        f2 = tmp_path / "single.json"
        f2.write_text(json.dumps({"text": "c"}))
        stats = bd.get_stats()
        assert stats["total_responses"] == 3


# ── Edge cases ──────────────────────────────────────────────────────

class TestEdgeCases:
    def test_no_text_field(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([{"model": "m"}]))
        result = bd.evaluate_latest()
        assert result["responses_analyzed"] == 1
        assert result["metrics"]["avg_length"] == 0

    def test_no_model_field(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([{"text": "hello"}]))
        stats = bd.get_stats()
        assert "unknown" in stats["models"]

    def test_very_long_text(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        long_text = "word " * 10000
        f.write_text(json.dumps([{"text": long_text, "model": "m"}]))
        result = bd.evaluate_latest()
        assert result["responses_analyzed"] == 1
        assert result["metrics"]["avg_length"] > 1000

    def test_special_characters(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([{"text": "hello! @#$%^&*() world", "model": "m"}]))
        result = bd.evaluate_latest()
        assert result["responses_analyzed"] == 1


# ── Additional coverage ──────────────────────────────────────────────

class TestBenchmarkResultExtended:
    def test_inequality_different_model(self):
        r1 = BenchmarkResult(
            timestamp="t", model="m1", num_responses=1, avg_length=1.0,
            length_std=0.0, repetition_rate=0.0, repetition_bigrams=0,
            avg_log_prob=0.0, unique_bigrams=1.0, unique_trigrams=1.0,
        )
        r2 = BenchmarkResult(
            timestamp="t", model="m2", num_responses=1, avg_length=1.0,
            length_std=0.0, repetition_rate=0.0, repetition_bigrams=0,
            avg_log_prob=0.0, unique_bigrams=1.0, unique_trigrams=1.0,
        )
        assert r1 != r2

    def test_inequality_different_values(self):
        r1 = BenchmarkResult(
            timestamp="t", model="m", num_responses=1, avg_length=1.0,
            length_std=0.0, repetition_rate=0.0, repetition_bigrams=0,
            avg_log_prob=0.0, unique_bigrams=1.0, unique_trigrams=1.0,
        )
        r2 = BenchmarkResult(
            timestamp="t", model="m", num_responses=2, avg_length=1.0,
            length_std=0.0, repetition_rate=0.0, repetition_bigrams=0,
            avg_log_prob=0.0, unique_bigrams=1.0, unique_trigrams=1.0,
        )
        assert r1 != r2

    def test_extreme_values(self):
        r = BenchmarkResult(
            timestamp="t", model="m", num_responses=0, avg_length=0.0,
            length_std=0.0, repetition_rate=1.0, repetition_bigrams=999,
            avg_log_prob=-100.0, unique_bigrams=0.0, unique_trigrams=0.0,
        )
        assert r.repetition_rate == 1.0
        assert r.avg_log_prob == -100.0

    def test_not_equal_to_non_benchmark_result(self):
        r = BenchmarkResult(
            timestamp="t", model="m", num_responses=1, avg_length=1.0,
            length_std=0.0, repetition_rate=0.0, repetition_bigrams=0,
            avg_log_prob=0.0, unique_bigrams=1.0, unique_trigrams=1.0,
        )
        assert r != "not a result"
        assert r != 42


class TestSingletonExtended:
    def test_multiple_resets(self):
        for _ in range(5):
            reset_benchmark_domain()
        a = get_benchmark_domain()
        reset_benchmark_domain()
        b = get_benchmark_domain()
        assert a is not b
        reset_benchmark_domain()

    def test_get_without_prior_reset(self):
        reset_benchmark_domain()
        a = get_benchmark_domain()
        b = get_benchmark_domain()
        assert a is b
        reset_benchmark_domain()


class TestGetStatsExtended:
    def test_stats_all_empty_text(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([
            {"text": "", "model": "m"},
            {"text": "", "model": "m"},
        ]))
        stats = bd.get_stats()
        assert stats["total_responses"] == 2
        assert stats["avg_length"] == 0

    def test_stats_mixed_empty_and_nonempty(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([
            {"text": "", "model": "m"},
            {"text": "four", "model": "m"},
        ]))
        stats = bd.get_stats()
        assert stats["total_responses"] == 2
        assert stats["avg_length"] == 2.0

    def test_stats_after_clear(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([{"text": "hello", "model": "m"}]))
        bd.clear_history()
        stats = bd.get_stats()
        assert stats["total_responses"] == 0

    def test_stats_non_json_files_ignored(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        (tmp_path / "readme.txt").write_text("not json")
        f = tmp_path / "data.json"
        f.write_text(json.dumps([{"text": "hello", "model": "m"}]))
        stats = bd.get_stats()
        assert stats["total_responses"] == 1


class TestEvaluateLatestExtended:
    def test_evaluate_default_limit(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        data = [{"text": f"resp {i}", "model": "m"} for i in range(60)]
        f = tmp_path / "resp.json"
        f.write_text(json.dumps(data))
        result = bd.evaluate_latest()
        assert result["responses_analyzed"] == 50

    def test_evaluate_high_repetition(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([
            {"text": "the the the the the the the the", "model": "m"},
        ]))
        result = bd.evaluate_latest()
        assert result["metrics"]["repetition_rate"] > 0.5

    def test_evaluate_single_word_texts(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([
            {"text": "hello", "model": "m"},
            {"text": "world", "model": "m"},
        ]))
        result = bd.evaluate_latest()
        assert result["responses_analyzed"] == 2
        assert result["metrics"]["avg_length"] == 5.0

    def test_evaluate_after_clear(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([{"text": "hello", "model": "m"}]))
        bd.clear_history()
        result = bd.evaluate_latest()
        assert result["responses_analyzed"] == 0

    def test_evaluate_non_dict_entries_skipped(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([
            {"text": "valid", "model": "m"},
            "not a dict",
            42,
        ]))
        result = bd.evaluate_latest()
        assert result["responses_analyzed"] == 1

    def test_evaluate_punctuation_only(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([{"text": "!@#$%^&*()", "model": "m"}]))
        result = bd.evaluate_latest()
        assert result["responses_analyzed"] == 1
        assert result["metrics"]["avg_length"] == 10

    def test_evaluate_unicode_text(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([{"text": "hello \u4e16\u754c", "model": "m"}]))
        result = bd.evaluate_latest()
        assert result["responses_analyzed"] == 1

    def test_evaluate_repeated_bigrams_low_unique(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([
            {"text": "a b a b a b a b", "model": "m"},
        ]))
        result = bd.evaluate_latest()
        assert result["metrics"]["unique_bigram_ratio"] < 0.5

    def test_evaluate_all_unique_bigrams(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([
            {"text": "one two three four five six", "model": "m"},
        ]))
        result = bd.evaluate_latest()
        assert result["metrics"]["unique_bigram_ratio"] == 1.0


class TestClearHistoryExtended:
    def test_clear_then_add(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        f = tmp_path / "resp.json"
        f.write_text(json.dumps([{"text": "old", "model": "m"}]))
        bd.clear_history()
        f2 = tmp_path / "new.json"
        f2.write_text(json.dumps([{"text": "new", "model": "m"}]))
        stats = bd.get_stats()
        assert stats["total_responses"] == 1

    def test_clear_non_json_files_also_removed(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        (tmp_path / "notes.txt").write_text("will be removed")
        f = tmp_path / "data.json"
        f.write_text(json.dumps([{"text": "hello"}]))
        bd.clear_history()
        assert not f.exists()
        assert not (tmp_path / "notes.txt").exists()


class TestMultipleFilesExtended:
    def test_many_files(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        for i in range(10):
            (tmp_path / f"f{i}.json").write_text(
                json.dumps([{"text": f"response {i}", "model": "m"}])
            )
        stats = bd.get_stats()
        assert stats["total_responses"] == 10

    def test_nested_non_json_ignored(self, tmp_path):
        bd = BenchmarkDomain()
        bd._responses_dir = tmp_path
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file.json").write_text(
            json.dumps([{"text": "nested"}])
        )
        f = tmp_path / "top.json"
        f.write_text(json.dumps([{"text": "top level"}]))
        stats = bd.get_stats()
        assert stats["total_responses"] == 1
