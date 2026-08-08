"""Tests for domains.feedback.lora_eval — LoRA evaluation and quality scoring."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from domains.feedback.lora_eval import (
    BLEUScorer,
    EvalResult,
    LoRAEvaluator,
    PersonalityScore,
    get_lora_evaluator,
)

import domains.feedback.lora_eval as mod


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset global singleton."""
    mod._global_eval = None
    yield
    mod._global_eval = None


@pytest.fixture
def evaluator(tmp_path):
    """Create a LoRAEvaluator with temp dirs and a mock generator."""
    gen = lambda prompt: f"Response to: {prompt}"
    return LoRAEvaluator(
        eval_dir=str(tmp_path / "eval"),
        generator=gen,
    )


class TestEvalResult:
    def test_to_dict(self):
        r = EvalResult(
            timestamp="2024-01-01T00:00:00", adapter_path=None,
            prompts=8, references=5, perplexity=10.0, bleu=25.0,
            avg_response_len=15.0, inference_time_sec=1.5,
            tokens_per_sec=10.0, personality_score=0.6,
        )
        d = r.to_dict()
        assert d["timestamp"] == "2024-01-01T00:00:00"
        assert d["perplexity"] == 10.0
        assert "quality_delta" not in d

    def test_to_dict_with_adapter(self):
        r = EvalResult(
            timestamp="t", adapter_path="/tmp/a.npz",
            prompts=1, references=0, perplexity=None, bleu=None,
            avg_response_len=5.0, inference_time_sec=0.1,
            tokens_per_sec=50.0, personality_score=0.3,
        )
        d = r.to_dict()
        assert d["adapter_path"] == "/tmp/a.npz"

    def test_quality_delta_default(self):
        r = EvalResult(
            timestamp="t", adapter_path=None,
            prompts=1, references=0, perplexity=1.0, bleu=1.0,
            avg_response_len=1.0, inference_time_sec=0.1,
            tokens_per_sec=10.0, personality_score=0.5,
        )
        assert r.quality_delta is None


class TestBLEUScorer:
    def test_identical_strings(self):
        score = BLEUScorer.score("hello world", "hello world")
        assert score > 80

    def test_completely_different(self):
        score = BLEUScorer.score("aaa bbb", "xxx yyy")
        assert score == 0.0

    def test_empty_candidate(self):
        assert BLEUScorer.score("", "hello") == 0.0

    def test_empty_reference(self):
        assert BLEUScorer.score("hello", "") == 0.0

    def test_both_empty(self):
        assert BLEUScorer.score("", "") == 0.0

    def test_partial_overlap(self):
        score = BLEUScorer.score("the cat sat", "the cat played")
        assert 0 < score < 100

    def test_short_text(self):
        score = BLEUScorer.score("hi", "hi there")
        assert score > 0

    def test_get_ngrams(self):
        ngrams = BLEUScorer._get_ngrams(["a", "b", "c"], 2)
        assert ("a", "b") in ngrams
        assert ("b", "c") in ngrams

    def test_single_word(self):
        score = BLEUScorer.score("hello", "hello world")
        assert score > 0


class TestPersonalityScore:
    def test_to_dict(self):
        ps = PersonalityScore(
            soul_name="assistant", warmth_score=0.8,
            creativity_score=0.5, formality_score=0.6,
            coherence_score=0.7, overall=0.65,
        )
        d = ps.to_dict()
        assert d["soul"] == "assistant"
        assert d["warmth"] == 0.8
        assert d["overall"] == 0.65
        assert len(d) == 6


class TestLoRAEvaluatorInit:
    def test_defaults(self):
        ev = LoRAEvaluator()
        assert ev.eval_prompts == ev.EVAL_PROMPTS
        assert ev._model is None
        assert ev._generator is None

    def test_custom_prompts(self):
        ev = LoRAEvaluator(eval_prompts=["p1", "p2"])
        assert ev.eval_prompts == ["p1", "p2"]

    def test_eval_dir_created(self, tmp_path):
        d = tmp_path / "new_eval_dir"
        LoRAEvaluator(eval_dir=str(d))
        assert d.is_dir()


class TestAvailable:
    def test_available_with_generator(self, evaluator):
        assert evaluator.available() is True

    def test_not_available_without_any(self, tmp_path):
        ev = LoRAEvaluator(base_model=str(tmp_path / "nonexistent.safetensors"))
        assert ev.available() is False

    def test_available_with_generator_override(self, evaluator):
        assert evaluator.available() is True


class TestSimulateGeneration:
    def test_returns_tuple(self, evaluator):
        text, latency, tps = evaluator._simulate_generation("test prompt")
        assert isinstance(text, str)
        assert latency > 0
        assert tps > 0

    def test_with_adapter(self, evaluator):
        _, lat1, _ = evaluator._simulate_generation("test", adapter_path=None)
        _, lat2, _ = evaluator._simulate_generation("test", adapter_path="/some/path")
        # Both should return valid results
        assert lat1 > 0
        assert lat2 > 0

    def test_deterministic(self, evaluator):
        t1, _, _ = evaluator._simulate_generation("same prompt")
        t2, _, _ = evaluator._simulate_generation("same prompt")
        assert t1 == t2


class TestScorePersonality:
    def test_assistant_soul(self, evaluator):
        ps = evaluator._score_personality("Thank you for helping me!", "assistant")
        assert isinstance(ps, PersonalityScore)
        assert ps.soul_name == "assistant"
        assert 0 <= ps.overall <= 2

    def test_creative_soul(self, evaluator):
        ps = evaluator._score_personality("Imagine a world of wonder", "creative")
        assert ps.soul_name == "creative"
        assert ps.creativity_score >= 0

    def test_unknown_soul_falls_back(self, evaluator):
        ps = evaluator._score_personality("test", "nonexistent")
        assert ps.soul_name == "nonexistent"

    def test_empty_text(self, evaluator):
        ps = evaluator._score_personality("", "assistant")
        assert isinstance(ps, PersonalityScore)


class TestFmt:
    def test_format_number(self):
        assert LoRAEvaluator._fmt(3.14159, 2) == "3.14"

    def test_format_none(self):
        assert LoRAEvaluator._fmt(None) == "n/a"

    def test_format_zero(self):
        assert LoRAEvaluator._fmt(0.0, 3) == "0.000"

    def test_format_integer(self):
        assert LoRAEvaluator._fmt(42.0, 0) == "42"


class TestCompare:
    def test_improved_verdict(self, evaluator):
        baseline = EvalResult(
            timestamp="t", adapter_path=None, prompts=1, references=1,
            perplexity=10.0, bleu=20.0, avg_response_len=10.0,
            inference_time_sec=1.0, tokens_per_sec=10.0, personality_score=0.3,
        )
        with_adapter = EvalResult(
            timestamp="t", adapter_path="/a", prompts=1, references=1,
            perplexity=5.0, bleu=40.0, avg_response_len=15.0,
            inference_time_sec=1.0, tokens_per_sec=12.0, personality_score=0.6,
        )
        delta = evaluator.compare(baseline, with_adapter)
        assert delta["verdict"] == "improved"
        assert delta["perplexity_delta"] < 0
        assert delta["bleu_delta"] > 0
        assert delta["throughput_delta"] > 0
        assert delta["personality_delta"] > 0

    def test_degraded_verdict(self, evaluator):
        baseline = EvalResult(
            timestamp="t", adapter_path=None, prompts=1, references=1,
            perplexity=None, bleu=40.0, avg_response_len=10.0,
            inference_time_sec=1.0, tokens_per_sec=10.0, personality_score=0.8,
        )
        with_adapter = EvalResult(
            timestamp="t", adapter_path="/a", prompts=1, references=1,
            perplexity=None, bleu=10.0, avg_response_len=5.0,
            inference_time_sec=1.0, tokens_per_sec=5.0, personality_score=0.2,
        )
        delta = evaluator.compare(baseline, with_adapter)
        assert delta["verdict"] == "degraded"

    def test_no_metrics(self, evaluator):
        baseline = EvalResult(
            timestamp="t", adapter_path=None, prompts=1, references=0,
            perplexity=None, bleu=None, avg_response_len=10.0,
            inference_time_sec=1.0, tokens_per_sec=None, personality_score=None,
        )
        with_adapter = EvalResult(
            timestamp="t", adapter_path="/a", prompts=1, references=0,
            perplexity=None, bleu=None, avg_response_len=10.0,
            inference_time_sec=1.0, tokens_per_sec=None, personality_score=None,
        )
        delta = evaluator.compare(baseline, with_adapter)
        assert delta["verdict"] == "degraded"

    def test_none_perplexity(self, evaluator):
        baseline = EvalResult(
            timestamp="t", adapter_path=None, prompts=1, references=0,
            perplexity=None, bleu=None, avg_response_len=10.0,
            inference_time_sec=1.0, tokens_per_sec=10.0, personality_score=0.5,
        )
        with_adapter = EvalResult(
            timestamp="t", adapter_path="/a", prompts=1, references=0,
            perplexity=None, bleu=None, avg_response_len=10.0,
            inference_time_sec=1.0, tokens_per_sec=12.0, personality_score=0.5,
        )
        delta = evaluator.compare(baseline, with_adapter)
        assert "perplexity_delta" not in delta
        assert "throughput_delta" in delta


class TestCompareWithReport:
    def test_generates_report(self, evaluator):
        baseline = EvalResult(
            timestamp="t", adapter_path=None, prompts=1, references=1,
            perplexity=10.0, bleu=20.0, avg_response_len=10.0,
            inference_time_sec=1.0, tokens_per_sec=10.0, personality_score=0.3,
        )
        with_adapter = EvalResult(
            timestamp="t", adapter_path="/a", prompts=1, references=1,
            perplexity=5.0, bleu=40.0, avg_response_len=15.0,
            inference_time_sec=1.0, tokens_per_sec=12.0, personality_score=0.6,
        )
        report = evaluator.compare_with_report(baseline, with_adapter)
        assert "LoRA EVALUATION REPORT" in report
        assert "VERDICT" in report
        assert "IMPROVED" in report


class TestRun:
    def test_run_with_generator(self, evaluator):
        result = evaluator.run(save=False)
        assert isinstance(result, EvalResult)
        assert result.prompts == len(evaluator.eval_prompts)
        assert result.inference_time_sec > 0

    def test_run_saves_results(self, evaluator, tmp_path):
        evaluator.run(save=True)
        files = list((tmp_path / "eval").glob("baseline_*.json"))
        assert len(files) > 0

    def test_run_with_adapter_path(self, evaluator, tmp_path):
        result = evaluator.run(adapter_path="/fake/adapter.npz", save=False)
        assert result.adapter_path == "/fake/adapter.npz"

    def test_run_all_prompts_processed(self, evaluator):
        result = evaluator.run(save=False)
        assert result.prompts == len(evaluator.eval_prompts)

    def test_run_with_custom_prompts(self, tmp_path):
        gen = lambda p: f"reply to {p}"
        ev = LoRAEvaluator(eval_dir=str(tmp_path), eval_prompts=["q1"], generator=gen)
        result = ev.run(save=False)
        assert result.prompts == 1


class TestGetHistory:
    def test_empty_history(self, evaluator):
        assert evaluator.get_history() == []

    def test_loads_saved_results(self, evaluator):
        evaluator.run(save=True)
        history = evaluator.get_history()
        assert len(history) >= 1

    def test_limit(self, evaluator):
        for i in range(3):
            evaluator.run(save=True)
            time.sleep(1.1)
        history = evaluator.get_history(limit=1)
        assert len(history) == 1

    def test_handles_corrupt_files(self, evaluator, tmp_path):
        corrupt = evaluator.eval_dir / "baseline_corrupt.json"
        corrupt.write_text("not json {{{")
        history = evaluator.get_history()
        assert history == []


class TestExportAdapterAsSou:
    def test_export_creates_soul_file(self, evaluator, tmp_path):
        adapter_path = tmp_path / "test_adapter.npz"
        np.savez(adapter_path, W_a=np.zeros((4, 16)), W_b=np.zeros((16, 4)), rank=4, alpha=8.0, source_users=["u1"])
        output = tmp_path / "output.soul"
        result = evaluator.export_adapter_as_sou(
            adapter_npz=str(adapter_path),
            soul_name="test_soul",
            eval_delta={"verdict": "improved", "perplexity_delta": -2.0},
            output_sou=str(output),
        )
        assert Path(result).exists()

    def test_export_default_path(self, evaluator, tmp_path):
        adapter_path = tmp_path / "my_adapter.npz"
        np.savez(adapter_path, W_a=np.zeros((2, 4)), W_b=np.zeros((4, 2)), rank=2, alpha=4.0)
        result = evaluator.export_adapter_as_sou(
            adapter_npz=str(adapter_path),
            soul_name="test",
            eval_delta={"verdict": "unknown"},
        )
        assert Path(result).exists()


class TestSingleton:
    def test_get_lora_evaluator_returns_same(self, tmp_path):
        a = get_lora_evaluator()
        b = get_lora_evaluator()
        assert a is b
