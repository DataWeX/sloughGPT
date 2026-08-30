"""Tests for lora_eval — BLEUScorer, EvalResult, PersonalityScore, LoRAEvaluator pure logic."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from domains.feedback.lora_eval import (
    BLEUScorer,
    EvalResult,
    LoRAEvaluator,
    PersonalityScore,
    get_lora_evaluator,
)


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------

class TestEvalResult:
    def test_basic_fields(self):
        r = EvalResult(
            timestamp="2025-01-01T00:00:00",
            adapter_path=None,
            prompts=5,
            references=3,
            perplexity=42.0,
            bleu=25.5,
            avg_response_len=12.0,
            inference_time_sec=1.23,
            tokens_per_sec=100.0,
            personality_score=0.75,
        )
        assert r.prompts == 5
        assert r.references == 3
        assert r.adapter_path is None
        assert r.quality_delta is None

    def test_to_dict_excludes_quality_delta(self):
        r = EvalResult(
            timestamp="2025-01-01T00:00:00",
            adapter_path="some.npz",
            prompts=2,
            references=1,
            perplexity=10.0,
            bleu=30.0,
            avg_response_len=8.0,
            inference_time_sec=0.5,
            tokens_per_sec=50.0,
            personality_score=0.6,
            quality_delta=5.0,
        )
        d = r.to_dict()
        assert "quality_delta" not in d
        assert d["adapter_path"] == "some.npz"
        assert d["perplexity"] == 10.0

    def test_to_dict_with_none_adapter(self):
        r = EvalResult(
            timestamp="2025-01-01T00:00:00",
            adapter_path=None,
            prompts=1,
            references=0,
            perplexity=None,
            bleu=None,
            avg_response_len=5.0,
            inference_time_sec=0.1,
            tokens_per_sec=None,
            personality_score=None,
        )
        d = r.to_dict()
        assert d["perplexity"] is None
        assert d["bleu"] is None

    def test_to_dict_contains_all_expected_keys(self):
        r = EvalResult(
            timestamp="t",
            adapter_path="a",
            prompts=1,
            references=1,
            perplexity=1.0,
            bleu=1.0,
            avg_response_len=1.0,
            inference_time_sec=1.0,
            tokens_per_sec=1.0,
            personality_score=1.0,
        )
        d = r.to_dict()
        expected = {
            "timestamp", "adapter_path", "prompts", "references",
            "perplexity", "bleu", "avg_response_len", "inference_time_sec",
            "tokens_per_sec", "personality_score",
        }
        assert expected == set(d.keys())

    def test_to_dict_preserves_none_quality_delta(self):
        r = EvalResult(
            timestamp="t", adapter_path=None, prompts=0, references=0,
            perplexity=None, bleu=None, avg_response_len=0.0,
            inference_time_sec=0.0, tokens_per_sec=None, personality_score=None,
            quality_delta=None,
        )
        d = r.to_dict()
        assert "quality_delta" not in d

    def test_to_dict_with_quality_delta_set(self):
        r = EvalResult(
            timestamp="t", adapter_path=None, prompts=0, references=0,
            perplexity=None, bleu=None, avg_response_len=0.0,
            inference_time_sec=0.0, tokens_per_sec=None, personality_score=None,
            quality_delta=3.14,
        )
        d = r.to_dict()
        assert "quality_delta" not in d


# ---------------------------------------------------------------------------
# BLEUScorer
# ---------------------------------------------------------------------------

class TestBLEUScorer:
    def test_identical_strings(self):
        score = BLEUScorer.score("hello world", "hello world")
        assert score == 100.0

    def test_no_overlap(self):
        score = BLEUScorer.score("foo bar baz", "qux quux corge")
        assert score == 0.0

    def test_empty_candidate(self):
        score = BLEUScorer.score("", "hello world")
        assert score == 0.0

    def test_empty_reference(self):
        score = BLEUScorer.score("hello world", "")
        assert score == 0.0

    def test_both_empty(self):
        score = BLEUScorer.score("", "")
        assert score == 0.0

    def test_partial_overlap(self):
        score = BLEUScorer.score("the cat sat on the mat", "the cat is on the mat")
        assert 0 < score < 100

    def test_max_n_clamped(self):
        score = BLEUScorer.score("a b", "a b", max_n=10)
        assert score == 100.0

    def test_single_word(self):
        score = BLEUScorer.score("hello", "hello")
        assert score == 100.0

    def test_score_is_percentage(self):
        score = BLEUScorer.score("a b c", "a b c")
        assert score == 100.0
        score_partial = BLEUScorer.score("a b c", "a b d")
        assert 0 < score_partial < 100

    def test_get_ngrams(self):
        ngrams = BLEUScorer._get_ngrams(["a", "b", "c"], 2)
        assert ("a", "b") in ngrams
        assert ("b", "c") in ngrams
        assert len(ngrams) == 2

    def test_get_ngrams_unigram(self):
        ngrams = BLEUScorer._get_ngrams(["x", "y"], 1)
        assert ("x",) in ngrams
        assert ("y",) in ngrams

    def test_whitespace_handling(self):
        score = BLEUScorer.score("  hello   world  ", "hello world")
        assert score == 100.0

    def test_max_n_1(self):
        score = BLEUScorer.score("a b c", "a b d", max_n=1)
        assert 0 < score <= 100

    def test_long_candidate_short_reference(self):
        score = BLEUScorer.score("a b c d e f g h", "a b")
        assert score >= 0

    def test_long_reference_short_candidate(self):
        score = BLEUScorer.score("a", "a b c d e f g h")
        assert score >= 0

    def test_identical_single_token(self):
        score = BLEUScorer.score("test", "test")
        assert score == 100.0

    def test_completely_different(self):
        score = BLEUScorer.score("apple banana cherry", "dog elephant frog")
        assert score == 0.0

    def test_get_ngrams_n_greater_than_tokens(self):
        ngrams = BLEUScorer._get_ngrams(["a"], 3)
        assert len(ngrams) == 0

    def test_get_ngrams_empty(self):
        ngrams = BLEUScorer._get_ngrams([], 1)
        assert len(ngrams) == 0


# ---------------------------------------------------------------------------
# PersonalityScore
# ---------------------------------------------------------------------------

class TestPersonalityScore:
    def test_to_dict(self):
        ps = PersonalityScore(
            soul_name="coder",
            warmth_score=0.8,
            creativity_score=0.6,
            formality_score=0.5,
            coherence_score=0.9,
            overall=0.7,
        )
        d = ps.to_dict()
        assert d["soul"] == "coder"
        assert d["warmth"] == 0.8
        assert d["overall"] == 0.7
        assert len(d) == 6

    def test_fields(self):
        ps = PersonalityScore(
            soul_name="assistant",
            warmth_score=0.0,
            creativity_score=0.0,
            formality_score=0.0,
            coherence_score=0.0,
            overall=0.0,
        )
        assert ps.soul_name == "assistant"
        assert ps.overall == 0.0

    def test_to_dict_all_keys(self):
        ps = PersonalityScore(
            soul_name="x", warmth_score=0.1, creativity_score=0.2,
            formality_score=0.3, coherence_score=0.4, overall=0.5,
        )
        d = ps.to_dict()
        assert set(d.keys()) == {"soul", "warmth", "creativity", "formality", "coherence", "overall"}

    def test_to_dict_preserves_values(self):
        ps = PersonalityScore(
            soul_name="test", warmth_score=1.0, creativity_score=0.0,
            formality_score=0.75, coherence_score=0.25, overall=0.5,
        )
        d = ps.to_dict()
        assert d["warmth"] == 1.0
        assert d["creativity"] == 0.0
        assert d["formality"] == 0.75
        assert d["coherence"] == 0.25


# ---------------------------------------------------------------------------
# LoRAEvaluator — _fmt
# ---------------------------------------------------------------------------

class TestLoRAEvaluatorFmt:
    def test_format_float(self):
        assert LoRAEvaluator._fmt(3.14159) == "3.14"

    def test_format_none(self):
        assert LoRAEvaluator._fmt(None) == "n/a"

    def test_format_custom_precision(self):
        assert LoRAEvaluator._fmt(1.23456, precision=4) == "1.2346"

    def test_format_zero(self):
        assert LoRAEvaluator._fmt(0.0) == "0.00"

    def test_format_negative(self):
        assert LoRAEvaluator._fmt(-5.5) == "-5.50"

    def test_format_large_number(self):
        assert LoRAEvaluator._fmt(999999.999, precision=1) == "1000000.0"

    def test_format_precision_0(self):
        assert LoRAEvaluator._fmt(3.7, precision=0) == "4"

    def test_format_precision_5(self):
        assert LoRAEvaluator._fmt(1.0, precision=5) == "1.00000"


# ---------------------------------------------------------------------------
# LoRAEvaluator — _score_personality
# ---------------------------------------------------------------------------

class TestLoRAEvaluatorScorePersonality:
    @pytest.fixture
    def evaluator(self, tmp_path):
        return LoRAEvaluator(eval_dir=str(tmp_path / "eval"))

    def test_assistant_keywords(self, evaluator):
        ps = evaluator._score_personality("I'm here to help you. Thank you for asking!", "assistant")
        assert ps.soul_name == "assistant"
        assert ps.warmth_score > 0
        assert ps.overall >= 0

    def test_coder_keywords(self, evaluator):
        ps = evaluator._score_personality("Define a function that returns the result.", "coder")
        assert ps.creativity_score > 0

    def test_teacher_keywords(self, evaluator):
        ps = evaluator._score_personality("First, understand the concept. Next, try the example.", "teacher")
        assert ps.creativity_score > 0

    def test_unknown_soul_falls_back(self, evaluator):
        ps = evaluator._score_personality("help and assist", "unknown_soul")
        assert ps.soul_name == "unknown_soul"

    def test_empty_text(self, evaluator):
        ps = evaluator._score_personality("", "assistant")
        assert ps.overall >= 0

    def test_warmth_capped_at_one(self, evaluator):
        text = "thank thank thank thank thank thank thank thank thank thank"
        ps = evaluator._score_personality(text, "assistant")
        assert ps.warmth_score <= 1.0

    def test_analyst_keywords(self, evaluator):
        text = "However, the evidence suggests the conclusion is correct based on the data."
        ps = evaluator._score_personality(text, "analyst")
        assert ps.creativity_score > 0

    def test_creative_keywords(self, evaluator):
        text = "Imagine the colors of the dream, perhaps a wonder of inspiration."
        ps = evaluator._score_personality(text, "creative")
        assert ps.creativity_score > 0

    def test_formality_increases_with_periods(self, evaluator):
        ps1 = evaluator._score_personality("hello", "assistant")
        ps2 = evaluator._score_personality("hello. world. test. another.", "assistant")
        assert ps2.formality_score >= ps1.formality_score

    def test_coherence_increases_with_sentences(self, evaluator):
        ps1 = evaluator._score_personality("one sentence", "assistant")
        ps2 = evaluator._score_personality("s1. s2. s3. s4. s5.", "assistant")
        assert ps2.coherence_score >= ps1.coherence_score

    def test_coherence_capped_at_one(self, evaluator):
        ps = evaluator._score_personality("s1. s2. s3. s4. s5. s6. s7. s8. s9. s10.", "assistant")
        assert ps.coherence_score <= 1.0

    def test_overall_weighted_average(self, evaluator):
        ps = evaluator._score_personality("help and assist thank", "assistant")
        # warmth is NOT capped before overall calc (capped only in return value)
        raw_warmth = sum(1 for k in ["thank", "great", "help", "appreciate", "wonderful"] if k in "help and assist thank") / max(len("help and assist thank".split()), 1) * 10
        expected = (
            raw_warmth * 0.3
            + ps.creativity_score * 0.3
            + ps.formality_score * 0.2
            + ps.coherence_score * 0.2
        )
        assert ps.overall == pytest.approx(expected)


# ---------------------------------------------------------------------------
# LoRAEvaluator — compare
# ---------------------------------------------------------------------------

class TestLoRAEvaluatorCompare:
    def _make_result(self, ppl=None, bleu=None, tps=None, personality=None):
        return EvalResult(
            timestamp="2025-01-01T00:00:00",
            adapter_path=None,
            prompts=5,
            references=3,
            perplexity=ppl,
            bleu=bleu,
            avg_response_len=10.0,
            inference_time_sec=1.0,
            tokens_per_sec=tps,
            personality_score=personality,
        )

    def test_improved_perplexity(self):
        ev = LoRAEvaluator.__new__(LoRAEvaluator)
        baseline = self._make_result(ppl=50.0, bleu=20.0, tps=100.0, personality=0.5)
        after = self._make_result(ppl=40.0, bleu=25.0, tps=110.0, personality=0.6)
        delta = ev.compare(baseline, after)
        assert delta["perplexity_delta"] == -10.0
        assert delta["bleu_delta"] == 5.0
        assert delta["throughput_delta"] == pytest.approx(10.0)
        assert delta["personality_delta"] == pytest.approx(0.1)
        assert delta["verdict"] == "improved"

    def test_degraded(self):
        ev = LoRAEvaluator.__new__(LoRAEvaluator)
        # perplexity=None to avoid perplexity_improvement_pct skewing the count
        baseline = self._make_result(ppl=None, bleu=30.0, tps=100.0, personality=0.8)
        after = self._make_result(ppl=None, bleu=10.0, tps=80.0, personality=0.3)
        delta = ev.compare(baseline, after)
        assert delta["bleu_delta"] == -20.0
        assert delta["throughput_delta"] == pytest.approx(-20.0)
        assert delta["personality_delta"] == pytest.approx(-0.5)
        assert delta["verdict"] == "degraded"

    def test_none_metrics_excluded(self):
        ev = LoRAEvaluator.__new__(LoRAEvaluator)
        baseline = self._make_result(ppl=None, bleu=None, tps=None, personality=None)
        after = self._make_result(ppl=None, bleu=None, tps=None, personality=None)
        delta = ev.compare(baseline, after)
        assert "perplexity_delta" not in delta
        assert "bleu_delta" not in delta
        assert "throughput_delta" not in delta
        assert "personality_delta" not in delta

    def test_mixed_results(self):
        ev = LoRAEvaluator.__new__(LoRAEvaluator)
        # perplexity=None to avoid perplexity_improvement_pct skewing the count
        # bleu improves, throughput degrades → 1 positive out of 2 total → mixed
        baseline = self._make_result(ppl=None, bleu=20.0, tps=100.0, personality=None)
        after = self._make_result(ppl=None, bleu=30.0, tps=80.0, personality=None)
        delta = ev.compare(baseline, after)
        assert delta["bleu_delta"] == 10.0
        assert delta["throughput_delta"] == pytest.approx(-20.0)
        assert delta["verdict"] == "mixed"

    def test_perplexity_improvement_overrides_mixed(self):
        ev = LoRAEvaluator.__new__(LoRAEvaluator)
        baseline = self._make_result(ppl=50.0, bleu=30.0, tps=100.0, personality=0.8)
        after = self._make_result(ppl=40.0, bleu=20.0, tps=80.0, personality=0.3)
        delta = ev.compare(baseline, after)
        assert delta["verdict"] == "improved"

    def test_only_personality_provided(self):
        ev = LoRAEvaluator.__new__(LoRAEvaluator)
        baseline = self._make_result(personality=0.5)
        after = self._make_result(personality=0.9)
        delta = ev.compare(baseline, after)
        assert delta["personality_delta"] == pytest.approx(0.4)
        assert delta["verdict"] == "improved"

    def test_only_throughput_provided(self):
        ev = LoRAEvaluator.__new__(LoRAEvaluator)
        baseline = self._make_result(tps=50.0)
        after = self._make_result(tps=100.0)
        delta = ev.compare(baseline, after)
        assert delta["throughput_delta"] == pytest.approx(100.0)
        assert delta["verdict"] == "improved"

    def test_only_bleu_provided(self):
        ev = LoRAEvaluator.__new__(LoRAEvaluator)
        baseline = self._make_result(bleu=10.0)
        after = self._make_result(bleu=20.0)
        delta = ev.compare(baseline, after)
        assert delta["bleu_delta"] == 10.0
        assert delta["verdict"] == "improved"

    def test_only_perplexity_worsened(self):
        ev = LoRAEvaluator.__new__(LoRAEvaluator)
        # Only perplexity provided — it worsens (positive delta)
        baseline = self._make_result(ppl=10.0)
        after = self._make_result(ppl=20.0)
        delta = ev.compare(baseline, after)
        assert delta["perplexity_delta"] == 10.0
        # perplexity_improvement_pct is -100.0 (negative), so positive=1 total=2, not > 50% → mixed
        assert delta["verdict"] == "mixed"

    def test_equal_metrics(self):
        ev = LoRAEvaluator.__new__(LoRAEvaluator)
        baseline = self._make_result(ppl=50.0, bleu=25.0, tps=100.0, personality=0.5)
        after = self._make_result(ppl=50.0, bleu=25.0, tps=100.0, personality=0.5)
        delta = ev.compare(baseline, after)
        assert delta["perplexity_delta"] == 0.0
        assert delta["bleu_delta"] == 0.0
        assert delta["throughput_delta"] == 0.0
        assert delta["personality_delta"] == 0.0

    def test_perplexity_improvement_pct(self):
        ev = LoRAEvaluator.__new__(LoRAEvaluator)
        baseline = self._make_result(ppl=100.0)
        after = self._make_result(ppl=80.0)
        delta = ev.compare(baseline, after)
        assert delta["perplexity_improvement_pct"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# LoRAEvaluator — compare_with_report
# ---------------------------------------------------------------------------

class TestLoRAEvaluatorCompareWithReport:
    def test_report_contains_verdict(self, tmp_path):
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"))
        baseline = EvalResult(
            timestamp="2025-01-01T00:00:00",
            adapter_path=None, prompts=2, references=1,
            perplexity=50.0, bleu=20.0, avg_response_len=10.0,
            inference_time_sec=1.0, tokens_per_sec=100.0, personality_score=0.5,
        )
        after = EvalResult(
            timestamp="2025-01-01T00:01:00",
            adapter_path="test.npz", prompts=2, references=1,
            perplexity=40.0, bleu=25.0, avg_response_len=12.0,
            inference_time_sec=0.8, tokens_per_sec=120.0, personality_score=0.6,
        )
        report = ev.compare_with_report(baseline, after)
        assert "IMPROVED" in report.upper()
        assert "LoRA EVALUATION REPORT" in report
        assert "Perplexity" in report
        assert "BLEU" in report
        assert "Throughput" in report
        assert "Personality" in report

    def test_report_degraded(self, tmp_path):
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"))
        baseline = EvalResult(
            timestamp="2025-01-01T00:00:00",
            adapter_path=None, prompts=1, references=1,
            perplexity=30.0, bleu=40.0, avg_response_len=10.0,
            inference_time_sec=1.0, tokens_per_sec=100.0, personality_score=0.9,
        )
        after = EvalResult(
            timestamp="2025-01-01T00:01:00",
            adapter_path="bad.npz", prompts=1, references=1,
            perplexity=60.0, bleu=10.0, avg_response_len=5.0,
            inference_time_sec=2.0, tokens_per_sec=50.0, personality_score=0.2,
        )
        report = ev.compare_with_report(baseline, after)
        assert "DEGRADED" in report.upper()

    def test_report_with_none_metrics(self, tmp_path):
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"))
        baseline = EvalResult(
            timestamp="2025-01-01T00:00:00",
            adapter_path=None, prompts=1, references=0,
            perplexity=None, bleu=None, avg_response_len=5.0,
            inference_time_sec=0.1, tokens_per_sec=None, personality_score=None,
        )
        after = EvalResult(
            timestamp="2025-01-01T00:01:00",
            adapter_path=None, prompts=1, references=0,
            perplexity=None, bleu=None, avg_response_len=5.0,
            inference_time_sec=0.1, tokens_per_sec=None, personality_score=None,
        )
        report = ev.compare_with_report(baseline, after)
        assert "n/a" in report
        assert "LoRA EVALUATION REPORT" in report

    def test_report_shows_adapter_paths(self, tmp_path):
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"))
        baseline = EvalResult(
            timestamp="t", adapter_path="base.npz", prompts=1, references=0,
            perplexity=None, bleu=None, avg_response_len=5.0,
            inference_time_sec=0.0, tokens_per_sec=None, personality_score=None,
        )
        after = EvalResult(
            timestamp="t", adapter_path="adapted.npz", prompts=1, references=0,
            perplexity=None, bleu=None, avg_response_len=5.0,
            inference_time_sec=0.0, tokens_per_sec=None, personality_score=None,
        )
        report = ev.compare_with_report(baseline, after)
        assert "base.npz" in report
        assert "adapted.npz" in report

    def test_report_mixed_verdict(self, tmp_path):
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"))
        baseline = EvalResult(
            timestamp="t", adapter_path=None, prompts=1, references=0,
            perplexity=50.0, bleu=30.0, avg_response_len=10.0,
            inference_time_sec=1.0, tokens_per_sec=100.0, personality_score=0.5,
        )
        after = EvalResult(
            timestamp="t", adapter_path="a.npz", prompts=1, references=0,
            perplexity=55.0, bleu=35.0, avg_response_len=10.0,
            inference_time_sec=1.0, tokens_per_sec=100.0, personality_score=0.5,
        )
        report = ev.compare_with_report(baseline, after)
        assert "MIXED" in report.upper()

    def test_report_perplexity_improvement_line(self, tmp_path):
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"))
        baseline = EvalResult(
            timestamp="t", adapter_path=None, prompts=1, references=0,
            perplexity=100.0, bleu=None, avg_response_len=10.0,
            inference_time_sec=1.0, tokens_per_sec=None, personality_score=None,
        )
        after = EvalResult(
            timestamp="t", adapter_path="a.npz", prompts=1, references=0,
            perplexity=80.0, bleu=None, avg_response_len=10.0,
            inference_time_sec=1.0, tokens_per_sec=None, personality_score=None,
        )
        report = ev.compare_with_report(baseline, after)
        assert "improved" in report.lower()
        assert "20.0%" in report


# ---------------------------------------------------------------------------
# LoRAEvaluator — available
# ---------------------------------------------------------------------------

class TestLoRAEvaluatorAvailable:
    def test_available_with_generator(self, tmp_path):
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"), generator=lambda p: "hi")
        assert ev.available() is True

    def test_not_available_without_model(self, tmp_path):
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"))
        assert ev.available() is False

    def test_available_with_model_file(self, tmp_path):
        model_path = tmp_path / "model.bin"
        model_path.touch()
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"), base_model=str(model_path))
        assert ev.available() is True


# ---------------------------------------------------------------------------
# LoRAEvaluator — _simulate_generation
# ---------------------------------------------------------------------------

class TestLoRAEvaluatorSimulateGeneration:
    def test_simulate_returns_tuple(self, tmp_path):
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"))
        text, latency, tps = ev._simulate_generation("test prompt")
        assert isinstance(text, str)
        assert latency > 0
        assert tps > 0

    def test_simulate_deterministic(self, tmp_path):
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"))
        t1, _, _ = ev._simulate_generation("same prompt")
        t2, _, _ = ev._simulate_generation("same prompt")
        assert t1 == t2

    def test_adapter_changes_length(self, tmp_path):
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"))
        np.random.seed(42)
        _, _, tps_no = ev._simulate_generation("prompt", adapter_path=None)
        np.random.seed(42)
        _, _, tps_with = ev._simulate_generation("prompt", adapter_path="fake.npz")
        assert tps_with != tps_no

    def test_simulate_contains_prompt(self, tmp_path):
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"))
        text, _, _ = ev._simulate_generation("hello world")
        assert "hello world" in text

    def test_simulate_different_prompts_different_text(self, tmp_path):
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"))
        t1, _, _ = ev._simulate_generation("prompt A")
        t2, _, _ = ev._simulate_generation("prompt B")
        assert t1 != t2

    def test_simulate_latency_range(self, tmp_path):
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"))
        latencies = []
        for _ in range(20):
            _, lat, _ = ev._simulate_generation("test")
            latencies.append(lat)
        assert all(0.05 <= l <= 0.15 for l in latencies)

    def test_simulate_tps_positive(self, tmp_path):
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"))
        for _ in range(10):
            _, _, tps = ev._simulate_generation("test")
            assert tps > 0


# ---------------------------------------------------------------------------
# LoRAEvaluator — run
# ---------------------------------------------------------------------------

class TestLoRAEvaluatorRun:
    def test_run_with_generator(self, tmp_path):
        responses = iter(["Hello there", "Python is great", "ML is cool",
                          "Roses are red", "I am an AI", "Use def keyword",
                          "42 is the answer", "Season of drifts"])
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            generator=lambda p: next(responses),
        )
        result = ev.run(save=True)
        assert result.prompts == 8
        assert result.inference_time_sec >= 0
        assert result.personality_score is not None

    def test_run_saves_files(self, tmp_path):
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            generator=lambda p: "test response",
        )
        ev.run(save=True)
        files = list((tmp_path / "eval").glob("*.json"))
        assert len(files) >= 2

    def test_run_no_save(self, tmp_path):
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            generator=lambda p: "test",
        )
        result = ev.run(save=False)
        files = list((tmp_path / "eval").glob("*.json"))
        assert len(files) == 0
        assert result.adapter_path is None

    def test_run_with_adapter(self, tmp_path):
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            generator=lambda p: "response with adapter",
        )
        result = ev.run(adapter_path=None, save=False)
        assert result.adapter_path is None

    def test_run_custom_prompts(self, tmp_path):
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            eval_prompts=["custom prompt"],
            generator=lambda p: "custom response",
        )
        result = ev.run(save=False)
        assert result.prompts == 1

    def test_run_with_reference_responses(self, tmp_path):
        gen_map = {
            "Hello, how are you?": "I'm doing well!",
            "What is Python?": "Python is a language.",
        }
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            eval_prompts=list(gen_map.keys()),
            generator=lambda p: gen_map[p],
        )
        result = ev.run(save=False)
        assert result.references == 2
        assert result.bleu is not None
        assert result.bleu >= 0

    def test_run_without_reference_responses(self, tmp_path):
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            eval_prompts=["no ref prompt"],
            generator=lambda p: "no ref response",
        )
        result = ev.run(save=False)
        assert result.references == 0
        assert result.bleu is None

    def test_run_multiple_save_files(self, tmp_path):
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            generator=lambda p: "resp",
        )
        counter = {"n": 0}
        def fake_strftime(fmt, *a, **kw):
            counter["n"] += 1
            return f"2025-01-01T00-00-{counter['n']:02d}"

        import domains.feedback.lora_eval as mod
        with pytest.MonkeyPatch.context() as m:
            m.setattr(mod.time, "strftime", fake_strftime)
            ev.run(save=True)
            ev.run(save=True)
        files = list((tmp_path / "eval").glob("*.json"))
        assert len(files) >= 4

    def test_run_result_timestamp_format(self, tmp_path):
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            generator=lambda p: "test",
        )
        result = ev.run(save=False)
        assert "T" in result.timestamp

    def test_run_inference_time_accumulates(self, tmp_path):
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            eval_prompts=["p1", "p2", "p3"],
            generator=lambda p: "r",
        )
        result = ev.run(save=False)
        assert result.inference_time_sec >= 0

    def test_run_tokens_per_sec(self, tmp_path):
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            generator=lambda p: "this is a test response",
        )
        result = ev.run(save=False)
        assert result.tokens_per_sec is not None
        assert result.tokens_per_sec > 0

    def test_run_avg_response_len(self, tmp_path):
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            eval_prompts=["p1"],
            generator=lambda p: "one two three four five",
        )
        result = ev.run(save=False)
        assert result.avg_response_len == 5.0

    def test_run_soul_name_forwarded(self, tmp_path):
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            eval_prompts=["test prompt"],
            generator=lambda p: "help me assist you thank you",
        )
        result = ev.run(save=False, soul_name="assistant")
        assert result.personality_score is not None


# ---------------------------------------------------------------------------
# LoRAEvaluator — _compute_perplexity
# ---------------------------------------------------------------------------

class TestLoRAEvaluatorComputePerplexity:
    def test_returns_none_without_model(self, tmp_path):
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"))
        pp = ev._compute_perplexity("hello", "hi")
        assert pp is None


# ---------------------------------------------------------------------------
# LoRAEvaluator — get_history
# ---------------------------------------------------------------------------

class TestLoRAEvaluatorGetHistory:
    def test_empty_history(self, tmp_path):
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"))
        history = ev.get_history()
        assert history == []

    def test_history_loads_saved(self, tmp_path):
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            generator=lambda p: "test",
        )
        ev.run(save=True)
        history = ev.get_history()
        assert len(history) == 1
        assert isinstance(history[0], EvalResult)

    def test_history_limit(self, tmp_path):
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            generator=lambda p: "test",
        )
        counter = {"n": 0}
        def fake_strftime(fmt, *a, **kw):
            counter["n"] += 1
            return f"2025-01-01T00-00-{counter['n']:02d}"

        import domains.feedback.lora_eval as mod
        with pytest.MonkeyPatch.context() as m:
            m.setattr(mod.time, "strftime", fake_strftime)
            for _ in range(5):
                ev.run(save=True)
        history = ev.get_history(limit=3)
        assert len(history) == 3

    def test_history_sorted_newest_first(self, tmp_path):
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            generator=lambda p: "test",
        )
        counter = {"n": 0}
        def fake_strftime(fmt, *a, **kw):
            counter["n"] += 1
            return f"2025-01-01T00-00-{counter['n']:02d}"

        import domains.feedback.lora_eval as mod
        with pytest.MonkeyPatch.context() as m:
            m.setattr(mod.time, "strftime", fake_strftime)
            ev.run(save=True)
            ev.run(save=True)
        history = ev.get_history()
        assert len(history) >= 2
        for i in range(len(history) - 1):
            assert history[i].timestamp >= history[i + 1].timestamp

    def test_history_ignores_detail_files(self, tmp_path):
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            generator=lambda p: "test",
        )
        ev.run(save=True)
        files = list((tmp_path / "eval").glob("baseline_*_detail.json"))
        assert len(files) >= 1
        history = ev.get_history()
        assert len(history) == 1

    def test_history_skips_corrupted_files(self, tmp_path):
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        (eval_dir / "baseline_2025-01-01T00-00-00.json").write_text("not json {{{")
        ev = LoRAEvaluator(eval_dir=str(eval_dir))
        history = ev.get_history()
        assert history == []


# ---------------------------------------------------------------------------
# LoRAEvaluator — _load_inference_engine
# ---------------------------------------------------------------------------

class TestLoRAEvaluatorLoadInferenceEngine:
    def test_no_model_sets_none(self, tmp_path):
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"))
        ev._load_inference_engine()
        assert ev._model is None

    def test_nonexistent_model_sets_none(self, tmp_path):
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            base_model="/nonexistent/path/model.bin",
        )
        ev._load_inference_engine()
        assert ev._model is None

    def test_warns_once(self, tmp_path):
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"))
        ev._load_inference_engine()
        assert ev._warned_no_model is True
        ev._load_inference_engine()
        assert ev._warned_no_model is True


# ---------------------------------------------------------------------------
# LoRAEvaluator — _generate
# ---------------------------------------------------------------------------

class TestLoRAEvaluatorGenerate:
    def test_generate_with_generator(self, tmp_path):
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            generator=lambda p: "generated text",
        )
        text, latency, tps = ev._generate("prompt")
        assert text == "generated text"
        assert latency >= 0
        assert tps >= 0

    def test_generate_fallback_without_model(self, tmp_path):
        ev = LoRAEvaluator(eval_dir=str(tmp_path / "eval"))
        text, latency, tps = ev._generate("prompt")
        assert isinstance(text, str)
        assert latency > 0

    def test_generate_with_generator_returns_empty_falls_back(self, tmp_path):
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            generator=lambda p: "",
        )
        text, latency, tps = ev._generate("prompt")
        assert isinstance(text, str)

    def test_generate_with_generator_returns_none_falls_back(self, tmp_path):
        ev = LoRAEvaluator(
            eval_dir=str(tmp_path / "eval"),
            generator=lambda p: None,
        )
        text, latency, tps = ev._generate("prompt")
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# LoRAEvaluator — export_adapter_as_sou (signature + config)
# ---------------------------------------------------------------------------

class TestLoRAEvaluatorExportConfig:
    def test_eval_prompts_class_default(self):
        assert len(LoRAEvaluator.EVAL_PROMPTS) == 8

    def test_reference_responses_keys(self):
        assert "Hello, how are you?" in LoRAEvaluator.REFERENCE_RESPONSES
        assert "What is Python?" in LoRAEvaluator.REFERENCE_RESPONSES

    def test_soul_keywords_keys(self):
        assert "assistant" in LoRAEvaluator.SOUL_KEYWORDS
        assert "coder" in LoRAEvaluator.SOUL_KEYWORDS
        assert "teacher" in LoRAEvaluator.SOUL_KEYWORDS

    def test_default_eval_prompts(self):
        ev = LoRAEvaluator.__new__(LoRAEvaluator)
        ev.eval_prompts = LoRAEvaluator.EVAL_PROMPTS
        assert len(ev.eval_prompts) == 8


# ---------------------------------------------------------------------------
# get_lora_evaluator singleton
# ---------------------------------------------------------------------------

class TestGetLoraEvaluatorSingleton:
    def test_returns_same_instance(self, tmp_path, monkeypatch):
        import domains.feedback.lora_eval as mod
        mod._global_eval = None
        ev1 = get_lora_evaluator()
        ev2 = get_lora_evaluator()
        assert ev1 is ev2
        mod._global_eval = None

    def test_singleton_reset(self, monkeypatch):
        import domains.feedback.lora_eval as mod
        mod._global_eval = None
        ev1 = get_lora_evaluator()
        mod._global_eval = None
        ev2 = get_lora_evaluator()
        assert ev1 is not ev2
        mod._global_eval = None
