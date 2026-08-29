"""
Model Benchmark Suite — tests for the benchmark comparison infrastructure.

TDD: Tests written first, implementation follows.
"""

import sys
import json
import time
import tempfile
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "core-py"))


class TestBenchmarkMetrics:
    """Test benchmark metric computation."""

    def test_perplexity_computation(self):
        from domains.feedback.lora_eval import LoRAEvaluator
        evaluator = LoRAEvaluator()
        # Compute perplexity on known text
        ppl = evaluator._compute_perplexity("hello world", "hello")
        # Should return None without a model, or a positive float with one
        assert ppl is None or ppl > 0

    def test_bleu_perfect_match(self):
        from domains.feedback.lora_eval import BLEUScorer
        score = BLEUScorer.score("hello world", "hello world")
        assert score == 100.0

    def test_bleu_partial_match(self):
        from domains.feedback.lora_eval import BLEUScorer
        score = BLEUScorer.score("hello world", "hello there world")
        assert 0 < score < 100

    def test_bleu_no_match(self):
        from domains.feedback.lora_eval import BLEUScorer
        score = BLEUScorer.score("foo bar", "baz qux")
        assert score == 0.0

    def test_bleu_empty_candidate(self):
        from domains.feedback.lora_eval import BLEUScorer
        score = BLEUScorer.score("", "hello world")
        assert score == 0.0

    def test_bleu_empty_reference(self):
        from domains.feedback.lora_eval import BLEUScorer
        score = BLEUScorer.score("hello world", "")
        assert score == 0.0

    def test_personality_scoring(self):
        from domains.feedback.lora_eval import LoRAEvaluator
        evaluator = LoRAEvaluator()
        score = evaluator._score_personality("I'm happy to help you with that!")
        assert 0 <= score.overall <= 1.0
        assert score.warmth_score >= 0

    def test_eval_result_creation(self):
        from domains.feedback.lora_eval import EvalResult
        result = EvalResult(
            timestamp="2024-01-01T00:00:00",
            adapter_path=None,
            prompts=5,
            references=3,
            perplexity=10.5,
            bleu=45.0,
            avg_response_len=25.0,
            inference_time_sec=1.2,
            tokens_per_sec=20.0,
            personality_score=0.7,
        )
        assert result.perplexity == 10.5
        assert result.bleu == 45.0
        d = result.to_dict()
        assert "perplexity" in d


class TestBenchmarkCompare:
    """Test model comparison infrastructure."""

    def test_model_result_properties(self):
        from scripts.benchmark_compare import ModelResult
        result = ModelResult(
            name="test_model",
            prompts=["prompt1", "prompt2"],
            responses=["response1", "response2"],
            latencies=[0.1, 0.2],
            token_counts=[10, 20],
        )
        assert abs(result.mean_latency - 0.15) < 1e-10
        assert result.total_tokens == 30
        assert result.tokens_per_sec > 0
        assert result.mean_response_len > 0

    def test_model_result_empty(self):
        from scripts.benchmark_compare import ModelResult
        result = ModelResult(
            name="empty",
            prompts=[],
            responses=[],
            latencies=[],
            token_counts=[],
        )
        assert result.mean_latency == 0.0
        assert result.total_tokens == 0

    def test_comparison_report_creation(self):
        from scripts.benchmark_compare import ModelResult, ComparisonReport
        r1 = ModelResult("m1", ["p1"], ["r1"], [0.1], [10])
        r2 = ModelResult("m2", ["p1"], ["r2"], [0.2], [15])
        report = ComparisonReport(models=[r1, r2], prompts=["p1"])
        assert len(report.models) == 2
        assert report.prompts == ["p1"]


class TestBenchmarkStability:
    """Test stability benchmark metrics."""

    def test_gold_standard_thresholds(self):
        # Gold standard: 0% crash, <=1.2x latency degradation, 0% empty, <=0.30 CV, 100% response rate
        thresholds = {
            "crash_rate": 0.0,
            "latency_degradation_max": 1.2,
            "empty_rate": 0.0,
            "response_length_cv_max": 0.30,
            "response_rate_min": 1.0,
        }
        # Simulate perfect results
        metrics = {
            "crash_rate": 0.0,
            "latency_degradation": 1.0,
            "empty_rate": 0.0,
            "response_length_cv": 0.15,
            "response_rate": 1.0,
        }
        assert metrics["crash_rate"] <= thresholds["crash_rate"]
        assert metrics["latency_degradation"] <= thresholds["latency_degradation_max"]
        assert metrics["empty_rate"] <= thresholds["empty_rate"]
        assert metrics["response_length_cv"] <= thresholds["response_length_cv_max"]
        assert metrics["response_rate"] >= thresholds["response_rate_min"]


class TestBenchmarkQualityScorer:
    """Test training pair quality scorer."""

    def test_quality_score_range(self):
        from domains.training.quality_scorer import score_pair
        score = score_pair("What is Python?", "Python is a programming language.")
        assert 0 <= score <= 5.0

    def test_quality_score_repetition_penalty(self):
        from domains.training.quality_scorer import score_pair
        good = score_pair("What is 2+2?", "4 is the answer.")
        bad = score_pair("What is 2+2?", "the the the the the the the the the the the the")
        assert good >= bad

    def test_quality_score_length_penalty(self):
        from domains.training.quality_scorer import score_pair
        # Too short
        short = score_pair("What is Python?", "Yes.")
        # Good length
        good = score_pair("What is Python?", "Python is a high-level programming language.")
        assert good >= short


class TestBenchmarkIntegration:
    """Integration tests for the benchmark suite."""

    def test_evaluator_with_simulated_model(self):
        from domains.feedback.lora_eval import LoRAEvaluator
        evaluator = LoRAEvaluator()
        # Should work without a real model (simulated)
        assert not evaluator.available()

    def test_evaluator_run_simulated(self):
        from domains.feedback.lora_eval import LoRAEvaluator
        evaluator = LoRAEvaluator()
        result = evaluator.run(save=False)
        assert result.prompts > 0
        assert result.inference_time_sec >= 0

    def test_compare_results(self):
        from domains.feedback.lora_eval import LoRAEvaluator, EvalResult
        evaluator = LoRAEvaluator()
        baseline = EvalResult(
            timestamp="2024-01-01T00:00:00",
            adapter_path=None,
            prompts=5,
            references=3,
            perplexity=12.0,
            bleu=40.0,
            avg_response_len=20.0,
            inference_time_sec=1.0,
            tokens_per_sec=20.0,
            personality_score=0.6,
        )
        adapted = EvalResult(
            timestamp="2024-01-01T01:00:00",
            adapter_path="test.npz",
            prompts=5,
            references=3,
            perplexity=10.0,
            bleu=50.0,
            avg_response_len=25.0,
            inference_time_sec=0.8,
            tokens_per_sec=25.0,
            personality_score=0.7,
        )
        delta = evaluator.compare(baseline, adapted)
        assert delta["verdict"] == "improved"
        assert delta["perplexity_delta"] < 0  # Lower is better
        assert delta["bleu_delta"] > 0  # Higher is better

    def test_export_adapter_as_sou(self):
        from domains.feedback.lora_eval import LoRAEvaluator
        evaluator = LoRAEvaluator()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, W_a=np.random.randn(8, 32), W_b=np.random.randn(32, 8), rank=8, alpha=16)
            adapter_path = f.name

        output_path = str(Path(tempfile.mkdtemp()) / "test_export.soul")
        result = evaluator.export_adapter_as_sou(
            adapter_path,
            soul_name="test_soul",
            eval_delta={"verdict": "improved", "perplexity_delta": -2.0, "bleu_delta": 10.0},
            output_sou=output_path,
        )
        assert Path(result).exists()
        Path(adapter_path).unlink()


class TestBenchmarkCompare:
    def test_compare_models_independence(self):
        from scripts.benchmark_model_comparison import ModelMetrics, evaluate_model
        responses_a = ["hello world", "foo bar"]
        responses_b = ["hello world", "baz qux"]
        latencies = [0.1, 0.2]
        token_counts = [2, 2]
        prompts = ["hello", "foo"]
        r1 = evaluate_model("ModelA", responses_a, latencies, token_counts, prompts)
        r2 = evaluate_model("ModelB", responses_b, latencies, token_counts, prompts)
        assert r1.model_name == "ModelA"
        assert r2.model_name == "ModelB"
        assert r1.diversity == r2.diversity  # same structure

    def test_evaluate_model_empty_responses(self):
        from scripts.benchmark_model_comparison import evaluate_model
        r = evaluate_model("empty", [], [], [], [])
        assert r.mean_latency_ms == 0.0
        assert r.tokens_per_sec == 0.0
        assert r.responses == []

    def test_run_native_inference_lstm(self):
        from scripts.benchmark_model_comparison import run_native_inference
        from domains.training.slonet import SloNet, SloEmbedding, SloLSTM, tensor
        net = SloNet(
            layers=[SloEmbedding(10, 8), SloLSTM(10, 8, 16, num_layers=1, dropout=0.0)],
            soul_name="test",
        )
        lstm = net.layers[1]
        stoi = {c: i + 1 for i, c in enumerate("abcdefghij")}
        itos = {i + 1: c for i, c in enumerate("abcdefghij")}
        encode = lambda t: np.array([stoi.get(c, 0) for c in t], dtype=np.int64)
        decode = lambda ids: "".join(itos.get(int(i), "?") for i in ids if i > 0)
        resp, lat, tokens = run_native_inference(net, lstm, encode, decode, "abc", max_new_tokens=10)
        assert isinstance(resp, str)
        assert lat > 0
        assert tokens >= 0

    def test_run_native_inference_transformer(self):
        from scripts.benchmark_model_comparison import run_native_inference
        from domains.training.slonet import SloTransformer
        net = SloTransformer(vocab_size=32, n_embed=32, n_layer=1, n_head=2, block_size=16)
        stoi = {c: i + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz012345")}
        itos = {i + 1: c for i, c in enumerate("abcdefghijklmnopqrstuvwxyz012345")}
        encode = lambda t: np.array([stoi.get(c, 0) for c in t], dtype=np.int64).reshape(1, -1)
        decode = lambda ids: "".join(itos.get(int(i), "?") for i in ids.flatten() if int(i) in itos)
        resp, lat, tokens = run_native_inference(net, None, encode, decode, "abc", max_new_tokens=10)
        assert isinstance(resp, str)
        assert lat > 0
        assert tokens >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
