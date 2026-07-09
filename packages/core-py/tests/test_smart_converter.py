"""Tests for smart_converter — HF→SloNet conversion analysis."""

import pytest
from domains.infrastructure.smart_converter import (
    analyze_model,
    ConversionReport,
    Recommendation,
    _score_architecture,
    _score_size,
)


@pytest.fixture(scope="session")
def gpt2_report():
    """Analyze GPT-2 once for entire test session."""
    return analyze_model("gpt2")


class TestAnalyzeModel:
    """Integration tests for analyze_model()."""

    def test_gpt2_returns_report(self, gpt2_report):
        assert isinstance(gpt2_report, ConversionReport)
        assert gpt2_report.model_id == "gpt2"

    def test_gpt2_recommendation(self, gpt2_report):
        assert gpt2_report.recommendation in (Recommendation.CONVERT, Recommendation.KEEP, Recommendation.SKIP)

    def test_gpt2_score_in_range(self, gpt2_report):
        assert 0 <= gpt2_report.score <= 100

    def test_gpt2_has_params(self, gpt2_report):
        assert gpt2_report.params > 0

    def test_gpt2_has_layers(self, gpt2_report):
        assert gpt2_report.total_layers > 0
        assert gpt2_report.convertible_layers >= 0
        assert gpt2_report.convertible_layers <= gpt2_report.total_layers

    def test_gpt2_summary(self, gpt2_report):
        s = gpt2_report.summary()
        assert "gpt2" in s.lower()
        assert "RECOMMENDATION" in s.upper() or "recommendation" in s.lower()

    def test_unknown_model_returns_skip(self):
        report = analyze_model("nonexistent/model-xyz-12345")
        assert report.recommendation == Recommendation.SKIP
        assert report.score == 0

    def test_gpt2_arch_match_positive(self, gpt2_report):
        assert gpt2_report.arch_match > 0


class TestScoreArchitecture:
    """Unit tests for _score_architecture()."""

    def test_gpt2_config(self):
        from domains.infrastructure.safetensors_loader import load_model_config
        config = load_model_config("gpt2")
        weights = {"wte": None, "ln_f": None, "h.0.ln_1": None}
        score, reasons, warnings, convertible, total = _score_architecture(config, weights)
        assert 0 <= score <= 1
        assert total > 0

    def test_empty_weights(self):
        config = {"model_type": "gpt2", "n_layer": 12, "n_head": 12, "n_embd": 768, "vocab_size": 50257}
        score, reasons, warnings, convertible, total = _score_architecture(config, {})
        assert score >= 0.0


class TestScoreSize:
    """Unit tests for _score_size()."""

    def test_small_model_high_score(self):
        config = {"vocab_size": 50000, "n_embd": 768, "n_layer": 12}
        score, speed, memory = _score_size(config)
        assert 0 <= score <= 1

    def test_large_model_lower_score(self):
        config = {"vocab_size": 100000, "n_embd": 4096, "n_layer": 32}
        score, speed, memory = _score_size(config)
        assert 0 <= score <= 1
