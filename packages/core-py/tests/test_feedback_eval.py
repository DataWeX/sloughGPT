"""Tests for domains.feedback — WorkflowConfig, OnlineLoRAUpdater config, EvalResult, BLEUScorer, PersonalityScore."""

from domains.feedback.workflow import WorkflowConfig
from domains.feedback.online_train import LoRAConfig
from domains.feedback.lora_eval import EvalResult, BLEUScorer, PersonalityScore


class TestWorkflowConfig:
    def test_defaults(self):
        cfg = WorkflowConfig()
        assert cfg.aggregate_interval_minutes == 60
        assert cfg.background_training_enabled is True
        assert cfg.export_format == "dpo"


class TestOnlineLoRAConfig:
    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.rank == 8
        assert cfg.alpha == 16
        assert cfg.dropout == 0.0
        assert "attn.c_attn" in cfg.target_modules


class TestEvalResult:
    def test_fields(self):
        er = EvalResult(
            timestamp="2024-01-01", adapter_path=None, prompts=10,
            references=10, perplexity=5.0, bleu=0.7, avg_response_len=20.0,
            inference_time_sec=1.0, tokens_per_sec=50.0, personality_score=0.8,
        )
        assert er.perplexity == 5.0
        assert er.bleu == 0.7

    def test_to_dict(self):
        er = EvalResult(
            timestamp="2024-01-01", adapter_path=None, prompts=10,
            references=10, perplexity=5.0, bleu=0.7, avg_response_len=20.0,
            inference_time_sec=1.0, tokens_per_sec=50.0, personality_score=0.8,
        )
        d = er.to_dict()
        assert isinstance(d, dict)
        assert "perplexity" in d


class TestBLEUScorer:
    def test_identical(self):
        score = BLEUScorer.score("hello world", "hello world")
        assert float(score) == 100.0
    def test_different(self):
        score = BLEUScorer.score("cat dog", "fish bird")
        assert float(score) == 0.0
    def test_empty(self):
        score = BLEUScorer.score("", "hello")
        assert float(score) == 0.0


class TestPersonalityScore:
    def test_fields(self):
        ps = PersonalityScore(
            soul_name="test", warmth_score=0.8, creativity_score=0.7,
            formality_score=0.5, coherence_score=0.9, overall=0.75,
        )
        assert ps.soul_name == "test"
        assert ps.overall == 0.75

    def test_to_dict(self):
        ps = PersonalityScore(
            soul_name="test", warmth_score=0.8, creativity_score=0.7,
            formality_score=0.5, coherence_score=0.9, overall=0.75,
        )
        d = ps.to_dict()
        assert d["soul"] == "test"
        assert d["warmth"] == 0.8
