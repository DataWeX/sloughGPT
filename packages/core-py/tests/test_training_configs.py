"""Tests for domains.training — DistillConfig, TextDataset, DistillEvalResult; domains.training.lr_schedulers — SchedulerConfig, CosineAnnealingConfig, WarmupConfig, OneCycleConfig, CyclicConfig."""

from domains.training.distill_gpt2 import DistillConfig, TextDataset, DistillEvalResult
from domains.training.lr_schedulers import (
    SchedulerConfig, CosineAnnealingConfig, WarmupConfig, OneCycleConfig, CyclicConfig,
)


class TestDistillConfig:
    def test_defaults(self):
        cfg = DistillConfig()
        assert cfg.n_embed == 128
        assert cfg.n_layer == 4
        assert cfg.temperature == 4.0
        assert cfg.alpha == 0.5
        assert cfg.teacher_model == "gpt2"


class TestTextDataset:
    def test_init(self):
        ds = TextDataset("hello world", block_size=4, stoi={"h": 0, "e": 1, "l": 2, "o": 3, " ": 4, "w": 5, "r": 6, "d": 7})
        assert len(ds) >= 1

    def test_len(self):
        ds = TextDataset("hello", block_size=2, stoi={"h": 0, "e": 1, "l": 2, "o": 3})
        assert len(ds) >= 1


class TestDistillEvalResult:
    def test_fields(self):
        er = DistillEvalResult(
            perplexity=5.0, bleu_vs_teacher=0.7, avg_response_len=10.0,
            teacher_samples=[], student_samples=[], eval_prompts=[],
            inference_time_sec=1.0,
        )
        assert er.perplexity == 5.0
        assert er.bleu_vs_teacher == 0.7

    def test_to_dict(self):
        er = DistillEvalResult(
            perplexity=5.0, bleu_vs_teacher=0.7, avg_response_len=10.0,
            teacher_samples=[], student_samples=[], eval_prompts=[],
            inference_time_sec=1.0,
        )
        d = er.to_dict()
        assert isinstance(d, dict)
        assert "perplexity" in d


class TestSchedulerConfig:
    def test_defaults(self):
        cfg = SchedulerConfig()
        assert cfg.name == "none"
        assert cfg.initial_lr == 1e-4


class TestCosineAnnealingConfig:
    def test_defaults(self):
        cfg = CosineAnnealingConfig()
        assert cfg.name == "cosine"
        assert cfg.min_lr == 1e-6
        assert cfg.total_steps == 10000


class TestWarmupConfig:
    def test_defaults(self):
        cfg = WarmupConfig()
        assert cfg.name == "warmup"
        assert cfg.warmup_steps == 500


class TestOneCycleConfig:
    def test_defaults(self):
        cfg = OneCycleConfig()
        assert cfg.name == "onecycle"
        assert cfg.max_lr == 1e-3
        assert cfg.pct_start == 0.1


class TestCyclicConfig:
    def test_defaults(self):
        cfg = CyclicConfig()
        assert cfg.name == "cyclic"
        assert cfg.base_lr == 1e-5
        assert cfg.max_lr == 1e-3
