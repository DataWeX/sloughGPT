"""Tests for domains.training — DistillConfig, TextDataset, DistillEvalResult; domains.training.lr_schedulers — SchedulerConfig, CosineAnnealingConfig, WarmupConfig, OneCycleConfig, CyclicConfig."""

import numpy as np
from domains.training.distill_gpt2 import (
    DistillConfig, TextDataset, DistillEvalResult,
    _softmax, _kl_div_loss, _cross_entropy_loss, _compute_perplexity, _bleu_score,
)
from domains.training.lr_schedulers import (
    SchedulerConfig, CosineAnnealingConfig, WarmupConfig, OneCycleConfig, CyclicConfig,
)


# ── DistillConfig ────────────────────────────────────────────────────────────


class TestDistillConfig:
    def test_defaults(self):
        cfg = DistillConfig()
        assert cfg.n_embed == 128
        assert cfg.n_layer == 4
        assert cfg.temperature == 4.0
        assert cfg.alpha == 0.5
        assert cfg.teacher_model == "gpt2"

    def test_custom_values(self):
        cfg = DistillConfig(
            n_embed=256, n_layer=6, n_head=8, block_size=256,
            dropout=0.2, epochs=20, lr=1e-3, batch_size=16,
            grad_clip=2.0, warmup_steps=200, temperature=2.0,
            alpha=0.3, beta=0.7, teacher_model="gpt2-medium",
            checkpoint_dir="/tmp/ckpts", eval_interval=25, log_interval=5,
        )
        assert cfg.n_embed == 256
        assert cfg.n_layer == 6
        assert cfg.n_head == 8
        assert cfg.block_size == 256
        assert cfg.dropout == 0.2
        assert cfg.epochs == 20
        assert cfg.lr == 1e-3
        assert cfg.batch_size == 16
        assert cfg.grad_clip == 2.0
        assert cfg.warmup_steps == 200
        assert cfg.temperature == 2.0
        assert cfg.alpha == 0.3
        assert cfg.beta == 0.7
        assert cfg.teacher_model == "gpt2-medium"
        assert cfg.checkpoint_dir == "/tmp/ckpts"
        assert cfg.eval_interval == 25
        assert cfg.log_interval == 5

    def test_resume_defaults(self):
        cfg = DistillConfig()
        assert cfg.resume_checkpoint is None
        assert cfg.resume_epoch == 0
        assert cfg.resume_step == 0

    def test_resume_custom(self):
        cfg = DistillConfig(resume_checkpoint="/tmp/ckpt.soul", resume_epoch=5, resume_step=100)
        assert cfg.resume_checkpoint == "/tmp/ckpt.soul"
        assert cfg.resume_epoch == 5
        assert cfg.resume_step == 100

    def test_alpha_beta_sum(self):
        cfg = DistillConfig()
        assert cfg.alpha + cfg.beta == 1.0

    def test_dataclass_fields(self):
        cfg = DistillConfig()
        assert hasattr(cfg, "n_embed")
        assert hasattr(cfg, "n_layer")
        assert hasattr(cfg, "n_head")
        assert hasattr(cfg, "block_size")
        assert hasattr(cfg, "dropout")
        assert hasattr(cfg, "epochs")
        assert hasattr(cfg, "lr")
        assert hasattr(cfg, "batch_size")

    def test_temperature_positive(self):
        cfg = DistillConfig()
        assert cfg.temperature > 0

    def test_epochs_positive(self):
        cfg = DistillConfig()
        assert cfg.epochs > 0

    def test_lr_positive(self):
        cfg = DistillConfig()
        assert cfg.lr > 0

    def test_dropout_range(self):
        cfg = DistillConfig()
        assert 0.0 <= cfg.dropout <= 1.0


# ── TextDataset ──────────────────────────────────────────────────────────────


class TestTextDataset:
    def test_init(self):
        ds = TextDataset("hello world", block_size=4, stoi={"h": 0, "e": 1, "l": 2, "o": 3, " ": 4, "w": 5, "r": 6, "d": 7})
        assert len(ds) >= 1

    def test_len(self):
        ds = TextDataset("hello", block_size=2, stoi={"h": 0, "e": 1, "l": 2, "o": 3})
        assert len(ds) >= 1

    def test_len_formula(self):
        text = "abcdefghij"
        block_size = 3
        stoi = {c: i for i, c in enumerate(text)}
        ds = TextDataset(text, block_size=block_size, stoi=stoi)
        expected = max(1, len(text) - block_size - 1)
        assert len(ds) == expected

    def test_get_batch_shapes(self):
        text = "a" * 50
        stoi = {"a": 0}
        ds = TextDataset(text, block_size=8, stoi=stoi)
        rng = np.random.default_rng(42)
        x, y = ds.get_batch(4, rng)
        assert x.shape == (4, 8)
        assert y.shape == (4, 8)

    def test_get_batch_dtype(self):
        text = "b" * 50
        stoi = {"b": 1}
        ds = TextDataset(text, block_size=8, stoi=stoi)
        rng = np.random.default_rng(0)
        x, y = ds.get_batch(2, rng)
        assert x.dtype == np.int32
        assert y.dtype == np.int32

    def test_get_batch_y_is_x_shifted(self):
        text = "c" * 50
        stoi = {"c": 2}
        ds = TextDataset(text, block_size=4, stoi=stoi)
        rng = np.random.default_rng(0)
        x, y = ds.get_batch(1, rng)
        # With uniform text, y should equal x
        np.testing.assert_array_equal(x, y)

    def test_get_batch_different_lengths(self):
        text = "d" * 100
        stoi = {"d": 3}
        ds = TextDataset(text, block_size=8, stoi=stoi)
        rng = np.random.default_rng(0)
        x1, _ = ds.get_batch(1, rng)
        x2, _ = ds.get_batch(1, rng)
        # Both should be valid batches
        assert x1.shape[1] == 8
        assert x2.shape[1] == 8

    def test_unknown_chars_map_to_zero(self):
        text = "xyz"
        stoi = {"x": 5}
        ds = TextDataset(text, block_size=1, stoi=stoi)
        # 'y' and 'z' not in stoi, should default to 0
        assert ds.ids[1] == 0
        assert ds.ids[2] == 0

    def test_min_samples_is_one(self):
        ds = TextDataset("a", block_size=100, stoi={"a": 0})
        assert len(ds) >= 1

    def test_ids_conversion(self):
        text = "abc"
        stoi = {"a": 10, "b": 20, "c": 30}
        ds = TextDataset(text, block_size=1, stoi=stoi)
        assert ds.ids == [10, 20, 30]


# ── DistillEvalResult ────────────────────────────────────────────────────────


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

    def test_to_dict_rounds_values(self):
        er = DistillEvalResult(
            perplexity=3.14159, bleu_vs_teacher=0.678, avg_response_len=12.345,
            teacher_samples=[], student_samples=[], eval_prompts=[],
            inference_time_sec=2.999,
        )
        d = er.to_dict()
        assert d["perplexity"] == round(3.14159, 4)
        assert d["bleu_vs_teacher"] == round(0.678, 2)
        assert d["avg_response_len"] == round(12.345, 1)
        assert d["inference_time_sec"] == round(2.999, 3)

    def test_to_dict_num_samples(self):
        er = DistillEvalResult(
            perplexity=1.0, bleu_vs_teacher=0.5, avg_response_len=5.0,
            teacher_samples=["a", "b", "c"], student_samples=["d", "e", "f"],
            eval_prompts=["p1", "p2", "p3"], inference_time_sec=0.1,
        )
        d = er.to_dict()
        assert d["num_samples"] == 3

    def test_to_dict_samples_structure(self):
        er = DistillEvalResult(
            perplexity=1.0, bleu_vs_teacher=0.5, avg_response_len=5.0,
            teacher_samples=["t1", "t2"], student_samples=["s1", "s2"],
            eval_prompts=["p1", "p2"], inference_time_sec=0.1,
        )
        d = er.to_dict()
        assert len(d["samples"]) == 2
        assert d["samples"][0]["prompt"] == "p1"
        assert d["samples"][0]["teacher"] == "t1"
        assert d["samples"][0]["student"] == "s1"

    def test_all_fields_stored(self):
        er = DistillEvalResult(
            perplexity=2.0, bleu_vs_teacher=0.8, avg_response_len=15.0,
            teacher_samples=["t"], student_samples=["s"],
            eval_prompts=["p"], inference_time_sec=3.0,
        )
        assert er.perplexity == 2.0
        assert er.bleu_vs_teacher == 0.8
        assert er.avg_response_len == 15.0
        assert er.teacher_samples == ["t"]
        assert er.student_samples == ["s"]
        assert er.eval_prompts == ["p"]
        assert er.inference_time_sec == 3.0

    def test_to_dict_empty_samples(self):
        er = DistillEvalResult(
            perplexity=1.0, bleu_vs_teacher=0.0, avg_response_len=0.0,
            teacher_samples=[], student_samples=[], eval_prompts=[],
            inference_time_sec=0.0,
        )
        d = er.to_dict()
        assert d["num_samples"] == 0
        assert d["samples"] == []


# ── _softmax ─────────────────────────────────────────────────────────────────


class TestSoftmax:
    def test_basic(self):
        x = np.array([[1.0, 2.0, 3.0]])
        s = _softmax(x)
        np.testing.assert_allclose(s.sum(), 1.0, atol=1e-6)

    def test_all_positive(self):
        x = np.array([[10.0, 20.0, 30.0]])
        s = _softmax(x)
        assert s.shape == x.shape
        np.testing.assert_allclose(s.sum(), 1.0, atol=1e-6)

    def test_uniform_input(self):
        x = np.array([[5.0, 5.0, 5.0]])
        s = _softmax(x)
        np.testing.assert_allclose(s, [[1 / 3, 1 / 3, 1 / 3]], atol=1e-6)

    def test_batch(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        s = _softmax(x)
        np.testing.assert_allclose(s.sum(axis=-1), [1.0, 1.0], atol=1e-6)

    def test_large_values_stable(self):
        x = np.array([[1000.0, 1001.0, 1002.0]])
        s = _softmax(x)
        assert np.all(np.isfinite(s))
        np.testing.assert_allclose(s.sum(), 1.0, atol=1e-6)

    def test_preserves_ranking(self):
        x = np.array([[1.0, 5.0, 2.0]])
        s = _softmax(x)
        assert s[0, 1] > s[0, 2] > s[0, 0]


# ── _kl_div_loss ─────────────────────────────────────────────────────────────


class TestKLDivLoss:
    def test_identical_distributions(self):
        p = np.array([[0.25, 0.25, 0.25, 0.25]])
        student_log_probs = np.log(p)
        loss = _kl_div_loss(student_log_probs, p)
        assert loss < 1e-6

    def test_finite(self):
        student_log_probs = np.array([[0.0, -1.0, -2.0]])
        teacher_probs = np.array([[0.5, 0.3, 0.2]])
        loss = _kl_div_loss(student_log_probs, teacher_probs)
        assert np.isfinite(loss)

    def test_batch(self):
        slp = np.array([[0.0, -1.0], [-1.0, 0.0]])
        tp = np.array([[0.5, 0.5], [0.5, 0.5]])
        loss = _kl_div_loss(slp, tp)
        assert np.isfinite(loss)

    def test_zero_teacher_clamped(self):
        student_log_probs = np.array([[0.0, -100.0]])
        teacher_probs = np.array([[0.0, 1.0]])
        loss = _kl_div_loss(student_log_probs, teacher_probs)
        assert np.isfinite(loss)


# ── _cross_entropy_loss ──────────────────────────────────────────────────────


class TestCrossEntropyLoss:
    def test_perfect_prediction(self):
        logits = np.array([[0.0, 100.0, 0.0]])
        targets = np.array([1])
        loss = _cross_entropy_loss(logits, targets)
        assert loss < 0.01

    def test_worse_prediction_higher_loss(self):
        good_logits = np.array([[0.0, 10.0, 0.0]])
        bad_logits = np.array([[10.0, 0.0, 0.0]])
        targets = np.array([1])
        loss_good = _cross_entropy_loss(good_logits, targets)
        loss_bad = _cross_entropy_loss(bad_logits, targets)
        assert loss_bad > loss_good

    def test_batch(self):
        logits = np.array([[0.0, 10.0], [10.0, 0.0]])
        targets = np.array([1, 0])
        loss = _cross_entropy_loss(logits, targets)
        assert np.isfinite(loss)

    def test_positive_loss(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        targets = np.array([0])
        loss = _cross_entropy_loss(logits, targets)
        assert loss > 0


# ── _compute_perplexity ──────────────────────────────────────────────────────


class TestComputePerplexity:
    def test_zero_loss(self):
        assert _compute_perplexity(0.0) == 1.0

    def test_positive_loss(self):
        ppl = _compute_perplexity(1.0)
        assert abs(ppl - np.exp(1.0)) < 1e-6

    def test_monotonic(self):
        ppl1 = _compute_perplexity(0.5)
        ppl2 = _compute_perplexity(2.0)
        assert ppl2 > ppl1

    def test_large_loss(self):
        ppl = _compute_perplexity(10.0)
        assert ppl == np.exp(10.0)


# ── _bleu_score ──────────────────────────────────────────────────────────────


class TestBleuScore:
    def test_identical(self):
        score = _bleu_score("the cat sat", "the cat sat")
        assert score > 90.0

    def test_empty_candidate(self):
        assert _bleu_score("", "hello") == 0.0

    def test_empty_reference(self):
        assert _bleu_score("hello", "") == 0.0

    def test_both_empty(self):
        assert _bleu_score("", "") == 0.0

    def test_no_overlap(self):
        score = _bleu_score("abc def", "xyz uvw")
        assert score == 0.0

    def test_partial_overlap(self):
        score = _bleu_score("the cat sat on", "the cat is on")
        assert 0 < score < 100

    def test_short_text(self):
        score = _bleu_score("a", "a")
        assert score > 0

    def test_max_n_1(self):
        score = _bleu_score("hello world", "hello world", max_n=1)
        assert score > 90.0

    def test_brevity_penalty(self):
        score_short = _bleu_score("the", "the cat sat on the mat")
        score_long = _bleu_score("the cat sat on the mat", "the cat sat on the mat")
        assert score_long >= score_short


# ── SchedulerConfig ──────────────────────────────────────────────────────────


class TestSchedulerConfig:
    def test_defaults(self):
        cfg = SchedulerConfig()
        assert cfg.name == "none"
        assert cfg.initial_lr == 1e-4

    def test_custom_values(self):
        cfg = SchedulerConfig(name="custom", initial_lr=0.01)
        assert cfg.name == "custom"
        assert cfg.initial_lr == 0.01

    def test_has_name_attr(self):
        cfg = SchedulerConfig()
        assert hasattr(cfg, "name")

    def test_has_initial_lr_attr(self):
        cfg = SchedulerConfig()
        assert hasattr(cfg, "initial_lr")

    def test_initial_lr_positive(self):
        cfg = SchedulerConfig()
        assert cfg.initial_lr > 0

    def test_name_is_string(self):
        cfg = SchedulerConfig()
        assert isinstance(cfg.name, str)


# ── CosineAnnealingConfig ────────────────────────────────────────────────────


class TestCosineAnnealingConfig:
    def test_defaults(self):
        cfg = CosineAnnealingConfig()
        assert cfg.name == "cosine"
        assert cfg.min_lr == 1e-6
        assert cfg.total_steps == 10000

    def test_inherits_scheduler_config(self):
        cfg = CosineAnnealingConfig()
        assert isinstance(cfg, SchedulerConfig)

    def test_custom_values(self):
        cfg = CosineAnnealingConfig(min_lr=1e-5, total_steps=5000, warmup_steps=100, num_cycles=1.0)
        assert cfg.min_lr == 1e-5
        assert cfg.total_steps == 5000
        assert cfg.warmup_steps == 100
        assert cfg.num_cycles == 1.0

    def test_warmup_steps_default(self):
        cfg = CosineAnnealingConfig()
        assert cfg.warmup_steps == 0

    def test_num_cycles_default(self):
        cfg = CosineAnnealingConfig()
        assert cfg.num_cycles == 0.5

    def test_initial_lr_inherited(self):
        cfg = CosineAnnealingConfig(initial_lr=3e-4)
        assert cfg.initial_lr == 3e-4


# ── WarmupConfig ─────────────────────────────────────────────────────────────


class TestWarmupConfig:
    def test_defaults(self):
        cfg = WarmupConfig()
        assert cfg.name == "warmup"
        assert cfg.warmup_steps == 500

    def test_inherits_scheduler_config(self):
        cfg = WarmupConfig()
        assert isinstance(cfg, SchedulerConfig)

    def test_custom_values(self):
        cfg = WarmupConfig(warmup_steps=200, warmup_start_lr=1e-6)
        assert cfg.warmup_steps == 200
        assert cfg.warmup_start_lr == 1e-6

    def test_warmup_start_lr_default(self):
        cfg = WarmupConfig()
        assert cfg.warmup_start_lr == 1e-7

    def test_initial_lr_inherited(self):
        cfg = WarmupConfig(initial_lr=5e-4)
        assert cfg.initial_lr == 5e-4

    def test_warmup_steps_positive(self):
        cfg = WarmupConfig()
        assert cfg.warmup_steps > 0


# ── OneCycleConfig ───────────────────────────────────────────────────────────


class TestOneCycleConfig:
    def test_defaults(self):
        cfg = OneCycleConfig()
        assert cfg.name == "onecycle"
        assert cfg.max_lr == 1e-3
        assert cfg.pct_start == 0.1

    def test_inherits_scheduler_config(self):
        cfg = OneCycleConfig()
        assert isinstance(cfg, SchedulerConfig)

    def test_custom_values(self):
        cfg = OneCycleConfig(max_lr=5e-3, pct_start=0.3, anneal_strategy="linear")
        assert cfg.max_lr == 5e-3
        assert cfg.pct_start == 0.3
        assert cfg.anneal_strategy == "linear"

    def test_anneal_strategy_default(self):
        cfg = OneCycleConfig()
        assert cfg.anneal_strategy == "cos"

    def test_pct_start_range(self):
        cfg = OneCycleConfig()
        assert 0.0 <= cfg.pct_start <= 1.0

    def test_max_lr_greater_than_initial(self):
        cfg = OneCycleConfig(initial_lr=1e-5, max_lr=1e-3)
        assert cfg.max_lr > cfg.initial_lr


# ── CyclicConfig ─────────────────────────────────────────────────────────────


class TestCyclicConfig:
    def test_defaults(self):
        cfg = CyclicConfig()
        assert cfg.name == "cyclic"
        assert cfg.base_lr == 1e-5
        assert cfg.max_lr == 1e-3

    def test_inherits_scheduler_config(self):
        cfg = CyclicConfig()
        assert isinstance(cfg, SchedulerConfig)

    def test_custom_values(self):
        cfg = CyclicConfig(base_lr=1e-4, max_lr=5e-3, step_size_up=500, step_size_down=500, mode="exp_range")
        assert cfg.base_lr == 1e-4
        assert cfg.max_lr == 5e-3
        assert cfg.step_size_up == 500
        assert cfg.step_size_down == 500
        assert cfg.mode == "exp_range"

    def test_step_size_down_default(self):
        cfg = CyclicConfig()
        assert cfg.step_size_down is None

    def test_mode_default(self):
        cfg = CyclicConfig()
        assert cfg.mode == "triangular2"

    def test_max_lr_greater_than_base(self):
        cfg = CyclicConfig()
        assert cfg.max_lr > cfg.base_lr


# ── Cross-module edge cases ──────────────────────────────────────────────────


class TestDistillConfigSerialization:
    def test_config_equality(self):
        cfg1 = DistillConfig(n_embed=64)
        cfg2 = DistillConfig(n_embed=64)
        assert cfg1.n_embed == cfg2.n_embed

    def test_config_inequality(self):
        cfg1 = DistillConfig(n_embed=64)
        cfg2 = DistillConfig(n_embed=128)
        assert cfg1.n_embed != cfg2.n_embed


class TestTextDatasetBatchVariations:
    def test_single_sample_batch(self):
        text = "x" * 20
        stoi = {"x": 0}
        ds = TextDataset(text, block_size=4, stoi=stoi)
        rng = np.random.default_rng(0)
        x, y = ds.get_batch(1, rng)
        assert x.shape == (1, 4)

    def test_batch_larger_than_dataset(self):
        text = "z" * 12
        stoi = {"z": 0}
        ds = TextDataset(text, block_size=4, stoi=stoi)
        rng = np.random.default_rng(0)
        # Should still work, picks random indices
        x, y = ds.get_batch(32, rng)
        assert x.shape[0] == 32

    def test_block_size_one(self):
        text = "m" * 10
        stoi = {"m": 0}
        ds = TextDataset(text, block_size=1, stoi=stoi)
        rng = np.random.default_rng(0)
        x, y = ds.get_batch(4, rng)
        assert x.shape == (4, 1)
        # With block_size=1, y is the next char
        assert y.shape == (4, 1)


class TestBleuScoreEdgeCases:
    def test_single_word_match(self):
        score = _bleu_score("hello", "hello")
        assert score > 0

    def test_longer_candidate(self):
        score = _bleu_score("the big brown cat sat", "the cat sat")
        assert score > 0

    def test_max_n_greater_than_length(self):
        score = _bleu_score("ab", "ab", max_n=10)
        assert score > 0


class TestSchedulerInheritance:
    def test_cosine_is_scheduler(self):
        assert issubclass(CosineAnnealingConfig, SchedulerConfig)

    def test_warmup_is_scheduler(self):
        assert issubclass(WarmupConfig, SchedulerConfig)

    def test_onecycle_is_scheduler(self):
        assert issubclass(OneCycleConfig, SchedulerConfig)

    def test_cyclic_is_scheduler(self):
        assert issubclass(CyclicConfig, SchedulerConfig)

    def test_all_have_name(self):
        for cls in [SchedulerConfig, CosineAnnealingConfig, WarmupConfig, OneCycleConfig, CyclicConfig]:
            cfg = cls()
            assert hasattr(cfg, "name")
            assert isinstance(cfg.name, str)
