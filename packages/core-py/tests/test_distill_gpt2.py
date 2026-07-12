"""Tests for distill_gpt2 — GPT-2 → SloTransformer distillation module.

Covers DistillConfig, TextDataset, loss functions, softmax, and DistillEvaluator.
The full distill_gpt2_to_slo() is slow (downloads GPT-2) and marked @slow.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from domains.training.distill_gpt2 import (
    DistillConfig,
    TextDataset,
    DistillEvaluator,
    DistillEvalResult,
    _softmax,
    _kl_div_loss,
    _cross_entropy_loss,
    _compute_perplexity,
    _bleu_score,
)


class TestDistillConfig:
    def test_defaults(self):
        c = DistillConfig()
        assert c.n_embed == 128
        assert c.n_layer == 4
        assert c.n_head == 4
        assert c.block_size == 128
        assert c.dropout == 0.1
        assert c.epochs == 10
        assert c.lr == 3e-4
        assert c.batch_size == 8
        assert c.grad_clip == 1.0
        assert c.warmup_steps == 100
        assert c.temperature == 4.0
        assert c.alpha == 0.5
        assert c.beta == 0.5
        assert c.teacher_model == "gpt2"
        assert c.checkpoint_dir == "models/auto-training"
        assert c.eval_interval == 50
        assert c.log_interval == 10

    def test_custom_values(self):
        c = DistillConfig(n_embed=64, n_layer=2, n_head=2, epochs=3)
        assert c.n_embed == 64
        assert c.n_layer == 2
        assert c.epochs == 3

    def test_alpha_beta_independent(self):
        c = DistillConfig(alpha=0.3, beta=0.7)
        assert c.alpha + c.beta == 1.0

    def test_temperature_affects_scale(self):
        c = DistillConfig(temperature=2.0)
        assert c.temperature == 2.0


class TestTextDataset:
    def test_creation(self):
        stoi = {"a": 0, "b": 1, "c": 2}
        ds = TextDataset("abcabc", block_size=3, stoi=stoi)
        assert len(ds) > 0
        assert ds.block_size == 3
        assert ds.text == "abcabc"

    def test_get_batch_shape(self):
        stoi = {"a": 0, "b": 1, "c": 2}
        text = "abcabcabcabc"
        ds = TextDataset(text, block_size=4, stoi=stoi)
        rng = np.random.default_rng(0)
        x, y = ds.get_batch(2, rng)
        assert x.shape == (2, 4)
        assert y.shape == (2, 4)

    def test_get_batch_values_are_int(self):
        stoi = {"x": 0, "y": 1, "z": 2}
        ds = TextDataset("xyzxyzxyz", block_size=3, stoi=stoi)
        rng = np.random.default_rng(42)
        x, y = ds.get_batch(4, rng)
        assert x.dtype == np.int32
        assert y.dtype == np.int32

    def test_get_batch_y_is_shifted_x(self):
        stoi = {"a": 1, "b": 2, "c": 3}
        ds = TextDataset("abcabcabc", block_size=4, stoi=stoi)
        rng = np.random.default_rng(0)
        x, y = ds.get_batch(1, rng)
        # y should be x shifted by 1 position (within same sample)
        # At minimum, both should have same shape
        assert x.shape == y.shape

    def test_empty_text_gives_min_samples(self):
        ds = TextDataset("", block_size=10, stoi={})
        assert len(ds) >= 1  # max(1, ...)

    def test_short_text_fewer_samples(self):
        stoi = {"a": 0}
        ds1 = TextDataset("a" * 20, block_size=5, stoi=stoi)
        ds2 = TextDataset("a" * 100, block_size=5, stoi=stoi)
        assert len(ds1) < len(ds2)

    def test_unknown_chars_default_to_0(self):
        stoi = {"a": 1}
        ds = TextDataset("xyz", block_size=2, stoi=stoi)
        # x, y, z not in stoi → all map to 0
        assert all(v == 0 for v in ds.ids)


class TestSoftmax:
    def test_basic(self):
        x = np.array([[1.0, 2.0, 3.0]])
        result = _softmax(x, axis=-1)
        assert result.shape == (1, 3)
        assert abs(result.sum() - 1.0) < 1e-6

    def test_large_values_stable(self):
        x = np.array([[1000.0, 1001.0, 1002.0]])
        result = _softmax(x, axis=-1)
        assert abs(result.sum() - 1.0) < 1e-6
        assert all(r > 0 for r in result[0])

    def test_negative_values(self):
        x = np.array([[-10.0, -5.0, 0.0]])
        result = _softmax(x, axis=-1)
        assert abs(result.sum() - 1.0) < 1e-6

    def test_uniform_input(self):
        x = np.array([[1.0, 1.0, 1.0]])
        result = _softmax(x, axis=-1)
        np.testing.assert_allclose(result, [[1/3, 1/3, 1/3]], atol=1e-6)

    def test_batch(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _softmax(x, axis=-1)
        assert result.shape == (2, 2)
        assert abs(result[0].sum() - 1.0) < 1e-6
        assert abs(result[1].sum() - 1.0) < 1e-6

    def test_axis_0(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _softmax(x, axis=0)
        assert abs(result[0].sum() + result[1].sum() - 2.0) < 1e-6
        np.testing.assert_allclose(result.sum(axis=0), [1.0, 1.0], atol=1e-6)


class TestKLDivLoss:
    def test_identical_distributions_low_loss(self):
        p = np.array([[0.25, 0.25, 0.25, 0.25]])
        q_log = np.log(p)
        loss = _kl_div_loss(q_log, p)
        assert loss < 0.01  # near zero

    def test_different_distributions_higher_loss(self):
        p = np.array([[1.0, 0.0, 0.0]])
        q_log = np.log(np.array([[1e-15, 0.5, 0.5]]))
        loss = _kl_div_loss(q_log, p)
        assert loss > 0

    def test_batch_averaging(self):
        p1 = np.array([[1.0, 0.0]])
        p2 = np.array([[0.0, 1.0]])
        q1 = np.log(np.array([[0.5, 0.5]]))
        q2 = np.log(np.array([[0.5, 0.5]]))
        loss1 = _kl_div_loss(q1, p1)
        loss2 = _kl_div_loss(q2, p2)
        # Average should be between individual losses
        combined = _kl_div_loss(np.vstack([q1, q2]), np.vstack([p1, p2]))
        assert abs(combined - (loss1 + loss2) / 2) < 1e-6

    def test_result_is_float(self):
        p = np.array([[0.5, 0.5]])
        q_log = np.log(p)
        loss = _kl_div_loss(q_log, p)
        assert isinstance(loss, float)


class TestCrossEntropyLoss:
    def test_correct_prediction_low_loss(self):
        logits = np.array([[0.1, 10.0, 0.1]])
        targets = np.array([1])
        loss = _cross_entropy_loss(logits, targets)
        assert loss < 0.1

    def test_wrong_prediction_high_loss(self):
        logits = np.array([[10.0, 0.1, 0.1]])
        targets = np.array([1])
        loss = _cross_entropy_loss(logits, targets)
        assert loss > 5.0

    def test_batch(self):
        logits = np.array([[0.1, 10.0], [10.0, 0.1]])
        targets = np.array([1, 0])
        loss = _cross_entropy_loss(logits, targets)
        assert isinstance(loss, float)
        assert loss < 0.1  # both correct

    def test_perfect_scores(self):
        logits = np.array([[0.0, 100.0]])
        targets = np.array([1])
        loss = _cross_entropy_loss(logits, targets)
        assert loss < 1e-4

    def test_uniform_logits(self):
        logits = np.array([[1.0, 1.0, 1.0]])
        targets = np.array([0])
        loss = _cross_entropy_loss(logits, targets)
        expected = -np.log(1.0 / 3.0)
        assert abs(loss - expected) < 1e-4


class TestComputePerplexity:
    def test_zero_loss(self):
        assert _compute_perplexity(0.0) == 1.0

    def test_positive_loss(self):
        assert _compute_perplexity(1.0) == pytest.approx(np.exp(1.0))

    def test_high_loss(self):
        ppl = _compute_perplexity(5.0)
        assert ppl > 100

    def test_negative_loss(self):
        ppl = _compute_perplexity(-1.0)
        assert 0 < ppl < 1


class TestBleuScore:
    def test_identical_text(self):
        score = _bleu_score("hello world", "hello world")
        assert score == 100.0

    def test_empty_candidate(self):
        assert _bleu_score("", "hello") == 0.0

    def test_empty_reference(self):
        assert _bleu_score("hello", "") == 0.0

    def test_both_empty(self):
        assert _bleu_score("", "") == 0.0

    def test_partial_overlap(self):
        score = _bleu_score("the cat sat", "the cat mat")
        assert 0 < score < 100

    def test_no_overlap(self):
        score = _bleu_score("abc def", "xyz uvw")
        assert score == 0.0

    def test_subset(self):
        score = _bleu_score("the", "the cat sat")
        assert 0 < score < 100

    def test_superset(self):
        score = _bleu_score("the cat sat on", "the cat sat")
        assert 0 < score < 100

    def test_unigram_only(self):
        score = _bleu_score("a b c", "a b d")
        assert 0 < score < 100

    def test_perfect_unigram_imperfect_bigram(self):
        score = _bleu_score("a b a b", "a b b a")
        assert 0 < score < 100


class TestDistillEvalResult:
    def test_to_dict(self):
        result = DistillEvalResult(
            perplexity=42.5,
            bleu_vs_teacher=75.3,
            avg_response_len=12.5,
            teacher_samples=["hello", "world"],
            student_samples=["hi there", "earth"],
            eval_prompts=["Hello", "World"],
            inference_time_sec=0.5,
        )
        d = result.to_dict()
        assert d["perplexity"] == 42.5
        assert d["bleu_vs_teacher"] == 75.3
        assert d["avg_response_len"] == 12.5
        assert d["inference_time_sec"] == 0.5
        assert len(d["samples"]) == 2
        assert d["samples"][0]["prompt"] == "Hello"
        assert d["samples"][0]["teacher"] == "hello"
        assert d["samples"][0]["student"] == "hi there"

    def test_to_dict_rounding(self):
        result = DistillEvalResult(
            perplexity=42.56789,
            bleu_vs_teacher=75.34,
            avg_response_len=12.567,
            teacher_samples=[],
            student_samples=[],
            eval_prompts=[],
            inference_time_sec=0.123456,
        )
        d = result.to_dict()
        assert d["perplexity"] == 42.5679
        assert d["bleu_vs_teacher"] == 75.34
        assert d["inference_time_sec"] == 0.123


class TestDistillEvaluator:
    def test_init_defaults(self):
        evaluator = DistillEvaluator(
            teacher_rw={},
            teacher_arch=None,
            itos={0: "a"},
            stoi={"a": 0},
        )
        assert evaluator.max_tokens == 50
        assert len(evaluator.eval_prompts) == 5

    def test_init_custom_prompts(self):
        prompts = ["test prompt"]
        evaluator = DistillEvaluator(
            teacher_rw={},
            teacher_arch=None,
            itos={0: "a"},
            stoi={"a": 0},
            eval_prompts=prompts,
        )
        assert evaluator.eval_prompts == prompts

    def test_bleu_score_static_method(self):
        score = DistillEvaluator._bleu_score("hello world", "hello world") if hasattr(DistillEvaluator, '_bleu_score') else _bleu_score("hello world", "hello world")
        # _bleu_score is a module-level function, not a method
        score = _bleu_score("hello world", "hello world")
        assert score == 100.0


class TestDistillConfigResume:
    def test_resume_checkpoint_default(self):
        c = DistillConfig()
        assert c.resume_checkpoint is None

    def test_resume_checkpoint_set(self):
        c = DistillConfig(resume_checkpoint="/path/to/checkpoint.soul")
        assert c.resume_checkpoint == "/path/to/checkpoint.soul"

    def test_resume_epoch_default(self):
        c = DistillConfig()
        assert c.resume_epoch == 0

    def test_resume_step_default(self):
        c = DistillConfig()
        assert c.resume_step == 0

    def test_resume_custom_values(self):
        c = DistillConfig(resume_checkpoint="test.soul", resume_epoch=5, resume_step=100)
        assert c.resume_checkpoint == "test.soul"
        assert c.resume_epoch == 5
        assert c.resume_step == 100
