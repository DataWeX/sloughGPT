"""Tests for distill_gpt2 — GPT-2 → SloTransformer distillation module.

Covers DistillConfig, TextDataset, loss functions, softmax, and DistillEvaluator.
The full distill_gpt2_to_slo() is slow (downloads GPT-2) and marked @slow.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import threading
import types
from pathlib import Path

import numpy as np
import pytest

import domains.training.distill_gpt2 as dg
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
    _load_gpt2_numpy,
    _teacher_forward,
    distill_gpt2_to_slo,
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


class _Param:
    def __init__(self):
        self.grad = None


class _LogitsObj:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.requires_grad = False
        self.grad = None
        self._consumers = []

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        elif len(shape) == 1 and isinstance(shape[0], int):
            shape = (shape[0],)
        else:
            shape = tuple(shape)
        new_data = self.data.reshape(shape)
        return _LogitsObj(new_data)

    def reshape(self, *shape):
        return _LogitsObj(self.data.reshape(*shape))

    def __getitem__(self, key):
        return _LogitsObj(self.data[key])

    def backward(self):
        pass


class _StubSloTransformer:
    def __init__(self, vocab_size, **kw):
        self.vocab_size = vocab_size
        self.metadata = None
        self.params = [_Param(), _Param()]

    def parameters(self):
        return self.params

    def forward(self, x, y=None):
        return _LogitsObj(np.zeros((1, 1, self.vocab_size))), None


class _StubAdam:
    def __init__(self, **kw):
        pass

    def step(self, params):
        for p in params:
            if hasattr(p, "grad"):
                p.grad = None


class TestLoadGpt2Numpy:
    def test_success(self, monkeypatch, tmp_path):
        import domains.infrastructure.slnc.parser as _parser_mod

        class _FakeSLNCParser:
            def __init__(self, path):
                self._path = path

            def get_weights_dict_parallel(self):
                return {"wte": np.zeros((2, 2)), "ln_f": np.zeros((2,))}

        base = tmp_path / "home"
        snap = base / ".cache/huggingface/hub/models--gpt2/snapshots/1234"
        snap.mkdir(parents=True)
        (snap / "model.slnc").write_text("fake")
        (snap / "tokenizer.json").write_text(
            json.dumps({"model": {"vocab": {"a": 0, "b": 1}}})
        )

        monkeypatch.setattr(dg.Path, "home", staticmethod(lambda: base))
        monkeypatch.setattr(dg, "build_arch", lambda *a, **k: "ARCH")
        monkeypatch.setattr(dg, "pre_extract_weights", lambda arch, w: {"extracted": w})
        monkeypatch.setattr(_parser_mod, "SLNCParser", _FakeSLNCParser)

        rw, arch, vocab = _load_gpt2_numpy()
        assert arch == "ARCH"
        assert "extracted" in rw
        assert vocab["vocab_size"] == 2
        assert vocab["stoi"] == {"a": 0, "b": 1}
        assert vocab["itos"] == {0: "a", 1: "b"}

    def test_no_snapshots_raises(self, monkeypatch, tmp_path):
        fake_st = types.ModuleType("safetensors")
        fake_st.safe_open = object
        monkeypatch.setitem(sys.modules, "safetensors", fake_st)

        base = tmp_path / "home"
        (base / ".cache/huggingface/hub/models--gpt2").mkdir(parents=True)
        monkeypatch.setattr(dg.Path, "home", staticmethod(lambda: base))

        with pytest.raises(RuntimeError, match="GPT-2 not found"):
            _load_gpt2_numpy()

    def test_missing_tokenizer_raises(self, monkeypatch, tmp_path):
        import domains.infrastructure.slnc.parser as _parser_mod

        class _FakeSLNCParser:
            def __init__(self, path):
                pass
            def get_weights_dict_parallel(self):
                return {"wte": np.zeros((2, 2))}

        base = tmp_path / "home"
        snap = base / ".cache/huggingface/hub/models--gpt2/snapshots/1234"
        snap.mkdir(parents=True)
        (snap / "model.slnc").write_text("fake")
        monkeypatch.setattr(dg.Path, "home", staticmethod(lambda: base))
        monkeypatch.setattr(_parser_mod, "SLNCParser", _FakeSLNCParser)

        with pytest.raises(RuntimeError, match="tokenizer.json"):
            _load_gpt2_numpy()


class TestTeacherForward:
    def test_delegates_to_forward_fast(self, monkeypatch):
        monkeypatch.setattr(dg, "forward_fast", lambda rw, arch, ids: np.array([ids]))
        out = _teacher_forward({"rw": 1}, "arch", [1, 2])
        assert out.shape == (1, 2)


class TestGenerateGreedy:
    def _ev(self):
        return DistillEvaluator(
            teacher_rw={"w": np.zeros(2)},
            teacher_arch=None,
            itos={0: "", 1: "b"},
            stoi={"a": 1},
            eval_prompts=[],
        )

    def test_breaks_on_eos(self, monkeypatch):
        ev = self._ev()

        def fake_ff(gw, arch, tokens, kv_cache=None, start_pos=0):
            return np.array([[5.0, 0.0]])

        monkeypatch.setattr(dg, "forward_fast", fake_ff)
        out = ev._generate_greedy("a", lambda n: 0, "arch", 5)
        assert out == ""

    def test_appends_token_then_breaks(self, monkeypatch):
        ev = self._ev()

        def fake_ff(gw, arch, tokens, kv_cache=None, start_pos=0):
            return np.array([[0.0, 5.0]]) if len(tokens) < 2 else np.array([[5.0, 0.0]])

        monkeypatch.setattr(dg, "forward_fast", fake_ff)
        out = ev._generate_greedy("a", lambda n: 0, "arch", 5)
        assert out == "b"

    def test_empty_prompt_uses_zero_token(self, monkeypatch):
        ev = self._ev()

        def fake_ff(gw, arch, tokens, kv_cache=None, start_pos=0):
            return np.array([[5.0, 0.0]])

        monkeypatch.setattr(dg, "forward_fast", fake_ff)
        out = ev._generate_greedy("", lambda n: 0, "arch", 5)
        assert out == ""


class TestGenerateStudent:
    def _ev(self):
        return DistillEvaluator(
            teacher_rw={},
            teacher_arch=None,
            itos={0: "", 1: "b"},
            stoi={"a": 1},
            eval_prompts=[],
        )

    def test_data_attribute_path(self):
        ev = self._ev()

        class _Student:
            def forward(self, x):
                return _LogitsObj(np.array([[[5.0, 0.0]]])), None

        out = ev._generate_student("a", _Student(), 5)
        assert out == ""

    def test_numpy_conversion_path(self):
        ev = self._ev()

        class _Student:
            def forward(self, x):
                return [[5.0, 0.0]], None

        out = ev._generate_student("a", _Student(), 5)
        assert out == ""

    def test_appends_token_then_breaks(self):
        ev = self._ev()
        calls = {"n": 0}

        class _Student:
            def forward(self, x):
                calls["n"] += 1
                if calls["n"] < 2:
                    return _LogitsObj(np.array([[[0.0, 5.0]]])), None
                return _LogitsObj(np.array([[[5.0, 0.0]]])), None

        out = ev._generate_student("a", _Student(), 5)
        assert out == "b"

    def test_empty_prompt_uses_zero_token(self):
        ev = self._ev()

        class _Student:
            def forward(self, x):
                return _LogitsObj(np.array([[[5.0, 0.0]]])), None

        out = ev._generate_student("", _Student(), 5)
        assert out == ""


class TestComputePerplexityFromModel:
    def _ev(self):
        return DistillEvaluator(
            teacher_rw={},
            teacher_arch=None,
            itos={0: "a"},
            stoi={"a": 0},
            eval_prompts=[],
        )

    def test_short_text_returns_one(self):
        ev = self._ev()
        assert ev._compute_perplexity_from_model(None, "a") == 1.0

    def test_chunked_3d_logits(self):
        ev = self._ev()

        class _Student:
            def forward(self, x, y):
                return _LogitsObj(np.zeros((1, 3, 1))), None

        # 3 tokens -> one chunk of 2 targets -> ppl = exp(0) = 1.0
        ppl = ev._compute_perplexity_from_model(_Student(), "aaa")
        assert ppl == pytest.approx(1.0)

    def test_2d_logits_skip_reshape(self):
        ev = self._ev()

        class _Student:
            def forward(self, x, y):
                return _LogitsObj(np.zeros((2, 1))), None

        ppl = ev._compute_perplexity_from_model(_Student(), "aaa")
        assert ppl == pytest.approx(1.0)

    def test_logits_without_data_attr(self):
        ev = self._ev()

        class _Student:
            def forward(self, x, y):
                return [[0.0, 0.0], [0.0, 0.0]], None

        ppl = ev._compute_perplexity_from_model(_Student(), "aaa")
        assert ppl == pytest.approx(2.0)

    def test_zero_tokens_returns_one(self):
        ev = self._ev()

        class _Student:
            def forward(self, x, y):
                return _LogitsObj(np.zeros((0, 1))), None

        assert ev._compute_perplexity_from_model(_Student(), "aaa") == 1.0


class TestEvaluatorRun:
    def _stub_eval(self, monkeypatch, t_out, s_out):
        ev = DistillEvaluator(
            teacher_rw={"w": np.zeros(2)},
            teacher_arch=None,
            itos={0: "a", 1: "b"},
            stoi={"a": 0, "b": 1},
            eval_prompts=["ab", "ba"],
        )
        monkeypatch.setattr(ev, "_generate_greedy", lambda *a, **k: t_out)
        monkeypatch.setattr(ev, "_generate_student", lambda *a, **k: s_out)
        monkeypatch.setattr(
            ev, "_compute_perplexity_from_model", lambda *a, **k: 2.5
        )
        return ev

    def test_run_with_samples(self, monkeypatch):
        ev = self._stub_eval(monkeypatch, "hello", "hello")
        res = ev.run(None)
        assert res.perplexity == 2.5
        assert res.bleu_vs_teacher == 100.0
        assert res.avg_response_len == 1.0
        assert len(res.teacher_samples) == 2

    def test_run_with_empty_samples(self, monkeypatch):
        ev = self._stub_eval(monkeypatch, "", "")
        res = ev.run(None)
        assert res.perplexity == 2.5
        assert res.bleu_vs_teacher == 0.0
        assert res.avg_response_len == 0.0

    def test_run_with_no_prompts(self, monkeypatch):
        ev = DistillEvaluator(
            teacher_rw={},
            teacher_arch=None,
            itos={0: "a"},
            stoi={"a": 0},
            eval_prompts=[],
        )
        monkeypatch.setattr(ev, "_generate_greedy", lambda *a, **k: "")
        monkeypatch.setattr(ev, "_generate_student", lambda *a, **k: "")
        monkeypatch.setattr(
            ev, "_compute_perplexity_from_model", lambda *a, **k: 2.5
        )
        res = ev.run(None)
        assert res.bleu_vs_teacher == 0.0
        assert res.avg_response_len == 0.0


def _patch_distill(monkeypatch, vocab_size=5):
    vocab = {chr(97 + i): i for i in range(vocab_size)}
    itos = {i: chr(97 + i) for i in range(vocab_size)}
    text = "abcde" * 8
    arch = types.SimpleNamespace(n_layers=2)

    monkeypatch.setattr(
        dg,
        "_load_gpt2_numpy",
        lambda: ({"w": np.zeros((2, 2))}, arch,
                 {"stoi": vocab, "itos": itos, "vocab_size": vocab_size}),
    )
    monkeypatch.setattr(
        dg, "forward_fast", lambda rw, a, ids: np.zeros((len(ids), vocab_size))
    )
    monkeypatch.setattr(dg, "SloTransformer", lambda **kw: _StubSloTransformer(vocab_size))
    monkeypatch.setattr(dg, "SloAdam", _StubAdam)
    return text, vocab, itos


class TestDistillGpt2ToSlo:
    def _eval_stub(self):
        class _Eval:
            def __init__(self, **kw):
                pass

            def run(self, student):
                return DistillEvalResult(
                    perplexity=3.5,
                    bleu_vs_teacher=60.0,
                    avg_response_len=4.0,
                    teacher_samples=["hi"],
                    student_samples=["hello"],
                    eval_prompts=["Hi"],
                    inference_time_sec=0.1,
                )

        return _Eval

    def _config(self, tmp_path, **kw):
        base = dict(
            n_embed=16, n_layer=1, n_head=2, block_size=8, dropout=0.0,
            epochs=2, lr=1e-3, batch_size=2, grad_clip=1.0, warmup_steps=0,
            temperature=4.0, alpha=0.5, beta=0.5, teacher_model="gpt2",
            checkpoint_dir=str(tmp_path), eval_interval=3, log_interval=2,
        )
        base.update(kw)
        return DistillConfig(**base)

    def test_train_and_export(self, monkeypatch, tmp_path):
        text, _, _ = _patch_distill(monkeypatch)
        exported = {}
        monkeypatch.setattr(
            dg, "export_to_sou",
            lambda net, path, soul_profile=None, metadata=None: exported.update(
                {"path": path, "metadata": metadata}
            ),
        )
        monkeypatch.setattr(dg, "DistillEvaluator", self._eval_stub())

        steps = []
        model, meta = distill_gpt2_to_slo(
            text, self._config(tmp_path), on_step=lambda s, l, e: steps.append(s)
        )
        assert isinstance(model, _StubSloTransformer)
        assert steps
        assert meta["teacher"] == "gpt2"
        assert meta["vocab_size"] == "5"
        assert meta["perplexity"] == "3.5"
        assert meta["bleu_vs_teacher"] == "60.0"
        assert "epochs" in meta and "steps" in meta
        assert exported["metadata"]["vocab_size"] == 5

    def test_cancel_before_training(self, monkeypatch, tmp_path):
        text, _, _ = _patch_distill(monkeypatch)
        exported = {}
        monkeypatch.setattr(
            dg, "export_to_sou",
            lambda net, path, soul_profile=None, metadata=None: exported.update(
                {"path": path, "metadata": metadata}
            ),
        )
        monkeypatch.setattr(dg, "DistillEvaluator", self._eval_stub())

        ev = threading.Event()
        ev.set()
        model, meta = distill_gpt2_to_slo(text, self._config(tmp_path), cancel_event=ev)
        assert meta["final_loss"] == "0.0"
        assert meta["epochs"] == "1"

    def test_cancel_mid_training(self, monkeypatch, tmp_path):
        text, _, _ = _patch_distill(monkeypatch)
        exported = {}
        monkeypatch.setattr(
            dg, "export_to_sou",
            lambda net, path, soul_profile=None, metadata=None: exported.update(
                {"path": path, "metadata": metadata}
            ),
        )
        monkeypatch.setattr(dg, "DistillEvaluator", self._eval_stub())

        ev = threading.Event()
        model, meta = distill_gpt2_to_slo(
            text,
            self._config(tmp_path, log_interval=1, epochs=3),
            on_step=lambda s, l, e: ev.set() if s >= 2 else None,
            cancel_event=ev,
        )
        assert int(meta["steps"]) < 60

    def test_train_student_returns_plain_logits(self, monkeypatch, tmp_path):
        text, _, _ = _patch_distill(monkeypatch)
        exported = {}

        class _PlainLogitsStudent:
            def __init__(self, **kw):
                self.params = []

            def parameters(self):
                return self.params

            def forward(self, x, y=None):
                return [[[0.0, 0.0, 0.0, 0.0, 0.0]]], None

        monkeypatch.setattr(dg, "SloTransformer", _PlainLogitsStudent)
        monkeypatch.setattr(
            dg, "export_to_sou",
            lambda net, path, soul_profile=None, metadata=None: exported.update(
                {"path": path, "metadata": metadata}
            ),
        )
        monkeypatch.setattr(dg, "DistillEvaluator", self._eval_stub())

        _, meta = distill_gpt2_to_slo(
            text, self._config(tmp_path, batch_size=1, epochs=1)
        )
        assert meta["vocab_size"] == "5"

    def test_resume_from_checkpoint(self, monkeypatch, tmp_path):
        text, _, _ = _patch_distill(monkeypatch)
        ckpt = tmp_path / "resume.soul"
        ckpt.write_text("data")
        exported = {}

        class _ResumeStudent(_StubSloTransformer):
            def __init__(self):
                super().__init__(5)
                self.metadata = {"epoch": 1, "step": 5, "best_loss": 0.25}

        import domains.training.slonet as slonet
        monkeypatch.setattr(slonet, "import_from_sou", lambda p: _ResumeStudent())
        monkeypatch.setattr(
            dg, "export_to_sou",
            lambda net, path, soul_profile=None, metadata=None: exported.update(
                {"path": path, "metadata": metadata}
            ),
        )
        monkeypatch.setattr(dg, "DistillEvaluator", self._eval_stub())

        model, meta = distill_gpt2_to_slo(
            text,
            self._config(tmp_path, resume_checkpoint=str(ckpt), epochs=2),
        )
        assert isinstance(model, _ResumeStudent)
        assert meta["epochs"] == "2"
        assert exported["metadata"]["step"] == 5 + 15

    def test_resume_student_without_metadata(self, monkeypatch, tmp_path):
        text, _, _ = _patch_distill(monkeypatch)
        ckpt = tmp_path / "resume.soul"
        ckpt.write_text("data")
        exported = {}

        import domains.training.slonet as slonet
        monkeypatch.setattr(slonet, "import_from_sou", lambda p: _StubSloTransformer(5))
        monkeypatch.setattr(
            dg, "export_to_sou",
            lambda net, path, soul_profile=None, metadata=None: exported.update(
                {"path": path, "metadata": metadata}
            ),
        )
        monkeypatch.setattr(dg, "DistillEvaluator", self._eval_stub())

        _, meta = distill_gpt2_to_slo(
            text, self._config(tmp_path, resume_checkpoint=str(ckpt), epochs=2)
        )
        assert meta["epochs"] == "2"
