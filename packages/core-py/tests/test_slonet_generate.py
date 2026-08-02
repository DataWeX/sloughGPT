"""Tests for SloTransformer generation paths, logit processors, state dicts, schedulers, and optimizers."""

import math
import numpy as np
import pytest
from types import SimpleNamespace

from domains.training import slonet as sn
from domains.training.slonet import (
    SloTransformer,
    _apply_temperature,
    _apply_top_k,
    _apply_top_p,
    _apply_repetition_penalty,
    _apply_frequency_penalty,
    _apply_presence_penalty,
    _sample_from_logits,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _cpu_only():
    sn._ACCELERATOR = None
    yield
    sn._ACCELERATOR = None


def _tiny(**kw):
    cfg = dict(
        vocab_size=32, n_embed=16, n_layer=1, n_head=2, block_size=16,
        max_seq_len=32, dropout=0.0, tie_weights=True, norm_type="rms_norm",
    )
    cfg.update(kw)
    return SloTransformer(**cfg)


PROMPT = np.array([[3, 7, 12, 5]], dtype=np.int64)


def _first_token(m, prompt=PROMPT):
    one = m.generate_numpy(prompt, max_new_tokens=1, temperature=0.0)
    return int(one[0, -1])


# ── Logit processors ─────────────────────────────────────────────────────────

class TestLogitProcessors:

    def test_apply_temperature_zero_is_identity(self):
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        np.testing.assert_allclose(_apply_temperature(logits, 0.0), logits)

    def test_apply_temperature_scales(self):
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        np.testing.assert_allclose(_apply_temperature(logits, 2.0), logits / 2.0)

    def test_apply_top_k_identity_when_k_invalid(self):
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        np.testing.assert_allclose(_apply_top_k(logits, 0), logits)
        np.testing.assert_allclose(_apply_top_k(logits, 3), logits)
        np.testing.assert_allclose(_apply_top_k(logits, -5), logits)

    def test_apply_top_k_masks_below_cutoff(self):
        logits = np.array([[1.0, 5.0, 3.0, 2.0, 4.0]], dtype=np.float32)
        out = _apply_top_k(logits, 2)
        assert out[0, 1] == 5.0 and out[0, 4] == 4.0
        assert out[0, 0] == -1e9 and out[0, 2] == -1e9 and out[0, 3] == -1e9

    def test_apply_top_p_identity_outside_range(self):
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        np.testing.assert_allclose(_apply_top_p(logits.copy(), 0.0), logits)
        np.testing.assert_allclose(_apply_top_p(logits.copy(), 1.0), logits)

    def test_apply_top_p_keeps_top_token_batch(self):
        logits = np.array(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
             [5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        out = _apply_top_p(logits.copy(), 0.01)
        # Each row must keep its single best token.
        assert out[0, 0] == 1.0
        assert out[1, 0] == 5.0
        assert np.count_nonzero(out[1] > -1e8) == 1

    def test_apply_repetition_penalty_identity(self):
        logits = np.array([[1.0, 2.0]], dtype=np.float32)
        np.testing.assert_allclose(
            _apply_repetition_penalty(logits.copy(), np.array([4], dtype=np.int64), 1.0),
            logits,
        )
        np.testing.assert_allclose(
            _apply_repetition_penalty(logits.copy(), np.array([], dtype=np.int64), 2.0),
            logits,
        )

    def test_apply_repetition_penalty_negative_and_positive_logits(self):
        logits = np.array([[-1.0, 2.0, 0.5]], dtype=np.float32)
        out = _apply_repetition_penalty(logits.copy(), np.array([0, 1, 50], dtype=np.int64), 2.0)
        # token 0 negative -> multiplied, token 1 positive -> divided, token 50 skipped
        assert out[0, 0] == pytest.approx(-2.0)
        assert out[0, 1] == pytest.approx(1.0)
        assert out[0, 2] == pytest.approx(0.5)

    def test_apply_frequency_penalty(self):
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = _apply_frequency_penalty(logits.copy(), np.array([0, 0, 1, 100], dtype=np.int64), 0.5)
        assert out[0, 0] == pytest.approx(0.0)  # 1.0 - 0.5*2
        assert out[0, 1] == pytest.approx(1.5)  # 2.0 - 0.5*1
        assert out[0, 2] == pytest.approx(3.0)
        # identity cases
        np.testing.assert_allclose(
            _apply_frequency_penalty(logits.copy(), np.array([], dtype=np.int64), 1.0), logits)
        np.testing.assert_allclose(
            _apply_frequency_penalty(logits.copy(), np.array([1], dtype=np.int64), 0.0), logits)

    def test_apply_presence_penalty(self):
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = _apply_presence_penalty(logits.copy(), np.array([0, 1, 0, 77], dtype=np.int64), 1.0)
        assert out[0, 0] == pytest.approx(0.0)
        assert out[0, 1] == pytest.approx(1.0)
        assert out[0, 2] == pytest.approx(3.0)
        np.testing.assert_allclose(
            _apply_presence_penalty(logits.copy(), np.array([], dtype=np.int64), 1.0), logits)

    def test_sample_from_logits_greedy_argmax(self):
        logits = np.array([[1.0, 5.0, 3.0]], dtype=np.float32)
        assert _sample_from_logits(logits, temperature=0.0) == 1

    def test_sample_from_logits_3d_input_squeezed(self):
        logits = np.zeros((1, 1, 5), dtype=np.float32)
        logits[0, 0, 3] = 10.0
        assert _sample_from_logits(logits, temperature=0.0) == 3

    def test_sample_from_logits_temperature_applied(self):
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = _sample_from_logits(logits, temperature=2.0, top_k=3)
        assert out in (0, 1, 2)

    def test_sample_from_logits_masks_eos(self):
        logits = np.array([[1.0, 5.0, 3.0]], dtype=np.float32)
        # eos=1 masked, argmax now 2
        assert _sample_from_logits(logits, temperature=0.0, eos_token=1) == 2

    def test_sample_from_logits_non_finite_replaced(self):
        logits = np.array([[np.nan, np.inf, 4.0, 2.0]], dtype=np.float32)
        assert _sample_from_logits(logits, temperature=0.0) == 2

    def test_sample_from_logits_top_k_and_top_p(self):
        np.random.seed(0)
        logits = np.array([[1.0, 2.0, 3.0, 0.5, 0.2]], dtype=np.float32)
        tok = _sample_from_logits(logits, temperature=1.0, top_k=2, top_p=0.95)
        assert tok in (1, 2)

    def test_sample_from_logits_seeded_deterministic(self):
        logits = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32)
        np.random.seed(7)
        a = _sample_from_logits(logits, temperature=1.0, top_k=6)
        np.random.seed(7)
        b = _sample_from_logits(logits, temperature=1.0, top_k=6)
        assert a == b


# ── generate_numpy ───────────────────────────────────────────────────────────

class TestGenerateNumpy:

    def test_greedy_shape_and_prompt_preserved(self):
        m = _tiny()
        out = m.generate_numpy(PROMPT, max_new_tokens=6, temperature=0.0)
        assert out.shape == (1, 10)
        assert out.dtype == np.int64
        np.testing.assert_array_equal(out[0, :4], PROMPT[0])

    def test_greedy_deterministic(self):
        m = _tiny()
        a = m.generate_numpy(PROMPT, max_new_tokens=6, temperature=0.0)
        b = m.generate_numpy(PROMPT, max_new_tokens=6, temperature=0.0)
        np.testing.assert_array_equal(a, b)

    def test_1d_input_reshaped(self):
        m = _tiny()
        out = m.generate_numpy(PROMPT[0], max_new_tokens=3, temperature=0.0)
        assert out.shape == (1, 7)

    def test_sampling_path(self):
        m = _tiny()
        out = m.generate_numpy(PROMPT, max_new_tokens=6, temperature=0.8, top_k=5, top_p=0.9)
        assert out.shape == (1, 10)

    def test_repetition_penalty_path(self):
        m = _tiny()
        out = m.generate_numpy(PROMPT, max_new_tokens=6, temperature=0.0, repetition_penalty=1.2)
        assert out.shape == (1, 10)

    def test_eos_stops_early(self):
        m = _tiny()
        eos = _first_token(m)
        out = m.generate_numpy(PROMPT, max_new_tokens=4, temperature=0.0, eos_token=eos)
        assert out.shape[1] == 5

    def test_max_seq_len_caps(self):
        m = _tiny(max_seq_len=8)
        out = m.generate_numpy(PROMPT, max_new_tokens=10, temperature=0.0)
        assert out.shape[1] == 8

    def test_quantized_fake_path(self):
        m = _tiny()
        m.blocks[0].attn.W_q._quant_info = SimpleNamespace(is_quantized=False)
        out = m.generate_numpy(PROMPT, max_new_tokens=5, temperature=0.0)
        assert out.shape == (1, 9)
        m2 = _tiny()
        m2.blocks[0].attn.W_q._quant_info = SimpleNamespace(is_quantized=False)
        s_out = list(m2.generate_numpy_stream(PROMPT, max_new_tokens=3, temperature=0.8, top_k=4, top_p=0.9))
        assert len(s_out) == 3

    def test_gqa_model(self):
        m = _tiny(n_kv_head=1)
        out = m.generate_numpy(PROMPT, max_new_tokens=5, temperature=0.0)
        assert out.shape == (1, 9)

    def test_layer_norm_model(self):
        m = _tiny(norm_type="layer_norm")
        out = m.generate_numpy(PROMPT, max_new_tokens=5, temperature=0.0)
        assert out.shape == (1, 9)

    def test_abs_pos_emb_model(self):
        m = _tiny(use_abs_pos_emb=True)
        out = m.generate_numpy(PROMPT, max_new_tokens=5, temperature=0.0)
        assert out.shape == (1, 9)

    def test_gelu_activation_and_emb_dropout(self):
        m = _tiny(dropout=0.1, activation="gelu")
        m.eval()
        out = m.generate_numpy(PROMPT, max_new_tokens=5, temperature=0.0)
        assert out.shape == (1, 9)
        assert isinstance(m.layers[1], sn.SloDropout)

    def test_no_rope_model(self):
        m = _tiny(use_rope=False)
        out = m.generate_numpy(PROMPT, max_new_tokens=5, temperature=0.0)
        assert out.shape == (1, 9)

    def test_matches_stream_for_greedy(self):
        m = _tiny()
        full = m.generate_numpy(PROMPT, max_new_tokens=5, temperature=0.0)
        toks = list(m.generate_numpy_stream(PROMPT, max_new_tokens=5, temperature=0.0))
        np.testing.assert_array_equal(full[0, 4:], np.array(toks))


# ── generate_numpy_stream ────────────────────────────────────────────────────

class TestGenerateNumpyStream:

    def test_yields_tokens(self):
        m = _tiny()
        toks = list(m.generate_numpy_stream(PROMPT, max_new_tokens=4, temperature=0.0))
        assert len(toks) == 4
        assert all(isinstance(t, (int, np.integer)) for t in toks)

    def test_1d_input(self):
        m = _tiny()
        toks = list(m.generate_numpy_stream(PROMPT[0], max_new_tokens=3, temperature=0.0))
        assert len(toks) == 3

    def test_sampling(self):
        m = _tiny()
        toks = list(m.generate_numpy_stream(
            PROMPT, max_new_tokens=4, temperature=0.8, top_k=5, top_p=0.9))
        assert len(toks) == 4

    def test_eos_stops_after_first_step(self):
        m = _tiny()
        eos = _first_token(m)
        toks = list(m.generate_numpy_stream(PROMPT, max_new_tokens=4, temperature=0.0, eos_token=eos))
        assert len(toks) == 1


# ── generate (Tensor path) ───────────────────────────────────────────────────

class TestGenerateTensor:

    def test_greedy_output(self):
        m = _tiny()
        out = m.generate(PROMPT, max_new_tokens=3, temperature=0.0)
        assert isinstance(out, sn.Tensor)
        assert out.data.shape == (1, 7)
        np.testing.assert_array_equal(out.data[0, :4].astype(np.int64), PROMPT[0])

    def test_tensor_input_and_kv_cache_cleared(self):
        m = _tiny()
        m._kv_caches[0] = (np.zeros((1, 4, 1, 8)), np.zeros((1, 4, 1, 8)))
        out = m.generate(sn.Tensor(PROMPT), max_new_tokens=2, temperature=0.0)
        assert out.data.shape == (1, 6)
        assert all(c is None for c in m._kv_caches)

    def test_eos_break(self):
        m = _tiny()
        eos = _first_token(m)
        out = m.generate(PROMPT, max_new_tokens=4, temperature=0.0, eos_token=eos)
        assert out.data.shape[1] == 5

    def test_sampling_with_penalties(self):
        m = _tiny()
        out = m.generate(
            PROMPT, max_new_tokens=3, temperature=0.7, top_k=8, top_p=0.9,
            repetition_penalty=1.1, frequency_penalty=0.2, presence_penalty=0.1)
        assert out.data.shape == (1, 7)

    def test_clear_kv_cache(self):
        m = _tiny()
        m._kv_caches = [("k", "v")] * m.n_layer
        m.clear_kv_cache()
        assert all(c is None for c in m._kv_caches)

    def test_call_delegates_to_forward(self):
        m = _tiny()
        logits, _ = m(PROMPT)
        assert isinstance(logits, sn.Tensor)


# ── State dict round trips ───────────────────────────────────────────────────

class TestSloTransformerStateDict:

    def test_named_parameters_keys(self):
        m = _tiny()
        keys = [n for n, _ in m._named_parameters()]
        assert "tok_emb.weight" in keys
        assert "blocks.0.attn_norm.weight" in keys
        assert "blocks.0.attn.q_proj.weight" in keys
        assert "blocks.0.attn.k_proj.weight" in keys
        assert "blocks.0.attn.v_proj.weight" in keys
        assert "blocks.0.attn.o_proj.weight" in keys
        assert "blocks.0.attn.q_proj.bias" in keys
        assert "blocks.0.ff_norm.weight" in keys
        assert "blocks.0.ff.w1.weight" in keys
        assert "blocks.0.ff.w2.weight" in keys
        assert "blocks.0.ff.w3.weight" in keys
        assert "norm.weight" in keys
        assert "lm_head.weight" in keys

    def test_layer_norm_bias_keys(self):
        m = _tiny(norm_type="layer_norm")
        keys = [n for n, _ in m._named_parameters()]
        assert "blocks.0.attn_norm.bias" in keys
        assert "blocks.0.ff_norm.bias" in keys
        assert "norm.bias" in keys

    def test_state_dict_round_trip(self):
        m1 = _tiny()
        m2 = _tiny()
        sd = m1.state_dict()
        missing = m2.load_state_dict(sd)
        assert missing == []
        for name, p in m2._named_parameters():
            np.testing.assert_array_equal(p.data, sd[name])

    def test_named_parameters_alias(self):
        m = _tiny()
        assert [n for n, _ in m.named_parameters()] == [n for n, _ in m._named_parameters()]

    def test_load_state_dict_partial_rows_2d(self):
        m = _tiny()
        sd = m.state_dict()
        slim = np.ones((16, 16), dtype=np.float32) * 3.0
        sd["lm_head.weight"] = slim
        missing = m.load_state_dict(sd, strict=False)
        assert "lm_head.weight" not in missing
        np.testing.assert_array_equal(m.layers[-1].weight.data[:16], slim)

    def test_load_state_dict_partial_1d(self):
        m = _tiny()
        sd = m.state_dict()
        sd["norm.weight"] = np.ones(8, dtype=np.float32) * 9.0
        missing = m.load_state_dict(sd, strict=False)
        assert "norm.weight" not in missing
        np.testing.assert_array_equal(m.norm.weight.data[:8], np.full(8, 9.0))

    def test_load_state_dict_strict_returns_missing(self):
        m = _tiny()
        sd = m.state_dict()
        sd["bogus.key"] = np.zeros(3, dtype=np.float32)
        missing = m.load_state_dict(sd, strict=True)
        assert "bogus.key" in missing

    def test_load_state_dict_dtype_cast(self):
        m = _tiny()
        sd = m.state_dict()
        sd["norm.weight"] = sd["norm.weight"].astype(np.float64)
        missing = m.load_state_dict(sd)
        assert missing == []
        assert m.norm.weight.data.dtype == np.float32

    def test_tie_weights_round_trip(self):
        m = _tiny(tie_weights=False)
        assert not np.array_equal(m.layers[0].weight.data, m.layers[-1].weight.data)
        m._tie_weights()
        np.testing.assert_array_equal(m.layers[0].weight.data, m.layers[-1].weight.data)

    def test_tie_weights_exception_silent(self):
        m = _tiny(tie_weights=False)
        m.layers[-1].weight.data = np.zeros((10, 10), dtype=np.float32)  # shape mismatch
        m._tie_weights()  # must not raise

    def test_pos_emb_in_state_dict(self):
        m = _tiny(use_abs_pos_emb=True)
        sd = m.state_dict()
        assert "pos_emb.weight" in sd
        assert m.pos_emb.num_embeddings == m.max_seq_len


# ── _rebuild_from_state_dict / _forward_state_dict ──────────────────────────

class TestSoulLibStateDict:

    def _gpt2_style_sd(self, vocab=16, hidden=32, ff_dim=64):
        sd = {}
        sd["tok_emb.weight"] = np.random.RandomState(1).randn(vocab, hidden).astype(np.float32)
        sd["blocks.0.norm1.weight"] = np.ones(hidden, dtype=np.float32)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            sd[f"blocks.0.attn.{name}.weight"] = (
                np.random.RandomState(2).randn(hidden, hidden).astype(np.float32))
        sd["blocks.0.norm2.weight"] = np.ones(hidden, dtype=np.float32)
        sd["blocks.0.mlp.w1.weight"] = np.random.RandomState(3).randn(ff_dim, hidden).astype(np.float32)
        sd["blocks.0.mlp.w2.weight"] = np.random.RandomState(4).randn(hidden, ff_dim).astype(np.float32)
        sd["blocks.0.mlp.w3.weight"] = np.random.RandomState(5).randn(ff_dim, hidden).astype(np.float32)
        sd["norm.weight"] = np.ones(hidden, dtype=np.float32)
        return sd

    def test_rebuild_builds_blocks(self):
        net = sn.SloNet()
        sd = self._gpt2_style_sd()
        net._rebuild_from_state_dict(sd)
        assert len(net.layers) == 2  # 1 block + output norm
        assert isinstance(net.layers[0], sn._SoulTransformerBlockSoulLib)
        assert net._sd is sd

    def test_forward_state_dict(self):
        net = sn.SloNet()
        net._rebuild_from_state_dict(self._gpt2_style_sd())
        x = sn.Tensor(np.array([[1, 2, 3, 4]], dtype=np.float32))
        out = net.forward(x)
        assert isinstance(out, sn.Tensor)
        assert out.data.shape == (1, 4, 16)

    def test_forward_without_state_dict_uses_layers(self):
        net = sn.SloNet(layers=[sn.SloLinear(4, 2)])
        out = net.forward(np.ones((3, 4), dtype=np.float32))
        assert out.data.shape == (3, 2)

    def test_get_weight(self):
        net = sn.SloNet()
        sd = self._gpt2_style_sd()
        net._rebuild_from_state_dict(sd)
        np.testing.assert_array_equal(net._get_weight("norm.weight"), sd["norm.weight"])
        assert net._get_weight("nope") is None

    def test_soul_signature(self):
        net = sn.SloNet()
        net._rebuild_from_state_dict(self._gpt2_style_sd())
        sig = net.soul_signature()
        assert sig["soul_name"] == "Slo"
        assert sig["layers"][0]["layer"] == "SloTransformerBlock"


# ── Schedulers ───────────────────────────────────────────────────────────────

class MockOpt:
    def __init__(self, lr=0.1):
        self.lr = lr
        self.param_groups = [{"lr": lr}]


class TestSloStepLR:

    def test_auto_step_on_init(self):
        opt = MockOpt(lr=0.1)
        sched = sn.SloStepLR(opt, step_size=2, gamma=0.5)
        assert sched.last_epoch == 0
        assert opt.lr == pytest.approx(0.1)

    def test_decay_by_gamma(self):
        opt = MockOpt(lr=0.1)
        sched = sn.SloStepLR(opt, step_size=2, gamma=0.5)
        sched.step()
        sched.step()  # last_epoch=2
        assert opt.lr == pytest.approx(0.05)
        sched.step()
        sched.step()  # last_epoch=4
        assert opt.lr == pytest.approx(0.025)

    def test_step_with_explicit_epoch(self):
        opt = MockOpt(lr=0.1)
        sched = sn.SloStepLR(opt, step_size=3, gamma=0.1)
        sched.step(epoch=6)
        assert opt.lr == pytest.approx(0.1 * 0.1 ** 2)

    def test_get_last_lr(self):
        opt = MockOpt(lr=0.1)
        sched = sn.SloStepLR(opt, step_size=1, gamma=0.5)
        sched.step()
        assert sched.get_last_lr() == [pytest.approx(0.05)]

    def test_state_dict_round_trip(self):
        opt = MockOpt(lr=0.1)
        sched = sn.SloStepLR(opt, step_size=2, gamma=0.5)
        sched.step()
        state = sched.state_dict()
        assert set(state) == {"last_epoch", "base_lrs"}
        sched2 = sn.SloStepLR(MockOpt(lr=0.1), step_size=2, gamma=0.5)
        sched2.load_state_dict(state)
        assert sched2.last_epoch == sched.last_epoch
        assert sched2.base_lrs == sched.base_lrs


class TestSloCosineAnnealingLR:

    def test_starts_at_base_lr(self):
        opt = MockOpt(lr=0.1)
        sched = sn.SloCosineAnnealingLR(opt, T_max=10)
        assert sched.get_lr() == [pytest.approx(0.1)]

    def test_decreases_then_hits_eta_min(self):
        opt = MockOpt(lr=0.1)
        sched = sn.SloCosineAnnealingLR(opt, T_max=10, eta_min=0.0)
        sched.step(epoch=5)
        assert sched.get_lr()[0] < 0.1
        sched.step(epoch=20)
        assert sched.get_lr()[0] == pytest.approx(0.0)

    def test_step_updates_optimizer_lr(self):
        opt = MockOpt(lr=0.1)
        sched = sn.SloCosineAnnealingLR(opt, T_max=10)
        sched.step(epoch=10)
        assert opt.lr == pytest.approx(0.0)


class TestSloReduceLROnPlateau:

    def test_min_mode_improves(self):
        opt = MockOpt(lr=0.1)
        sched = sn.SloReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
        sched.step(10.0)
        assert sched.best == 10.0
        sched.step(8.0)
        assert sched.best == 8.0
        assert opt.lr == pytest.approx(0.1)

    def test_min_mode_reduces_after_patience(self):
        opt = MockOpt(lr=0.1)
        sched = sn.SloReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
        for _ in range(4):
            sched.step(10.0)
        assert opt.lr == pytest.approx(0.05)
        assert sched.last_lr == pytest.approx(0.05)

    def test_cooldown_resets_bad_epochs(self):
        opt = MockOpt(lr=0.1)
        sched = sn.SloReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1, cooldown=2)
        sched.step(10.0)
        sched.step(10.0)  # bad epoch 1
        assert sched.num_bad_epochs == 1
        sched.step(10.0)  # cooldown active, resets
        assert sched.num_bad_epochs == 0

    def test_max_mode(self):
        opt = MockOpt(lr=0.1)
        sched = sn.SloReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
        sched.step(1.0)
        sched.step(0.5)  # worse for max mode
        assert sched.num_bad_epochs == 1
        sched.step(2.0)  # better
        assert sched.best == 2.0

    def test_abs_threshold_mode(self):
        opt = MockOpt(lr=0.1)
        sched = sn.SloReduceLROnPlateau(opt, mode="min", threshold=0.1, threshold_mode="abs", patience=1)
        sched.step(10.0)
        sched.step(10.05)  # within abs threshold → still better (below 10.0-0.1? no: 10.05 > 9.9 → not better)
        assert sched.num_bad_epochs == 1

    def test_min_lr_floor(self):
        opt = MockOpt(lr=1e-6)
        sched = sn.SloReduceLROnPlateau(opt, mode="min", factor=0.5, patience=0, min_lr=1e-7)
        sched.step(5.0)
        sched.step(5.0)
        assert opt.lr >= 1e-7

    def test_state_dict_round_trip(self):
        opt = MockOpt(lr=0.1)
        sched = sn.SloReduceLROnPlateau(opt, mode="min", patience=1)
        sched.step(10.0)
        state = sched.state_dict()
        sched2 = sn.SloReduceLROnPlateau(MockOpt(lr=0.1), mode="min", patience=1)
        sched2.load_state_dict(state)
        assert sched2.best == 10.0
        assert sched2.num_bad_epochs == sched.num_bad_epochs


class TestOtherSchedulers:

    def test_constant_lr(self):
        opt = MockOpt(lr=0.1)
        sched = sn.SloConstantLR(opt)
        sched.step()
        sched.step()
        assert opt.lr == pytest.approx(0.1)

    def test_one_cycle_warmup_then_decay(self):
        opt = MockOpt(lr=0.1)
        sched = sn.SloOneCycleLR(opt, max_lr=0.5, total_steps=10, pct_start=0.2)
        sched.step(epoch=0)  # phase < pct_start
        assert opt.lr <= 0.5
        sched.step(epoch=9)
        assert opt.lr <= 0.5

    def test_warmup_cosine(self):
        opt = MockOpt(lr=0.1)
        sched = sn.WarmupCosineScheduler(opt, warmup_steps=10, total_steps=100)
        sched.step(epoch=0)
        assert opt.lr == pytest.approx(0.0)
        sched.step(epoch=5)
        assert opt.lr == pytest.approx(0.05)
        sched.step(epoch=100)
        assert opt.lr < 0.1

    def test_polynomial_decay(self):
        opt = MockOpt(lr=0.1)
        sched = sn.PolynomialDecayScheduler(opt, total_steps=10, min_lr=0.0, power=1.0)
        sched.step(epoch=0)
        assert opt.lr == pytest.approx(0.1)
        sched.step(epoch=10)
        assert opt.lr == pytest.approx(0.0)
        sched.step(epoch=50)
        assert opt.lr == pytest.approx(0.0)

    def test_linear_warmup_hold_and_decay(self):
        opt = MockOpt(lr=0.1)
        sched = sn.LinearWarmupScheduler(
            opt, warmup_steps=10, base_lr=0.01, hold_steps=5,
            total_steps=20, decay_type="linear", min_lr=0.0)
        sched.step(epoch=0)
        assert opt.lr == pytest.approx(0.0)
        sched.step(epoch=10)
        assert opt.lr == pytest.approx(0.01)
        sched.step(epoch=12)
        assert opt.lr == pytest.approx(0.01)
        sched.step(epoch=20)
        assert opt.lr < 0.01

    def test_linear_warmup_cosine_decay(self):
        opt = MockOpt(lr=0.1)
        sched = sn.LinearWarmupScheduler(
            opt, warmup_steps=5, base_lr=0.02, hold_steps=5,
            total_steps=20, decay_type="cosine", min_lr=0.0)
        sched.step(epoch=20)
        assert opt.lr == pytest.approx(0.0)

    def test_linear_warmup_no_decay(self):
        opt = MockOpt(lr=0.1)
        sched = sn.LinearWarmupScheduler(
            opt, warmup_steps=5, base_lr=0.02, hold_steps=0,
            total_steps=20, decay_type="none")
        sched.step(epoch=20)
        assert opt.lr == pytest.approx(0.02)


# ── Optimizer state dicts ────────────────────────────────────────────────────

class TestOptimizerStateDict:

    def _two_params(self):
        p1 = sn.Tensor(np.array([1.0, 2.0]), requires_grad=True)
        p2 = sn.Tensor(np.array([0.5]), requires_grad=True)
        p1.grad = sn.Tensor(np.array([0.1, 0.1]))
        p2.grad = sn.Tensor(np.array([0.2]))
        return [p1, p2]

    def test_sgd_state_dict_without_params(self):
        opt = sn.SloSGD(lr=0.01, momentum=0.9)
        state = opt.state_dict()
        assert state["hyperparameters"]["lr"] == 0.01
        assert state["state"] == {}

    def test_sgd_state_dict_round_trip_with_momentum(self):
        params = self._two_params()
        opt = sn.SloSGD(lr=0.01, momentum=0.9, max_grad_norm=1.0)
        opt.step(params)
        params[0].grad = sn.Tensor(np.array([0.1, 0.1]))
        opt.step(params)
        state = opt.state_dict(params)

        opt2 = sn.SloSGD(lr=0.01, momentum=0.9)
        opt2.load_state_dict(state, params)
        assert len(opt2._v) == 2
        assert opt2.lr == 0.01

    def test_sgd_load_state_dict_without_params(self):
        opt = sn.SloSGD()
        opt.load_state_dict({"hyperparameters": {"lr": 0.5}, "state": {}})
        assert opt.lr == 0.5

    def test_adam_state_dict_round_trip(self):
        params = self._two_params()
        opt = sn.SloAdam(lr=0.001, weight_decay=0.01)
        opt.step(params)
        params[0].grad = sn.Tensor(np.array([0.1, 0.1]))
        params[1].grad = sn.Tensor(np.array([0.2]))
        opt.step(params)
        state = opt.state_dict(params)
        assert state["t"] == 2
        assert "param_0" in state["state"]

        opt2 = sn.SloAdam(lr=0.001, weight_decay=0.01)
        opt2.load_state_dict(state, params)
        assert len(opt2._m) == 2
        assert len(opt2._v) == 2
        assert opt2._t == 2

    def test_adam_state_dict_without_params(self):
        opt = sn.SloAdam()
        assert opt.state_dict()["state"] == {}
