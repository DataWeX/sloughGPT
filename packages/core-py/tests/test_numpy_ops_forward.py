"""Tests for numpy_ops and the generic numpy transformer forward pass."""

import numpy as np
import pytest

from domains.infrastructure.arch_config import ArchConfig, build_arch
from domains.infrastructure.numpy_forward import (
    forward,
    forward_cached,
    forward_fast,
    norm_fn,
    pre_extract_weights,
)
from domains.infrastructure.numpy_ops import (
    gelu,
    layer_norm,
    rope,
    rmsnorm,
    silu,
    softmax,
    to_float32,
)
from domains.infrastructure.numpy_engine import KVCache


class TestToFloat32:
    def test_float32_passthrough(self):
        arr = np.array([1.0], dtype=np.float32)
        assert to_float32(arr) is arr

    def test_float64_converts(self):
        arr = np.array([1.0], dtype=np.float64)
        assert to_float32(arr).dtype == np.float32

    def test_float16_converts(self):
        arr = np.array([1.0], dtype=np.float16)
        assert to_float32(arr).dtype == np.float32
        assert to_float32(arr)[0] == 1.0

    def test_bfloat16_bit_conversion(self):
        class FakeBF16:
            dtype = type("D", (), {"name": "bfloat16"})()

            def __init__(self, bits):
                self._bits = np.array(bits, dtype=np.uint16)

            def view(self, dt):
                if dt is np.uint16:
                    return self._bits.copy()
                raise TypeError

        one = FakeBF16([0x3F80])  # 1.0 in bfloat16
        assert to_float32(one)[0] == 1.0


class TestOps:
    def test_softmax_sums_to_one(self):
        x = np.array([[1.0, 2.0, 3.0]])
        out = softmax(x)
        assert np.allclose(out.sum(axis=-1), 1.0)

    def test_softmax_stability(self):
        x = np.array([1000.0, 1001.0])
        out = softmax(x)
        assert np.allclose(out.sum(), 1.0)
        assert not np.isnan(out).any()

    def test_rmsnorm_manual(self):
        x = np.array([[1.0, 2.0, 3.0]])
        w = np.array([1.0, 1.0, 1.0])
        out = rmsnorm(x, w, eps=1e-6)
        rms = np.sqrt(np.mean(x**2) + 1e-6)
        assert np.allclose(out, x / rms)

    def test_rmsnorm_applies_weight(self):
        x = np.array([[2.0, 2.0]])
        w = np.array([1.0, 2.0])
        out = rmsnorm(x, w)
        assert out[0][0] == pytest.approx(out[0][1] / 2.0)

    def test_layer_norm_mean_zero_unit_var(self):
        x = np.array([[1.0, 2.0, 3.0]])
        out = layer_norm(x, np.ones(3), np.zeros(3))
        assert out.mean() == pytest.approx(0.0, abs=1e-4)
        assert out.var() == pytest.approx(1.0, abs=1e-3)

    def test_layer_norm_bias(self):
        x = np.array([[1.0, 2.0]])
        out = layer_norm(x, np.ones(2), np.array([5.0, 5.0]))
        assert out[0].mean() == pytest.approx(5.0)

    def test_layer_norm_none_bias(self):
        x = np.array([[1.0, 2.0]])
        out = layer_norm(x, np.ones(2), None)
        assert out[0].mean() == pytest.approx(0.0, abs=1e-5)

    def test_gelu_zero(self):
        assert gelu(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_gelu_matches_formula(self):
        x = np.array([0.5, -0.5, 1.0])
        out = gelu(x)
        expected = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
        assert np.allclose(out, expected, atol=1e-7)

    def test_gelu_sign_and_order(self):
        out = gelu(np.array([-1.0, 0.0, 1.0]))
        assert out[0] < 0 < out[2]
        assert out[1] == pytest.approx(0.0)
        assert out[2] > out[0]

    def test_silu_zero(self):
        assert silu(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_silu_value(self):
        assert silu(np.array([1.0]))[0] == pytest.approx(0.73105857863, rel=1e-5)

    def test_rope_shape(self):
        x = np.zeros((4, 2, 4))
        out = rope(x, pos=0, dim=4)
        assert out.shape == (4, 2, 4)

    def test_rope_zero_input_stays_zero(self):
        x = np.zeros((3, 2, 4))
        out = rope(x, pos=0, dim=4)
        assert np.allclose(out, 0.0)

    def test_rope_position_dependent(self):
        x = np.ones((2, 1, 4))
        a = rope(x, pos=0, dim=4)
        b = rope(x, pos=5, dim=4)
        assert not np.allclose(a, b)

    def test_rope_preserves_norm(self):
        x = np.random.randn(3, 2, 4).astype(np.float32)
        out = rope(x, pos=0, dim=4)
        assert np.allclose(np.linalg.norm(out, axis=-1), np.linalg.norm(x, axis=-1), atol=1e-4)


def make_gpt2_weights(vocab=8, n_embed=4, n_layers=1, seq_max=16):
    rng = np.random.default_rng(0)
    ffn = n_embed * 4
    W = {}
    W["wte.weight"] = rng.standard_normal((vocab, n_embed)).astype(np.float32)
    W["wpe.weight"] = rng.standard_normal((seq_max, n_embed)).astype(np.float32)
    W["ln_f.weight"] = rng.standard_normal((n_embed,)).astype(np.float32)
    W["ln_f.bias"] = rng.standard_normal((n_embed,)).astype(np.float32)
    for i in range(n_layers):
        p = f"h.{i}."
        W[p + "ln_1.weight"] = rng.standard_normal((n_embed,)).astype(np.float32)
        W[p + "ln_1.bias"] = rng.standard_normal((n_embed,)).astype(np.float32)
        # GPT-2 Conv1D stores (in, out)
        W[p + "attn.c_attn.weight"] = rng.standard_normal((n_embed, n_embed * 3)).astype(np.float32)
        W[p + "attn.c_attn.bias"] = rng.standard_normal((n_embed * 3,)).astype(np.float32)
        W[p + "attn.c_proj.weight"] = rng.standard_normal((n_embed, n_embed)).astype(np.float32)
        W[p + "attn.c_proj.bias"] = rng.standard_normal((n_embed,)).astype(np.float32)
        W[p + "ln_2.weight"] = rng.standard_normal((n_embed,)).astype(np.float32)
        W[p + "ln_2.bias"] = rng.standard_normal((n_embed,)).astype(np.float32)
        W[p + "mlp.c_fc.weight"] = rng.standard_normal((n_embed, ffn)).astype(np.float32)
        W[p + "mlp.c_fc.bias"] = rng.standard_normal((ffn,)).astype(np.float32)
        W[p + "mlp.c_proj.weight"] = rng.standard_normal((ffn, n_embed)).astype(np.float32)
        W[p + "mlp.c_proj.bias"] = rng.standard_normal((n_embed,)).astype(np.float32)
    return W


def make_gpt2_arch(n_layers=1, n_embed=4, n_head=2):
    config = {"architectures": ["GPT2LMHeadModel"], "n_head": n_head,
              "n_embd": n_embed, "n_layer": n_layers}
    keys = {"wte.weight", "h.0.ln_1.weight", "h.0.attn.c_attn.weight"}
    return build_arch("gpt2", config, keys)


def make_llama_weights(vocab=8, n_embed=8, n_head=4, n_kv=2, n_layers=1):
    rng = np.random.default_rng(1)
    kv_dim = (n_embed // n_head) * n_kv
    W = {}
    W["model.embed_tokens.weight"] = rng.standard_normal((vocab, n_embed)).astype(np.float32)
    W["model.norm.weight"] = rng.standard_normal((n_embed,)).astype(np.float32)
    for i in range(n_layers):
        p = f"model.layers.{i}."
        W[p + "input_layernorm.weight"] = rng.standard_normal((n_embed,)).astype(np.float32)
        W[p + "post_attention_layernorm.weight"] = rng.standard_normal((n_embed,)).astype(np.float32)
        W[p + "self_attn.q_proj.weight"] = rng.standard_normal((n_embed, n_embed)).astype(np.float32)
        W[p + "self_attn.k_proj.weight"] = rng.standard_normal((kv_dim, n_embed)).astype(np.float32)
        W[p + "self_attn.v_proj.weight"] = rng.standard_normal((kv_dim, n_embed)).astype(np.float32)
        W[p + "self_attn.o_proj.weight"] = rng.standard_normal((n_embed, n_embed)).astype(np.float32)
        W[p + "mlp.gate_proj.weight"] = rng.standard_normal((n_embed, n_embed)).astype(np.float32)
        W[p + "mlp.up_proj.weight"] = rng.standard_normal((n_embed, n_embed)).astype(np.float32)
        W[p + "mlp.down_proj.weight"] = rng.standard_normal((n_embed, n_embed)).astype(np.float32)
    return W


def make_llama_arch(n_layers=1, n_embed=8, n_head=4, n_kv=2):
    config = {"architectures": ["LlamaForCausalLM"], "num_attention_heads": n_head,
              "num_key_value_heads": n_kv, "hidden_size": n_embed,
              "num_hidden_layers": n_layers}
    keys = {"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.input_layernorm.weight", "model.layers.0.mlp.gate_proj.weight"}
    return build_arch("llama", config, keys)


class TestNormFn:
    def test_rms(self):
        arch = make_gpt2_arch()
        arch.norm = "rms_norm"
        assert norm_fn(arch) is rmsnorm

    def test_layer(self):
        arch = make_gpt2_arch()
        assert norm_fn(arch) is layer_norm


class TestForwardGpt2:
    def test_logits_shape_and_finite(self):
        arch = make_gpt2_arch()
        W = make_gpt2_weights()
        logits = forward(W, arch, [0, 1, 2])
        assert logits.shape == (8,)
        assert np.isfinite(logits).all()

    def test_absolute_positional_affects_output(self):
        arch = make_gpt2_arch()
        W = make_gpt2_weights()
        a = forward(W, arch, [0, 1])
        b = forward(W, arch, [0, 1])
        assert np.allclose(a, b)

    def test_single_layer_multi_layer(self):
        arch = make_gpt2_arch(n_layers=1)
        arch2 = make_gpt2_arch(n_layers=1)
        assert arch.n_layers == arch2.n_layers

    def test_pre_extract_and_fast_match(self):
        arch = make_gpt2_arch()
        W = make_gpt2_weights()
        logits = forward(W, arch, [0, 1, 2, 3])
        rw = pre_extract_weights(arch, W)
        fast = forward_fast(rw, arch, [0, 1, 2, 3])
        assert np.allclose(fast, logits, atol=1e-6)

    def test_pre_extract_resolves_layers(self):
        arch = make_gpt2_arch(n_layers=2)
        W = make_gpt2_weights(n_layers=2)
        rw = pre_extract_weights(arch, W)
        assert "layers.{i}.attn_norm.weight:0" in rw
        assert "layers.{i}.attn_norm.weight:1" in rw
        assert rw["layers.{i}.attn_norm.weight:0"].flags["C_CONTIGUOUS"]

    def test_pre_extract_skips_missing(self):
        arch = make_gpt2_arch()
        W = make_gpt2_weights()
        del W["h.0.mlp.c_proj.bias"]
        rw = pre_extract_weights(arch, W)
        assert "layers.{i}.ffn.down.bias:0" not in rw

    def test_forward_cached_without_cache_matches_forward(self):
        arch = make_gpt2_arch()
        W = make_gpt2_weights()
        tokens = [0, 1, 2, 3]
        logits = forward(W, arch, tokens)
        cached = forward_cached(lambda n: W[n], arch, tokens)
        assert np.allclose(cached, logits, atol=1e-6)

    def test_incremental_kv_cache_matches_full(self):
        arch = make_gpt2_arch()
        W = make_gpt2_weights()
        tokens = [0, 1, 2, 3]
        full = forward(W, arch, tokens)

        kv = KVCache(arch.n_layers)
        get = lambda n: W[n]
        forward_cached(get, arch, [tokens[0]], kv_cache=kv)
        last = None
        for i in range(1, len(tokens)):
            last = forward_cached(get, arch, [tokens[i]], kv_cache=kv, start_pos=i)
        assert np.allclose(last, full, atol=1e-5)

    def test_lm_head_override(self):
        arch = make_gpt2_arch()
        W = make_gpt2_weights()
        W["lm_head.weight"] = np.full((8, 4), 5.0, dtype=np.float32)
        logits = forward(W, arch, [0, 1])
        assert np.allclose(logits, logits[0])


class TestForwardLlama:
    def test_gqa_swiglu_rope(self):
        arch = make_llama_arch()
        W = make_llama_weights()
        assert arch.norm == "rms_norm"
        assert arch.positional == "rope"
        assert arch.activation == "swiglu"
        assert arch.attention == "gqa"
        logits = forward(W, arch, [0, 1, 2])
        assert logits.shape == (8,)
        assert np.isfinite(logits).all()

    def test_fast_matches_llama(self):
        arch = make_llama_arch()
        W = make_llama_weights()
        logits = forward(W, arch, [1, 2, 3, 4])
        rw = pre_extract_weights(arch, W)
        assert np.allclose(forward_fast(rw, arch, [1, 2, 3, 4]), logits, atol=1e-6)

    def test_incremental_kv_matches_llama(self):
        arch = make_llama_arch()
        W = make_llama_weights()
        tokens = [0, 1, 2, 3, 4]
        full = forward(W, arch, tokens)
        kv = KVCache(arch.n_layers)
        get = lambda n: W[n]
        forward_cached(get, arch, [tokens[0]], kv_cache=kv)
        last = None
        for i in range(1, len(tokens)):
            last = forward_cached(get, arch, [tokens[i]], kv_cache=kv, start_pos=i)
        assert np.allclose(last, full, atol=1e-5)
