"""Portability contract tests — one handler for all models (card 061).

Invariant under test: model buffer shapes derive from config dims through
ONE alignment contract (``align_dim_ff``); the loader zero-pads FFN
checkpoints to that contract and raises ``ShapeMismatchError`` for
anything else — never a raw numpy broadcast mid-copy.
"""

import numpy as np
import pytest

from domain.infrastructure._internal.weight_loader import (
    ShapeMismatchError,
    _assign_checked,
    _is_align_pad,
    build_load_plan,
    build_model_from_config,
    dims_from_config,
    get_shape_policy,
    infer_arch_from_state_dict,
    load_into_model,
)
from domain.shared import FF_ALIGN, align_dim_ff

# ── align_dim_ff ──────────────────────────────────────────────────────


class TestAlignDimFf:
    @pytest.mark.parametrize(
        "raw,aligned",
        [
            (32, 64),
            (48, 64),
            (40, 64),
            (64, 64),
            (4864, 4864),
            (3072, 3072),
            (65, 128),
            (1, 64),
        ],
    )
    def test_rounds_up_to_ff_align(self, raw, aligned):
        assert align_dim_ff(raw) == aligned == aligned // FF_ALIGN * FF_ALIGN


# ── dims_from_config ──────────────────────────────────────────────────


class TestDimsFromConfig:
    def test_hf_qwen_keys(self):
        d = dims_from_config(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 8,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": 2,
                "vocab_size": 152064,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
            }
        )
        assert d["n_embed"] == 8
        assert d["n_head"] == 4
        assert d["n_kv_head"] == 2
        assert d["n_layer"] == 2
        assert d["intermediate_size"] == 32
        assert d["vocab_size"] == 152064
        assert d["activation"] == "silu"
        assert d["norm_type"] == "rms_norm"
        assert d["use_abs_pos"] is False  # rope_theta present
        assert d["rope_base"] == 1000000.0
        assert d["eps"] == 1e-6

    def test_gpt2_keys_and_defaults(self):
        d = dims_from_config({"n_embd": 768, "n_layer": 12, "n_head": 12})
        assert d["n_embed"] == 768
        assert d["vocab_size"] == 50257
        assert d["intermediate_size"] == 768 * 4
        assert d["use_abs_pos"] is True  # no rope keys
        assert d["activation"] == "gelu"
        assert d["norm_type"] == "layer_norm"


# ── shape helpers / policy registry ───────────────────────────────────


class TestShapeHelpers:
    def test_is_align_pad_cases(self):
        assert _is_align_pad((32,), (64,))
        assert _is_align_pad((8, 32), (8, 64))
        assert _is_align_pad((48,), (64,))
        assert not _is_align_pad((32,), (32,))  # no diff
        assert not _is_align_pad((32,), (96,))  # 96 != align(32)
        assert not _is_align_pad((8, 32), (16, 32))  # not FFN alignment
        assert not _is_align_pad((8, 32), (8, 32, 1))  # ndim differs
        assert not _is_align_pad((64,), (32,))  # shrinking never pads

    def test_policy_registry(self):
        qwen = get_shape_policy("Qwen2ForCausalLM")
        assert "layers.{i}.ffn.gate.weight" in qwen.ffn_pad_canonicals
        assert "layers.{i}.ffn.down.weight" in qwen.ffn_pad_canonicals
        gpt2 = get_shape_policy("GPT2LMHeadModel")
        assert "layers.{i}.ffn.down.weight" in gpt2.ffn_pad_canonicals
        assert "layers.{i}.ffn.gate.weight" not in gpt2.ffn_pad_canonicals
        unknown = get_shape_policy("TotallyNewArch")
        assert "layers.{i}.ffn.down.weight" in unknown.ffn_pad_canonicals

    def test_assign_checked_pads_ffn_and_raises_elsewhere(self):
        class _P:
            pass

        policy = get_shape_policy("Qwen2ForCausalLM")
        p = _P()
        p.data = np.zeros((8, 64), dtype=np.float32)
        src = np.ones((8, 32), dtype=np.float32)
        padded = _assign_checked(
            p, src, "blocks.0.ff.w2.weight", "layers.{i}.ffn.down.weight", policy
        )
        assert padded is True
        np.testing.assert_array_equal(p.data[:, :32], src)
        np.testing.assert_array_equal(p.data[:, 32:], 0)

        p2 = _P()
        p2.data = np.zeros((8, 64), dtype=np.float32)
        with pytest.raises(ShapeMismatchError) as ei:
            _assign_checked(p2, src, "blocks.0.attn.q_proj.weight", "layers.{i}.q.weight", policy)
        assert "blocks.0.attn.q_proj.weight" in str(ei.value)
        assert "(8, 32)" in str(ei.value) and "(8, 64)" in str(ei.value)


# ── synthetic checkpoints ─────────────────────────────────────────────


def _qwen2_sd(h, inter, n_layer, vocab, heads, kv_heads, seed=0):
    """HF-style Qwen2 state dict + config with given dims."""
    rng = np.random.default_rng(seed)
    head_dim = h // heads
    sd = {
        "model.embed_tokens.weight": rng.standard_normal((vocab, h)).astype(np.float32),
        "model.norm.weight": np.ones(h, dtype=np.float32),
    }
    for i in range(n_layer):
        sd.update(
            {
                f"model.layers.{i}.input_layernorm.weight": np.ones(h, np.float32),
                f"model.layers.{i}.post_attention_layernorm.weight": np.ones(h, np.float32),
                f"model.layers.{i}.self_attn.q_proj.weight": rng.standard_normal(
                    (heads * head_dim, h)
                ).astype(np.float32),
                f"model.layers.{i}.self_attn.q_proj.bias": np.zeros(
                    heads * head_dim, np.float32
                ),
                f"model.layers.{i}.self_attn.k_proj.weight": rng.standard_normal(
                    (kv_heads * head_dim, h)
                ).astype(np.float32),
                f"model.layers.{i}.self_attn.v_proj.weight": rng.standard_normal(
                    (kv_heads * head_dim, h)
                ).astype(np.float32),
                f"model.layers.{i}.self_attn.o_proj.weight": rng.standard_normal(
                    (h, h)
                ).astype(np.float32),
                f"model.layers.{i}.mlp.gate_proj.weight": rng.standard_normal(
                    (inter, h)
                ).astype(np.float32),
                f"model.layers.{i}.mlp.up_proj.weight": rng.standard_normal(
                    (inter, h)
                ).astype(np.float32),
                f"model.layers.{i}.mlp.down_proj.weight": rng.standard_normal(
                    (h, inter)
                ).astype(np.float32),
            }
        )
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": h,
        "intermediate_size": inter,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "num_hidden_layers": n_layer,
        "vocab_size": vocab,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
    }
    return sd, config


# ── end-to-end load: the (8,32)->(8,64) regression ────────────────────


class TestPortableLoad:
    def test_unaligned_intermediate_pads_and_loads(self):
        """The trl-tiny-Qwen2 case: intermediate 32 allocates as 64 (SIMD);
        loader must zero-pad, not broadcast-fail."""
        sd, config = _qwen2_sd(h=8, inter=32, n_layer=2, vocab=128, heads=4, kv_heads=2)
        model = build_model_from_config(config, _lazy=True)
        params = dict(model._named_parameters())
        assert params["blocks.0.ff.w2.weight"].data.shape == (8, 64)
        assert params["blocks.0.ff.w1.weight"].data.shape == (64, 8)

        plan = build_load_plan(sd, n_layer=2, config=config)
        result = load_into_model(model, plan, sd)
        assert result.success is True

        w2 = params["blocks.0.ff.w2.weight"].data
        np.testing.assert_array_equal(w2[:, :32], sd["model.layers.0.mlp.down_proj.weight"])
        np.testing.assert_array_equal(w2[:, 32:], 0)
        w1 = params["blocks.0.ff.w1.weight"].data
        np.testing.assert_array_equal(
            w1[:32], sd["model.layers.0.mlp.gate_proj.weight"]
        )
        np.testing.assert_array_equal(w1[32:], 0)

    def test_qwen25_dims_exact_match_no_padding(self):
        """Known-good production dims: already 64-aligned → exact writes."""
        sd, config = _qwen2_sd(
            h=896, inter=4864, n_layer=2, vocab=512, heads=14, kv_heads=2
        )
        model = build_model_from_config(config, _lazy=True)
        params = dict(model._named_parameters())
        assert params["blocks.0.ff.w2.weight"].data.shape == (896, 4864)

        plan = build_load_plan(sd, n_layer=2, config=config)
        result = load_into_model(model, plan, sd)
        assert result.success is True
        np.testing.assert_array_equal(
            params["blocks.0.ff.w2.weight"].data,
            sd["model.layers.0.mlp.down_proj.weight"],
        )

    def test_mismatch_raises_shape_error_before_any_write(self):
        """Config/weights disagree on hidden → structured error, both
        issues listed, not a numpy broadcast."""
        sd, config = _qwen2_sd(h=8, inter=32, n_layer=2, vocab=128, heads=4, kv_heads=2)
        bad_config = dict(config, hidden_size=16)
        model = build_model_from_config(bad_config, _lazy=True)
        params = dict(model._named_parameters())
        tok_before = params["tok_emb.weight"].data.copy()

        plan = build_load_plan(sd, n_layer=2, config=bad_config)
        with pytest.raises(ShapeMismatchError) as ei:
            load_into_model(model, plan, sd)
        msg = str(ei.value)
        assert "tok_emb.weight" in msg
        assert "shape mismatch" in msg
        assert len(ei.value.issues) >= 2  # embed + norms collected, not first-only
        # plan-time validation: no partial write happened
        np.testing.assert_array_equal(params["tok_emb.weight"].data, tok_before)


# ── infer_arch_from_state_dict: config-first ──────────────────────────


class TestInferArchConfigFirst:
    def test_config_overrides_shapes(self):
        sd = {"tok_emb.weight": np.zeros((100, 8), dtype=np.float32)}
        arch = infer_arch_from_state_dict(
            sd, config={"hidden_size": 64, "vocab_size": 5000, "num_hidden_layers": 4}
        )
        assert arch["n_embed"] == 64
        assert arch["vocab_size"] == 5000
        assert arch["n_layer"] == 4

    def test_shapes_without_config_keep_historical_behavior(self):
        sd = {"tok_emb.weight": np.zeros((100, 8), dtype=np.float32)}
        arch = infer_arch_from_state_dict(sd)
        assert arch["n_embed"] == 8
        assert arch["vocab_size"] == 100

    def test_empty_everything_returns_defaults(self):
        arch = infer_arch_from_state_dict({})
        assert arch["n_embed"] == 128
        assert arch["intermediate_size"] == 512
