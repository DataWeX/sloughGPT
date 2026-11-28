"""LoRA merge at the checkpoint boundary.

A fine-tune is saved as separable adapters beside the frozen base
(``<prefix>.lora_A`` / ``.lora_B``). Nothing in the read paths rebuilt LoRA
modules, so both the .slnc compiler and ``from_soul`` silently served the
base model — the trained delta vanished while every gate stayed green.
These tests pin the merge that closes that hole, including the name mapping
(attribute path ``W_q`` → canonical state-dict ``q_proj``).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from domain.training._internal.lora import merge_lora_state_dict


def _adapter(prefix: str, *, out_f: int = 6, in_f: int = 4, rank: int = 2, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        f"{prefix}.weight": rng.normal(size=(out_f, in_f)).astype(np.float32),
        f"{prefix}.lora_A": rng.normal(size=(rank, in_f)).astype(np.float32),
        f"{prefix}.lora_B": rng.normal(size=(out_f, rank)).astype(np.float32),
    }


class TestMergeMath:
    def test_folds_delta_into_base(self):
        sd = _adapter("blocks.0.attn.q_proj")
        base = sd["blocks.0.attn.q_proj.weight"].copy()
        a = sd["blocks.0.attn.q_proj.lora_A"]
        b = sd["blocks.0.attn.q_proj.lora_B"]

        merged, info = merge_lora_state_dict(sd, rank=2, alpha=8.0)

        expected = base + (b @ a) * (8.0 / 2)
        np.testing.assert_allclose(
            merged["blocks.0.attn.q_proj.weight"], expected, rtol=0, atol=1e-6
        )
        assert info["merged"] == 1
        assert info["scale"] == 4.0
        assert "blocks.0.attn.q_proj.lora_A" not in merged
        assert "blocks.0.attn.q_proj.lora_B" not in merged

    def test_attribute_path_maps_to_canonical_projection(self):
        # The trainer names adapters by attribute (attn.W_q); base weights are
        # saved canonically (attn.q_proj). Merging verbatim would miss the base.
        sd = _adapter("blocks.0.attn.W_q")
        base = sd["blocks.0.attn.W_q.weight"].copy()
        a = sd["blocks.0.attn.W_q.lora_A"]
        b = sd["blocks.0.attn.W_q.lora_B"]

        merged, _ = merge_lora_state_dict(sd, rank=2, alpha=2.0)

        np.testing.assert_allclose(
            merged["blocks.0.attn.W_q.weight"], base + (b @ a) * 1.0, rtol=0, atol=1e-6
        )

    def test_no_adapters_is_a_no_op(self):
        sd = {"tok_emb.weight": np.ones((4, 4), dtype=np.float32)}
        merged, info = merge_lora_state_dict(sd)
        assert info["merged"] == 0
        np.testing.assert_array_equal(merged["tok_emb.weight"], sd["tok_emb.weight"])

    def test_ia3_scales_rows(self):
        sd = {
            "blocks.0.ff.w1.weight": np.ones((3, 2), dtype=np.float32),
            "blocks.0.ff.w1.lora_s": np.array([2.0, 3.0, 4.0], dtype=np.float32),
        }
        merged, info = merge_lora_state_dict(sd)
        np.testing.assert_allclose(merged["blocks.0.ff.w1.weight"], [[2, 2], [3, 3], [4, 4]])
        assert info["ia3"] == 1
        assert "blocks.0.ff.w1.lora_s" not in merged

    def test_rank_inferred_from_a_when_omitted(self):
        sd = _adapter("blocks.0.attn.W_k", rank=3)
        _, info = merge_lora_state_dict(sd, alpha=6.0)
        assert info["rank"] == 3
        assert info["scale"] == 2.0


class TestMergeRefusesPartialWork:
    def test_unpaired_adapter_raises(self):
        sd = _adapter("blocks.0.attn.W_v")
        del sd["blocks.0.attn.W_v.lora_B"]
        with pytest.raises(ValueError, match="unpaired"):
            merge_lora_state_dict(sd, rank=2, alpha=2.0)

    def test_declared_rank_contradiction_raises(self):
        sd = _adapter("blocks.0.attn.W_o", rank=2)
        with pytest.raises(ValueError, match="contradicts"):
            merge_lora_state_dict(sd, rank=4, alpha=2.0)

    def test_missing_base_weight_raises(self):
        sd = _adapter("blocks.0.attn.W_q")
        del sd["blocks.0.attn.W_q.weight"]
        with pytest.raises(ValueError, match="no base weight"):
            merge_lora_state_dict(sd, rank=2, alpha=2.0)

    def test_shape_mismatch_raises(self):
        sd = _adapter("blocks.0.attn.W_q", out_f=6, in_f=4, rank=2)
        sd["blocks.0.attn.W_q.lora_B"] = np.zeros((5, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="delta"):
            merge_lora_state_dict(sd, rank=2, alpha=2.0)


class TestCompilerDoesNotDropSilently:
    def test_unknown_tensor_is_named_in_a_warning(self, tmp_path, caplog):
        from domain.infrastructure._internal.slnc.compiler import SLNCCompiler

        weights = {
            "tok_emb.weight": np.zeros((8, 8), dtype=np.float32),
            "mystery.weight": np.zeros((2, 2), dtype=np.float32),
        }
        config = {
            "n_layer": 1,
            "n_embd": 8,
            "n_head": 2,
            "vocab_size": 8,
            "n_inner": 16,
            "max_position_embeddings": 8,
            "n_positions": 8,
        }
        with caplog.at_level(logging.WARNING, logger="slo.infrastructure.slnc.compiler"):
            SLNCCompiler().compile_from_dict(config, weights, str(tmp_path / "x.slnc"))
        assert "mystery.weight" in caplog.text
        assert "dropped" in caplog.text


class TestSoulToSlncCarriesTheFineTune:
    def _soul_with_adapter(self, tmp_path):
        from domain.inference._internal.slo_format import create_soul_profile, save_soul
        from domain.infrastructure._internal.weight_loader import build_model_from_config
        from domain.training._internal.lora import LoRAConfig, apply_lora_to_model

        cfg = {
            "vocab_size": 16,
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "max_position_embeddings": 16,
            "block_size": 16,
            "intermediate_size": 16,
            "use_rope": True,
            "norm_type": "rms_norm",
        }
        model = build_model_from_config(cfg, _lazy=True)
        model = apply_lora_to_model(
            model,
            LoRAConfig(
                rank=2, alpha=4.0, target_modules=["W_q", "W_k", "W_v", "W_o", "w1", "w2", "w3"]
            ),
            freeze_base=True,
        )
        for name, param in model.named_parameters():
            if name.endswith("lora_B"):
                param.data[:] = 0.05
        soul = create_soul_profile(name="merge-test", base_model="sloughgpt", lineage="sloughgpt")
        # from_soul reads the native key names the trainer writes.
        soul.metadata["config"] = {
            "n_embed": 8,
            "n_layer": 1,
            "n_head": 2,
            "block_size": 16,
            "vocab_size": 16,
            "model_type": "sloughgpt",
            "use_rope": True,
            "norm_type": "rms_norm",
            "max_seq_len": 16,
            "intermediate_size": 16,
        }
        soul.metadata["vocab_size"] = 16
        soul.metadata["stoi"] = {chr(97 + i): i for i in range(16)}
        soul.metadata["itos"] = {i: chr(97 + i) for i in range(16)}
        soul.metadata["lora"] = {"rank": 2, "alpha": 4.0, "merged": False}
        path = tmp_path / "adapter.soul"
        save_soul(model, str(path), soul_profile=soul)
        return model, path

    def test_compiled_weights_are_the_merged_base(self, tmp_path):
        from domain.inference._internal.slo_format import load_soul
        from domain.infrastructure._internal.slnc.parser import SLNCParser
        from domain.infrastructure._internal.soul_to_slnc import soul_to_slnc
        from domain.training._internal.lora import merge_lora_state_dict

        _model, soul_path = self._soul_with_adapter(tmp_path)
        out = tmp_path / "adapter.slnc"
        soul_to_slnc(soul_path, out)

        _soul, sd = load_soul(str(soul_path))
        effective, info = merge_lora_state_dict(sd, rank=2, alpha=4.0)
        assert info["merged"] >= 1

        parser = SLNCParser(str(out))
        names = parser.tensor_names
        assert not [n for n in names if "lora" in n], (
            "adapters must be merged, not written as tensors"
        )
        assert parser.config.get("lora_merged") == info["merged"]
        np.testing.assert_allclose(
            parser.get_tensor("blocks.0.attn.q_proj.weight"),
            effective["blocks.0.attn.q_proj.weight"],
            rtol=0,
            atol=1e-6,
        )

    def test_from_soul_applies_the_adapter(self, tmp_path):
        from domain.inference._internal.slo_format import load_soul
        from domain.inference._internal.slonet_provider import SloNetChatProvider
        from domain.training._internal.lora import merge_lora_state_dict

        model, soul_path = self._soul_with_adapter(tmp_path)
        _soul, sd = load_soul(str(soul_path))
        effective, _ = merge_lora_state_dict(sd, rank=2, alpha=4.0)

        provider = SloNetChatProvider.from_soul(str(soul_path))
        loaded = dict(provider._model._named_parameters())
        np.testing.assert_allclose(
            loaded["blocks.0.attn.q_proj.weight"].data,
            effective["blocks.0.attn.q_proj.weight"],
            rtol=0,
            atol=1e-6,
        )
