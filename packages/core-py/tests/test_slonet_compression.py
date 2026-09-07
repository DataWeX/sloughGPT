"""Tests for training.slonet — compression and weight handling."""

from __future__ import annotations

import pytest
import numpy as np
from domains.training.slonet import (
    Tensor,
    SloLinear,
    SloNet,
    SloTransformer,
    no_grad,
)


# ── SloLinear weight operations ─────────────────────────────────────────────


class TestSloLinearWeightOps:

    def test_get_weight_T(self):
        lin = SloLinear(10, 5)
        wt = lin._get_weight_T()
        assert wt.shape == (10, 5)

    def test_get_weight_T_cached(self):
        lin = SloLinear(10, 5)
        wt1 = lin._get_weight_T()
        wt2 = lin._get_weight_T()
        assert wt1 is wt2

    def test_get_weight_T_contig(self):
        lin = SloLinear(10, 5)
        wt = lin._get_weight_T_contig()
        assert wt.shape == (10, 5)
        assert wt.flags["C_CONTIGUOUS"]

    def test_get_weight_T_contig_cached(self):
        lin = SloLinear(10, 5)
        wt1 = lin._get_weight_T_contig()
        wt2 = lin._get_weight_T_contig()
        assert np.array_equal(wt1, wt2)

    def test_set_quantized_weight_clears_cache(self):
        lin = SloLinear(10, 5)
        lin._get_weight_T_contig()

        class _MockQuantInfo:
            is_quantized = True
            array = None

            def __init__(self):
                self.meta = type("M", (), {"bits": 8, "scale": 1.0, "zero_point": 0, "original_shape": (5, 10)})()

        lin.set_quantized_weight(_MockQuantInfo())
        assert lin._weight_T_contig is None


# ── SloNet parameters ──────────────────────────────────────────────────────


class TestSloNetParameters:

    def test_num_parameters(self):
        net = SloNet(layers=[SloLinear(10, 5)])
        assert net.num_parameters() == 10 * 5 + 5

    def test_num_parameters_multiple_layers(self):
        net = SloNet(layers=[SloLinear(10, 5), SloLinear(5, 3)])
        assert net.num_parameters() == (10 * 5 + 5) + (5 * 3 + 3)

    def test_state_dict_keys(self):
        net = SloNet(layers=[SloLinear(10, 5)])
        sd = net.state_dict()
        assert "p0" in sd
        assert "p1" in sd

    def test_state_dict_values_are_arrays(self):
        net = SloNet(layers=[SloLinear(10, 5)])
        sd = net.state_dict()
        for v in sd.values():
            assert isinstance(v, np.ndarray)


# ── SloTransformer parameters ──────────────────────────────────────────────


class TestSloTransformerParameters:

    def test_num_parameters(self):
        model = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2)
        assert model.num_parameters() > 0

    def test_num_parameters_increases_with_layers(self):
        model1 = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2)
        model2 = SloTransformer(vocab_size=100, n_embed=32, n_layer=2, n_head=2)
        assert model2.num_parameters() > model1.num_parameters()

    def test_state_dict(self):
        model = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2)
        sd = model.state_dict()
        assert len(sd) > 0

    def test_metadata(self):
        model = SloTransformer(
            vocab_size=100, n_embed=32, n_layer=2, n_head=4,
            soul_name="TestModel",
        )
        assert model.metadata["vocab_size"] == 100
        assert model.metadata["n_layer"] == 2
        assert model.metadata["n_head"] == 4


# ── Gradient checkpointing ─────────────────────────────────────────────────


class TestGradientCheckpointing:

    def test_apply_gradient_checkpointing(self):
        model = SloTransformer(vocab_size=100, n_embed=32, n_layer=2, n_head=2)
        model.apply_gradient_checkpointing()
        for block in model.blocks:
            assert block.use_checkpoint is True


# ── Weight tying ────────────────────────────────────────────────────────────


class TestWeightTying:

    def test_tied_weights(self):
        model = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2, tie_weights=True)
        assert np.array_equal(model.tok_emb.weight.data, model.lm_head.weight.data)

    def test_untied_weights(self):
        model = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2, tie_weights=False)
        # Weights may differ
        assert model.lm_head.weight.shape == (100, 32)


# ── Forward/backward integration ───────────────────────────────────────────


class TestForwardBackward:

    def test_forward_no_grad(self):
        model = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2)
        x = Tensor(np.array([[1, 2, 3]]))
        with no_grad():
            logits, loss = model(x)
            assert logits.requires_grad is False

    def test_forward_with_targets(self):
        model = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2)
        x = Tensor(np.array([[1, 2, 3]]))
        y = Tensor(np.array([[2, 3, 4]]))
        logits, loss = model(x, targets=y)
        assert loss.item() > 0

    def test_generate(self):
        model = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2, max_seq_len=50)
        x = np.array([[1, 2, 3]])
        out = model.generate(x, max_new_tokens=5)
        assert out.data.shape[1] == 8
