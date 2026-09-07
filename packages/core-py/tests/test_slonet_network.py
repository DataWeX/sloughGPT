"""Tests for training.slonet — SloNet and SloTransformer."""

from __future__ import annotations

import math
import pytest
import numpy as np
from domains.training.slonet import (
    Tensor,
    SloNet,
    SloTransformer,
    SloLayer,
    SloLinear,
    SloEmbedding,
    SloTransformerBlock,
    SloAdapterLayer,
    no_grad,
)


# ── SloNet ──────────────────────────────────────────────────────────────────


class TestSloNet:

    def test_init(self):
        net = SloNet()
        assert net.soul_name == "Slo"
        assert net.layers == []

    def test_init_with_layers(self):
        layers = [SloLinear(10, 5)]
        net = SloNet(layers=layers, soul_name="TestNet")
        assert net.soul_name == "TestNet"
        assert len(net.layers) == 1

    def test_forward(self):
        layer = SloLinear(10, 5)
        net = SloNet(layers=[layer])
        x = Tensor(np.ones((1, 10)))
        out = net(x)
        assert out.shape == (1, 5)

    def test_forward_numpy(self):
        layer = SloLinear(10, 5)
        net = SloNet(layers=[layer])
        x = np.ones((1, 10))
        out = net(x)
        assert out.shape == (1, 5)

    def test_train(self):
        net = SloNet()
        net.train(True)
        net.train(False)
        net.eval()

    def test_parameters(self):
        layer = SloLinear(10, 5)
        net = SloNet(layers=[layer])
        params = net.parameters()
        assert len(params) == 2

    def test_state_dict(self):
        layer = SloLinear(10, 5)
        net = SloNet(layers=[layer])
        sd = net.state_dict()
        assert "p0" in sd
        assert "p1" in sd

    def test_num_parameters(self):
        layer = SloLinear(10, 5)
        net = SloNet(layers=[layer])
        assert net.num_parameters() == 10 * 5 + 5

    def test_soul_signature(self):
        net = SloNet(soul_name="Test")
        sig = net.soul_signature()
        assert sig["soul_name"] == "Test"

    def test_named_modules(self):
        net = SloNet()
        modules = net.named_modules()
        assert len(modules) == 1

    def test_named_children(self):
        layer = SloLinear(10, 5)
        net = SloNet(layers=[layer])
        children = net.named_children()
        assert len(children) == 1

    def test_user_adapter(self):
        net = SloNet()
        adapter = net.get_user_adapter("user1", dim=64, rank=8)
        assert isinstance(adapter, SloAdapterLayer)
        # Get same adapter again
        adapter2 = net.get_user_adapter("user1", dim=64, rank=8)
        assert adapter is adapter2

    def test_remove_user_adapter(self):
        net = SloNet()
        net.get_user_adapter("user1", dim=64, rank=8)
        net.remove_user_adapter("user1")
        assert "user1" not in net._user_adapters

    def test_set_active_user(self):
        net = SloNet()
        net.set_active_user("user1")
        assert net._active_user_id == "user1"

    def test_fit(self):
        from domains.training.slonet import SloAdam
        net = SloNet(layers=[SloLinear(2, 3)])
        X = Tensor(np.random.randn(20, 2).astype(np.float32))
        y = Tensor(np.random.randint(0, 3, (20,)))
        optimizer = SloAdam(lr=0.01)
        losses = net.fit(X, y, optimizer, epochs=2, batch_size=10)
        assert len(losses) == 2

    def test_gradient_checkpointing(self):
        block = SloTransformerBlock(d_model=64, n_heads=4)
        net = SloNet(layers=[block])
        net.apply_gradient_checkpointing()
        assert block.use_checkpoint is True

    def test_no_grad(self):
        layer = SloLinear(10, 5)
        net = SloNet(layers=[layer])
        x = Tensor(np.ones((1, 10)))
        with no_grad():
            out = net(x)
            assert out.requires_grad is False


# ── SloTransformer ──────────────────────────────────────────────────────────


class TestSloTransformer:

    def test_init(self):
        model = SloTransformer(vocab_size=100, n_embed=64, n_layer=2, n_head=4)
        assert model.vocab_size == 100
        assert model.n_embed == 64
        assert model.n_layer == 2
        assert model.n_head == 4

    def test_forward(self):
        model = SloTransformer(vocab_size=100, n_embed=64, n_layer=2, n_head=4)
        x = Tensor(np.array([[1, 2, 3, 4]]))
        logits, loss = model(x)
        assert logits.shape == (1, 4, 100)
        assert loss is None

    def test_forward_numpy(self):
        model = SloTransformer(vocab_size=100, n_embed=64, n_layer=2, n_head=4)
        x = np.array([[1, 2, 3, 4]])
        logits, loss = model(x)
        assert logits.shape == (1, 4, 100)

    def test_forward_with_targets(self):
        model = SloTransformer(vocab_size=100, n_embed=64, n_layer=2, n_head=4)
        x = Tensor(np.array([[1, 2, 3, 4]]))
        y = Tensor(np.array([[2, 3, 4, 5]]))
        out, loss = model(x, targets=y)
        assert out.shape == (1, 4, 100)
        assert loss.item() > 0

    def test_forward_with_cache(self):
        model = SloTransformer(vocab_size=100, n_embed=64, n_layer=2, n_head=4)
        x = Tensor(np.array([[1, 2, 3, 4]]))
        out1, _ = model(x, use_cache=True)
        assert model._kv_caches[0] is not None

    def test_clear_kv_cache(self):
        model = SloTransformer(vocab_size=100, n_embed=64, n_layer=2, n_head=4)
        x = Tensor(np.array([[1, 2, 3, 4]]))
        model(x, use_cache=True)
        model.clear_kv_cache()
        assert all(c is None for c in model._kv_caches)

    def test_tok_emb(self):
        model = SloTransformer(vocab_size=100, n_embed=64, n_layer=2, n_head=4)
        assert isinstance(model.tok_emb, SloEmbedding)

    def test_blocks(self):
        model = SloTransformer(vocab_size=100, n_embed=64, n_layer=2, n_head=4)
        assert len(model.blocks) == 2

    def test_norm(self):
        model = SloTransformer(vocab_size=100, n_embed=64, n_layer=2, n_head=4)
        assert model.norm is not None

    def test_lm_head(self):
        model = SloTransformer(vocab_size=100, n_embed=64, n_layer=2, n_head=4)
        assert isinstance(model.lm_head, SloLinear)
        assert model.lm_head.out_features == 100

    def test_num_parameters(self):
        model = SloTransformer(vocab_size=100, n_embed=64, n_layer=2, n_head=4)
        assert model.num_parameters() > 0

    def test_metadata(self):
        model = SloTransformer(vocab_size=100, n_embed=64, n_layer=2, n_head=4)
        assert model.metadata["vocab_size"] == 100
        assert model.metadata["n_layer"] == 2

    def test_tie_weights(self):
        model = SloTransformer(vocab_size=100, n_embed=64, n_layer=2, n_head=4, tie_weights=True)
        assert np.array_equal(model.tok_emb.weight.data, model.lm_head.weight.data)

    def test_no_tie_weights(self):
        model = SloTransformer(vocab_size=100, n_embed=64, n_layer=2, n_head=4, tie_weights=False)
        # Weights may differ after init
        assert model.lm_head.weight.shape == (100, 64)

    def test_layer_norm_type(self):
        model = SloTransformer(vocab_size=100, n_embed=64, n_layer=2, n_head=4, norm_type="layer_norm")
        x = Tensor(np.array([[1, 2, 3, 4]]))
        logits, loss = model(x)
        assert logits.shape == (1, 4, 100)

    def test_silu_activation(self):
        model = SloTransformer(vocab_size=100, n_embed=64, n_layer=2, n_head=4, activation="silu")
        x = Tensor(np.array([[1, 2, 3, 4]]))
        logits, loss = model(x)
        assert logits.shape == (1, 4, 100)

    def test_gqa(self):
        model = SloTransformer(vocab_size=100, n_embed=64, n_layer=2, n_head=4, n_kv_head=2)
        x = Tensor(np.array([[1, 2, 3, 4]]))
        logits, loss = model(x)
        assert logits.shape == (1, 4, 100)

    def test_abs_pos_emb(self):
        model = SloTransformer(vocab_size=100, n_embed=64, n_layer=2, n_head=4, use_abs_pos_emb=True)
        assert model.pos_emb is not None
        x = Tensor(np.array([[1, 2, 3, 4]]))
        logits, loss = model(x)
        assert logits.shape == (1, 4, 100)

    def test_no_grad(self):
        model = SloTransformer(vocab_size=100, n_embed=64, n_layer=2, n_head=4)
        x = Tensor(np.array([[1, 2, 3, 4]]))
        with no_grad():
            logits, loss = model(x)
            assert logits.requires_grad is False
