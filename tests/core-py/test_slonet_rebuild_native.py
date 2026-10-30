"""Native state_dict family: rebuild + forward must consume what SloTransformer emits.

SloNet._rebuild_from_state_dict / _forward_state_dict historically understood
only the LEGACY key family (``blocks.{i}.norm1`` / ``blocks.{i}.mlp.w1`` — as
pinned by the hand-built dicts in test_slonet_wave_a and test_slonet_legacy).
Fed the CURRENT native family that ``SloTransformer.state_dict()`` actually
produces (``blocks.{i}.attn_norm`` / ``blocks.{i}.ff.w1``), they mis-read
structure (hidden fell back to 384, ff to hidden*4, heads to a shape guess)
and the forward pass crashed with ``float * None`` because the norm lookup
could not see ``attn_norm``.

These tests pin the composition contract: a state_dict from a real
SloTransformer rebuilds and serves — dims equal to the source model, metadata
preferred over shape guesses (GQA head counts are underdetermined from
square-matrix shapes), and the layer-norm spy always receives real weights.
The three legacy key families stay green via their existing suites.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

import domain.training._internal.slonet as slonet
from domain.training._internal.slonet import (
    SloNet,
    SloTransformer,
    Tensor,
    _SoulTransformerBlockSoulLib,
)


def _native_source() -> tuple[SloTransformer, dict[str, np.ndarray]]:
    """A real SloTransformer whose dims differ from every legacy fallback."""
    model = SloTransformer(
        vocab_size=64,
        n_embed=64,  # fallback default is 384 — must not appear
        n_layer=2,
        n_head=4,  # shape-guess would say 6 (or 1); metadata says 4
        n_kv_head=2,
        block_size=32,
        max_seq_len=64,
        dropout=0,
        _lazy=True,
    )
    return model, model.state_dict()


def _blocks(net: SloNet) -> list[_SoulTransformerBlockSoulLib]:
    return [layer for layer in net.layers if isinstance(layer, _SoulTransformerBlockSoulLib)]


class TestRebuildNativeDims:
    def test_rebuild_dims_match_source_model(self):
        """Structure probes must see the native key family, not fall back."""
        model, sd = _native_source()
        ff_dim = sd["blocks.0.ff.w1.weight"].shape[0]  # truth from the weights

        net = SloNet(soul_name="native-rebuild", metadata={})
        net._rebuild_from_state_dict(sd)
        blocks = _blocks(net)

        assert len(blocks) == model.n_layer == 2
        assert {b.hidden for b in blocks} == {model.n_embed} == {64}
        assert {b.ff_dim for b in blocks} == {ff_dim} == {192}

    def test_metadata_is_preferred_over_shape_guesses(self):
        """n_head is underdetermined from q/k shapes under GQA — ask metadata."""
        _, sd = _native_source()
        net = SloNet(
            soul_name="native-rebuild",
            metadata={
                "n_embed": 64,
                "n_head": 4,
                "n_kv_head": 2,
                "n_layer": 2,
                "intermediate_size": 192,
            },
        )
        net._rebuild_from_state_dict(sd)
        blocks = _blocks(net)

        assert {b.n_heads for b in blocks} == {4}
        assert {b.hidden for b in blocks} == {64}
        assert {b.ff_dim for b in blocks} == {192}

    def test_tok_emb_is_the_hidden_source_without_metadata(self):
        """tok_emb is (vocab, hidden) by definition — never a 384 fallback."""
        _, sd = _native_source()
        net = SloNet(soul_name="native-rebuild", metadata={})
        net._rebuild_from_state_dict(sd)

        assert net._get_weight("tok_emb.weight").shape[1] == 64
        assert {b.hidden for b in _blocks(net)} == {64}


class TestForwardNativeStateDict:
    def test_forward_serves_native_keys(self):
        """The native family must forward — it used to TypeError at the norm."""
        _, sd = _native_source()
        net = SloNet(soul_name="native-forward", metadata={})
        net._rebuild_from_state_dict(sd)

        out = net.forward(Tensor(np.array([[1, 2]], dtype=np.int64)))

        assert out.data.shape == (1, 2, 64)  # (batch, seq, vocab)
        assert np.isfinite(out.data).all()

    def test_forward_serves_any_sequence_length_under_gqa(self):
        """GQA k/v carry fewer features than q — odd N used to ValueError.

        The head heuristic reshaped k/v as (B, -1, H, E) with q's head count,
        folding the sequence axis (even N: garbled attention; odd N: crash).
        """
        _, sd = _native_source()
        net = SloNet(soul_name="native-forward", metadata={})
        net._rebuild_from_state_dict(sd)

        for n in (1, 2, 3, 4, 7):
            out = net.forward(Tensor(np.arange(n, dtype=np.int64).reshape(1, n)))
            assert out.data.shape == (1, n, 64), f"seq len {n}"
            assert np.isfinite(out.data).all(), f"seq len {n}"

    def test_layernorm_receives_real_never_none_weights(self):
        """White-box: the norm spy must see attn_norm/ff_norm, not None."""
        _, sd = _native_source()
        net = SloNet(soul_name="native-forward", metadata={})
        net._rebuild_from_state_dict(sd)

        received: list[np.ndarray | None] = []
        real = slonet._layernorm_state_dict

        def spy(x, weight, eps=1e-5):
            received.append(weight)
            return real(x, weight, eps)

        with patch.object(slonet, "_layernorm_state_dict", side_effect=spy):
            net.forward(Tensor(np.array([[3, 4]], dtype=np.int64)))

        # Per block: attn_norm then ff_norm (2 blocks) + final norm.weight.
        assert len(received) == 5
        assert all(w is not None for w in received)
        np.testing.assert_array_equal(received[0], sd["blocks.0.attn_norm.weight"])
        np.testing.assert_array_equal(received[1], sd["blocks.0.ff_norm.weight"])
        np.testing.assert_array_equal(received[-1], sd["norm.weight"])

    def test_layernorm_none_returns_input_unchanged(self):
        """Defense: a missing norm must degrade to identity, not TypeError."""
        x = Tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))

        out = slonet._layernorm_state_dict(x, None)

        np.testing.assert_array_equal(out.data, x.data)
