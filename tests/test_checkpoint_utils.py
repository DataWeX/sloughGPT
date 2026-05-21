"""Tests for domains.training.checkpoint_utils."""

import torch

from domains.models import SloughGPTModel
from domains.training.checkpoint_utils import (
    KEY_MODEL_STATE,
    extract_state_dict,
    load_sloughgpt_from_checkpoint,
    normalize_raw_checkpoint,
    resolve_sloughgpt_hyperparams,
    tokenizer_maps_from_bundle,
)


def test_normalize_wraps_flat_state_dict():
    flat = {"tok_emb.weight": torch.zeros(3, 8)}
    out = normalize_raw_checkpoint(flat)
    assert KEY_MODEL_STATE in out
    assert out[KEY_MODEL_STATE] is flat


def test_extract_and_resolve_roundtrip():
    m = SloughGPTModel(vocab_size=12, n_embed=16, n_layer=1, n_head=2, block_size=8)
    bundle = {
        "model_state_dict": m.state_dict(),
        "training_info": {
            "vocab_size": 12,
            "n_embed": 16,
            "n_layer": 1,
            "n_head": 2,
            "block_size": 8,
        },
    }
    sd = extract_state_dict(bundle)
    assert "tok_emb.weight" in sd
    hp = resolve_sloughgpt_hyperparams(
        bundle,
        fallback_vocab_size=99,
        fallback_n_embed=99,
        fallback_n_layer=99,
        fallback_n_head=99,
        fallback_block_size=99,
    )
    assert hp["vocab_size"] == 12
    assert hp["n_embed"] == 16


def test_load_sloughgpt_from_checkpoint_cpu():
    m0 = SloughGPTModel(vocab_size=10, n_embed=24, n_layer=1, n_head=2, block_size=6)
    bundle = normalize_raw_checkpoint(
        {
            "model_state_dict": m0.state_dict(),
            "training_info": {
                "vocab_size": 10,
                "n_embed": 24,
                "n_layer": 1,
                "n_head": 2,
                "block_size": 6,
            },
            "stoi": {"a": 0},
            "itos": {0: "a"},
        }
    )
    m1, hp = load_sloughgpt_from_checkpoint(bundle, device="cpu")
    assert m1.vocab_size == hp["vocab_size"] == 10
    assert tokenizer_maps_from_bundle(bundle)[0] == {"a": 0}


def test_vocab_nonpositive_uses_fallback():
    bundle = {"training_info": {"vocab_size": 0}, "model_state_dict": {}}
    hp = resolve_sloughgpt_hyperparams(
        bundle,
        fallback_vocab_size=7,
        fallback_n_embed=8,
        fallback_n_layer=1,
        fallback_n_head=1,
        fallback_block_size=4,
    )
    assert hp["vocab_size"] == 7
