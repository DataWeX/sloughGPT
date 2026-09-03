"""Tests for checkpoint utilities — normalize, extract, resolve hyperparams."""
from __future__ import annotations

import numpy as np

from domains.training.checkpoint_utils import (
    KEY_MODEL_LEGACY,
    KEY_MODEL_STATE,
    KEY_TRAINING_INFO,
    extract_state_dict,
    normalize_raw_checkpoint,
    resolve_sloughgpt_hyperparams,
    tokenizer_maps_from_bundle,
)


class TestNormalizeRawCheckpoint:
    def test_passthrough_bundled(self):
        bundle = {KEY_MODEL_STATE: {"w": np.array([1.0])}, KEY_TRAINING_INFO: {}}
        result = normalize_raw_checkpoint(bundle)
        assert result is bundle

    def test_passthrough_legacy(self):
        bundle = {KEY_MODEL_LEGACY: {"w": np.array([1.0])}}
        result = normalize_raw_checkpoint(bundle)
        assert result is bundle

    def test_wraps_flat_dict(self):
        flat = {"layer.weight": np.array([1.0]), "layer.bias": np.array([0.0])}
        result = normalize_raw_checkpoint(flat)
        assert KEY_MODEL_STATE in result
        assert result[KEY_MODEL_STATE] is flat

    def test_wraps_tok_emb(self):
        flat = {"tok_emb": np.array([1.0])}
        result = normalize_raw_checkpoint(flat)
        assert KEY_MODEL_STATE in result

    def test_passthrough_unknown(self):
        raw = {"unknown_key": 42}
        result = normalize_raw_checkpoint(raw)
        assert result is raw


class TestExtractStateDict:
    def test_from_model_state(self):
        state = {"w": np.array([1.0, 2.0])}
        bundle = {KEY_MODEL_STATE: state}
        result = extract_state_dict(bundle)
        assert "w" in result
        np.testing.assert_array_equal(result["w"], state["w"])

    def test_from_legacy(self):
        state = {"w": np.array([3.0])}
        bundle = {KEY_MODEL_LEGACY: state}
        result = extract_state_dict(bundle)
        assert "w" in result

    def test_from_raw_dict(self):
        state = {"w": np.array([1.0])}
        result = extract_state_dict(state)
        assert "w" in result

    def test_invalid_raises(self):
        try:
            extract_state_dict({KEY_MODEL_STATE: "not a dict"})
            assert False, "should raise"
        except ValueError:
            pass


class TestResolveSloughgptHyperparams:
    def test_uses_fallbacks(self):
        hp = resolve_sloughgpt_hyperparams({}, fallback_vocab_size=100, fallback_n_embed=64, fallback_n_layer=4, fallback_n_head=8, fallback_block_size=32)
        assert hp["vocab_size"] == 100
        assert hp["n_embed"] == 64
        assert hp["n_layer"] == 4

    def test_bundle_overrides(self):
        bundle = {KEY_TRAINING_INFO: {"vocab_size": 200, "n_embed": 128}}
        hp = resolve_sloughgpt_hyperparams(bundle, fallback_vocab_size=100, fallback_n_embed=64, fallback_n_layer=4, fallback_n_head=8, fallback_block_size=32)
        assert hp["vocab_size"] == 200
        assert hp["n_embed"] == 128

    def test_chars_sets_vocab_size(self):
        bundle = {"chars": list("abcde")}
        hp = resolve_sloughgpt_hyperparams(bundle, fallback_vocab_size=100, fallback_n_embed=64, fallback_n_layer=4, fallback_n_head=8, fallback_block_size=32)
        assert hp["vocab_size"] == 5

    def test_config_dict_merges(self):
        bundle = {"config": {"n_layer": 8}, KEY_TRAINING_INFO: {"n_layer": 16}}
        hp = resolve_sloughgpt_hyperparams(bundle, fallback_vocab_size=100, fallback_n_embed=64, fallback_n_layer=4, fallback_n_head=8, fallback_block_size=32)
        assert hp["n_layer"] == 16  # training_info wins


class TestTokenizerMapsFromBundle:
    def test_returns_stoi_itos(self):
        bundle = {"stoi": {"a": 0}, "itos": {0: "a"}}
        stoi, itos = tokenizer_maps_from_bundle(bundle)
        assert stoi == {"a": 0}
        assert itos == {0: "a"}

    def test_none_when_missing(self):
        stoi, itos = tokenizer_maps_from_bundle({})
        assert stoi is None
        assert itos is None
