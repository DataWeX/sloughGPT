"""Tests for domains/training/checkpoint_utils.py — shared checkpoint helpers.

Exercises bundle normalization, state-dict extraction, hyperparameter
resolution and real SloughGPTModel load round-trips against the numpy
SloNet/torch-shim stack (no real PyTorch, no weights, no network).
"""

import sys
import subprocess
import textwrap

import numpy as np
import pytest

import domains.training.checkpoint_utils as cu
from domains.models import SloughGPTModel


def _make_model(**overrides):
    kw = dict(vocab_size=64, n_embed=32, n_layer=2, n_head=4, block_size=8, dropout=0.1)
    kw.update(overrides)
    return SloughGPTModel(**kw)


def _state_dict(model):
    return {k: np.asarray(v) for k, v in model.state_dict().items()}


def _hp(vocab=64, embed=32, layer=2, head=4, block=8, dropout=0.1):
    return {
        "vocab_size": vocab,
        "n_embed": embed,
        "n_layer": layer,
        "n_head": head,
        "block_size": block,
        "dropout": dropout,
    }


class TestNormalizeRawCheckpoint:
    def test_bundled_passthrough_model_state_dict(self):
        raw = {"model_state_dict": {"a": np.zeros(2)}, "training_info": {}}
        assert cu.normalize_raw_checkpoint(raw) is raw

    def test_bundled_passthrough_legacy_model(self):
        raw = {"model": {"a": np.zeros(2)}}
        assert cu.normalize_raw_checkpoint(raw) is raw

    def test_flat_state_dict_wrapped(self):
        raw = {"tok_emb.weight": np.zeros((3, 2)), "blocks.0.attn_norm.weight": np.zeros(2)}
        out = cu.normalize_raw_checkpoint(raw)
        assert out["model_state_dict"] is raw
        assert out["training_info"] == {}

    def test_bias_only_dict_not_wrapped(self):
        raw = {"anything.else": np.zeros(2)}
        assert cu.normalize_raw_checkpoint(raw) is raw

    def test_arbitrary_dict_passthrough(self):
        raw = {"foo": "bar"}
        assert cu.normalize_raw_checkpoint(raw) is raw


class TestExtractStateDict:
    def test_from_legacy_model_key(self):
        state = {"a": np.zeros(2)}
        out = cu.extract_state_dict({"model": state})
        np.testing.assert_array_equal(out["a"], state["a"])

    def test_from_model_state_dict(self):
        state = {"a": np.zeros(2)}
        out = cu.extract_state_dict({"model_state_dict": state})
        np.testing.assert_array_equal(out["a"], state["a"])

    def test_from_flat_bundle(self):
        state = {"a": np.zeros(2)}
        out = cu.extract_state_dict(state)
        np.testing.assert_array_equal(out["a"], state["a"])

    def test_non_dict_state_raises(self):
        with pytest.raises(ValueError):
            cu.extract_state_dict({"model_state_dict": "not-a-dict"})

    def test_returns_numpy_copy(self):
        arr = np.arange(4, dtype=np.float64)
        out = cu.extract_state_dict({"model_state_dict": {"w": arr}})
        assert isinstance(out["w"], np.ndarray)
        out["w"][0] = 999
        assert arr[0] == 0


class TestToNumpyDict:
    def test_tensor_like_converted(self):
        class _T:
            def cpu(self):
                return self

            def numpy(self):
                return np.arange(3)

        out = cu._to_numpy_dict({"w": _T()})
        np.testing.assert_array_equal(out["w"], np.arange(3))

    def test_float32_passthrough_same_object(self):
        arr = np.arange(3, dtype=np.float32)
        assert cu._to_numpy_dict({"w": arr})["w"] is arr

    def test_other_dtype_copied(self):
        arr = np.arange(3, dtype=np.int32)
        out = cu._to_numpy_dict({"w": arr})["w"]
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, arr)

    def test_nested_dict_recursed(self):
        out = cu._to_numpy_dict({"outer": {"inner": np.zeros(2)}})
        np.testing.assert_array_equal(out["outer"]["inner"], np.zeros(2))

    def test_scalar_passthrough(self):
        assert cu._to_numpy_dict({"n": 5})["n"] == 5


class TestResolveHyperparams:
    def test_training_info_overrides_top_level(self):
        bundle = {
            "training_info": {"n_embed": 96, "n_layer": 5},
            "n_embed": 16,
        }
        out = cu.resolve_sloughgpt_hyperparams(bundle, fallback_vocab_size=256, fallback_n_embed=32,
                                              fallback_n_layer=2, fallback_n_head=4, fallback_block_size=8)
        assert out["n_embed"] == 96
        assert out["n_layer"] == 5

    def test_config_dict_merged_below_info(self):
        bundle = {"config": {"n_head": 3}, "training_info": {"n_embed": 64}}
        out = cu.resolve_sloughgpt_hyperparams(bundle, fallback_vocab_size=256, fallback_n_embed=32,
                                              fallback_n_layer=2, fallback_n_head=4, fallback_block_size=8)
        assert out["n_head"] == 3
        assert out["n_embed"] == 64

    def test_chars_drives_vocab_size(self):
        bundle = {"chars": list("abc")}
        out = cu.resolve_sloughgpt_hyperparams(bundle, fallback_vocab_size=256, fallback_n_embed=32,
                                              fallback_n_layer=2, fallback_n_head=4, fallback_block_size=8)
        assert out["vocab_size"] == 3

    def test_fallbacks_used_when_missing(self):
        out = cu.resolve_sloughgpt_hyperparams({}, fallback_vocab_size=256, fallback_n_embed=32,
                                               fallback_n_layer=2, fallback_n_head=4, fallback_block_size=8,
                                               fallback_dropout=0.3)
        assert out == {"vocab_size": 256, "n_embed": 32, "n_layer": 2, "n_head": 4,
                       "block_size": 8, "dropout": 0.3}

    def test_invalid_vocab_falls_back(self):
        bundle = {"training_info": {"vocab_size": -5}}
        out = cu.resolve_sloughgpt_hyperparams(bundle, fallback_vocab_size=256, fallback_n_embed=32,
                                              fallback_n_layer=2, fallback_n_head=4, fallback_block_size=8)
        assert out["vocab_size"] == 256


class TestTokenizerMapsFromBundle:
    def test_with_maps(self):
        bundle = {"stoi": {"a": 0}, "itos": {0: "a"}}
        assert cu.tokenizer_maps_from_bundle(bundle) == ({"a": 0}, {0: "a"})

    def test_without_maps(self):
        assert cu.tokenizer_maps_from_bundle({"model_state_dict": {}}) == (None, None)


class TestTorchLoadCheckpoint:
    def test_pt_loader_path(self, monkeypatch, tmp_path):
        import domains.infrastructure.pt_loader as pt

        class _T:
            def cpu(self):
                return self

            def numpy(self):
                return np.zeros(3)

        def _fake_load(path, map_location="cpu"):
            return {"w": _T()}

        monkeypatch.setattr(pt, "load_pt_file", _fake_load)
        p = tmp_path / "ckpt.pt"
        p.write_bytes(b"\0")
        out = cu.torch_load_checkpoint(str(p))
        np.testing.assert_array_equal(out["w"], np.zeros(3))

    def test_torch_load_via_pt_loader(self, tmp_path):
        import numpy as np
        from domains.infrastructure.pt_loader import load_pt_file

        p = tmp_path / "ckpt.pt"
        p.write_bytes(b"\0")
        with pytest.raises(Exception):
            load_pt_file(str(p))

    def test_non_dict_result_raises(self, monkeypatch, tmp_path):
        import domains.infrastructure.pt_loader as pt
        monkeypatch.setattr(pt, "load_pt_file", lambda *a, **k: np.zeros(2))
        p = tmp_path / "ckpt.pt"
        p.write_bytes(b"\0")
        with pytest.raises(TypeError):
            cu.torch_load_checkpoint(str(p))


class TestLoadSloughgptFromCheckpoint:
    def test_roundtrip_preserves_weights(self):
        src = _make_model()
        sd = _state_dict(src)
        bundle = {"model_state_dict": sd, "training_info": _hp()}
        model, hp = cu.load_sloughgpt_from_checkpoint(bundle, device="cpu")
        assert hp == {"vocab_size": 64, "n_embed": 32, "n_layer": 2, "n_head": 4,
                      "block_size": 8, "dropout": 0.1}
        x = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        y = np.array([[2, 3, 4, 5, 6, 7, 8, 0]])
        src.eval()
        model.eval()
        o_src, _ = src(x, y)
        o_dst, _ = model(x, y)
        np.testing.assert_allclose(o_src.numpy(), o_dst.numpy(), atol=1e-6)

    def test_hyperparams_from_bundle_top_level(self):
        src = _make_model()
        bundle = {"model_state_dict": _state_dict(src), **{k: v for k, v in _hp().items()}}
        model, hp = cu.load_sloughgpt_from_checkpoint(bundle, device="cpu")
        assert hp["n_embed"] == 32
        assert hp["block_size"] == 8
        assert hp["dropout"] == 0.1

    def test_runtime_error_when_model_unavailable(self, monkeypatch):
        monkeypatch.setattr(cu, "SloughGPTModel", None)
        with pytest.raises(RuntimeError):
            cu.load_sloughgpt_from_checkpoint({"model_state_dict": {}}, device="cpu")

    def test_numpy_bundle_accepted(self):
        src = _make_model()
        sd = _state_dict(src)
        bundle = {"model_state_dict": {k: np.asarray(v) for k, v in sd.items()},
                  "training_info": _hp()}
        model, _ = cu.load_sloughgpt_from_checkpoint(bundle, device="cpu")
        assert model is not None


def test_import_fallback_when_domains_models_missing():
    """Reload the module with domains.models blocked so the module-level
    ImportError fallback runs (SloughGPTModel is None, load raises RuntimeError)."""
    import importlib
    import types as _types
    real_models = sys.modules.get("domains.models")
    try:
        sys.modules["domains.models"] = _types.ModuleType("domains.models")
        importlib.reload(cu)
        assert cu.SloughGPTModel is None
        with pytest.raises(RuntimeError):
            cu.load_sloughgpt_from_checkpoint({"model_state_dict": {}}, device="cpu")
    finally:
        if real_models is not None:
            sys.modules["domains.models"] = real_models
        else:
            sys.modules.pop("domains.models", None)
        importlib.reload(cu)
