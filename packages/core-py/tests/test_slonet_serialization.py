"""Tests for training.slonet — serialization (save/load checkpoint, state_dict)."""

from __future__ import annotations

import os
import json
import tempfile
import pytest
import numpy as np
from domains.training.slonet import (
    Tensor,
    SloNet,
    SloTransformer,
    SloLinear,
    save_checkpoint_npz,
    load_checkpoint_npz,
    _state_dict_to_numpy,
)


# ── _state_dict_to_numpy ────────────────────────────────────────────────────


class TestStateDictToNumpy:

    def test_numpy_array(self):
        sd = {"w": np.array([1.0, 2.0])}
        result = _state_dict_to_numpy(sd)
        assert isinstance(result["w"], np.ndarray)

    def test_tensor(self):
        sd = {"w": Tensor(np.array([1.0, 2.0]))}
        result = _state_dict_to_numpy(sd)
        assert isinstance(result["w"], np.ndarray)

    def test_nested_dict(self):
        sd = {"a": {"b": np.array([1.0])}}
        result = _state_dict_to_numpy(sd)
        assert isinstance(result["a"]["b"], np.ndarray)

    def test_list(self):
        sd = {"w": [1.0, 2.0]}
        result = _state_dict_to_numpy(sd)
        assert isinstance(result["w"], np.ndarray)


# ── save_checkpoint_npz / load_checkpoint_npz ────────────────────────────────


class TestCheckpointNPZ:

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npz")
            sd = {"layer.weight": np.array([1.0, 2.0, 3.0])}
            meta = {"soul_name": "Test"}
            save_checkpoint_npz(path, sd, meta)
            assert os.path.exists(path)

            loaded = load_checkpoint_npz(path)
            assert loaded["soul_name"] == "Test"
            assert "model_state_dict" in loaded
            assert np.allclose(loaded["model_state_dict"]["layer.weight"], [1.0, 2.0, 3.0])

    def test_save_adds_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test")
            sd = {"w": np.array([1.0])}
            save_checkpoint_npz(path, sd)
            assert os.path.exists(path + ".npz")

    def test_load_missing_meta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npz")
            # Save without _meta_json
            np.savez_compressed(path, w=np.array([1.0]))
            with pytest.raises(ValueError, match="missing"):
                load_checkpoint_npz(path)

    def test_meta_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npz")
            sd = {"w": np.array([1.0])}
            meta = {"version": 1, "config": {"lr": 0.001}}
            save_checkpoint_npz(path, sd, meta)
            loaded = load_checkpoint_npz(path)
            assert loaded["version"] == 1
            assert loaded["config"]["lr"] == 0.001


# ── SloNet.state_dict ──────────────────────────────────────────────────────


class TestSloNetStateDict:

    def test_state_dict(self):
        net = SloNet(layers=[SloLinear(10, 5)])
        sd = net.state_dict()
        assert "p0" in sd
        assert "p1" in sd

    def test_state_dict_values(self):
        net = SloNet(layers=[SloLinear(10, 5)])
        sd = net.state_dict()
        for k, v in sd.items():
            assert isinstance(v, np.ndarray)


# ── SloTransformer.state_dict ──────────────────────────────────────────────


class TestSloTransformerStateDict:

    def test_state_dict(self):
        model = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2)
        sd = model.state_dict()
        assert len(sd) > 0

    def test_roundtrip(self):
        model = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2)
        sd = model.state_dict()
        model2 = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2)
        model2.load_state_dict(sd, strict=False)
        # Weights should be similar
        for k in sd:
            if k in model2.state_dict():
                assert np.allclose(model.state_dict()[k], model2.state_dict()[k], atol=1e-5)


# ── Export/Import roundtrip ─────────────────────────────────────────────────


class TestExportImportRoundtrip:

    def test_sou_roundtrip(self):
        from domains.training.slonet import export_to_sou, import_from_sou
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.soul")
            model = SloTransformer(
                vocab_size=100, n_embed=32, n_layer=1, n_head=2,
                soul_name="RoundtripTest",
            )
            export_to_sou(model, path, include_weights=True)
            loaded = import_from_sou(path)
            assert loaded.soul_name == "RoundtripTest"
            assert isinstance(loaded, SloTransformer)
