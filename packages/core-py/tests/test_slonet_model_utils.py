"""Tests for training.slonet — model utility functions."""

from __future__ import annotations

import os
import tempfile
import pytest
import numpy as np
from domains.training.slonet import (
    Tensor,
    SloNet,
    SloTransformer,
    SloLinear,
    no_grad,
    compute_sensitivity,
    souls_from_directory,
    _rotate_half,
    _apply_rope,
)


# ── _rotate_half ────────────────────────────────────────────────────────────


class TestRotateHalf:

    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = _rotate_half(x)
        assert np.allclose(result, [-3.0, -4.0, 1.0, 2.0])

    def test_odd_dim(self):
        x = np.array([1.0, 2.0, 3.0])
        result = _rotate_half(x)
        assert result.shape == x.shape

    def test_2d(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _rotate_half(x)
        assert result.shape == x.shape


# ── _apply_rope ─────────────────────────────────────────────────────────────


class TestApplyRope:

    def test_basic(self):
        q = np.ones((2, 4, 8))
        k = np.ones((2, 4, 8))
        cos = np.ones((2, 4, 1))
        sin = np.ones((2, 4, 1))
        q_out, k_out = _apply_rope(q, k, cos, sin)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape


# ── compute_sensitivity ─────────────────────────────────────────────────────


class TestComputeSensitivity:

    def test_no_grad_params(self):
        model = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2)
        x = Tensor(np.array([[1, 2, 3]]))
        with no_grad():
            logits, _ = model(x)
        result = compute_sensitivity(logits, {"params": []})
        assert result == {}

    def test_empty_param_groups(self):
        model = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2)
        x = Tensor(np.array([[1, 2, 3]]))
        logits, _ = model(x)
        result = compute_sensitivity(logits, {})
        assert result == {}


# ── souls_from_directory ────────────────────────────────────────────────────


class TestSoulsFromDirectory:

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = souls_from_directory(tmpdir)
            assert result == []

    def test_nonexistent_dir(self):
        result = souls_from_directory("/nonexistent/path/12345")
        assert result == []

    def test_with_sou_files(self):
        from domains.training.slonet import export_to_sou
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SloTransformer(
                vocab_size=100, n_embed=32, n_layer=1, n_head=2,
                soul_name="TestSou",
            )
            export_to_sou(model, os.path.join(tmpdir, "test.soul"))
            result = souls_from_directory(tmpdir)
            assert len(result) == 1
            assert result[0].soul_name == "TestSou"


# ── SloNet.load_state_dict ─────────────────────────────────────────────────


class TestSloNetLoadStateDict:

    def test_state_dict_roundtrip(self):
        net = SloNet(layers=[SloLinear(10, 5)])
        sd = net.state_dict()
        assert "p0" in sd
        assert "p1" in sd
        assert isinstance(sd["p0"], np.ndarray)


# ── SloTransformer.load_state_dict ─────────────────────────────────────────


class TestSloTransformerLoadStateDict:

    def test_load_state_dict(self):
        model = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2)
        sd = model.state_dict()
        model2 = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2)
        model2.load_state_dict(sd, strict=False)
        assert np.allclose(model.state_dict()["tok_emb.weight"], model2.state_dict()["tok_emb.weight"])
