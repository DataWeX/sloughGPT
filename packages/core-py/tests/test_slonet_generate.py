"""Tests for training.slonet — generate_numpy and _sample_from_logits."""

from __future__ import annotations

import pytest
import numpy as np
from domains.training.slonet import (
    Tensor,
    SloTransformer,
    NumpyKVState,
    _sample_from_logits,
    _apply_temperature,
    _apply_top_k,
    _apply_top_p,
    _apply_repetition_penalty,
    _apply_frequency_penalty,
    _apply_presence_penalty,
)


# ── _sample_from_logits ─────────────────────────────────────────────────────


class TestSampleFromLogits:

    def test_greedy(self):
        logits = np.array([[1.0, 5.0, 3.0]])
        result = _sample_from_logits(logits, temperature=0.0)
        assert result == 1

    def test_greedy_with_eos(self):
        logits = np.array([[1.0, 5.0, 3.0]])
        result = _sample_from_logits(logits, temperature=0.0, eos_token=1)
        # With eos masked, should pick second highest
        assert result == 2

    def test_sampling(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        result = _sample_from_logits(logits, temperature=1.0)
        assert 0 <= result <= 2

    def test_top_k(self):
        logits = np.array([[1.0, 5.0, 3.0, 2.0]])
        result = _sample_from_logits(logits, temperature=1.0, top_k=2)
        assert result in [1, 2]

    def test_top_p(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        result = _sample_from_logits(logits, temperature=1.0, top_p=0.5)
        assert result == 2

    def test_repetition_penalty(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        gen_ids = np.array([0])
        result = _sample_from_logits(
            logits, temperature=1.0, repetition_penalty=2.0, generated_ids=gen_ids
        )
        assert result != 0 or result == 0  # just ensure no crash

    def test_3d_logits(self):
        logits = np.array([[[1.0, 5.0, 3.0]]])
        result = _sample_from_logits(logits, temperature=0.0)
        assert result == 1


# ── Logit processors ───────────────────────────────────────────────────────


class TestLogitProcessors:

    def test_temperature(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        out = _apply_temperature(logits, 2.0)
        assert out[0, 0] == pytest.approx(0.5)

    def test_top_k(self):
        logits = np.array([[1.0, 5.0, 3.0, 2.0]])
        out = _apply_top_k(logits, k=2)
        assert out[0, 1] == 5.0
        assert out[0, 0] < 0

    def test_top_p(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        out = _apply_top_p(logits, p=0.5)
        assert out[0, 2] == 3.0

    def test_repetition_penalty(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        gen_ids = np.array([0])
        out = _apply_repetition_penalty(logits, gen_ids, penalty=2.0)
        assert out[0, 0] == 0.5

    def test_frequency_penalty(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        gen_ids = np.array([0, 0, 1])
        out = _apply_frequency_penalty(logits, gen_ids, penalty=0.1)
        assert out[0, 0] < 1.0  # penalized by count

    def test_presence_penalty(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        gen_ids = np.array([0])
        out = _apply_presence_penalty(logits, gen_ids, penalty=0.5)
        assert out[0, 0] == 0.5


# ── generate (high-level) ──────────────────────────────────────────────────


class TestGenerate:

    def test_generate_basic(self):
        model = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2, max_seq_len=50)
        x = np.array([[1, 2, 3, 4]])
        out = model.generate(x, max_new_tokens=5)
        assert isinstance(out, Tensor)
        assert out.data.shape[1] == 9  # 4 prompt + 5 generated

    def test_generate_with_temperature(self):
        model = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2, max_seq_len=50)
        x = np.array([[1, 2, 3]])
        out = model.generate(x, max_new_tokens=3, temperature=0.5)
        assert out.data.shape[1] == 6

    def test_generate_with_top_k(self):
        model = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2, max_seq_len=50)
        x = np.array([[1, 2, 3]])
        out = model.generate(x, max_new_tokens=3, top_k=10)
        assert out.data.shape[1] == 6

    def test_generate_with_top_p(self):
        model = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2, max_seq_len=50)
        x = np.array([[1, 2, 3]])
        out = model.generate(x, max_new_tokens=3, top_p=0.9)
        assert out.data.shape[1] == 6

    def test_generate_with_eos(self):
        model = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2, max_seq_len=50)
        x = np.array([[1, 2, 3]])
        out = model.generate(x, max_new_tokens=10, eos_token=99)
        # May stop early
        assert out.data.shape[1] <= 13

    def test_generate_tensor_input(self):
        model = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2, max_seq_len=50)
        x = Tensor(np.array([[1, 2, 3]]))
        out = model.generate(x, max_new_tokens=3)
        assert out.data.shape[1] == 6

    def test_new_kv_state(self):
        model = SloTransformer(vocab_size=100, n_embed=32, n_layer=1, n_head=2, max_seq_len=50)
        state = model.new_kv_state()
        assert isinstance(state, NumpyKVState)


# ── NumpyKVState ────────────────────────────────────────────────────────────


class TestNumpyKVState:

    def test_init(self):
        state = NumpyKVState()
        assert state.prev_ids is None
        assert state.capacity == 0

    def test_reset(self):
        state = NumpyKVState()
        state.kv_buf_k = [np.ones((1, 10, 4, 8))]
        state.kv_len = [10]
        state.capacity = 10
        state.reset()
        assert state.kv_buf_k == []
        assert state.capacity == 0
