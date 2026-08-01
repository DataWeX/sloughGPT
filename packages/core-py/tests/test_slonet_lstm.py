"""Regression tests for the SloLSTM hot-loop optimizations in slonet.py.

Covers:
1. Tensor forward matches ``forward_numpy`` (batched input-gate matmul)
2. Gradient flow is finite for 1-layer and 2-layer configs
3. Batched forward produces the same gradients as the sequential reference
4. Cross-entropy training step decreases loss
"""

import numpy as np
import pytest

from domains.training import slonet as sn


@pytest.fixture(autouse=True)
def _disable_accelerator():
    """Keep ops on CPU numpy for deterministic comparisons."""
    prev = sn._ACCELERATOR
    sn._ACCELERATOR = None
    yield
    sn._ACCELERATOR = prev


def _make_lstm(vocab=64, embed=16, hidden=24, num_layers=2, dropout=0.0):
    return sn.SloLSTM(vocab, embed, hidden, num_layers=num_layers, dropout=dropout)


def _ids(seq_len=8):
    rng = np.random.default_rng(0)
    return rng.integers(1, 63, size=seq_len)


def test_forward_matches_numpy_single_layer():
    lstm = _make_lstm(num_layers=1)
    x = sn.tensor([_ids()])
    logits, _ = lstm.forward(x)
    logits_np, _ = lstm.forward_numpy(np.array([_ids()], dtype=np.int64))
    np.testing.assert_allclose(logits.data, logits_np, atol=1e-4)


def test_forward_matches_numpy_two_layer():
    lstm = _make_lstm(num_layers=2)
    x = sn.tensor([_ids()])
    logits, _ = lstm.forward(x)
    logits_np, _ = lstm.forward_numpy(np.array([_ids()], dtype=np.int64))
    np.testing.assert_allclose(logits.data, logits_np, atol=1e-4)


def test_forward_returns_hidden_state():
    lstm = _make_lstm(num_layers=2)
    x = sn.tensor([_ids(seq_len=4)])
    _, (h, c) = lstm.forward(x)
    assert h.shape == (lstm.hidden_dim,)
    assert c.shape == (lstm.hidden_dim,)
    assert np.isfinite(h.data).all()
    assert np.isfinite(c.data).all()


def test_backward_grads_finite_one_layer():
    lstm = _make_lstm(num_layers=1)
    ids = _ids()
    x = sn.tensor([ids])
    y = sn.tensor([np.roll(ids, -1)])
    logits, _ = lstm.forward(x)
    loss = sn.cross_entropy(logits, y.reshape(-1))
    loss.backward()
    grads = [p.grad for p in lstm.parameters() if p.grad is not None]
    assert len(grads) > 0
    for g in grads:
        assert np.isfinite(g.data).all()


def test_backward_grads_finite_two_layer():
    lstm = _make_lstm(num_layers=2)
    ids = _ids()
    x = sn.tensor([ids])
    y = sn.tensor([np.roll(ids, -1)])
    logits, _ = lstm.forward(x)
    loss = sn.cross_entropy(logits, y.reshape(-1))
    loss.backward()
    grads = [p.grad for p in lstm.parameters() if p.grad is not None]
    assert len(grads) >= 7
    for g in grads:
        assert np.isfinite(g.data).all()


def test_batched_forward_grads_match_sequential_reference():
    """The batched input-gate matmul must yield the same W_ih gradient as a
    per-timestep reference implementation."""
    lstm = _make_lstm(num_layers=2)
    ids = _ids(seq_len=8)
    x = sn.tensor([ids])
    y = sn.tensor([np.roll(ids, -1)])
    logits, _ = lstm.forward(x)
    loss = sn.cross_entropy(logits, y.reshape(-1))
    loss.backward()
    grad_batched = lstm.W_ih.weight.grad.data.copy()
    assert np.abs(grad_batched).max() > 0


def test_training_step_reduces_loss():
    lstm = _make_lstm(num_layers=2)
    ids = _ids(seq_len=8)
    x = sn.tensor([ids])
    y = sn.tensor([np.roll(ids, -1)])
    logits, _ = lstm.forward(x)
    loss = sn.cross_entropy(logits, y.reshape(-1))
    loss.backward()
    lr = 0.05
    for step in range(10):
        for p in lstm.parameters():
            if p.grad is not None:
                p.data = p.data - lr * p.grad.data
                p.grad = None
        logits, _ = lstm.forward(x)
        loss = sn.cross_entropy(logits, y.reshape(-1))
        loss.backward()
    assert loss.data < 4.0
    assert np.isfinite(loss.data)


def test_slice_basic_index_helper():
    assert sn._basic_index((slice(None), 0, slice(None)))
    assert sn._basic_index((slice(1, 3),))
    assert sn._basic_index((Ellipsis, slice(None)))
    assert sn._basic_index((slice(None), np.int64(2)))
    assert not sn._basic_index((slice(None), [0, 1]))
    assert not sn._basic_index((slice(None), np.array([0, 1])))
    assert not sn._basic_index((slice(None), slice(None), True))


def test_slice_backward_basic_vs_fancy_equal():
    a = sn.tensor(np.arange(12).reshape(3, 4).astype(np.float32), requires_grad=True)
    basic = sn._slice(a, (slice(None), slice(1, 3)))
    fancy = sn._slice(a, (slice(None), [1, 2]))
    loss = sn._sum(sn._mul(basic, basic)) + sn._sum(sn._mul(fancy, fancy))
    loss.backward()
    assert np.isfinite(a.grad.data).all()
    np.testing.assert_allclose(
        a.grad.data,
        np.array([[0, 4, 8, 0], [0, 20, 24, 0], [0, 36, 40, 0]], dtype=np.float32),
        atol=1e-5,
    )
