"""Tests for compute_sensitivity — forward-mode AD parameter importance."""

import sys, math
sys.path.insert(0, "packages/core-py")

import numpy as np
import pytest
from domains.training.slonet import (
    Tensor, cross_entropy, SloLinear, SloSGD,
    compute_sensitivity,
    tensor, relu,
)


def test_sensitivity_basic():
    """Single linear layer: sensitivity > 0 for trainable params."""
    lin = SloLinear(4, 2)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert sens["linear"] > 0


def test_sensitivity_zero_for_frozen_params():
    """Params with requires_grad=False yield zero contribution."""
    lin = SloLinear(4, 2)
    for p in lin.parameters():
        p.requires_grad = False
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert sens == {}


def test_sensitivity_cross_entropy():
    """Cross-entropy loss: loss tangent norm reflects output sensitivity."""
    lin = SloLinear(8, 4)
    x = tensor(np.random.randn(2, 8).astype(np.float32), requires_grad=True)
    logits = lin.forward(x)
    targets = tensor(np.array([1, 2], dtype=np.int64), requires_grad=False)
    loss = cross_entropy(logits, targets)
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


def test_sensitivity_multiple_groups():
    """Two separate linear layers return independent sensitivity scores."""
    lin1 = SloLinear(4, 3)
    lin2 = SloLinear(3, 2)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    h = relu(lin1.forward(x))
    out = lin2.forward(h)
    loss = out.sum()
    sens = compute_sensitivity(loss, {"layer1": lin1.parameters(), "layer2": lin2.parameters()})
    assert "layer1" in sens
    assert "layer2" in sens
    assert math.isfinite(sens["layer1"])
    assert math.isfinite(sens["layer2"])


def test_sensitivity_deterministic_seed():
    """Same seed produces identical results."""
    lin = SloLinear(4, 2)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out1 = lin.forward(x)
    loss1 = out1.sum()
    s1 = compute_sensitivity(loss1, {"linear": lin.parameters()}, seed=42)

    lin2 = SloLinear(4, 2)
    lin2.weight.data[:] = lin.weight.data.copy()
    lin2.bias.data[:] = lin.bias.data.copy()
    out2 = lin2.forward(x)
    loss2 = out2.sum()
    s2 = compute_sensitivity(loss2, {"linear": lin2.parameters()}, seed=42)
    assert s1["linear"] == pytest.approx(s2["linear"], rel=1e-5)


def test_sensitivity_empty_groups():
    """Empty param_groups returns empty dict."""
    lin = SloLinear(4, 2)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    sens = compute_sensitivity(loss, {})
    assert sens == {}


def test_sensitivity_gradient_alignment():
    """JVP tangent = sum(param_grad · seed_tangent) for random direction."""
    lin = SloLinear(4, 2)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    loss.backward()

    rng = np.random.RandomState(0)
    seed_t = {}
    for p in lin.parameters():
        v = rng.standard_normal(p.shape).astype(p.data.dtype)
        v_norm = np.linalg.norm(v)
        seed_t[p.id] = v / v_norm if v_norm > 0 else np.zeros_like(p.data)

    result = loss.forward_grad(seed_t)
    jvp = float(result.get(loss.id, np.zeros(1)))

    rng2 = np.random.RandomState(0)
    expected = 0.0
    for p in lin.parameters():
        v2 = rng2.standard_normal(p.shape).astype(p.data.dtype)
        v2 /= np.linalg.norm(v2) + 1e-12
        expected += float(np.sum(p.grad.data * v2))

    assert abs(jvp - expected) < 1e-4, f"jvp={jvp} expected={expected}"


def test_sensitivity_backward_preserved():
    """compute_sensitivity does NOT corrupt subsequent backward()."""
    lin = SloLinear(4, 2)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()

    _ = compute_sensitivity(loss, {"linear": lin.parameters()})

    loss.backward()
    assert all(p.grad is not None for p in lin.parameters())
    assert all(np.all(np.isfinite(p.grad.data)) for p in lin.parameters())


def test_sensitivity_training_tracking():
    """Sensitivity score evolves as model trains (params become less sensitive)."""
    lin = SloLinear(4, 2)
    opt = SloSGD(lr=0.1)
    x = tensor(np.random.randn(16, 4).astype(np.float32), requires_grad=True)
    targets = tensor(np.random.randint(0, 2, size=(16,)).astype(np.int64), requires_grad=False)

    sensitivities = []
    for _ in range(20):
        logits = lin.forward(x)
        loss = cross_entropy(logits, targets)
        sens = compute_sensitivity(loss, {"linear": lin.parameters()})
        sensitivities.append(sens["linear"])
        loss.backward()
        opt.step(lin.parameters())
        # zero grad for next iteration
        for p in lin.parameters():
            p.grad = None

    # Sensitivity should generally decrease or stay bounded
    assert sensitivities[-1] < max(sensitivities[:3]) * 1.5
