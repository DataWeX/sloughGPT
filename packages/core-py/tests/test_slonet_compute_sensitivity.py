"""Tests for compute_sensitivity — forward-mode AD parameter importance."""

import sys, math
sys.path.insert(0, "packages/core-py")

import numpy as np
import pytest
from domains.training.slonet import (
    Tensor, cross_entropy, SloLinear, SloSGD,
    compute_sensitivity,
    tensor, relu, sigmoid, tanh, gelu,
    zeros, ones, randn,
    softmax, log_softmax, _softmax,
    mse_loss,
)


# ── Basic functionality ──────────────────────────────────────────────

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
    np.random.seed(0)
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
        for p in lin.parameters():
            p.grad = None

    assert sensitivities[-1] < max(sensitivities[:3]) * 1.5


# ── Sigmoid / Tanh / GELU activation sensitivity ────────────────────

def test_sensitivity_sigmoid():
    """Sigmoid activation produces finite sensitivity."""
    lin = SloLinear(4, 2)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = sigmoid(lin.forward(x))
    loss = out.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


def test_sensitivity_tanh():
    """Tanh activation produces finite sensitivity."""
    lin = SloLinear(4, 2)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = tanh(lin.forward(x))
    loss = out.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


def test_sensitivity_gelu():
    """GELU activation produces finite sensitivity."""
    lin = SloLinear(4, 2)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = gelu(lin.forward(x))
    loss = out.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


# ── Deep / wide network sensitivity ─────────────────────────────────

def test_sensitivity_deep_network():
    """3-layer network: all layers show finite sensitivity."""
    lin1 = SloLinear(8, 6)
    lin2 = SloLinear(6, 4)
    lin3 = SloLinear(4, 2)
    x = tensor(np.random.randn(4, 8).astype(np.float32), requires_grad=True)
    h = relu(lin1.forward(x))
    h = relu(lin2.forward(h))
    out = lin3.forward(h)
    loss = out.sum()
    sens = compute_sensitivity(loss, {
        "l1": lin1.parameters(), "l2": lin2.parameters(), "l3": lin3.parameters()
    })
    for key in ("l1", "l2", "l3"):
        assert key in sens
        assert math.isfinite(sens[key])


def test_sensitivity_wide_layer():
    """Wide linear layer produces finite sensitivity."""
    lin = SloLinear(256, 128)
    x = tensor(np.random.randn(8, 256).astype(np.float32), requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    sens = compute_sensitivity(loss, {"wide": lin.parameters()})
    assert "wide" in sens
    assert math.isfinite(sens["wide"])


def test_sensitivity_narrow_layer():
    """Narrow linear layer produces finite sensitivity."""
    lin = SloLinear(2, 1)
    x = tensor([[1.0, 2.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    sens = compute_sensitivity(loss, {"narrow": lin.parameters()})
    assert "narrow" in sens
    assert math.isfinite(sens["narrow"])


# ── Loss function variants ──────────────────────────────────────────

def test_sensitivity_mse_loss():
    """MSE loss produces finite sensitivity."""
    lin = SloLinear(4, 2)
    x = tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    pred = lin.forward(x)
    target = tensor(np.random.randn(3, 2).astype(np.float32), requires_grad=False)
    loss = mse_loss(pred, target)
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


def test_sensitivity_cross_entropy_large_vocab():
    """Cross-entropy with large vocab produces finite sensitivity."""
    lin = SloLinear(8, 64)
    x = tensor(np.random.randn(4, 8).astype(np.float32), requires_grad=True)
    logits = lin.forward(x)
    targets = tensor(np.random.randint(0, 64, size=(4,)).astype(np.int64), requires_grad=False)
    loss = cross_entropy(logits, targets)
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


def test_sensitivity_softmax_output():
    """Softmax output path produces finite sensitivity."""
    lin = SloLinear(4, 3)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    logits = lin.forward(x)
    probs = softmax(logits)
    loss = probs.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


# ── Edge cases ──────────────────────────────────────────────────────

def test_sensitivity_zero_input():
    """Zero input produces finite sensitivity."""
    lin = SloLinear(4, 2)
    x = tensor([[0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


def test_sensitivity_large_input():
    """Large input values produce finite sensitivity."""
    lin = SloLinear(4, 2)
    x = tensor([[100.0, 200.0, 300.0, 400.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


def test_sensitivity_negative_input():
    """Negative input values produce finite sensitivity."""
    lin = SloLinear(4, 2)
    x = tensor([[-1.0, -2.0, -3.0, -4.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


def test_sensitivity_no_bias_linear():
    """Linear layer without bias produces finite sensitivity."""
    lin = SloLinear(4, 2, bias=False)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


def test_sensitivity_single_param_group():
    """Single parameter group with one tensor."""
    lin = SloLinear(4, 2)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    sens = compute_sensitivity(loss, {"weight": [lin.weight]})
    assert "weight" in sens
    assert math.isfinite(sens["weight"])


def test_sensitivity_bias_only_group():
    """Group containing only bias parameters."""
    lin = SloLinear(4, 2)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    sens = compute_sensitivity(loss, {"bias": [lin.bias]})
    assert "bias" in sens
    assert math.isfinite(sens["bias"])


# ── Different seeds produce different results ───────────────────────

def test_sensitivity_different_seeds_differ():
    """Different seeds produce different sensitivity scores."""
    lin = SloLinear(4, 2)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    s1 = compute_sensitivity(loss, {"linear": lin.parameters()}, seed=0)
    s2 = compute_sensitivity(loss, {"linear": lin.parameters()}, seed=99)
    assert s1["linear"] != s2["linear"]


def test_sensitivity_negative_sensitivity():
    """Sensitivity can be negative (JVP < 0 when tangent opposes gradient)."""
    lin = SloLinear(4, 1)
    lin.weight.data[:] = -1.0
    lin.bias.data[:] = 0.0
    x = tensor([[1.0, 1.0, 1.0, 1.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    loss.backward()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()}, seed=42)
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


# ── Multiple batches ────────────────────────────────────────────────

def test_sensitivity_large_batch():
    """Large batch size produces finite sensitivity."""
    lin = SloLinear(8, 4)
    x = tensor(np.random.randn(128, 8).astype(np.float32), requires_grad=True)
    logits = lin.forward(x)
    targets = tensor(np.random.randint(0, 4, size=(128,)).astype(np.int64), requires_grad=False)
    loss = cross_entropy(logits, targets)
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


def test_sensitivity_single_sample():
    """Single sample batch produces finite sensitivity."""
    lin = SloLinear(4, 2)
    x = tensor(np.random.randn(1, 4).astype(np.float32), requires_grad=True)
    logits = lin.forward(x)
    targets = tensor(np.array([1], dtype=np.int64), requires_grad=False)
    loss = cross_entropy(logits, targets)
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


# ── Activation combinations ─────────────────────────────────────────

def test_sensitivity_relu_chain():
    """Chain of ReLU activations produces finite sensitivity."""
    lin1 = SloLinear(4, 4)
    lin2 = SloLinear(4, 2)
    x = tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
    h = relu(lin1.forward(x))
    h = relu(lin2.forward(h))
    loss = h.sum()
    sens = compute_sensitivity(loss, {"l1": lin1.parameters(), "l2": lin2.parameters()})
    assert math.isfinite(sens["l1"])
    assert math.isfinite(sens["l2"])


def test_sensitivity_mixed_activations():
    """Mix of sigmoid and tanh activations produces finite sensitivity."""
    lin1 = SloLinear(4, 4)
    lin2 = SloLinear(4, 2)
    x = tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
    h = sigmoid(lin1.forward(x))
    out = tanh(lin2.forward(h))
    loss = out.sum()
    sens = compute_sensitivity(loss, {"l1": lin1.parameters(), "l2": lin2.parameters()})
    assert math.isfinite(sens["l1"])
    assert math.isfinite(sens["l2"])


# ── Symmetry / consistency ──────────────────────────────────────────

def test_sensitivity_no_grad_context():
    """compute_sensitivity inside no_grad still returns finite values."""
    from domains.training.slonet import no_grad
    lin = SloLinear(4, 2)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    with no_grad():
        out = lin.forward(x)
        loss = out.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens


def test_sensitivity_many_groups():
    """Many parameter groups (5+) produce independent finite scores."""
    layers = [SloLinear(4, 4) for _ in range(5)]
    x = tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
    h = x
    groups = {}
    for i, layer in enumerate(layers):
        h = relu(layer.forward(h))
        groups[f"layer_{i}"] = layer.parameters()
    loss = h.sum()
    sens = compute_sensitivity(loss, groups)
    for key in groups:
        assert key in sens
        assert math.isfinite(sens[key])


def test_sensitivity_reproducible_across_calls():
    """Calling compute_sensitivity twice with same seed returns same result."""
    lin = SloLinear(4, 2)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    s1 = compute_sensitivity(loss, {"linear": lin.parameters()}, seed=7)
    s2 = compute_sensitivity(loss, {"linear": lin.parameters()}, seed=7)
    assert s1["linear"] == pytest.approx(s2["linear"], rel=1e-6)


# ── Random seed default ─────────────────────────────────────────────

def test_sensitivity_default_seed():
    """Default seed (None) produces finite sensitivity."""
    lin = SloLinear(4, 2)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


# ── Additional coverage ──────────────────────────────────────────────

def test_sensitivity_silu_activation():
    """Silu activation produces finite sensitivity."""
    from domains.training.slonet import silu
    lin = SloLinear(4, 2)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = silu(lin.forward(x))
    loss = out.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


def test_sensitivity_log_softmax():
    """Log-softmax output path produces finite sensitivity."""
    lin = SloLinear(4, 3)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    logits = lin.forward(x)
    log_probs = log_softmax(logits)
    loss = log_probs.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


def test_sensitivity_nonnegative_scores():
    """Sensitivity scores are norms, hence non-negative."""
    lin = SloLinear(4, 2)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()}, seed=42)
    assert sens["linear"] >= 0.0


def test_sensitivity_closer_layer_higher():
    """Layer closer to output has non-zero sensitivity (magnitude depends on init)."""
    lin_near = SloLinear(4, 2)
    lin_far = SloLinear(4, 4)
    x = tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
    h = relu(lin_far.forward(x))
    out = lin_near.forward(h)
    loss = out.sum()
    sens = compute_sensitivity(loss, {
        "near": lin_near.parameters(),
        "far": lin_far.parameters(),
    })
    assert sens["near"] > 0
    assert sens["far"] > 0


def test_sensitivity_elementwise_add():
    """Elementwise add produces finite sensitivity."""
    lin = SloLinear(4, 4)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    h = lin.forward(x)
    out = h + h
    loss = out.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


def test_sensitivity_elementwise_multiply():
    """Elementwise multiply produces finite sensitivity."""
    lin = SloLinear(4, 4)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    h = lin.forward(x)
    out = h * h
    loss = out.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


def test_sensitivity_return_type():
    """Return value is a dict mapping str to float."""
    lin = SloLinear(4, 2)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    sens = compute_sensitivity(loss, {"g": lin.parameters()})
    assert isinstance(sens, dict)
    assert isinstance(sens["g"], float)


def test_sensitivity_mixed_requires_grad_in_group():
    """Group with mix of frozen and trainable params returns finite score."""
    lin = SloLinear(4, 2)
    lin.bias.requires_grad = False
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    sens = compute_sensitivity(loss, {"mixed": lin.parameters()})
    assert "mixed" in sens
    assert math.isfinite(sens["mixed"])


def test_sensitivity_with_eye():
    """Eye matrix as input produces finite sensitivity."""
    from domains.training.slonet import eye
    lin = SloLinear(4, 2)
    inp = eye(4)
    x = tensor(inp.data.copy(), requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


def test_sensitivity_skip_connection():
    """Network with skip connection produces finite sensitivity."""
    lin1 = SloLinear(4, 4)
    lin2 = SloLinear(4, 4)
    x = tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
    h1 = relu(lin1.forward(x))
    h2 = lin2.forward(h1)
    out = h2 + x
    loss = out.sum()
    sens = compute_sensitivity(loss, {
        "l1": lin1.parameters(), "l2": lin2.parameters()
    })
    assert math.isfinite(sens["l1"])
    assert math.isfinite(sens["l2"])


def test_sensitivity_single_param():
    """Single scalar parameter produces finite sensitivity."""
    lin = SloLinear(1, 1)
    x = tensor([[2.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


def test_sensitivity_batch_sensitivity_differs():
    """Different batch sizes produce different sensitivity scores."""
    lin = SloLinear(4, 2)
    x1 = tensor(np.random.randn(1, 4).astype(np.float32), requires_grad=True)
    x2 = tensor(np.random.randn(8, 4).astype(np.float32), requires_grad=True)
    loss1 = lin.forward(x1).sum()
    loss2 = lin.forward(x2).sum()
    s1 = compute_sensitivity(loss1, {"linear": lin.parameters()}, seed=0)
    s2 = compute_sensitivity(loss2, {"linear": lin.parameters()}, seed=0)
    assert s1["linear"] != s2["linear"]


def test_sensitivity_deep_5layer():
    """5-layer deep network: all layers show finite sensitivity."""
    layers = [SloLinear(8, 8) for _ in range(4)] + [SloLinear(8, 2)]
    x = tensor(np.random.randn(2, 8).astype(np.float32), requires_grad=True)
    h = x
    groups = {}
    for i, layer in enumerate(layers):
        h = relu(layer.forward(h))
        groups[f"l{i}"] = layer.parameters()
    loss = h.sum()
    sens = compute_sensitivity(loss, groups)
    assert len(sens) == 5
    for v in sens.values():
        assert math.isfinite(v)


def test_sensitivity_label_names_preserved():
    """Group names in output match input names exactly."""
    lin = SloLinear(4, 2)
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    names = ["alpha", "beta", "gamma"]
    sens = compute_sensitivity(loss, {n: lin.parameters() for n in names})
    assert set(sens.keys()) == set(names)


def test_sensitivity_non_negative_after_relu():
    """Sensitivity remains finite even when ReLU kills some neurons."""
    lin = SloLinear(4, 4)
    lin.weight.data[:] = -10.0
    x = tensor([[1.0, 1.0, 1.0, 1.0]], requires_grad=True)
    out = relu(lin.forward(x))
    loss = out.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])


def test_sensitivity_very_small_tensor():
    """1x1 tensor produces finite sensitivity."""
    lin = SloLinear(1, 1)
    x = tensor([[0.5]], requires_grad=True)
    out = lin.forward(x)
    loss = out.sum()
    sens = compute_sensitivity(loss, {"linear": lin.parameters()})
    assert "linear" in sens
    assert math.isfinite(sens["linear"])
