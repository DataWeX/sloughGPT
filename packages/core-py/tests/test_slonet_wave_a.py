"""Wave A coverage tests for slonet.py — accelerator dispatch, module import
fallbacks, Tensor utility surface, primitive-op backward/forward-grad, and
low-level layer behaviors.

Each test locks in a specific behavior so a regression shows up as a single
failing assertion instead of a silent drift.
"""

import os
import struct
import sys
import types

import numpy as np
import pytest

import domains.training.slonet as slonet
from domains.training.slonet import (
    SloAdapterLayer,
    SloBatchNorm2D,
    SloConv2D,
    SloDataLoader,
    SloEmbedding,
    SloLayerNorm,
    SloLinear,
    SloMultiHeadAttention,
    SloNet,
    SloOneCycleLR,
    SloReduceLROnPlateau,
    SloRotaryEmbedding,
    SloTransformer,
    SloTransformerBlock,
    SloAdam,
    Tensor,
    _accel_op,
    _apply_rope,
    _apply_rope_t,
    _check_numba,
    _get_accelerator,
    _mean,
    _neg,
    _pow,
    _to_np,
    compute_sensitivity,
    cross_entropy,
    cpu,
    cuda,
    export_to_sou,
    gelu,
    is_cuda,
    is_mps,
    isfinite,
    mse_loss,
    no_grad,
    train_soul_transformer,
)

from unittest.mock import patch


@pytest.fixture(autouse=True)
def _reset_module_globals():
    slonet._ACCELERATOR = None
    slonet._NUMBA_AVAILABLE = None
    yield
    slonet._ACCELERATOR = None
    slonet._NUMBA_AVAILABLE = None
    slonet._NO_GRAD = False


class _FakeAcc:
    name = "metal"


# ---------------------------------------------------------------------------
# Module import / accelerator dispatch
# ---------------------------------------------------------------------------


def test_check_numba_false_branch(monkeypatch):
    old = sys.modules.pop("numba", None)
    try:
        assert slonet._check_numba() is False
    finally:
        if old is not None:
            sys.modules["numba"] = old


def test_check_numba_true_branch(monkeypatch):
    fake = types.ModuleType("numba")
    fake.njit = lambda f: f
    fake.__version__ = "0.60.0"
    old = sys.modules.get("numba")
    sys.modules["numba"] = fake
    try:
        assert slonet._check_numba() is True
    finally:
        if old is None:
            sys.modules.pop("numba", None)
        else:
            sys.modules["numba"] = old


def test_get_accelerator_slolib_metal(monkeypatch):
    slolib = types.ModuleType("domains.slolib")
    slolib_gpu = types.ModuleType("domains.slolib.gpu")
    slolib_gpu.get_accelerator = lambda: _FakeAcc()
    monkeypatch.setitem(sys.modules, "domains.slolib", slolib)
    monkeypatch.setitem(sys.modules, "domains.slolib.gpu", slolib_gpu)
    assert _get_accelerator().name == "metal"


def test_get_accelerator_slolib_raise_falls_back_to_old_backend(monkeypatch):
    slolib = types.ModuleType("domains.slolib")
    slolib_gpu = types.ModuleType("domains.slolib.gpu")
    slolib_gpu.get_accelerator = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "domains.slolib", slolib)
    monkeypatch.setitem(sys.modules, "domains.slolib.gpu", slolib_gpu)
    old_pkg = types.ModuleType("domains.training.gpu")
    old_mod = types.ModuleType("domains.training.gpu.accelerator")
    old_mod.get_accelerator = lambda: None
    monkeypatch.setitem(sys.modules, "domains.training.gpu", old_pkg)
    monkeypatch.setitem(sys.modules, "domains.training.gpu.accelerator", old_mod)
    assert _get_accelerator() is None


def test_get_accelerator_old_backend(monkeypatch):
    slolib = types.ModuleType("domains.slolib")
    slolib_gpu = types.ModuleType("domains.slolib.gpu")
    slolib_gpu.get_accelerator = lambda: (_ for _ in ()).throw(ImportError("no"))
    monkeypatch.setitem(sys.modules, "domains.slolib", slolib)
    monkeypatch.setitem(sys.modules, "domains.slolib.gpu", slolib_gpu)
    old_pkg = types.ModuleType("domains.training.gpu")
    old_mod = types.ModuleType("domains.training.gpu.accelerator")
    old_mod.get_accelerator = lambda: _FakeAcc()
    monkeypatch.setitem(sys.modules, "domains.training.gpu", old_pkg)
    monkeypatch.setitem(sys.modules, "domains.training.gpu.accelerator", old_mod)
    assert _get_accelerator().name == "metal"


def test_get_accelerator_old_returns_none(monkeypatch):
    slolib = types.ModuleType("domains.slolib")
    slolib_gpu = types.ModuleType("domains.slolib.gpu")
    slolib_gpu.get_accelerator = lambda: (_ for _ in ()).throw(ImportError("no"))
    monkeypatch.setitem(sys.modules, "domains.slolib", slolib)
    monkeypatch.setitem(sys.modules, "domains.slolib.gpu", slolib_gpu)
    old_pkg = types.ModuleType("domains.training.gpu")
    old_mod = types.ModuleType("domains.training.gpu.accelerator")
    old_mod.get_accelerator = lambda: None
    monkeypatch.setitem(sys.modules, "domains.training.gpu", old_pkg)
    monkeypatch.setitem(sys.modules, "domains.training.gpu.accelerator", old_mod)
    assert _get_accelerator() is None


def test_accel_op_below_threshold_uses_numpy():
    a = np.ones(64, dtype=np.float32)
    b = np.ones(64, dtype=np.float32)
    with patch.object(slonet, "_get_accelerator", return_value=_FakeAcc()):
        out = _accel_op("add", a, b, lambda x, y: x + y)
    assert np.array_equal(out, a + b)


def test_accel_op_no_accelerator_uses_numpy():
    a = np.ones(5000, dtype=np.float32)
    b = np.ones(5000, dtype=np.float32)
    with patch.object(slonet, "_get_accelerator", return_value=None):
        out = _accel_op("add", a, b, lambda x, y: x + y)
    assert np.array_equal(out, a + b)


def test_no_grad_context_and_decorator():
    with no_grad():
        t = Tensor([1.0, 2.0], requires_grad=True)
        assert t.requires_grad is False

    @no_grad()
    def build():
        return Tensor([3.0], requires_grad=True)

    assert build().requires_grad is False
    # sanity: outside the context requires_grad is honored again
    assert Tensor([1.0], requires_grad=True).requires_grad is True


# ---------------------------------------------------------------------------
# Tensor utility surface
# ---------------------------------------------------------------------------


def test_tensor_repr_and_shape_api():
    t = Tensor(np.ones((2, 3), dtype=np.float32))
    assert repr(t) == "Tensor(shape=(2, 3))"
    assert t.dim() == 2
    assert t.numel() == 6
    assert t.size() == (2, 3)
    assert t.size(1) == 3
    assert t.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


def test_tensor_setitem_and_getitem():
    t = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    t[0, 1] = 99.0
    assert t.data[0, 1] == 99.0
    assert t[1, 2].item() == 5.0


def test_tensor_torchlike_ctor():
    class _FakeTorch:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array([1.0, 2.0], dtype=np.float32)

    t = Tensor(_FakeTorch())
    assert t.data.shape == (2,)


def test_tensor_to_dtype_and_type():
    class _Dtype:
        pass

    t = Tensor([1.0, 2.0]).to(dtype=_Dtype())
    assert isinstance(t, Tensor)


def test_tensor_float_non_float32():
    t = Tensor(np.array([1.0, 2.0], dtype=np.float64))
    f = t.float()
    assert f.data.dtype == np.float32


def test_tensor_sqrt_clamp_and_math_helpers():
    t = Tensor([4.0, 9.0])
    assert np.allclose(t.sqrt().data, [2.0, 3.0])
    c = t.clamp(5.0, 8.0)
    assert np.allclose(c.data, [5.0, 8.0])


def test_tensor_selection_and_softmax_helpers():
    t = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert t.argmax(dim=1).data.tolist() == [2, 2]
    assert t.argmin(dim=1).data.tolist() == [0, 0]
    top_vals, top_idx = t.topk(2)
    assert top_vals.data.shape == (1, 2)
    assert t.softmax(dim=1).data.shape == (2, 3)
    assert t.log_softmax(dim=1).data.shape == (2, 3)


def test_tensor_t_2d_only():
    t = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    assert t.t().data.shape == (3, 2)
    with pytest.raises(RuntimeError):
        Tensor([1.0, 2.0]).t()


def test_tensor_bool_scalar_only():
    assert bool(Tensor(np.array(1.0))) is True
    with pytest.raises(RuntimeError):
        bool(Tensor([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Primitive ops: backward + forward-grad
# ---------------------------------------------------------------------------


def test_neg_backward():
    a = Tensor([1.0, 2.0], requires_grad=True)
    y = _neg(a)
    y.backward()
    assert np.allclose(a.grad.data, [-1.0, -1.0])


def test_pow_backward():
    a = Tensor([2.0, 3.0], requires_grad=True)
    y = _pow(a, 2)
    y.backward()
    assert np.allclose(a.grad.data, [4.0, 6.0])


def test_mean_backward():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = _mean(a)
    y.backward()
    assert np.allclose(a.grad.data, [1.0 / 3.0] * 3)


def test_gelu_numpy_and_tangents():
    with patch.object(slonet, "_get_accelerator", return_value=None):
        raw = gelu(np.array([0.0, 1.0], dtype=np.float32))
        assert isinstance(raw, np.ndarray)

        x = Tensor([1.0, 2.0], requires_grad=True)
        y = gelu(x)
        y.backward()
        assert x.grad is not None
        assert np.all(np.isfinite(x.grad.data))

    assert np.allclose(y._forward_fn(None), np.zeros_like(y.data))
    t = y._forward_fn(np.array([1.0, 1.0], dtype=np.float32))
    assert np.all(np.isfinite(t))

    tangents = y.forward_grad({x.id: np.array([1.0, 1.0], dtype=np.float32)})
    assert tangents.get(y.id) is not None


def test_cross_entropy_3d_backward_and_forward_grad():
    logits = Tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)
    targets = Tensor(np.array([[0, 1, 2], [3, 0, 1]], dtype=np.int64))
    loss = cross_entropy(logits, targets)
    assert loss.data.ndim == 0 or loss.data.size == 1
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.data.shape == logits.data.shape

    tangents = loss.forward_grad({logits.id: np.ones((2, 3, 4), dtype=np.float32)})
    assert tangents.get(loss.id) is not None


def test_mse_loss_and_to_np():
    pred = Tensor([1.0, 2.0, 3.0])
    target = Tensor([1.0, 3.0, 3.0])
    assert mse_loss(pred, target).data.item() == pytest.approx(1.0 / 3.0)
    assert _to_np(np.float32(3.0)).shape == ()
    assert isinstance(_to_np(np.zeros(2)), np.ndarray)


def test_module_helpers():
    t = Tensor([1.0, np.inf])
    assert isfinite(t).tolist() == [True, False]
    assert is_cuda(t) is False
    assert is_mps(t) is False
    assert cuda() is None
    assert cpu(t) is t


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


def test_lazy_linear_and_embedding():
    lin = SloLinear(4, 4, _lazy=True)
    assert np.allclose(lin.weight.data, 0.0)
    emb = SloEmbedding(8, 4, _lazy=True)
    assert np.allclose(emb.weight.data, 0.0)


def test_embedding_2d_double_backward():
    emb = SloEmbedding(10, 4)
    idx = Tensor(np.array([[0, 1], [2, 3]], dtype=np.int64))
    out = emb.forward(idx)
    assert out.data.shape == (2, 2, 4)
    out.backward()
    g1 = emb.weight.grad.data.copy()
    assert g1.sum() > 0
    out.backward()
    assert np.allclose(emb.weight.grad.data, 2 * g1)


def test_embedding_forward_3d_tensor_path():
    emb = SloEmbedding(10, 4)
    idx = Tensor(np.array([[[0, 1, 2]]], dtype=np.int64))
    out = emb.forward(idx)
    assert out.data.shape == (1, 3, 4)


def test_embedding_forward_numpy_3d_branches():
    emb = SloEmbedding(10, 4)
    a = emb.forward_numpy(np.array([[[0], [1]]], dtype=np.int64))
    assert a.shape == (1, 2, 4)
    b = emb.forward_numpy(np.array([[[0, 1], [2, 3]]], dtype=np.int64))
    assert b.shape == (1, 4, 4)


def test_adapter_forward_and_parameters():
    ad = SloAdapterLayer(dim=4, rank=2)
    assert len(ad.parameters()) == 2
    out = ad.forward(Tensor([1.0, 2.0, 3.0, 4.0]))
    assert out.data.shape == (4,)


def test_layer_norm_parameters():
    ln = SloLayerNorm(4)
    assert len(ln.parameters()) == 2
    assert ln.parameters()[0].data.shape == (4,)


def test_transformer_block_train_eval():
    block = SloTransformerBlock(16, 4, dim_ff=16, use_rope=True, max_seq_len=16, dropout=0.1)
    block.train(False)
    assert block.drop.training is False
    block.train(True)
    assert block.drop.training is True


def test_rope_precompute_early_return():
    rope = SloRotaryEmbedding(dim=8, max_seq_len=16)
    rope._precompute(8)
    assert rope._cached_seq_len == 8
    rope._precompute(4)
    assert rope._cached_seq_len == 8


def test_apply_rope_numpy():
    q = np.random.randn(2, 3, 8).astype(np.float32)
    k = np.random.randn(2, 3, 8).astype(np.float32)
    cos = np.ones((3, 8), dtype=np.float32)
    sin = np.zeros((3, 8), dtype=np.float32)
    q_out, k_out = _apply_rope(q, k, cos, sin)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_apply_rope_t_double_backward():
    Q = Tensor(np.random.randn(1, 2, 4, 8).astype(np.float32), requires_grad=True)
    K = Tensor(np.random.randn(1, 2, 4, 8).astype(np.float32), requires_grad=True)
    cos = np.ones((1, 1, 1, 8), dtype=np.float32)
    sin = np.zeros((1, 1, 1, 8), dtype=np.float32)
    qo, ko = _apply_rope_t(Q, K, cos, sin)
    qo.backward()
    ko.backward()
    g1 = Q.grad.data.copy()
    g2 = K.grad.data.copy()
    qo.backward()
    ko.backward()
    assert np.allclose(Q.grad.data, 2 * g1)
    assert np.allclose(K.grad.data, 2 * g2)


def test_mha_grad_determinism_across_passes():
    mha = SloMultiHeadAttention(8, 2)
    q = Tensor(np.random.randn(1, 4, 8).astype(np.float32), requires_grad=True)
    k = Tensor(np.random.randn(1, 4, 8).astype(np.float32), requires_grad=True)
    v = Tensor(np.random.randn(1, 4, 8).astype(np.float32), requires_grad=True)
    out, _ = mha.forward(q, k, v)
    out.backward()
    assert q.grad is not None
    assert np.all(np.isfinite(q.grad.data))
    q1 = q.grad.data.copy()
    k1 = k.grad.data.copy()
    v1 = v.grad.data.copy()
    q.grad = None
    k.grad = None
    v.grad = None
    out2, _ = mha.forward(q, k, v)
    out2.backward()
    assert np.allclose(q.grad.data, q1)
    assert np.allclose(k.grad.data, k1)
    assert np.allclose(v.grad.data, v1)


def test_mha_forward_numpy_mask():
    mha = SloMultiHeadAttention(8, 2)
    q = np.random.randn(1, 4, 8).astype(np.float32)
    k = np.random.randn(1, 4, 8).astype(np.float32)
    v = np.random.randn(1, 4, 8).astype(np.float32)
    mask = np.zeros((1, 2, 4, 4), dtype=np.float32)
    out, cache = mha.forward_numpy(q, k, v, mask=mask)
    assert out.shape == (1, 4, 8)


def test_conv2d_tuple_padding():
    conv = SloConv2D(2, 4, kernel_size=3, padding=(1, 2))
    x = Tensor(np.random.randn(1, 2, 8, 8).astype(np.float32))
    out = conv.forward(x)
    assert out.data.shape[2] == 8 and out.data.shape[3] == 10

    conv0 = SloConv2D(2, 4, kernel_size=3, padding=(0, 0))
    out0 = conv0.forward(x)
    assert out0.data.shape[2] == 6 and out0.data.shape[3] == 6


def test_batchnorm_eval_forward_double_backward():
    bn = SloBatchNorm2D(2)
    bn._train = False
    x = Tensor(np.random.randn(1, 2, 4, 4).astype(np.float32), requires_grad=True)
    out = bn.forward(x)
    out.backward()
    assert x.grad is not None
    assert np.all(np.isfinite(x.grad.data))
    g1 = x.grad.data.copy()
    x.grad = None
    out2 = bn.forward(x)
    out2.backward()
    assert np.allclose(x.grad.data, g1)


# ---------------------------------------------------------------------------
# SloNet core
# ---------------------------------------------------------------------------


def test_slonet_forward_coercions():
    net = SloNet()
    assert net.forward(np.array([1.0, 2.0], dtype=np.float64)).data.dtype == np.float32
    assert net.forward([1.0, 2.0]).data.shape == (2,)
    mv = memoryview(np.array([1.0, 2.0], dtype=np.float32))
    assert net.forward(mv).data.shape == (2,)
    t = Tensor([1.0, 2.0])
    assert net.forward(t).data.shape == (2,)
    assert net.forward(5).data.shape == ()


def test_slonet_call_and_train():
    net = SloNet(layers=[slonet.SloDropout(0.1), lambda x: x])
    net.train(False)
    assert net.layers[0].training is False
    out = net([1.0, 2.0])
    assert out.data.shape == (2,)


def test_slonet_rebuild_from_state_dict():
    sd = {
        "tok_emb.weight": np.random.randn(16, 32).astype(np.float32),
        "blocks.0.norm1.weight": np.ones(32, dtype=np.float32),
        "blocks.0.norm2.weight": np.ones(32, dtype=np.float32),
        "blocks.0.attn.q_proj.weight": np.ones((32, 32), dtype=np.float32),
        "blocks.0.attn.k_proj.weight": np.ones((32, 32), dtype=np.float32),
        "blocks.0.attn.v_proj.weight": np.ones((32, 32), dtype=np.float32),
        "blocks.0.attn.o_proj.weight": np.ones((32, 32), dtype=np.float32),
        "blocks.0.mlp.w1.weight": np.ones((128, 32), dtype=np.float32),
        "blocks.0.mlp.w2.weight": np.ones((32, 128), dtype=np.float32),
        "blocks.0.mlp.w3.weight": np.ones((128, 32), dtype=np.float32),
        "norm.weight": np.ones(32, dtype=np.float32),
    }
    net = SloNet()
    net._rebuild_from_state_dict(sd)
    assert len(net.layers) == 2
    assert net._get_weight("norm.weight") is not None
    out = net.forward(Tensor(np.array([[0, 1]], dtype=np.int64)))
    assert out.data.shape == (1, 2, 16)


# ---------------------------------------------------------------------------
# SloTransformer
# ---------------------------------------------------------------------------


def _tiny_transformer(**kw):
    params = dict(
        vocab_size=16,
        n_embed=16,
        n_layer=1,
        n_head=4,
        block_size=8,
        max_seq_len=16,
        dropout=0.0,
        use_rope=False,
        tie_weights=False,
    )
    params.update(kw)
    return SloTransformer(**params)


def test_transformer_tie_weights():
    net = _tiny_transformer(vocab_size=16, n_embed=16, tie_weights=True)
    assert np.array_equal(net.lm_head.weight.data, net.tok_emb.weight.data)
    net._tie_weights()
    assert np.array_equal(net.lm_head.weight.data, net.tok_emb.weight.data)
    assert np.allclose(net.lm_head.bias.data, 0.0)


def test_transformer_call_with_and_without_targets():
    net = _tiny_transformer()
    x = Tensor(np.array([[0, 1, 2, 3]], dtype=np.int64))
    logits, loss = net(x, targets=None)
    assert logits is not None and loss is None
    logits2, loss2 = net(x, targets=np.array([[0, 1, 2, 3]], dtype=np.int64))
    assert loss2 is not None and loss2.data.size == 1


def test_transformer_generate_tensor_input_and_eos():
    net = _tiny_transformer()
    prompt = Tensor(np.array([[0, 1, 2, 3]], dtype=np.int64))
    with patch.object(slonet, "_sample_from_logits", return_value=7):
        out = net.generate(prompt, max_new_tokens=5, eos_token=7, temperature=0.0)
    assert out.data.shape == (1, 5)
    assert out.data[0, 4] == 7


def test_transformer_generate_numpy_1d_and_stream():
    net = _tiny_transformer()
    out = net.generate_numpy(np.array([0, 1, 2, 3]), max_new_tokens=3, temperature=0.0)
    assert out.shape == (1, 7)
    assert out.dtype == np.int64

    streamed = list(
        net.generate_numpy_stream(
            input_ids=np.array([0, 1, 2]), max_new_tokens=2, temperature=0.0
        )
    )
    assert len(streamed) == 2


def test_transformer_named_parameters():
    net = _tiny_transformer(norm_type="layer_norm", use_abs_pos_emb=True)
    names = [n for n, _ in net._named_parameters()]
    assert "blocks.0.attn_norm.bias" in names
    assert "blocks.0.ff_norm.bias" in names
    assert "norm.bias" in names
    assert "pos_emb.weight" in names
    prefixed = [n for n, _ in net.named_parameters(prefix="x.")]
    assert any(n.startswith("x.") for n in prefixed)


def test_transformer_load_state_dict_1d_reshape():
    net = _tiny_transformer(norm_type="layer_norm")
    missing = net.load_state_dict({"norm.bias": np.zeros(5, dtype=np.float64)})
    assert missing == []
    assert net.norm.bias.data.shape == (16,)
    assert np.allclose(net.norm.bias.data[5:], 0.0)


# ---------------------------------------------------------------------------
# SloAdam / schedulers / training helpers
# ---------------------------------------------------------------------------


class _FakeTorchTensor:
    def __init__(self, shape, dtype=None):
        self.shape = shape
        self.dtype = dtype

    def new_zeros(self, shape, dtype=None):
        return np.zeros(shape)

    def sqrt(self):
        return np.ones(self.shape)


def test_slo_adam_static_stubs():
    assert isinstance(SloAdam._zeros_like(np.zeros(3)), np.ndarray)
    assert np.array_equal(SloAdam._sqrt(np.array([4.0])), [2.0])
    assert np.array_equal(SloAdam._zeros_like(_FakeTorchTensor((2, 2))), np.zeros((2, 2)))
    assert np.array_equal(SloAdam._sqrt(_FakeTorchTensor((3,))), np.ones((3,)))


def test_slo_adam_load_state_dict_params_none():
    opt = SloAdam(lr=0.01)
    opt.load_state_dict({"hyperparameters": {"lr": 0.02}}, params=None)
    assert opt.lr == 0.02


def test_slo_adam_step_updates_weight():
    opt = SloAdam(lr=0.1)
    p = Tensor([1.0, 1.0], requires_grad=True)
    p.grad = Tensor([1.0, 0.5])
    opt.step([p])
    assert p.data[0] < 1.0


def test_plateau_max_mode_init():
    opt = SloAdam(lr=0.1)
    sched = SloReduceLROnPlateau(opt, mode="max")
    assert sched.best == -float("inf")


def test_plateau_abs_threshold():
    opt = SloAdam(lr=0.1)
    sched = SloReduceLROnPlateau(opt, mode="min", threshold=0.5, threshold_mode="abs")
    sched.step(10.0)
    assert sched.best == 10.0
    sched.step(10.2)
    assert sched.best == 10.0


def test_plateau_cooldown_resets_bad_epochs():
    opt = SloAdam(lr=0.1)
    sched = SloReduceLROnPlateau(opt, mode="min", patience=1, cooldown=2)
    sched.step(5.0)
    sched.step(6.0)
    sched.step(7.0)
    assert sched.cooldown_counter == 2
    sched.step(8.0)
    assert sched.cooldown_counter == 1
    assert sched.num_bad_epochs == 0


def test_onecycle_cos_decay_formula():
    opt = SloAdam(lr=0.001)
    sched = SloOneCycleLR(opt, max_lr=0.01, total_steps=100, pct_start=0.1)
    sched.last_epoch = 50
    lr = sched.get_lr()[0]
    progress = (0.5 - 0.1) / (1 - 0.1)
    cos_val = 0.5 * (1 + np.cos(np.pi * progress))
    expected = 0.001 * 25.0 * ((1 - 1e-4) * cos_val + 1e-4)
    assert lr == pytest.approx(expected)
    sched.last_epoch = 99
    lr99 = sched.get_lr()[0]
    assert lr99 < lr


def test_compute_sensitivity():
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)
    out = (a * b).sum()
    scores = compute_sensitivity(out, {"a": [a], "b": [b]}, seed=0)
    assert set(scores.keys()) == {"a", "b"}
    assert all(np.isfinite(v) for v in scores.values())


def test_train_soul_transformer_empty_responses(monkeypatch, tmp_path):
    captured = {}

    def fake_export(net, path, **kw):
        captured["net"] = net
        return path

    monkeypatch.setattr(slonet, "export_to_sou", fake_export)
    net = train_soul_transformer(gpt_fn=lambda topic, temp: "", epochs=1)
    assert isinstance(net, SloTransformer)


# ---------------------------------------------------------------------------
# SOU export
# ---------------------------------------------------------------------------


def test_export_to_sou_rename_failure_cleans_temp(tmp_path, monkeypatch):
    net = SloNet(soul_name="Test")

    def _boom(src, dst):
        raise OSError("rename failed")

    def _fake_unlink(path):
        raise OSError("unlink failed")

    monkeypatch.setattr(os, "rename", _boom)
    monkeypatch.setattr(os, "unlink", _fake_unlink)
    target = str(tmp_path / "x.soul")
    with pytest.raises(OSError):
        export_to_sou(net, target, include_weights=False)


# ---------------------------------------------------------------------------
# SloDataLoader
# ---------------------------------------------------------------------------


def test_slo_data_loader_basic():
    data = np.arange(40, dtype=np.float32).reshape(10, 4)
    dl = SloDataLoader(data, batch_size=3, shuffle=False)
    batches = list(dl)
    assert sum(len(b) for b in batches) == 10
    assert len(batches[0]) == 3
    assert batches[0][0].shape == (4,)


# NOTE: this test runs in a subprocess — reloading the module in-process would
# invalidate class identities referenced by every other slonet test file.
def test_kernel_import_fallback():
    import subprocess
    import textwrap

    core_py = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {core_py!r})
        sys.modules["domains.training.slonet_kernels"] = None
        from domains.training import slonet
        assert slonet._KERNELS_AVAILABLE is False
        print("KERNELS_FALLBACK_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "KERNELS_FALLBACK_OK" in result.stdout
