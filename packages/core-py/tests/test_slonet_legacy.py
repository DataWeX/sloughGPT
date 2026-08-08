"""Tests for legacy SloNet surfaces: SOU export/import, flat-param rebuild,
demo trainers, SloDataLoader, silu numpy path, Tensor.to, GQA repeat backward."""

import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from domains.training import slonet
from domains.training.slonet import (
    SloAdapterLayer,
    SloDataLoader,
    SloEmbedding,
    SloLSTM,
    SloLinear,
    SloMultiHeadAttention,
    SloNet,
    SloTransformer,
    Tensor,
    _accel_op,
    _load_pytorch_zip_weights,
    _rebuild_net_from_params,
    _sanitize,
    export_to_sou,
    import_from_sou,
    kl_div_loss,
    log_softmax,
    multinomial,
    normalize,
    pairwise_distance,
    silu,
    softmax,
    souls_from_directory,
    train_char_lstm_from_gpt,
    train_soul_transformer,
)


class _ListDataset:
    def __init__(self, data):
        self._d = data

    def __len__(self):
        return len(self._d)

    def __getitem__(self, i):
        return self._d[i]


def _small_transformer():
    return SloTransformer(vocab_size=32, n_embed=16, n_layer=1, n_head=2, block_size=16)


class TestSOURoundTrip:
    def test_export_import_transformer(self, tmp_path):
        net = _small_transformer()
        path = export_to_sou(net, str(tmp_path / "net.soul"))
        assert (tmp_path / "net.soul.meta.json").exists()
        net2 = import_from_sou(path)
        assert isinstance(net2, SloTransformer)
        assert net2.vocab_size == 32
        assert net2.n_layer == 1
        assert net2.n_head == 2
        assert np.allclose(net2.tok_emb.weight.data, net.tok_emb.weight.data)

    def test_export_metadata_merge(self, tmp_path):
        net = _small_transformer()
        export_to_sou(net, str(tmp_path / "net.soul"), metadata={"extra": 7})
        meta = json.loads((tmp_path / "net.soul.meta.json").read_text())
        assert meta["metadata"]["extra"] == 7

    def test_export_sanitizes_nan(self, tmp_path):
        net = _small_transformer()
        export_to_sou(net, str(tmp_path / "nan.soul"), metadata={"bad": float("nan"), "ok": 1.5})
        meta = json.loads((tmp_path / "nan.soul.meta.json").read_text())
        assert meta["metadata"]["bad"] is None
        assert meta["metadata"]["ok"] == 1.5

    def test_sanitize_nested(self):
        assert _sanitize({"a": float("nan"), "b": [float("inf"), 3]}) == {"a": None, "b": [None, 3]}
        assert _sanitize({"c": "plain"}) == {"c": "plain"}

    def test_export_no_weights(self, tmp_path):
        net = _small_transformer()
        export_to_sou(net, str(tmp_path / "now.soul"), include_weights=False)
        net2 = import_from_sou(str(tmp_path / "now.soul"))
        assert net2.vocab_size == 32
        assert net2.n_layer == 1

    def test_import_bad_magic(self, tmp_path):
        bad = tmp_path / "bad.soul"
        bad.write_bytes(b"NOPE....")
        with pytest.raises(ValueError):
            import_from_sou(str(bad))

    def test_import_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            import_from_sou(str(tmp_path / "nope.soul"))

    def test_import_v1_json_weights(self, tmp_path):
        meta = json.dumps({"soul_name": "v1", "lineage": "slonet"}).encode()
        raw = slonet.SOU_MAGIC + struct.pack("<I", 1) + struct.pack("<I", len(meta))
        raw += meta + struct.pack("<I", 2) + b"{}"
        path = tmp_path / "v1.soul"
        path.write_bytes(raw)
        net = import_from_sou(str(path))
        assert isinstance(net, SloNet)
        assert net.soul_name == "v1"

    def test_import_v1_zip_weights_error_path(self, tmp_path):
        meta = json.dumps({"soul_name": "v1z", "lineage": "slonet"}).encode()
        raw = slonet.SOU_MAGIC + struct.pack("<I", 1) + struct.pack("<I", len(meta))
        raw += meta + b"PKgarbage-not-a-real-zip"
        path = tmp_path / "v1z.soul"
        path.write_bytes(raw)
        net = import_from_sou(str(path))
        assert net.soul_name == "v1z"

    def test_load_pytorch_zip_weights_garbage(self):
        assert _load_pytorch_zip_weights(b"garbage-bytes") == {}

    def test_souls_from_directory(self, tmp_path):
        d = tmp_path / "souls"
        d.mkdir()
        export_to_sou(_small_transformer(), str(d / "a.soul"))
        (d / "bad.soul").write_bytes(b"NOPE")
        souls = souls_from_directory(str(d))
        assert len(souls) == 1
        assert isinstance(souls[0], SloTransformer)


class TestFlatParamRebuild:
    def _make_net(self, num_layers):
        return SloNet(
            [SloEmbedding(12, 8, "embed"), SloLSTM(12, 8, 4, num_layers=num_layers, dropout=0.0)],
            soul_name="flat",
            lineage="legacy",
            metadata={},
        )

    def test_roundtrip_single_layer(self, tmp_path):
        net = self._make_net(1)
        path = export_to_sou(net, str(tmp_path / "flat.soul"))
        net2 = import_from_sou(str(path))
        assert len(net2.layers) == 2
        lstm = net2.layers[1]
        assert isinstance(lstm, SloLSTM)
        assert np.allclose(lstm.W_ih.weight.data, net.layers[1].W_ih.weight.data)
        assert np.allclose(lstm.W_hh.bias.data, net.layers[1].W_hh.bias.data)
        assert np.allclose(lstm.fc_out.weight.data, net.layers[1].fc_out.weight.data)

    def test_roundtrip_two_layers(self, tmp_path):
        net = self._make_net(2)
        path = export_to_sou(net, str(tmp_path / "flat2.soul"))
        net2 = import_from_sou(str(path))
        assert len(net2.layers) == 2
        lstm = net2.layers[1]
        assert np.allclose(lstm.W_ih2.weight.data, net.layers[1].W_ih2.weight.data)
        assert np.allclose(lstm.W_hh2.bias.data, net.layers[1].W_hh2.bias.data)

    def test_empty_weights_noop(self):
        net = SloNet(soul_name="x")
        _rebuild_net_from_params(net, {})
        assert len(net.layers) == 0

    def test_1d_first_array_aborts(self):
        net = SloNet(soul_name="x")
        _rebuild_net_from_params(net, {"p0": np.zeros((5,), dtype=np.float32)})
        assert len(net.layers) == 0

    def test_too_few_lstm_arrays_aborts(self):
        net = SloNet(soul_name="x")
        weights = {
            "p0": np.zeros((12, 8), dtype=np.float32),
            "p1": np.zeros((8, 8), dtype=np.float32),
            "p2": np.zeros((16, 8), dtype=np.float32),
            "p3": np.zeros((16,), dtype=np.float32),
        }
        _rebuild_net_from_params(net, weights)
        assert len(net.layers) == 1
        assert isinstance(net.layers[0], SloEmbedding)


def _fake_gpt(topic, temperature):
    return "hello world this is a short test response about machine learning today"


class TestDemoTrainers:
    def test_train_char_lstm(self, monkeypatch):
        monkeypatch.setattr(slonet, "export_to_sou", lambda *a, **k: None)
        losses = []
        net = train_char_lstm_from_gpt(
            _fake_gpt,
            soul_name="LegacyLSTM",
            epochs=1,
            lr=0.001,
            embed_dim=16,
            hidden_dim=16,
            on_step=lambda s, l, e: losses.append(float(l)),
        )
        assert isinstance(net, SloNet)
        assert len(losses) > 0
        assert all(np.isfinite(l) for l in losses)

    def test_train_soul_transformer(self, monkeypatch):
        monkeypatch.setattr(slonet, "export_to_sou", lambda *a, **k: None)
        losses = []
        state = {"n": 0}

        def flaky_gpt(topic, temperature):
            state["n"] += 1
            if state["n"] == 1:
                return ""
            if state["n"] == 2:
                return "hi"
            return _fake_gpt(topic, temperature)

        net = train_soul_transformer(
            flaky_gpt,
            soul_name="LegacyTransformer",
            epochs=1,
            lr=0.001,
            vocab_size=44,
            n_embed=16,
            n_layer=1,
            n_head=2,
            on_step=lambda s, l, e: losses.append(float(l)),
        )
        assert isinstance(net, SloTransformer)
        assert len(losses) > 0
        assert all(np.isfinite(l) for l in losses)


class TestSloDataLoader:
    def test_basic_batching(self):
        loader = SloDataLoader(_ListDataset(list(range(7))), batch_size=3)
        assert len(loader) == 3
        batches = [b for b in loader]
        assert batches == [[0, 1, 2], [3, 4, 5], [6]]

    def test_drop_last(self):
        loader = SloDataLoader(_ListDataset(list(range(7))), batch_size=3, drop_last=True)
        assert len(loader) == 2
        assert [b for b in loader] == [[0, 1, 2], [3, 4, 5]]

    def test_len_exact_and_remainder(self):
        assert len(SloDataLoader(_ListDataset(list(range(6))), batch_size=3)) == 2
        assert len(SloDataLoader(_ListDataset(list(range(6))), batch_size=4)) == 2

    def test_shuffle_preserves_items(self):
        data = list(range(20))
        np.random.seed(0)
        loader = SloDataLoader(_ListDataset(data), batch_size=5, shuffle=True)
        batches = [b for b in loader]
        flat = [x for b in batches for x in b]
        assert sorted(flat) == data
        assert flat != data

    def test_collate_fn(self):
        loader = SloDataLoader(
            _ListDataset([1, 2, 3]), batch_size=2, collate_fn=lambda b: sum(b)
        )
        assert [b for b in loader] == [3, 3]

    def test_reset(self):
        loader = SloDataLoader(_ListDataset(list(range(4))), batch_size=2)
        first = [b for b in loader]
        loader.reset()
        second = [b for b in loader]
        assert first == second == [[0, 1], [2, 3]]

    def test_empty_dataset(self):
        loader = SloDataLoader(_ListDataset([]))
        assert len(loader) == 1
        with pytest.raises(StopIteration):
            next(iter(loader))


class TestSiluNumpyPath:
    def test_numpy_input_returns_ndarray(self):
        out = silu(np.array([0.0, 1.0, -1.0], dtype=np.float32))
        assert isinstance(out, np.ndarray)
        expected = np.array([0.0, 0.73105858, -0.26894142], dtype=np.float32)
        assert np.allclose(out, expected, atol=1e-6)

    def test_tensor_input_returns_tensor(self):
        t = Tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
        out = silu(t)
        assert isinstance(out, Tensor)
        assert out.requires_grad

    def test_tensor_backward(self):
        t = Tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
        out = silu(t)
        (out * out).sum().backward()
        assert t.grad is not None
        assert np.all(np.isfinite(t.grad.data))
        assert t.grad.data.shape == (2,)

    def test_tensor_forward_grad(self):
        t = Tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
        out = silu(t)
        tangents = out.forward_grad({t.id: np.ones(2, dtype=np.float32)})
        assert tangents.get(out.id) is not None
        assert tangents[out.id].shape == (2,)


class TestTensorTo:
    def test_float_dtype_pins_float32(self):
        t = Tensor(np.array([1, 2, 3]))
        t.to(dtype=np.float64)
        assert t.data.dtype == np.float32

    def test_int_dtype_converts(self):
        t = Tensor(np.array([1, 2, 3]))
        t.to(dtype=np.int32)
        assert t.data.dtype == np.int32
        assert t.data.tolist() == [1, 2, 3]

    def test_int_dtype_string(self):
        t = Tensor(np.array([1, 2, 3]))
        t.to(dtype="int64")
        assert t.data.dtype == np.int64

    def test_device_noop(self):
        t = Tensor(np.array([1.0]))
        t.to("cpu")
        assert t.data.dtype == np.float32

    def test_device_positional_dtype(self):
        t = Tensor(np.array([1.0, 2.0]))
        t.to(np.float32)
        assert t.data.dtype == np.float32

    def test_meta_device(self):
        class _Dev:
            type = "meta"

        t = Tensor(np.array([1.0]))
        meta = t.to(_Dev())
        assert type(meta).__name__ == "_MetaTensor"


class TestGQARepeatBackward:
    def test_n_rep_gt_one_backward(self):
        mha = SloMultiHeadAttention(d_model=16, n_heads=4, n_kv_head=2)
        rng = np.random.default_rng(0)
        q = Tensor(rng.standard_normal((2, 3, 16)).astype(np.float32), requires_grad=True)
        k = Tensor(rng.standard_normal((2, 3, 16)).astype(np.float32), requires_grad=True)
        v = Tensor(rng.standard_normal((2, 3, 16)).astype(np.float32), requires_grad=True)
        out_t, _ = mha.forward(q, k, v)
        assert out_t.data.shape == (2, 3, 16)
        (out_t * out_t).sum().backward()
        assert q.grad is not None
        assert np.all(np.isfinite(q.grad.data))
        assert np.all(np.isfinite(mha.W_q.weight.grad.data))


class TestLossUtils:
    def test_log_softmax_numpy(self):
        x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        lp = log_softmax(x, dim=-1)
        assert isinstance(lp, np.ndarray)
        assert np.allclose(np.exp(lp).sum(axis=-1), np.ones(2), atol=1e-5)

    def test_log_softmax_tensor_backward(self):
        t = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
        out = log_softmax(t)
        assert isinstance(out, Tensor)
        (out * out).sum().backward()
        assert t.grad is not None
        assert np.all(np.isfinite(t.grad.data))

    def test_log_softmax_forward_grad(self):
        t = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
        out = log_softmax(t)
        tangents = out.forward_grad({t.id: np.ones(3, dtype=np.float32)})
        assert tangents.get(out.id) is not None

    def test_kl_div_loss(self):
        ilp = np.log(np.array([[0.5, 0.5]], dtype=np.float32))
        tp = np.array([[0.3, 0.7]], dtype=np.float32)
        loss = kl_div_loss(ilp, tp)
        assert isinstance(loss, (np.ndarray, float)) or hasattr(loss, "data")
        val = loss.data[()] if hasattr(loss, "data") else loss
        assert np.isfinite(val)

    def test_pairwise_distance_numpy(self):
        x1 = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        x2 = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        dist = pairwise_distance(x1, x2)
        assert np.allclose(dist[0], 0.0, atol=1e-4)
        assert np.allclose(dist[1], 5.0, atol=1e-3)

    def test_pairwise_distance_tensor_backward(self):
        x1 = Tensor(np.array([[1.0, 2.0]], dtype=np.float32), requires_grad=True)
        x2 = Tensor(np.array([[4.0, 6.0]], dtype=np.float32), requires_grad=True)
        dist = pairwise_distance(x1, x2)
        assert isinstance(dist, Tensor)
        dist.sum().backward()
        assert x1.grad is not None
        assert x2.grad is not None
        assert np.all(np.isfinite(x1.grad.data))


class TestSoftmaxNumpyPath:
    def test_numpy_input(self):
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = softmax(x)
        assert isinstance(out, np.ndarray)
        assert np.allclose(out.sum(axis=-1), np.ones(1), atol=1e-5)

    def test_tensor_input_delegates(self):
        t = Tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), requires_grad=True)
        out = softmax(t)
        assert isinstance(out, Tensor)
        assert np.allclose(out.data.sum(axis=-1), np.ones(1), atol=1e-5)


class TestSloNetFit:
    def test_fit_returns_epoch_losses(self):
        net = SloNet([SloLinear(4, 2, "lin")], soul_name="fit_test")
        rng = np.random.default_rng(0)
        X = Tensor(rng.standard_normal((12, 4)).astype(np.float32))
        y = Tensor(np.asarray(rng.integers(0, 2, 12), dtype=np.float32))
        opt = slonet.SloAdam(lr=0.05)
        losses = net.fit(X, y, opt, epochs=2, batch_size=4)
        assert len(losses) == 2
        assert all(np.isfinite(l) for l in losses)
        assert net._step == 6

    def test_fit_on_step_callback(self):
        net = SloNet([SloLinear(4, 2, "lin")])
        rng = np.random.default_rng(1)
        X = Tensor(rng.standard_normal((8, 4)).astype(np.float32))
        y = Tensor(np.asarray(rng.integers(0, 2, 8), dtype=np.float32))
        opt = slonet.SloAdam(lr=0.05)
        calls = []
        net.fit(X, y, opt, epochs=1, batch_size=4, on_step=lambda step, loss, ep: calls.append((step, ep)))
        assert len(calls) == 2
        assert calls[0] == (1, 0)


class TestUserAdapters:
    def test_cache_hit(self):
        net = SloNet([SloLinear(4, 2, "lin")])
        a = SloAdapterLayer(dim=4, rank=2, name="adapter_u1")
        net._user_adapters["u1"] = a
        assert net.get_user_adapter("u1", dim=4, rank=2) is a

    def test_fresh_identity_adapter(self):
        net = SloNet([SloLinear(4, 2, "lin")])
        a = net.get_user_adapter("u2", dim=4, rank=2)
        assert isinstance(a, SloAdapterLayer)
        assert np.all(a.up_proj.weight.data == 0)
        assert net._user_adapters["u2"] is a

    def test_load_from_npz(self, monkeypatch, tmp_path):
        net = SloNet([SloLinear(4, 2, "lin")])
        ref = net.get_user_adapter("u3", dim=4, rank=2)
        dw = np.ones_like(ref.down_proj.weight.data)
        uw = np.ones_like(ref.up_proj.weight.data)
        monkeypatch.chdir(tmp_path)
        out_dir = tmp_path / "data" / "user_adapters"
        out_dir.mkdir(parents=True)
        np.savez(str(out_dir / "u3_adapter.npz"), down_weight=dw, up_weight=uw)
        net._user_adapters.clear()
        a = net.get_user_adapter("u3", dim=4, rank=2)
        assert np.allclose(a.down_proj.weight.data, dw)
        assert np.allclose(a.up_proj.weight.data, uw)

    def test_shape_mismatch_keeps_fresh(self, monkeypatch, tmp_path):
        net = SloNet([SloLinear(4, 2, "lin")])
        ref = net.get_user_adapter("u4", dim=4, rank=2)
        dw = np.ones((2, 2), dtype=np.float32)
        uw = np.ones((4, 2), dtype=np.float32)
        monkeypatch.chdir(tmp_path)
        out_dir = tmp_path / "data" / "user_adapters"
        out_dir.mkdir(parents=True)
        np.savez(str(out_dir / "u4_adapter.npz"), down_weight=dw, up_weight=uw)
        net._user_adapters.clear()
        a = net.get_user_adapter("u4", dim=4, rank=2)
        assert a.down_proj.weight.data.shape == ref.down_proj.weight.data.shape
        assert np.all(a.up_proj.weight.data == 0)

    def test_apply_adapter_in_forward(self):
        net = SloNet([SloLinear(4, 4, "lin")])
        net.set_active_user("u5")
        a = net.get_user_adapter("u5", dim=4, rank=2)
        a.down_proj.weight.data = np.ones_like(a.down_proj.weight.data) * 0.1
        a.up_proj.weight.data = np.ones_like(a.up_proj.weight.data) * 0.1
        x = Tensor(np.ones((2, 4), dtype=np.float32))
        out = net.forward(x)
        assert out.data.shape == (2, 4)
        assert np.all(np.isfinite(out.data))


class TestMHAForwardNumpy:
    def test_batch_greater_than_one_einsum(self):
        mha = SloMultiHeadAttention(d_model=16, n_heads=4, n_kv_head=2)
        rng = np.random.default_rng(0)
        q = rng.standard_normal((2, 3, 16)).astype(np.float32)
        k = rng.standard_normal((2, 3, 16)).astype(np.float32)
        v = rng.standard_normal((2, 3, 16)).astype(np.float32)
        out, kv = mha.forward_numpy(q, k, v)
        assert out.shape == (2, 3, 16)
        assert np.all(np.isfinite(out))
        assert kv[0].shape == (2, 3, 4, 4)

    def test_batch_greater_than_one_no_gqa(self):
        mha = SloMultiHeadAttention(d_model=16, n_heads=4, n_kv_head=4)
        rng = np.random.default_rng(3)
        q = rng.standard_normal((2, 3, 16)).astype(np.float32)
        k = rng.standard_normal((2, 3, 16)).astype(np.float32)
        v = rng.standard_normal((2, 3, 16)).astype(np.float32)
        out, kv = mha.forward_numpy(q, k, v)
        assert out.shape == (2, 3, 16)
        assert kv[0].shape == (2, 3, 4, 4)

    def test_single_token_fused_kernel(self):
        mha = SloMultiHeadAttention(d_model=16, n_heads=4, n_kv_head=2, use_rope=True)
        rng = np.random.default_rng(1)
        q = rng.standard_normal((1, 1, 16)).astype(np.float32)
        k = rng.standard_normal((1, 1, 16)).astype(np.float32)
        v = rng.standard_normal((1, 1, 16)).astype(np.float32)
        cache = (np.zeros((1, 3, 2, 4), dtype=np.float32), np.zeros((1, 3, 2, 4), dtype=np.float32))
        out, kv = mha.forward_numpy(q, k, v, kv_cache=cache, start_pos=3)
        assert out.shape == (1, 1, 16)
        assert np.all(np.isfinite(out))
        assert kv[0].shape == (1, 4, 2, 4)

    def test_multi_token_fused_kernel_with_rope_and_cache(self):
        mha = SloMultiHeadAttention(d_model=16, n_heads=4, n_kv_head=2, use_rope=True)
        rng = np.random.default_rng(2)
        q = rng.standard_normal((1, 2, 16)).astype(np.float32)
        k = rng.standard_normal((1, 2, 16)).astype(np.float32)
        v = rng.standard_normal((1, 2, 16)).astype(np.float32)
        cache = (np.zeros((1, 1, 2, 4), dtype=np.float32), np.zeros((1, 1, 2, 4), dtype=np.float32))
        out, kv = mha.forward_numpy(q, k, v, kv_cache=cache, start_pos=1)
        assert out.shape == (1, 2, 16)
        assert kv[0].shape == (1, 3, 2, 4)


class TestLSTMNumbaFallback:
    def test_forward_numba_without_numba(self, monkeypatch):
        monkeypatch.setattr(slonet, "_check_numba", lambda: False)
        net = SloLSTM(vocab_size=16, embed_dim=8, hidden_dim=8, num_layers=1)
        x = np.array([[1, 2, 3]], dtype=np.int64)
        out, hidden = net.forward_numba(x)
        assert out.shape == (1, 16)
        assert hidden[0].shape == (8,)


class TestAccelOpDispatch:
    class _FakeAcc:
        name = "gpu"

        def silu(self, d):
            return d * 2

        def to_device(self, a):
            return a

        def from_device(self, r):
            return r

    def test_dispatch_uses_accel(self, monkeypatch):
        monkeypatch.setattr(slonet, "_get_accelerator", lambda: self._FakeAcc())
        arr = np.array([1.0, 2.0], dtype=np.float32)
        out = _accel_op("silu", arr, lambda d: d, threshold=1)
        assert np.allclose(out, arr * 2)

    def test_missing_acc_fn_falls_back(self, monkeypatch):
        class _NoOp:
            name = "gpu"

        monkeypatch.setattr(slonet, "_get_accelerator", lambda: _NoOp())
        arr = np.array([1.0, 2.0], dtype=np.float32)
        out = _accel_op("silu", arr, lambda d: d * 3, threshold=1)
        assert np.allclose(out, arr * 3)

    def test_accel_exception_falls_back(self, monkeypatch):
        class _Boom:
            name = "gpu"

            def silu(self, d):
                raise RuntimeError("boom")

            def to_device(self, a):
                return a

            def from_device(self, r):
                return r

        monkeypatch.setattr(slonet, "_get_accelerator", lambda: _Boom())
        arr = np.array([1.0, 2.0], dtype=np.float32)
        out = _accel_op("silu", arr, lambda d: d * 3, threshold=1)
        assert np.allclose(out, arr * 3)


class TestNormalize:
    def test_numpy_path(self):
        x = np.array([[1.0, 1.0], [3.0, 4.0]], dtype=np.float32)
        nd = normalize(x)
        assert np.allclose(np.linalg.norm(nd, axis=1), np.ones(2), atol=1e-5)

    def test_tensor_path_backward(self):
        x = Tensor(np.array([[1.0, 2.0]], dtype=np.float32), requires_grad=True)
        out = normalize(x)
        assert isinstance(out, Tensor)
        out.sum().backward()
        assert x.grad is not None
        assert np.all(np.isfinite(x.grad.data))

    def test_zero_norm_no_divzero(self):
        x = Tensor(np.array([[0.0, 0.0]], dtype=np.float32), requires_grad=True)
        out = normalize(x)
        assert np.all(np.isfinite(out.data))


class TestSiluAccelerator:
    def test_accel_success(self, monkeypatch):
        class _A:
            name = "gpu"

            def silu(self, d):
                return d

        monkeypatch.setattr(slonet, "_get_accelerator", lambda: _A())
        t = Tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
        out = silu(t)
        assert np.allclose(out.data, t.data)

    def test_accel_exception_falls_back(self, monkeypatch):
        class _A:
            name = "gpu"

            def silu(self, d):
                raise RuntimeError("boom")

        monkeypatch.setattr(slonet, "_get_accelerator", lambda: _A())
        t = Tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
        out = silu(t)
        assert out.data.shape == (2,)


class TestSoftmaxAccelerator:
    def test_accel_success(self, monkeypatch):
        class _A:
            name = "gpu"

            def softmax(self, d, axis=-1):
                return d * 0 + 0.25

        monkeypatch.setattr(slonet, "_get_accelerator", lambda: _A())
        t = Tensor(np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32))
        out = softmax(t)
        assert isinstance(out, Tensor)
        assert np.allclose(out.data, 0.25)

    def test_accel_exception_falls_back(self, monkeypatch):
        class _A:
            name = "gpu"

            def softmax(self, d, axis=-1):
                raise RuntimeError("boom")

        monkeypatch.setattr(slonet, "_get_accelerator", lambda: _A())
        t = Tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        out = softmax(t)
        assert np.allclose(out.data.sum(axis=-1), np.ones(1), atol=1e-5)


class TestSloCyclicLR:
    def test_down_phase_and_cycle_halving(self):
        opt = slonet.SloAdam(lr=0.1)
        sched = slonet.SloCyclicLR(
            opt, base_lr=0.1, max_lr=0.5, step_size_up=4, step_size_down=6,
            mode="triangular2", last_epoch=-1,
        )
        sched.step(7)
        assert sched.get_last_lr()[0] == pytest.approx(0.3)
        sched.step(17)
        assert sched.get_last_lr()[0] == pytest.approx(0.2)


class TestInvalidateGpuCache:
    def test_runs_with_cpu_backend(self):
        slonet._invalidate_gpu_cache()


class TestExportExceptionBranch:
    def test_rename_failure_cleans_temp_and_reraises(self, monkeypatch, tmp_path):
        net = _small_transformer()
        out = tmp_path / "soul.sou"

        def _boom(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(slonet.os, "rename", _boom)
        with pytest.raises(OSError, match="disk full"):
            export_to_sou(net, str(out))
        assert list(tmp_path.glob("*.tmp")) == []


class TestInt4QuantUnpack:
    def test_lazy_unpack_int4(self):
        from domains.infrastructure.quantization import QuantMeta, TensorInfo

        lin = SloLinear(4, 2, "quant")
        original = np.array([[-8, 2, 3, -4], [5, -6, 7, 1]], dtype=np.int8)
        flat = original.reshape(-1)
        packed = np.array([
            (flat[i] & 0x0F) | ((flat[i + 1] & 0x0F) << 4)
            for i in range(0, len(flat), 2)
        ], dtype=np.int8)
        meta = QuantMeta(
            scale=1.0, zero_point=0, bits=4, mode="symmetric",
            dtype_code=0, original_shape=(2, 4), original_dtype="float32",
        )
        info = TensorInfo(name="w", array=packed, meta=meta)
        lin.set_quantized_weight(info)
        unpacked = lin._get_quant_array()
        assert unpacked.dtype == np.int8
        assert unpacked.shape == (2, 4)
        assert np.array_equal(unpacked, original)
        assert lin._get_quant_array() is unpacked

    def test_no_quant_returns_none(self):
        lin = SloLinear(4, 2, "plain")
        assert lin._get_quant_array() is None


class TestImportFromPoints:
    def test_points_fallback(self, tmp_path):
        import base64

        import json as _json

        from domains.infrastructure.pugqeep.library import PointLibrary
        from domains.infrastructure.pugqeep.point import Point

        base = tmp_path / "soul_test.sou"
        arr = np.arange(256 * 64, dtype=np.float32).reshape(256, 64) / 1000.0
        lib = PointLibrary(name="soul_test")
        lib.add(Point(
            identity="soul_test.tok_emb.weight",
            function_type="raw",
            params={
                "data_b64": base64.b64encode(arr.tobytes()).decode(),
                "shape": (256, 64),
                "dtype": "float32",
            },
        ))
        lib.save(base.with_suffix(".points.json"))
        (base.with_suffix(".meta.json")).write_text(
            _json.dumps({"metadata": {"weight_shapes": {"tok_emb.weight": [256, 64]}}})
        )
        net = import_from_sou(str(base))
        assert isinstance(net, SloTransformer)
        loaded = net.state_dict()["tok_emb.weight"]
        assert np.allclose(loaded, arr)


class TestMultinomial:
    def test_samples_without_replacement(self):
        t = Tensor(np.array([0.1, 0.2, 0.7], dtype=np.float32))
        out = multinomial(t, num_samples=2)
        assert out.data.shape == (1, 2)
        assert out.data.size == 2

    def test_zero_total_uniform(self):
        t = Tensor(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        out = multinomial(t, num_samples=1)
        assert out.data.shape == (1, 1)


class TestTensorScatter:
    def test_scatter_tensor_src(self):
        t = Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
        idx = Tensor(np.array([0, 2]))
        t.scatter_(0, idx, np.array([9.0, 8.0]))
        assert t.data.tolist() == [9.0, 2.0, 8.0, 4.0]
        assert isinstance(t.scatter_(0, idx, np.array([1.0, 1.0])), Tensor)


class TestFeedForward:
    def test_forward_numpy_silu_swiglu(self):
        from domains.training.slonet import SloFeedForward, silu_np

        ff = SloFeedForward(8, 32, name="test_ff", activation="silu")
        x = np.random.randn(3, 8).astype(np.float32)
        manual = ff.w2.forward_numpy(
            silu_np(ff.w1.forward_numpy(x)) * ff.w3.forward_numpy(x)
        )
        assert np.allclose(ff.forward_numpy(x), manual)

    def test_forward_numpy_gelu_swiglu(self):
        from domains.training.slonet import SloFeedForward, gelu_np

        ff = SloFeedForward(8, 32, name="test_ff", activation="gelu")
        x = np.random.randn(3, 8).astype(np.float32)
        manual = ff.w2.forward_numpy(
            gelu_np(ff.w1.forward_numpy(x)) * ff.w3.forward_numpy(x)
        )
        assert np.allclose(ff.forward_numpy(x), manual)

    def test_forward_numpy_unknown_activation_defaults_to_gelu(self):
        from domains.training.slonet import SloFeedForward, gelu_np

        ff = SloFeedForward(8, 32, activation="relu")
        x = np.random.randn(3, 8).astype(np.float32)
        manual = ff.w2.forward_numpy(
            gelu_np(ff.w1.forward_numpy(x)) * ff.w3.forward_numpy(x)
        )
        assert np.allclose(ff.forward_numpy(x), manual)

    def test_forward_tensor_grad_flow(self):
        from domains.training.slonet import SloFeedForward

        ff = SloFeedForward(8, 32, activation="silu")
        x = Tensor(np.random.randn(3, 8).astype(np.float32), requires_grad=True)
        out = ff.forward(x)
        out.backward()
        assert out.data.shape == (3, 8)
        assert np.all(np.isfinite(out.data))
        assert all(np.all(np.isfinite(p.grad.data)) for p in ff.parameters())

    def test_forward_tensor_matches_numpy(self):
        from domains.training.slonet import SloFeedForward

        ff = SloFeedForward(8, 32, activation="silu")
        x_np = np.random.randn(2, 8).astype(np.float32)
        via_tensor = ff.forward(Tensor(x_np, requires_grad=False))
        assert np.allclose(via_tensor.data, ff.forward_numpy(x_np))

    def test_parameters_three_linears(self):
        from domains.training.slonet import SloFeedForward

        ff = SloFeedForward(8, 32, name="test_ff")
        assert len(ff.parameters()) == 6
        assert [p.data.shape for p in ff.parameters()] == [
            (32, 8), (32,), (8, 32), (8,), (32, 8), (32,),
        ]

    def test_default_name(self):
        from domains.training.slonet import SloFeedForward

        assert SloFeedForward(8, 32).name == "FF8"


class _FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)
        self.device = "cpu"

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class TestCheckpointNpz:
    def test_state_dict_to_numpy_handles_fake_tensor(self):
        from domains.training.slonet import _state_dict_to_numpy

        out = _state_dict_to_numpy({"a": _FakeTensor(np.arange(4.0))})
        assert isinstance(out["a"], np.ndarray)
        assert out["a"].tolist() == [0.0, 1.0, 2.0, 3.0]

    def test_state_dict_to_numpy_recurses_nested_dict(self):
        from domains.training.slonet import _state_dict_to_numpy

        out = _state_dict_to_numpy({"mod": {"w": np.array([1, 2])}})
        assert isinstance(out["mod"], dict)
        assert out["mod"]["w"].tolist() == [1, 2]

    def test_state_dict_to_numpy_wraps_scalars(self):
        from domains.training.slonet import _state_dict_to_numpy

        out = _state_dict_to_numpy({"lr": 0.001, "flag": True})
        assert out["lr"].item() == 0.001
        assert bool(out["flag"].item()) is True

    def test_round_trip_with_meta(self, tmp_path):
        from domains.training.slonet import (
            load_checkpoint_npz,
            save_checkpoint_npz,
        )

        path = str(tmp_path / "model")
        weights = {"w": np.arange(24.0).reshape(4, 6), "b": np.array([1.0, 2.0, 3.0])}
        meta = {"loss": 1.5, "nested": {"steps": [1, 2, 3]}}
        saved = save_checkpoint_npz(path, weights, meta=meta)
        assert saved.endswith(".npz")
        loaded = load_checkpoint_npz(saved)
        assert loaded["loss"] == 1.5
        assert loaded["nested"] == {"steps": [1, 2, 3]}
        assert np.allclose(loaded["model_state_dict"]["w"], weights["w"])
        assert np.allclose(loaded["model_state_dict"]["b"], weights["b"])

    def test_round_trip_without_meta(self, tmp_path):
        from domains.training.slonet import (
            load_checkpoint_npz,
            save_checkpoint_npz,
        )

        saved = save_checkpoint_npz(str(tmp_path / "plain.npz"), {"x": np.array([7.0])})
        loaded = load_checkpoint_npz(saved)
        assert loaded["model_state_dict"]["x"].tolist() == [7.0]

    def test_save_accepts_tensor_like_values(self, tmp_path):
        from domains.training.slonet import (
            load_checkpoint_npz,
            save_checkpoint_npz,
        )

        saved = save_checkpoint_npz(str(tmp_path / "t"), {"w": _FakeTensor(np.arange(6.0))})
        loaded = load_checkpoint_npz(saved)
        assert loaded["model_state_dict"]["w"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


class _NamedTensor(Tensor):
    def __init__(self, data, name="", **kw):
        super().__init__(data, **kw)
        self.name = name


class TestSloSGD:
    def test_step_applies_lr(self):
        w = Tensor(np.array([1.0]), requires_grad=True)
        sgd = slonet.SloSGD(lr=0.1)
        w.grad = Tensor(np.array([1.0]))
        sgd.step([w])
        assert w.data[0] == pytest.approx(0.9)
        assert w.grad is None

    def test_momentum_accumulates(self):
        w = Tensor(np.array([1.0]), requires_grad=True)
        sgd = slonet.SloSGD(lr=0.1, momentum=0.9)
        w.grad = Tensor(np.array([1.0]))
        sgd.step([w])
        assert w.data[0] == pytest.approx(0.9)
        w.grad = Tensor(np.array([1.0]))
        sgd.step([w])
        assert w.data[0] == pytest.approx(0.71)

    def test_max_grad_norm_clips(self):
        w = Tensor(np.array([0.0]), requires_grad=True)
        sgd = slonet.SloSGD(lr=1.0, max_grad_norm=1.0)
        w.grad = Tensor(np.array([10.0]))
        sgd.step([w])
        assert w.data[0] == pytest.approx(-1.0)

    def test_skips_params_without_grad(self):
        w = Tensor(np.array([1.0]), requires_grad=True)
        sgd = slonet.SloSGD(lr=0.1)
        sgd.step([w])
        assert w.data[0] == pytest.approx(1.0)

    def test_state_dict_round_trip(self):
        w = _NamedTensor(np.array([1.0]), name="w", requires_grad=True)
        sgd = slonet.SloSGD(lr=0.1, momentum=0.9)
        w.grad = Tensor(np.array([1.0]))
        sgd.step([w])
        state = sgd.state_dict([w])
        assert state["hyperparameters"]["lr"] == 0.1
        assert state["state"]["w"] == [1.0]
        restored = slonet.SloSGD()
        restored.load_state_dict(state, [w])
        assert restored.lr == 0.1 and restored.momentum == 0.9
        assert np.allclose(restored._v[id(w)], [1.0])

    def test_state_dict_without_params(self):
        sgd = slonet.SloSGD(lr=0.05)
        assert sgd.state_dict()["state"] == {}


class TestSloAdam:
    def test_step_lowers_loss_direction(self):
        w = Tensor(np.array([1.0]), requires_grad=True)
        adam = slonet.SloAdam(lr=0.1)
        w.grad = Tensor(np.array([1.0]))
        adam.step([w])
        assert w.data[0] == pytest.approx(0.9)

    def test_weight_decay_adds_to_grad(self):
        w = Tensor(np.array([1.0]), requires_grad=True)
        adam = slonet.SloAdam(lr=0.1, weight_decay=0.5)
        w.grad = Tensor(np.array([0.0]))
        adam.step([w])
        assert w.data[0] == pytest.approx(0.9)

    def test_max_grad_norm_clips(self):
        w = Tensor(np.array([0.0]), requires_grad=True)
        adam = slonet.SloAdam(lr=0.1, max_grad_norm=1.0)
        w.grad = Tensor(np.array([100.0]))
        adam.step([w])
        assert w.data[0] == pytest.approx(-0.1)

    def test_state_dict_round_trip(self):
        w = _NamedTensor(np.array([1.0]), name="w", requires_grad=True)
        adam = slonet.SloAdam(lr=0.01, weight_decay=0.1)
        w.grad = Tensor(np.array([1.0]))
        adam.step([w])
        state = adam.state_dict([w])
        assert state["t"] == 1
        assert set(state["state"]["w"].keys()) == {"m", "v"}
        restored = slonet.SloAdam()
        restored.load_state_dict(state, [w])
        assert restored._t == 1
        assert np.allclose(restored._m[id(w)], state["state"]["w"]["m"])

    def test_second_step_uses_bias_correction(self):
        w = Tensor(np.array([1.0]), requires_grad=True)
        adam = slonet.SloAdam(lr=0.1)
        for _ in range(2):
            w.grad = Tensor(np.array([1.0]))
            adam.step([w])
        assert w.data[0] == pytest.approx(0.8, abs=1e-4)


class TestClipGradNorm:
    def test_clips_to_max_norm(self):
        p = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        p.grad = Tensor(np.array([6.0, 8.0]))
        total = slonet.clip_grad_norm_([p], max_norm=5.0)
        assert total == pytest.approx(10.0)
        assert np.allclose(p.grad.data, [3.0, 4.0])

    def test_returns_zero_for_no_grads(self):
        assert slonet.clip_grad_norm_([Tensor(np.array([1.0]))], 1.0) == 0.0

    def test_no_scale_when_below_max_norm(self):
        p = Tensor(np.array([1.0]), requires_grad=True)
        p.grad = Tensor(np.array([0.5]))
        total = slonet.clip_grad_norm_([p], max_norm=5.0)
        assert total == pytest.approx(0.5)
        assert p.grad.data[0] == pytest.approx(0.5)


class TestLRSchedulers:
    def test_warmup_linear_then_cosine(self):
        sched = slonet.WarmupCosineScheduler(
            slonet.SloAdam(lr=1.0), warmup_steps=10, total_steps=100, last_epoch=-1
        )
        sched.step(0)
        assert sched.get_last_lr()[0] == pytest.approx(0.0)
        sched.step(10)
        assert sched.get_last_lr()[0] == pytest.approx(1.0)
        sched.step(50)
        mid = sched.get_last_lr()[0]
        assert 0.0 < mid < 1.0

    def test_warmup_zero_steps_skips_linear_phase(self):
        sched = slonet.WarmupCosineScheduler(
            slonet.SloAdam(lr=1.0), warmup_steps=0, total_steps=100, last_epoch=-1
        )
        sched.step(50)
        assert sched.get_last_lr()[0] == pytest.approx(0.5 * (1 + 2 ** -0.5))

    def test_polynomial_decay_floor(self):
        sched = slonet.PolynomialDecayScheduler(
            slonet.SloAdam(lr=1.0), total_steps=10, min_lr=0.1, power=2.0, last_epoch=-1
        )
        sched.step(10)
        assert sched.get_last_lr()[0] == pytest.approx(0.1)
        sched.step(5)
        assert sched.get_last_lr()[0] == pytest.approx(0.775)

    def test_linear_warmup_hold_decay(self):
        sched = slonet.LinearWarmupScheduler(
            slonet.SloAdam(lr=0.1), warmup_steps=5, base_lr=0.1, hold_steps=3,
            decay_type="linear", min_lr=0.0, total_steps=15, last_epoch=-1,
        )
        sched.step(2)
        assert sched.get_last_lr()[0] == pytest.approx(0.04)
        sched.step(6)
        assert sched.get_last_lr()[0] == pytest.approx(0.1)
        sched.step(10)
        assert sched.get_last_lr()[0] == pytest.approx(0.1 / 1.4)

    def test_linear_warmup_cosine_decay(self):
        sched = slonet.LinearWarmupScheduler(
            slonet.SloAdam(lr=0.1), warmup_steps=5, base_lr=0.1, hold_steps=0,
            decay_type="cosine", min_lr=0.01, total_steps=15, last_epoch=-1,
        )
        sched.step(10)
        lr = sched.get_last_lr()[0]
        assert 0.01 <= lr <= 0.1

    def test_linear_warmup_no_decay_holds(self):
        sched = slonet.LinearWarmupScheduler(
            slonet.SloAdam(lr=0.1), warmup_steps=5, base_lr=0.1, hold_steps=0,
            decay_type="none", last_epoch=-1,
        )
        sched.step(20)
        assert sched.get_last_lr()[0] == pytest.approx(0.1)

    def test_constant_lr(self):
        sched = slonet.SloConstantLR(slonet.SloAdam(lr=0.3), last_epoch=-1)
        sched.step(5)
        assert sched.get_last_lr()[0] == pytest.approx(0.3)

    def test_one_cycle_up_phase(self):
        sched = slonet.SloOneCycleLR(
            slonet.SloAdam(lr=0.1), max_lr=1.0, total_steps=100,
            pct_start=0.2, anneal_strategy="cos", last_epoch=-1,
        )
        sched.step(10)
        assert sched.get_last_lr()[0] == pytest.approx(1.25)

    def test_one_cycle_linear_anneal(self):
        sched = slonet.SloOneCycleLR(
            slonet.SloAdam(lr=0.1), max_lr=1.0, total_steps=100,
            pct_start=0.2, anneal_strategy="linear", last_epoch=-1,
        )
        sched.step(50)
        assert sched.get_last_lr()[0] == pytest.approx(1.5625, abs=1e-3)

    def test_step_lr_decay(self):
        sched = slonet.SloStepLR(slonet.SloAdam(lr=0.1), step_size=3, gamma=0.5, last_epoch=-1)
        sched.step(6)
        assert sched.get_last_lr()[0] == pytest.approx(0.025)

    def test_cosine_annealing(self):
        sched = slonet.SloCosineAnnealingLR(
            slonet.SloAdam(lr=0.2), T_max=10, eta_min=0.01, last_epoch=-1
        )
        sched.step(5)
        assert sched.get_last_lr()[0] == pytest.approx(0.105)
        sched.step(11)
        assert sched.get_last_lr()[0] == pytest.approx(0.01)

    def test_reduce_lr_on_plateau_patience(self):
        opt = slonet.SloAdam(lr=0.1)
        rp = slonet.SloReduceLROnPlateau(opt, patience=2, factor=0.5, min_lr=0.001)
        for _ in range(4):
            rp.step(1.0)
        assert opt.lr == pytest.approx(0.05)

    def test_reduce_lr_on_plateau_better_resets(self):
        opt = slonet.SloAdam(lr=0.1)
        rp = slonet.SloReduceLROnPlateau(opt, patience=2, factor=0.5, min_lr=0.001)
        rp.step(1.0)
        rp.step(0.5)
        rp.step(0.4)
        rp.step(0.2)
        assert opt.lr == pytest.approx(0.1)

    def test_create_scheduler_factory(self):
        assert isinstance(
            slonet.create_scheduler(slonet.SloAdam(lr=1.0), "none"), slonet.SloConstantLR
        )
        assert isinstance(
            slonet.create_scheduler(slonet.SloAdam(lr=1.0), "constant"), slonet.SloConstantLR
        )
        assert isinstance(
            slonet.create_scheduler(slonet.SloAdam(lr=1.0), "cosine", total_steps=100),
            slonet.WarmupCosineScheduler,
        )
        assert isinstance(
            slonet.create_scheduler(slonet.SloAdam(lr=1.0), "warmup", total_steps=100, max_lr=0.5),
            slonet.LinearWarmupScheduler,
        )

    def test_base_state_dict_round_trip(self):
        sched = slonet.SloCosineAnnealingLR(slonet.SloAdam(lr=0.2), T_max=10, last_epoch=-1)
        sched.step(3)
        state = sched.state_dict()
        assert state["last_epoch"] == 3
        restored = slonet.SloCosineAnnealingLR(slonet.SloAdam(lr=0.2), T_max=10, last_epoch=-1)
        restored.load_state_dict(state)
        assert restored.last_epoch == 3


class TestTensorUtilOps:
    def test_argmax_argmin(self):
        x = Tensor(np.array([[0.5, 0.2], [0.1, 0.9]], dtype=np.float32))
        assert slonet.argmax(x).data.tolist() == [0.0, 1.0]
        assert slonet.argmin(x).data.tolist() == [1.0, 0.0]
        assert slonet.argmax(x).data.dtype == np.float32

    def test_squeeze_tensor_and_ndarray(self):
        assert slonet.squeeze(Tensor(np.array([[[1.0], [2.0]]]))).data.shape == (2,)
        assert slonet.squeeze(Tensor(np.array([[1.0], [2.0]])), dim=1).data.shape == (2,)
        assert slonet.squeeze(np.array([[1.0, 2.0]]), dim=0).shape == (2,)
        assert slonet.squeeze(np.array([[[1.0, 2.0]]])).shape == (2,)

    def test_unsqueeze_tensor_and_ndarray(self):
        assert slonet.unsqueeze(Tensor(np.array([1.0, 2.0])), 0).data.shape == (1, 2)
        assert slonet.unsqueeze(Tensor(np.array([1.0, 2.0])), 1).data.shape == (2, 1)
        assert slonet.unsqueeze(np.array([1.0, 2.0]), 1).shape == (2, 1)

    def test_cat_and_concatenate(self):
        a = Tensor(np.array([1.0, 2.0]))
        b = Tensor(np.array([3.0, 4.0]))
        assert slonet.cat([a, b]).data.tolist() == [1.0, 2.0, 3.0, 4.0]
        assert slonet.concatenate([a, b], dim=0).data.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_eye(self):
        assert np.array_equal(slonet.eye(3).data, np.eye(3, dtype=np.float32))
        assert slonet.eye(2, 3).data.shape == (2, 3)

    def test_stack(self):
        out = slonet.stack([Tensor(np.array([1.0, 2.0])), Tensor(np.array([3.0, 4.0]))])
        assert out.data.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_exp(self):
        assert slonet.exp(Tensor(np.array([0.0, np.log(2.0)]))).data.tolist() == pytest.approx([1.0, 2.0])

    def test_where(self):
        cond = Tensor(np.array([True, False]))
        a = Tensor(np.array([1.0, 2.0]))
        b = Tensor(np.array([9.0, 8.0]))
        assert slonet.where(cond, a, b).data.tolist() == [1.0, 8.0]

    def test_normalize_and_backward(self):
        a = Tensor(np.array([[3.0, 4.0]]), requires_grad=True)
        out = normalize(a)
        assert np.allclose(out.data[0], [0.6, 0.8])
        out.backward()
        assert np.allclose(a.grad.data, [0.2, 0.2])

    def test_normalize_zero_vector_safe(self):
        a = Tensor(np.array([[0.0, 0.0]]))
        out = normalize(a)
        assert np.allclose(out.data, [[0.0, 0.0]])

    def test_pairwise_distance_backward(self):
        a = Tensor(np.array([[0.0, 0.0]]), requires_grad=True)
        b = Tensor(np.array([[3.0, 4.0]]), requires_grad=True)
        out = pairwise_distance(a, b)
        assert out.data[0] == pytest.approx(5.0)
        out.backward()
        assert np.allclose(a.grad.data, [[-0.6, -0.8]])
        assert np.allclose(b.grad.data, [[0.6, 0.8]])


class _FakeAcc:
    name = "metal"

    def matmul(self, a, b):
        raise RuntimeError("boom")

    def gelu(self, d):
        raise RuntimeError("boom")

    def layer_norm(self, *a):
        raise RuntimeError("boom")


class TestTensorTypeAndJvp:
    def test_type_casts(self):
        t = Tensor(np.array([1.0, 2.0]))
        t.type(np.float64)
        assert t.data.dtype == np.float64
        t.type("torch.FloatTensor")
        assert t.data.dtype == np.float32
        t.type(np.float32)
        assert t.data.dtype == np.float32
        t.type("bogus")
        assert t.data.dtype == np.float32

    def test_jvp_and_forward_grad(self):
        x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        y = x * x
        z = y + y
        res = z.forward_grad({x.id: np.array([1.0, 1.0])})
        assert np.allclose(res[x.id], [1.0, 1.0])
        assert np.allclose(res[y.id], [4.0, 6.0])
        assert np.allclose(res[z.id], [8.0, 12.0])
        j = z.jvp(Tensor(np.array([0.5, 0.5])))
        assert np.allclose(j.data, [0.5, 0.5])


class TestModuleTensorFuncs:
    def test_topk(self):
        t = Tensor(np.array([3.0, 1.0, 4.0, 2.0]))
        vals, idx = slonet.topk(t, 2)
        assert vals.shape == (1, 2)
        assert idx.shape == (1, 2)
        assert vals.data[0, 0] == 4.0

    def test_concatenate_where_stack(self):
        a = Tensor(np.ones((2, 3)))
        b = Tensor(np.zeros((2, 3)))
        c = slonet.concatenate([a, b], dim=-1)
        assert c.shape == (2, 6)
        w = slonet.where(
            Tensor(np.array([True, False])),
            Tensor(np.array([1.0, 2.0])),
            Tensor(np.array([3.0, 4.0])),
        )
        assert np.array_equal(w.data, [1.0, 4.0])
        s = slonet.stack([a, b])
        assert s.shape == (2, 2, 3)

    def test_squeeze_unsqueeze_tensor(self):
        x = Tensor(np.array([[1.0, 2.0]]))
        assert slonet.squeeze(x).shape == (2,)
        assert slonet.unsqueeze(x, 0).shape == (1, 1, 2)
        assert slonet.squeeze(np.array([[1.0, 2.0]])).shape == (2,)
        assert slonet.unsqueeze(np.array([1.0, 2.0]), 0).shape == (1, 2)


class TestAccelFallbacks:
    def test_matmul_fallback(self, monkeypatch):
        monkeypatch.setattr(slonet, "_get_accelerator", lambda: _FakeAcc())
        rng = np.random.default_rng(0)
        a = rng.standard_normal((300, 300)).astype(np.float32)
        b = rng.standard_normal((300, 300)).astype(np.float32)
        out = slonet._matmul(a, b)
        assert np.allclose(out.data, a @ b)

    def test_gelu_fallback(self, monkeypatch):
        monkeypatch.setattr(slonet, "_get_accelerator", lambda: _FakeAcc())
        out = slonet.gelu(Tensor(np.array([0.0, 1.0, -1.0])))
        assert np.allclose(out.data, [0.0, 0.84119, -0.15881], atol=1e-3)

    def test_state_dict_acc_fallbacks(self, monkeypatch):
        monkeypatch.setattr(slonet, "_get_accelerator", lambda: _FakeAcc())
        a = Tensor(np.random.randn(5, 8).astype(np.float32))
        b = np.random.randn(8, 3).astype(np.float32)
        r1 = slonet._matmul_state_dict(a, b)
        assert np.allclose(r1.data, a.data @ b)
        r2 = slonet._layernorm_state_dict(a, np.ones(8))
        assert r2.shape == (5, 8)

    def test_conv2d_fallback(self, monkeypatch):
        monkeypatch.setattr(slonet, "_get_accelerator", lambda: _FakeAcc())
        x = Tensor(np.random.randn(1, 1, 8, 8).astype(np.float32))
        w = Tensor(np.random.randn(2, 1, 3, 3).astype(np.float32))
        b = Tensor(np.random.randn(2).astype(np.float32))
        out = slonet._conv2d(x, w, b, stride=1, padding=1)
        assert out.shape == (1, 2, 8, 8)


class TestEmbedding3DInput:
    def test_squeeze_axis_1(self):
        emb = SloEmbedding(32, 16)
        out = emb.forward_numpy(np.array([[[1, 2, 3]]], dtype=np.int64))
        assert out.shape == (1, 3, 16)

    def test_squeeze_axis_2(self):
        emb = SloEmbedding(32, 16)
        out = emb.forward_numpy(np.array([[[1], [2], [3]]], dtype=np.int64))
        assert out.shape == (1, 3, 16)

    def test_reshape_3d_fallback(self):
        emb = SloEmbedding(32, 16)
        out = emb.forward_numpy(np.array([[[1, 2], [3, 4]]], dtype=np.int64))
        assert out.shape == (1, 4, 16)


class TestLSTMShapesAndAdapter:
    def _lstm(self):
        return SloLSTM(vocab_size=32, embed_dim=16, hidden_dim=8, num_layers=1, dropout=0.0)

    def test_skip_embed_2d(self):
        lstm = self._lstm()
        out, (h, c) = lstm.forward_numpy(np.random.randn(4, 16).astype(np.float32), skip_embed=True)
        assert out.shape == (1, 32)
        assert h.shape == (8,)

    def test_skip_embed_3d(self):
        lstm = self._lstm()
        out, (h, c) = lstm.forward_numpy(np.random.randn(2, 4, 16).astype(np.float32), skip_embed=True)
        assert out.shape == (1, 32)

    def test_adapter_applied(self):
        lstm = self._lstm()
        adapter = SloAdapterLayer(dim=8, rank=2)
        out, _ = lstm.forward_numpy(np.random.randn(4, 16).astype(np.float32), adapter=adapter, skip_embed=True)
        assert out.shape == (1, 32)

    def test_zero_grad_clears_grads(self):
        lstm = self._lstm()
        for p in lstm.parameters():
            p.grad = Tensor(np.ones_like(p.data))
        lstm.zero_grad()
        assert all(p.grad is None for p in lstm.parameters())


class TestSloNetStateDict:
    def test_state_dict_and_load_weights(self):
        net = SloNet(layers=[SloLinear(4, 2)])
        sd = net.state_dict()
        assert set(sd) == {"p0", "p1"}
        net._load_weights({"p0": np.zeros((2, 4), dtype=np.float32)})
        assert np.allclose(net.layers[0].weight.data, 0)

    def test_empty_load_noop(self):
        SloNet()._load_weights({})


class TestCrossAttentionJvp:
    def test_forward_grad_through_einsum(self):
        ca = slonet.SloCrossAttention(d_model=16, n_heads=4)
        rng = np.random.default_rng(0)
        x = Tensor(rng.standard_normal((2, 3, 16)).astype(np.float32), requires_grad=True)
        ctx = Tensor(rng.standard_normal((2, 5, 16)).astype(np.float32), requires_grad=True)
        out = ca.forward(x, ctx)
        tangents = out.forward_grad({
            x.id: np.ones_like(x.data),
            ctx.id: np.ones_like(ctx.data),
        })
        assert out.id in tangents
        assert np.all(np.isfinite(tangents[out.id]))


class TestKLDivGradBranch:
    def test_backward_batchmean(self):
        ilp = Tensor(np.log(np.array([[0.5, 0.5], [0.2, 0.8]])), requires_grad=True)
        tp = np.array([[0.3, 0.7], [0.6, 0.4]])
        loss = kl_div_loss(ilp, tp, reduction="batchmean")
        loss.backward()
        assert ilp.grad is not None
        assert np.allclose(ilp.grad.data, -tp / 2)

    def test_backward_sum_reduction(self):
        ilp = Tensor(np.log(np.array([[0.5, 0.5]])), requires_grad=True)
        tp = np.array([[0.3, 0.7]])
        loss = kl_div_loss(ilp, tp, reduction="sum")
        loss.backward()
        assert ilp.grad is not None
        assert np.allclose(ilp.grad.data, -tp)


class TestLayerNormTransformerGeneration:
    def _layer_norm_transformer(self):
        return SloTransformer(
            vocab_size=16, n_embed=16, n_layer=1, n_head=2,
            block_size=16, norm_type="layer_norm",
        )

    def test_generate_numpy_layernorm(self):
        m = self._layer_norm_transformer()
        prompt = np.array([[1, 2, 3, 4]])
        out = m.generate_numpy(prompt, max_new_tokens=3, temperature=0.0)
        assert out.shape == (1, 7)

    def test_generate_numpy_stream_layernorm(self):
        m = self._layer_norm_transformer()
        prompt = np.array([[1, 2, 3, 4]])
        toks = list(m.generate_numpy_stream(prompt, max_new_tokens=3, temperature=0.0))
        assert len(toks) == 3


class _MockOpt:
    def __init__(self, lr=1.0):
        self.lr = lr


class _ParamGroupsOpt:
    def __init__(self, lr=0.2):
        self.param_groups = [{"lr": lr}]


class _EmptyGroupsOpt:
    def __init__(self):
        self.param_groups = []


class _BareScheduler(slonet.SloLRScheduler):
    pass


class _RangeDataset(slonet.SloDataset):
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return idx * 10


class TestSloDatasetAbstract:
    def test_abstract_len_raises(self):
        with pytest.raises(NotImplementedError):
            len(slonet.SloDataset())

    def test_abstract_getitem_raises(self):
        with pytest.raises(NotImplementedError):
            slonet.SloDataset()[0]

    def test_iter_yields_items(self):
        assert list(iter(_RangeDataset())) == [0, 10, 20]


class TestKLDivMeanReduction:
    def test_mean_reduction_forward(self):
        ilp = np.log(np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32))
        tp = np.array([[0.3, 0.7], [0.6, 0.4]], dtype=np.float32)
        loss = kl_div_loss(ilp, tp, reduction="mean")
        batchmean = kl_div_loss(ilp, tp, reduction="batchmean")
        assert loss.data == pytest.approx(batchmean.data)
        assert loss.data == pytest.approx(0.178060, abs=1e-5)

    def test_mean_reduction_backward_no_scale(self):
        ilp = Tensor(np.log(np.array([[0.5, 0.5], [0.25, 0.75]])), requires_grad=True)
        tp = np.array([[0.3, 0.7], [0.6, 0.4]])
        loss = kl_div_loss(ilp, tp, reduction="mean")
        loss.backward()
        assert np.allclose(ilp.grad.data, -tp)

    def test_ndarray_input_no_backward_grad(self):
        ilp = np.log(np.array([[0.5, 0.5]], dtype=np.float32))
        tp = np.array([[0.3, 0.7]], dtype=np.float32)
        loss = kl_div_loss(ilp, tp)
        loss.backward()
        assert loss._children == ()


class TestSchedulerBaseEdges:
    def test_param_groups_only_optimizer(self):
        opt = _ParamGroupsOpt(lr=0.2)
        sched = slonet.SloConstantLR(opt, last_epoch=-1)
        assert sched.base_lrs == [0.2]
        assert opt.lr == 0.2

    def test_no_lr_no_param_groups_defaults_zero(self):
        opt = _EmptyGroupsOpt()
        sched = slonet.SloConstantLR(opt, last_epoch=-1)
        assert sched.base_lrs == [0.0]
        assert opt.lr == 0.0

    def test_abstract_get_lr_raises(self):
        sched = _BareScheduler(_MockOpt(lr=1.0), last_epoch=0)
        with pytest.raises(NotImplementedError):
            sched.get_lr()

    def test_get_last_lr_fallback_to_get_lr(self):
        opt = _MockOpt(lr=0.1)
        sched = slonet.SloConstantLR(opt, last_epoch=0)
        assert sched.get_last_lr() == [0.1]

    def test_step_with_explicit_epoch_sets_lr(self):
        opt = _MockOpt(lr=0.1)
        sched = slonet.SloConstantLR(opt, last_epoch=0)
        sched.step(5)
        assert sched.last_epoch == 5
        assert opt.lr == 0.1
        assert sched.get_last_lr() == [0.1]


class TestSchedulerFactoryBranches:
    def test_onecycle_branch(self):
        assert isinstance(
            slonet.create_scheduler(slonet.SloAdam(lr=1.0), "onecycle", total_steps=100),
            slonet.SloOneCycleLR,
        )

    def test_cyclic_branch(self):
        assert isinstance(
            slonet.create_scheduler(slonet.SloAdam(lr=1.0), "cyclic", total_steps=100),
            slonet.SloCyclicLR,
        )

    def test_polynomial_branch(self):
        assert isinstance(
            slonet.create_scheduler(slonet.SloAdam(lr=1.0), "polynomial", total_steps=100),
            slonet.PolynomialDecayScheduler,
        )

    def test_step_branch(self):
        assert isinstance(
            slonet.create_scheduler(slonet.SloAdam(lr=1.0), "step", total_steps=100),
            slonet.SloStepLR,
        )

    def test_plateau_branch(self):
        assert isinstance(
            slonet.create_scheduler(slonet.SloAdam(lr=1.0), "plateau", total_steps=100),
            slonet.SloReduceLROnPlateau,
        )

    def test_cosine_annealing_branch(self):
        assert isinstance(
            slonet.create_scheduler(slonet.SloAdam(lr=1.0), "cosine_annealing", total_steps=100),
            slonet.SloCosineAnnealingLR,
        )

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            slonet.create_scheduler(slonet.SloAdam(lr=1.0), "bogus", total_steps=100)


class _TorchLike:
    """Minimal torch-like tensor shim (cpu/detach/numpy) for compat paths."""

    def __init__(self, arr):
        self._a = arr

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self._a


class _GoodAcc:
    name = "metal"

    def layer_norm(self, d, w, b, eps):
        mu = d.mean(axis=-1, keepdims=True)
        var = d.var(axis=-1, keepdims=True)
        return (d - mu) / np.sqrt(var + eps) * w + b

    def rms_norm(self, d, w, eps):
        return d / np.sqrt(np.mean(d**2, axis=-1, keepdims=True) + eps) * w


class TestTensorCompatMethods:
    def test_comparisons_scalar_and_tensor(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal((t > 2).data, [0, 0, 1])
        np.testing.assert_array_equal((t < 2).data, [1, 0, 0])
        np.testing.assert_array_equal((t >= 2).data, [0, 1, 1])
        np.testing.assert_array_equal((t <= 2).data, [1, 1, 0])
        o = Tensor(np.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal((t == o).data, [1, 0, 0])
        np.testing.assert_array_equal((t != o).data, [0, 1, 1])
        np.testing.assert_array_equal((t > o).data, [0, 1, 1])
        np.testing.assert_array_equal((t < o).data, [0, 0, 0])
        np.testing.assert_array_equal((t >= o).data, [1, 1, 1])
        np.testing.assert_array_equal((t <= o).data, [1, 0, 0])

    def test_named_comparison_and_aggregate_methods(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        o = Tensor(np.array([2.0, 2.0, 2.0]))
        np.testing.assert_array_equal(t.eq(o).data, [0, 1, 0])
        np.testing.assert_array_equal(t.ne(o).data, [1, 0, 1])
        np.testing.assert_array_equal(t.gt(o).data, [0, 0, 1])
        np.testing.assert_array_equal(t.lt(o).data, [1, 0, 0])
        np.testing.assert_array_equal(t.ge(o).data, [0, 1, 1])
        np.testing.assert_array_equal(t.le(o).data, [1, 1, 0])
        np.testing.assert_array_equal(t.all().data, [1.0])
        np.testing.assert_array_equal(t.any().data, [1.0])
        np.testing.assert_array_equal(t.all(dim=0).data, [1.0])

    def test_bool_and_len(self):
        assert bool(Tensor(np.array(1.0)))
        assert not bool(Tensor(np.array(0.0)))
        with pytest.raises(RuntimeError, match="ambiguous"):
            bool(Tensor(np.array([1.0, 2.0])))
        assert len(Tensor(np.array([1.0, 2.0, 3.0]))) == 3
        with pytest.raises(TypeError):
            len(Tensor(np.array(1.0)))

    def test_t_method(self):
        m = Tensor(np.arange(6.0).reshape(2, 3))
        assert m.t().shape == (3, 2)
        with pytest.raises(RuntimeError, match="2D"):
            Tensor(np.zeros((2, 2, 2))).t()

    def test_type_casts_and_mutators(self):
        i = Tensor(np.array([1, 2], dtype=np.int64))
        assert i.float().data.dtype == np.float32
        z = Tensor(np.ones(3, dtype=np.float32))
        assert z.zero_() is z
        np.testing.assert_array_equal(z.data, np.zeros(3))
        z.fill_(7.0)
        np.testing.assert_array_equal(z.data, np.full(3, 7.0))
        dst = Tensor(np.zeros(3, dtype=np.float32))
        dst.copy_(np.array([9.0, 8.0, 7.0]))
        np.testing.assert_array_equal(dst.data, [9, 8, 7])
        dst.copy_(Tensor(np.array([1.0, 2.0, 3.0])))
        np.testing.assert_array_equal(dst.data, [1, 2, 3])
        np.testing.assert_array_equal(Tensor(np.array([1.0, -2.0])).abs().data, [1, 2])
        np.testing.assert_array_equal(Tensor(np.array([1.0, 2.0])).expand(2, 2).data, [[1, 2], [1, 2]])
        np.testing.assert_array_equal(Tensor(np.arange(6.0).reshape(2, 3)).transpose(0, 1).data.shape, (3, 2))
        np.testing.assert_array_equal(Tensor(np.arange(6.0).reshape(2, 3)).permute(1, 0).data.shape, (3, 2))
        np.testing.assert_array_equal(Tensor(np.array([3.0, 1.0, 2.0])).argsort().data, [1, 2, 0])
        np.testing.assert_array_equal(Tensor(np.array([3.0, 1.0, 2.0])).argsort(descending=True).data, [0, 2, 1])
        rg = Tensor(np.ones(2))
        assert not rg.requires_grad
        assert rg.requires_grad_() is rg
        assert rg.requires_grad


class TestMiscTensorBackwardForwardGrad:
    def test_slice_basic_and_fancy_backward(self):
        x = Tensor(np.array([1.0, 2.0, 3.0, 4.0]), requires_grad=True)
        s = x[1:3]
        s.grad = Tensor(np.array([10.0, 20.0]))
        s.backward()
        np.testing.assert_array_equal(x.grad.data, [0, 10, 20, 0])
        x2 = Tensor(np.array([1.0, 2.0, 3.0, 4.0]), requires_grad=True)
        s2 = x2[[0, 2]]
        s2.grad = Tensor(np.array([7.0, 8.0]))
        s2.backward()
        np.testing.assert_array_equal(x2.grad.data, [7, 0, 8, 0])
        res = s.forward_grad({x.id: np.array([1.0, 1.0, 1.0, 1.0])})
        np.testing.assert_array_equal(res[x.id], np.array([1.0, 1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(res[s.id], np.array([1.0, 1.0]))

    def test_max_and_sum_forward_grad(self):
        x = Tensor(np.array([1.0, 5.0, 3.0, 5.0]), requires_grad=True)
        m = x.max()
        m.backward()
        np.testing.assert_array_equal(x.grad.data, [0, 1, 0, 1])
        res = m.forward_grad({x.id: np.array([0.0, 1.0, 0.0, 0.0])})
        np.testing.assert_array_equal(res[x.id], np.array([0.0, 1.0, 0.0, 0.0]))
        s = x.sum()
        sres = s.forward_grad({x.id: np.array([1.0, 1.0, 1.0, 1.0])})
        np.testing.assert_array_equal(sres[x.id], np.array([1.0, 1.0, 1.0, 1.0]))

    def test_randint_and_no_grad_decorator(self):
        t = slonet.randint(0, 10, (2, 3))
        assert t.shape == (2, 3)
        assert t.data.dtype == np.float32
        assert (t.data >= 0).all() and (t.data < 10).all()
        called = []
        fn = slonet._NoGrad()(lambda: called.append(1) or 42)
        assert fn() == 42
        assert called == [1]

    def test_conv_tuple_padding_backward_and_jvp(self):
        rng = np.random.RandomState(0)
        x = Tensor(rng.randn(1, 2, 6, 6).astype(np.float32), requires_grad=True)
        w = Tensor(rng.randn(3, 2, 3, 3).astype(np.float32), requires_grad=True)
        b = Tensor(rng.randn(3).astype(np.float32), requires_grad=True)
        out = slonet._conv2d(x, w, b, stride=1, padding=(1, 1))
        assert out.data.shape == (1, 3, 6, 6)
        out.backward()
        assert x.grad is not None
        assert w.grad is not None
        assert b.grad is not None
        assert np.all(np.isfinite(x.grad.data))
        res = out.forward_grad({x.id: np.ones((1, 2, 6, 6), dtype=np.float32)})
        assert x.id in res


class TestNormAccAndNd:
    def test_layernorm_3d_and_acc_branches(self, monkeypatch):
        ln = slonet.SloLayerNorm(16)
        x = Tensor(np.random.RandomState(0).randn(2, 5, 16).astype(np.float32), requires_grad=True)
        monkeypatch.setattr(slonet, "_get_accelerator", lambda: _GoodAcc())
        out = ln.forward(x)
        assert out.data.shape == (2, 5, 16)
        out.backward()
        assert x.grad is not None
        monkeypatch.setattr(slonet, "_get_accelerator", lambda: _FakeAcc())
        out2 = ln.forward(x)
        assert np.all(np.isfinite(out2.data))
        out2.backward()

    def test_rmsnorm_acc_branches(self, monkeypatch):
        rn = slonet.SloRMSNorm(64)
        x = Tensor(np.random.RandomState(1).randn(2, 64, 64).astype(np.float32), requires_grad=True)
        monkeypatch.setattr(slonet, "_get_accelerator", lambda: _GoodAcc())
        out = rn.forward(x)
        out.backward()
        assert x.grad is not None
        monkeypatch.setattr(slonet, "_get_accelerator", lambda: _FakeAcc())
        out2 = rn.forward(x)
        assert np.all(np.isfinite(out2.data))
        out2.backward()

    def test_rmsnorm_forward_numpy(self):
        rn = slonet.SloRMSNorm(8)
        out = rn.forward_numpy(np.random.randn(3, 8).astype(np.float32))
        assert out.shape == (3, 8)

    def test_layernorm_forward_numpy_both_paths(self, monkeypatch):
        ln = slonet.SloLayerNorm(8)
        x = np.random.randn(3, 8).astype(np.float32)
        with_kernels = ln.forward_numpy(x)
        assert with_kernels.shape == (3, 8)
        monkeypatch.setattr(slonet, "_KERNELS_AVAILABLE", False)
        no_kernels = ln.forward_numpy(x)
        np.testing.assert_allclose(with_kernels, no_kernels, atol=1e-4)


class TestQuantizedLinearForward:
    def test_int8_quantized_forward(self):
        from domains.infrastructure.quantization import Quantine
        lin = SloLinear(8, 4, "q8")
        info = Quantine(bits=8, mode="symmetric").quantize("w", lin.weight.data)
        lin.set_quantized_weight(info)
        x = np.random.randn(3, 8).astype(np.float32)
        f = lin.forward_numpy(x)
        t = lin.forward(Tensor(x))
        assert f.shape == (3, 4)
        np.testing.assert_allclose(f, t.data, atol=1e-3)

    def test_int4_quantized_forward(self):
        from domains.infrastructure.quantization import Quantine
        lin = SloLinear(8, 4, "q4")
        info = Quantine(bits=4, mode="symmetric").quantize("w", lin.weight.data)
        lin.set_quantized_weight(info)
        x = np.random.randn(3, 8).astype(np.float32)
        f = lin.forward_numpy(x)
        t = lin.forward(Tensor(x))
        assert f.shape == (3, 4)
        np.testing.assert_allclose(f, t.data, atol=1e-3)


class TestSloNetMiscMethods:
    def test_train_eval_and_checkpointing(self):
        m = _small_transformer()
        m.train(False)
        m.eval()
        m.train()
        m.apply_gradient_checkpointing()
        assert m.layers[2].use_checkpoint
        assert m.num_parameters() > 0
        assert m.named_modules() == [("", m)]
        children = m.named_children()
        assert len(children) == len(m.layers)
        assert m._get_weights_dict()
        assert "soul_name" in m.soul_signature()

    def test_get_user_adapter_disk_error(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        import pathlib
        target = pathlib.Path("data/user_adapters")
        target.mkdir(parents=True, exist_ok=True)
        bad = target / "opencode_bad_adapter.npz"
        bad.write_bytes(b"not a zip archive")
        net = slonet.SloNet()
        adapter = net.get_user_adapter("opencode_bad", dim=768, rank=8)
        assert adapter is not None
        assert adapter is net.get_user_adapter("opencode_bad")
        net.remove_user_adapter("opencode_bad")
        assert "opencode_bad" not in net._user_adapters

    def test_invalidate_gpu_cache(self, monkeypatch):
        class _HasClear:
            def clear_cache(self):
                return None
        monkeypatch.setattr("domains.slolib.gpu.get_accelerator", lambda: _HasClear())
        slonet._invalidate_gpu_cache()
        monkeypatch.setattr("domains.slolib.gpu.get_accelerator", _boom if False else (lambda: (_ for _ in ()).throw(RuntimeError("no gpu"))))
        slonet._invalidate_gpu_cache()


class TestTransformerCompatInputs:
    def test_forward_torchlike_and_targets(self):
        m = _small_transformer()
        arr = np.array([[1, 2, 3]])
        logits, loss = m.forward(_TorchLike(arr), targets=_TorchLike(arr))
        assert logits.data.shape == (1, 3, 32)
        assert loss.data.ndim == 0
        logits2, loss2 = m.forward(np.array([[1, 2, 3]]), targets=np.array([[1, 2, 3]]))
        assert logits2.data.shape == (1, 3, 32)
        assert loss2 is not None
        logits3, none3 = m.forward(np.array([[1, 2, 3]]))
        assert none3 is None

    def test_forward_pos_emb(self):
        m = SloTransformer(vocab_size=32, n_embed=16, n_layer=1, n_head=2,
                           block_size=16, max_seq_len=32, use_abs_pos_emb=True)
        logits, _ = m.forward(np.array([[1, 2, 3]]))
        assert logits.data.shape == (1, 3, 32)

    def test_forward_pass(self):
        m = _small_transformer()
        r1 = m.forward_pass(np.array([1, 2, 3]))
        assert r1.logits.shape == (1, 3, 32)
        r2 = m.forward_pass(np.array([[1, 2, 3]]))
        assert r2.logits.shape == (1, 3, 32)

    def test_generate_torchlike_and_pos_emb(self):
        m = _small_transformer()
        out = m.generate(_TorchLike(np.array([[1, 2, 3]])), max_new_tokens=2)
        assert isinstance(out, Tensor)
        assert out.data.shape[1] == 5
        mp = SloTransformer(vocab_size=32, n_embed=16, n_layer=1, n_head=2,
                            block_size=16, max_seq_len=32, use_abs_pos_emb=True)
        outp = mp.generate(np.array([[1, 2, 3]]), max_new_tokens=2)
        assert outp.data.shape[1] == 5

    def test_forward_state_dict_legacy_fallbacks(self):
        sd = {}
        hidden, ff_dim = 32, 64
        sd["tok_emb.weight"] = np.random.RandomState(1).randn(16, hidden).astype(np.float32)
        sd["blocks.0.norm1.weight"] = np.ones(hidden, dtype=np.float32)
        sd["blocks.0.q_proj.weight"] = np.random.RandomState(2).randn(hidden, hidden).astype(np.float32)
        sd["blocks.0.k_proj.weight"] = np.random.RandomState(3).randn(hidden, hidden).astype(np.float32)
        sd["blocks.0.v_proj.weight"] = np.random.RandomState(4).randn(hidden, hidden).astype(np.float32)
        sd["blocks.0.proj.weight"] = np.random.RandomState(5).randn(hidden, hidden).astype(np.float32)
        sd["blocks.0.norm2.weight"] = np.ones(hidden, dtype=np.float32)
        sd["blocks.0.w1.weight"] = np.random.RandomState(6).randn(ff_dim, hidden).astype(np.float32)
        sd["blocks.0.w2.weight"] = np.random.RandomState(7).randn(hidden, ff_dim).astype(np.float32)
        sd["blocks.0.w3.weight"] = np.random.RandomState(8).randn(ff_dim, hidden).astype(np.float32)
        sd["norm.weight"] = np.ones(hidden, dtype=np.float32)
        sd["lm_head.weight"] = np.random.RandomState(9).randn(16, hidden).astype(np.float32)
        net = slonet.SloNet()
        net._rebuild_from_state_dict(sd)
        out = net.forward(slonet.Tensor(np.array([[1, 2, 3, 4]], dtype=np.float32)))
        assert isinstance(out, slonet.Tensor)
        assert out.data.shape == (1, 4, 16)
        assert np.all(np.isfinite(out.data))


class TestGenerateNumpyNoKernels:
    def _ln_model(self):
        return SloTransformer(vocab_size=32, n_embed=16, n_layer=1, n_head=2,
                              block_size=16, norm_type="layer_norm")

    def test_generate_numpy_layer_norm_fallback(self, monkeypatch):
        monkeypatch.setattr(slonet, "_KERNELS_AVAILABLE", False)
        out = self._ln_model().generate_numpy(np.array([[1, 2, 3]]), max_new_tokens=3, temperature=0.0)
        assert out.shape == (1, 6)

    def test_generate_numpy_rms_norm_fallback(self, monkeypatch):
        monkeypatch.setattr(slonet, "_KERNELS_AVAILABLE", False)
        out = _small_transformer().generate_numpy(np.array([[1, 2, 3]]), max_new_tokens=3, temperature=0.0)
        assert out.shape == (1, 6)

    def test_generate_numpy_pos_emb(self):
        m = SloTransformer(vocab_size=32, n_embed=16, n_layer=1, n_head=2,
                           block_size=16, max_seq_len=32, use_abs_pos_emb=True)
        out = m.generate_numpy(np.array([[1, 2, 3]]), max_new_tokens=3, temperature=0.0)
        assert out.shape == (1, 6)

    def test_generate_numpy_stream_layer_norm_fallback(self, monkeypatch):
        monkeypatch.setattr(slonet, "_KERNELS_AVAILABLE", False)
        toks = list(self._ln_model().generate_numpy_stream(np.array([[1, 2, 3]]), max_new_tokens=3, temperature=0.0))
        assert len(toks) == 3

    def test_generate_numpy_stream_rms_norm_fallback(self, monkeypatch):
        monkeypatch.setattr(slonet, "_KERNELS_AVAILABLE", False)
        toks = list(_small_transformer().generate_numpy_stream(np.array([[1, 2, 3]]), max_new_tokens=3, temperature=0.0))
        assert len(toks) == 3

    def test_generate_numpy_stream_gqa_kernel_and_fallback(self, monkeypatch):
        m = SloTransformer(vocab_size=32, n_embed=16, n_layer=1, n_head=4, n_kv_head=2,
                           block_size=16, tie_weights=False)
        toks = list(m.generate_numpy_stream(np.array([[1, 2, 3]]), max_new_tokens=4, temperature=0.0))
        assert len(toks) == 4
        monkeypatch.setattr(slonet, "_KERNELS_AVAILABLE", False)
        toks2 = list(m.generate_numpy_stream(np.array([[1, 2, 3]]), max_new_tokens=4, temperature=0.0))
        assert len(toks2) == 4

    def test_generate_numpy_stream_pos_emb(self):
        m = SloTransformer(vocab_size=32, n_embed=16, n_layer=1, n_head=2,
                           block_size=16, max_seq_len=32, use_abs_pos_emb=True)
        toks = list(m.generate_numpy_stream(np.array([[1, 2, 3]]), max_new_tokens=3, temperature=0.0))
        assert len(toks) == 3


class TestTensorFormatMethods:
    def test_shape_and_scalar_queries(self):
        t = Tensor(np.arange(6.0).reshape(2, 3))
        assert t.dim() == 2
        assert t.numel() == 6
        assert t.size() == (2, 3)
        assert t.size(1) == 3
        assert t.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        assert t.item() == 0.0
        assert Tensor(np.array(7.0)).item() == 7.0

    def test_any_and_reduction_dim(self):
        z = Tensor(np.zeros((2, 3)))
        np.testing.assert_array_equal(z.any().data, [0.0])
        np.testing.assert_array_equal(z.any(dim=1).data, [0.0, 0.0])
        np.testing.assert_array_equal(z.any(dim=0, keepdim=True).data, [[0.0, 0.0, 0.0]])

    def test_reshape_repeat_gather_squeeze(self):
        t = Tensor(np.arange(6.0).reshape(2, 3))
        assert t.reshape(3, 2).shape == (3, 2)
        np.testing.assert_array_equal(t.repeat(2, 1).data, np.tile(t.data, (2, 1)))
        idx = Tensor(np.array([[0, 2], [1, 2]]))
        np.testing.assert_array_equal(t.gather(1, idx).data, [[0.0, 2.0], [4.0, 5.0]])
        sq = Tensor(np.array([[1.0, 2.0]]))
        assert sq.squeeze().shape == (2,)
        assert sq.squeeze(0).shape == (2,)
        assert Tensor(np.array([1.0, 2.0])).unsqueeze(0).shape == (1, 2)
        assert Tensor(np.array([1.0, 2.0])).unsqueeze(-1).shape == (2, 1)

    def test_dtype_casts_and_copies(self):
        i = Tensor(np.array([1, 2], dtype=np.int64))
        np.testing.assert_array_equal(i.long().data, [1, 2])
        np.testing.assert_array_equal(i.int().data, [1, 2])
        np.testing.assert_array_equal(i.half().data, [1, 2])
        np.testing.assert_array_equal(i.double().data, [1, 2])
        assert i.float().data.dtype == np.float32
        det = i.detach()
        assert not det.requires_grad
        assert i.numpy() is i.data
        assert i.cpu() is i
        cl = i.clone()
        assert cl is not i
        np.testing.assert_array_equal(cl.data, [1, 2])
        assert i.contiguous() is i
        assert i.flatten().shape == (2,)
        assert i.view(1, 2).shape == (1, 2)
        assert i.view(-1).shape == (2,)
        np.testing.assert_array_equal(i.tolist(), [1, 2])


class TestSloLayerBaseMethods:
    def test_base_slo_layer(self):
        lay = slonet.SloLayer("test")
        lay.eval()
        lay.train()
        assert lay.parameters() == []
        assert lay.named_children() == []
        assert lay.named_modules() == [("", lay)]
        assert lay.soul_signature()["name"] == "test"

    def test_slonet_train_eval_module(self):
        net = slonet.SloNet()
        net.train()
        net.eval()

    def test_transformer_block_forward_numpy(self):
        m = _small_transformer()
        block = m.layers[2]
        x = np.random.RandomState(0).randn(1, 8, 16).astype(np.float32)
        out, cache = block.forward_numpy(x, kv_cache=None)
        assert out.shape == (1, 8, 16)
        assert np.all(np.isfinite(out))

    def test_load_state_dict_roundtrip(self):
        m = _small_transformer()
        sd = {k: p.data.copy() for k, p in m._named_parameters()}
        assert m.load_state_dict(sd) == []
        sd["blocks.0.attn.q_proj.weight"] = sd["blocks.0.attn.q_proj.weight"][:8]
        m.load_state_dict(sd, strict=False)


class _FakeCPUTensor:
    """Stand-in for a PyTorch-style tensor exposing .cpu().detach().numpy()."""

    def __init__(self, arr):
        self._a = arr

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self._a


class TestSloTransformerNoSoul:
    def _tiny(self, **kw):
        cfg = dict(vocab_size=32, n_embed=16, n_layer=1, n_head=2,
                   block_size=16, max_seq_len=32, dropout=0.0, tie_weights=True)
        cfg.update(kw)
        return SloTransformer(**cfg)

    def test_default_soul_metadata(self):
        m = self._tiny()
        sig = m.soul_signature()
        assert m.soul_name == "SloTransformer"
        assert m.lineage == "soultransformer"
        assert sig["soul_name"] == "SloTransformer"
        assert sig["lineage"] == "soultransformer"
        assert sig["system_prompt"] == ""
        assert "soul_traits" in sig and "layers" in sig and "step" in sig
        assert m.metadata["model_type"] == "sloughgpt"

    def test_explicit_no_soul_name(self):
        m = self._tiny(soul_name="no soul", soul_traits={"creativity": 0.9})
        assert m.soul_name == "no soul"
        assert m.soul_signature()["soul_name"] == "no soul"
        assert m.soul_signature()["soul_traits"] == {"creativity": 0.9}
        out = m.generate_numpy(np.array([[1, 2, 3]]), max_new_tokens=3, temperature=0.0)
        assert out.shape == (1, 6)

    def test_properties_and_tie_weights(self):
        m = self._tiny()
        assert isinstance(m.tok_emb, SloEmbedding)
        assert isinstance(m.norm, slonet.SloRMSNorm)
        assert isinstance(m.lm_head, SloLinear)
        assert len(m.blocks) == 1 and isinstance(m.blocks[0], slonet.SloTransformerBlock)
        np.testing.assert_array_equal(m.layers[0].weight.data, m.layers[-1].weight.data)

    def test_norm_property_fallback_for_layer_norm(self):
        m = self._tiny(norm_type="layer_norm")
        assert isinstance(m.norm, slonet.SloLayerNorm)
        assert m.norm is m.layers[-2]

    def test_tie_weights_exception_is_silent(self):
        m = self._tiny()
        m.layers[0] = object()
        m._tie_weights()

    def test_forward_use_cache_stores_kv(self):
        m = self._tiny()
        logits, loss = m.forward(np.array([[1, 2, 3]]), use_cache=True)
        assert logits.data.shape == (1, 3, 32)
        assert all(k is not None for k in m._kv_caches)
        m.clear_kv_cache()
        assert m._kv_caches == [None]

    def test_generate_1d_input_reshaped(self):
        m = self._tiny()
        out = m.generate(np.array([1, 2, 3]), max_new_tokens=3, temperature=0.0)
        assert out.data.shape == (1, 6)

    def test_generate_accepts_cpu_style_input(self):
        m = self._tiny()
        out = m.generate(_FakeCPUTensor(np.array([[1, 2, 3]])), max_new_tokens=3, temperature=0.0)
        assert out.data.shape == (1, 6)

    def test_forward_accepts_cpu_style_inputs(self):
        m = self._tiny()
        logits, loss = m.forward(_FakeCPUTensor(np.array([[1, 2, 3]])))
        assert logits.data.shape == (1, 3, 32)
        logits, loss = m.forward(np.array([[1, 2, 3]]),
                                 targets=_FakeCPUTensor(np.array([[1, 2, 3]])))
        assert loss is not None

    def test_forward_list_input_no_cpu(self):
        m = self._tiny()
        logits, loss = m.forward([[1, 2, 3]])
        assert logits.data.shape == (1, 3, 32)
        assert loss is None

    def test_forward_pass_1d_and_2d(self):
        m = self._tiny()
        fp = m.forward_pass(np.array([1, 2, 3]))
        assert fp.logits.shape == (1, 3, 32)
        assert fp.engine == "numpy"
        fp2 = m.forward_pass(np.array([[1, 2, 3]]))
        assert fp2.logits.shape == (1, 3, 32)

    def test_forward_and_generate_abs_pos_emb(self):
        m = self._tiny(use_abs_pos_emb=True)
        logits, loss = m.forward(np.array([[1, 2, 3]]))
        assert logits.data.shape == (1, 3, 32)
        out = m.generate(np.array([[1, 2, 3]]), max_new_tokens=3, temperature=0.0)
        assert out.data.shape == (1, 6)

    def test_generate_numpy_gqa_nokernel_repeat(self, monkeypatch):
        m = self._tiny(n_kv_head=1)
        monkeypatch.setattr(slonet, "_KERNELS_AVAILABLE", False)
        out = m.generate_numpy(np.array([[1, 2, 3]]), max_new_tokens=4, temperature=0.0)
        assert out.shape == (1, 7)

    def test_load_state_dict_reports_shape_mismatch(self):
        m = self._tiny()
        lm = np.random.RandomState(0).randn(16, 32).astype(np.float32)
        missing = m.load_state_dict({"lm_head.weight": lm}, strict=True)
        assert missing == ["lm_head.weight"]
        np.testing.assert_array_equal(m.layers[-1].weight.data, m.layers[0].weight.data)

    def test_load_state_dict_emb_drop_alias(self):
        m = self._tiny(dropout=0.1)
        assert isinstance(m.layers[1], slonet.SloDropout)
        missing = m.load_state_dict({"emb_drop.weight": np.zeros((4, 4))})
        assert missing == ["emb_drop.weight"]

    def test_to_train_eval_and_context_manager(self):
        m = self._tiny()
        assert m.to("cpu") is m
        assert m.train(True) is m
        assert m.train(False) is m
        with m:
            pass
        m.__exit__(None, None, None)

    def test_quantized_nokernel_generate_numpy(self, monkeypatch):
        m = self._tiny()
        m.blocks[0].attn.W_q._quant_info = SimpleNamespace(is_quantized=False)
        monkeypatch.setattr(slonet, "_KERNELS_AVAILABLE", False)
        out = m.generate_numpy(np.array([[1, 2, 3]]), max_new_tokens=5, temperature=0.0)
        assert out.shape == (1, 8)

    def test_quantized_nokernel_stream(self, monkeypatch):
        m = self._tiny()
        m.blocks[0].attn.W_q._quant_info = SimpleNamespace(is_quantized=False)
        monkeypatch.setattr(slonet, "_KERNELS_AVAILABLE", False)
        out = list(m.generate_numpy_stream(np.array([[1, 2, 3]]), max_new_tokens=3, temperature=0.0))
        assert len(out) == 3

    def test_quantized_kernel_stream_greedy(self):
        m = self._tiny()
        m.blocks[0].attn.W_q._quant_info = SimpleNamespace(is_quantized=False)
        out = list(m.generate_numpy_stream(np.array([[1, 2, 3]]), max_new_tokens=3, temperature=0.0))
        assert len(out) == 3

    def test_quantized_int8_lm_head_argmax(self):
        m = self._tiny()
        info = SimpleNamespace(
            is_quantized=True,
            meta=SimpleNamespace(bits=8, is_per_channel=False, scale=np.float32(0.1), zero_point=0),
            array=np.zeros((16, 16), dtype=np.int8),
        )
        m.blocks[0].attn.W_q._quant_info = info
        m.layers[-1]._quant_info = SimpleNamespace(
            is_quantized=True,
            meta=SimpleNamespace(bits=8, is_per_channel=False, scale=np.float32(0.1), zero_point=0),
            array=np.zeros((32, 16), dtype=np.int8),
        )
        out = m.generate_numpy(np.array([[1, 2, 3]]), max_new_tokens=5, temperature=0.0)
        assert out.shape == (1, 8)
        m2 = self._tiny()
        m2.blocks[0].attn.W_q._quant_info = info
        m2.layers[-1]._quant_info = m.layers[-1]._quant_info
        s = list(m2.generate_numpy_stream(np.array([[1, 2, 3]]), max_new_tokens=3, temperature=0.0))
        assert len(s) == 3


class TestGetAcceleratorOldBackendFailure:
    def test_both_backends_fail_returns_none(self, monkeypatch):
        import domains.training.gpu.accelerator as _old_acc_mod
        monkeypatch.setattr(slonet, "_ACCELERATOR", None)
        monkeypatch.setattr(
            _old_acc_mod,
            "get_accelerator",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert slonet._get_accelerator() is None
        assert slonet._ACCELERATOR == "none"


class TestTensorDtypeSurface:
    def test_to_with_np_dtype_object(self):
        class _FakeDtype:
            _np = np.float64

        t = Tensor(np.array([1, 2, 3], dtype=np.int64))
        t.to(dtype=_FakeDtype())
        assert t.data.dtype == np.float32

    def test_float_casts_non_float32(self):
        t = Tensor(np.array([1, 2, 3], dtype=np.int64))
        t.data = t.data.astype(np.float64)
        t.float()
        assert t.data.dtype == np.float32


class TestPointWeightLayerMethods:
    def test_set_and_get_point_weight(self):
        lin = SloLinear(4, 4, name="pw_lin")
        assert lin.get_point_weight() is None
        pw = lin.compress_to_point(method="cluster", n_clusters=4)
        assert pw is lin.get_point_weight()
        assert lin.weight.data.shape == (4, 4)
        np.testing.assert_allclose(lin.weight.data, pw.generate(), atol=1e-3)


class TestLstmForwardAdapterAndSkipEmbed1D:
    def _lstm(self, layers, hidden):
        return SloLSTM(vocab_size=32, embed_dim=16, hidden_dim=hidden, num_layers=layers, dropout=0.0)

    def test_tensor_forward_with_adapter(self):
        lstm = self._lstm(layers=2, hidden=16)
        adapter = SloAdapterLayer(dim=16, rank=2)
        x = Tensor(np.array([[1, 2, 3, 4]], dtype=np.int64))
        out, (h, c) = lstm.forward(x, adapter=adapter)
        assert out.shape == (1, 32)
        assert np.isfinite(h.data).all()
        assert np.isfinite(c.data).all()

    def test_forward_numpy_skip_embed_1d(self):
        lstm = self._lstm(layers=1, hidden=16)
        out, _ = lstm.forward_numpy(np.random.randn(16).astype(np.float32), skip_embed=True)
        assert out.shape == (1, 32)


class TestAttention4DGradAccumulation:
    def test_double_backward_accumulates(self):
        rng = np.random.default_rng(0)
        q = Tensor(rng.standard_normal((1, 2, 2, 4)).astype(np.float32), requires_grad=True)
        k = Tensor(rng.standard_normal((1, 2, 2, 4)).astype(np.float32), requires_grad=True)
        v = Tensor(rng.standard_normal((1, 2, 2, 4)).astype(np.float32), requires_grad=True)
        out = SloMultiHeadAttention._attention_4d(q, k, v, None, 1.0)
        loss = (out * out).sum()
        loss.backward()
        loss.backward()
        assert np.all(np.isfinite(q.grad.data))
        assert np.all(np.isfinite(k.grad.data))
        assert np.all(np.isfinite(v.grad.data))


class TestBatchNormEvalForwardGrad:
    def test_eval_forward_grad(self):
        rng = np.random.default_rng(0)
        x = Tensor(rng.standard_normal((1, 3, 4, 4)).astype(np.float32), requires_grad=True)
        g = Tensor(np.ones(3, dtype=np.float32), requires_grad=True)
        b = Tensor(np.zeros(3, dtype=np.float32), requires_grad=True)
        out = slonet._batchnorm2d(x, g, b, np.zeros(3), np.ones(3), 1e-5, training=False)
        tangents = out.forward_grad({x.id: np.ones_like(x.data)})
        assert out.id in tangents


class TestSoulLibBlock:
    def test_forward_identity_and_no_params(self):
        blk = slonet._SoulTransformerBlockSoulLib(8, 2, 16, name="blk")
        x = Tensor(np.random.randn(1, 5, 8).astype(np.float32))
        out = blk.forward(x)
        assert out.data.shape == (1, 5, 8)
        assert blk.parameters() == []


class TestSloNetNonCallableLayer:
    def test_non_callable_layer_uses_forward(self):
        class _Layer:
            def forward(self, x):
                return x

        net = SloNet(layers=[_Layer()])
        out = net.forward(Tensor(np.arange(4).reshape(1, 4).astype(np.float32)))
        assert out.data.shape == (1, 4)


class TestStateDictAccelHelpers:
    def test_matmul_and_layernorm_accel_success(self, monkeypatch):
        class _FakeAcc:
            name = "metal"

            def matmul(self, a, b):
                return a @ b

            def layer_norm(self, x, w, b, eps):
                mean = x.mean(axis=-1, keepdims=True)
                var = x.var(axis=-1, keepdims=True)
                return ((x - mean) / np.sqrt(var + eps)) * w

        monkeypatch.setattr(slonet, "_ACCELERATOR", _FakeAcc())
        a = Tensor(np.random.randn(3, 4).astype(np.float32))
        res = slonet._matmul_state_dict(a, np.random.randn(4, 5).astype(np.float32))
        assert res.data.shape == (3, 5)
        x = Tensor(np.random.randn(2, 3).astype(np.float32))
        res2 = slonet._layernorm_state_dict(x, np.ones(3, dtype=np.float32))
        assert res2.data.shape == (2, 3)
        assert np.isfinite(res2.data).all()


class TestImportFromSouVariants:
    def test_pytorch_zip_lineage(self, tmp_path, monkeypatch):
        meta = {"lineage": "slonet", "soul_name": "X", "system_prompt": ""}
        mj = json.dumps(meta).encode()
        raw = b"SOUL" + struct.pack("<I", 2) + struct.pack("<I", len(mj)) + mj + b"PKjunk"
        p = tmp_path / "pk.soul"
        p.write_bytes(raw)
        monkeypatch.setattr(slonet, "_load_pytorch_zip_weights", lambda b: {"w0": np.ones(2)})
        net = import_from_sou(str(p))
        assert net.lineage == "slolib-pytorch"

    def test_tok_emb_rebuild(self, tmp_path):
        meta = {"lineage": "slonet", "soul_name": "X", "system_prompt": ""}
        mj = json.dumps(meta).encode()
        wj = json.dumps({"tok_emb.weight": [[0.1, 0.2], [0.3, 0.4]]}).encode()
        raw = (
            b"SOUL" + struct.pack("<I", 2) + struct.pack("<I", len(mj)) + mj
            + struct.pack("<I", len(wj)) + wj
        )
        p = tmp_path / "tok.soul"
        p.write_bytes(raw)
        net = import_from_sou(str(p))
        assert net.lineage == "slonet"
        assert "tok_emb.weight" in net._sd

    def test_rebuild_hidden_dim_nonpositive(self):
        net = SloNet(soul_name="x", metadata={})
        weights = {
            "w0": np.ones((4, 4), dtype=np.float32),
            "w1": np.ones((4, 4), dtype=np.float32),
            "w2": np.ones((8, 4), dtype=np.float32),
            "w3": np.ones((2,), dtype=np.float32),
            "w4": np.ones((8, 8), dtype=np.float32),
            "w5": np.ones((8,), dtype=np.float32),
            "w6": np.ones((4, 8), dtype=np.float32),
            "w7": np.ones((4,), dtype=np.float32),
        }
        _rebuild_net_from_params(net, weights)
        assert len(net.layers) == 1


class TestStateDictToNumpyBranch:
    def test_numpy_method_object(self):
        class _ArrLike:
            def numpy(self):
                return np.arange(3)

        result = slonet._state_dict_to_numpy({"a": _ArrLike()})
        assert np.array_equal(result["a"], np.arange(3))


class TestTrainSoulTransformerNoneLoss:
    def test_none_loss_skips_step(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            slonet.SloTransformer,
            "forward",
            lambda self, x, targets=None, **kw: (None, None),
        )
        monkeypatch.setattr(slonet, "export_to_sou", lambda net, path: None)
        net = train_soul_transformer(
            lambda topic, temperature: "hello world!",
            epochs=1,
            on_step=lambda *a: calls.append(a),
        )
        assert isinstance(net, SloTransformer)
        assert calls == []


class TestKernelsImportFallback:
    def test_slonet_kernels_import_failure_falls_back(self):
        import os
        import subprocess
        import sys
        from pathlib import Path

        core_py = Path(__file__).resolve().parents[1]
        code = (
            "import sys\n"
            "class _Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'domains.training.slonet_kernels':\n"
            "            raise ImportError('blocked for test')\n"
            "sys.meta_path.insert(0, _Block())\n"
            "from domains.training import slonet\n"
            "assert slonet._KERNELS_AVAILABLE is False\n"
            "print('KERNELS_FALLBACK_OK')\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(core_py) + os.pathsep + env.get("PYTHONPATH", "")
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
        )
        assert res.returncode == 0, res.stderr
        assert "KERNELS_FALLBACK_OK" in res.stdout
