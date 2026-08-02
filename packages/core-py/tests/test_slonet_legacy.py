"""Tests for legacy SloNet surfaces: SOU export/import, flat-param rebuild,
demo trainers, SloDataLoader, silu numpy path, Tensor.to, GQA repeat backward."""

import json
import struct

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
        mha = SloMultiHeadAttention(d_model=16, n_heads=4, n_kv_head=4)
        rng = np.random.default_rng(0)
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
