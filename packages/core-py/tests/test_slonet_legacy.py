"""Tests for legacy SloNet surfaces: SOU export/import, flat-param rebuild,
demo trainers, SloDataLoader, silu numpy path, Tensor.to, GQA repeat backward."""

import json
import struct

import numpy as np
import pytest

from domains.training import slonet
from domains.training.slonet import (
    SloDataLoader,
    SloEmbedding,
    SloLSTM,
    SloMultiHeadAttention,
    SloNet,
    SloTransformer,
    Tensor,
    _load_pytorch_zip_weights,
    _rebuild_net_from_params,
    _sanitize,
    export_to_sou,
    import_from_sou,
    silu,
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
