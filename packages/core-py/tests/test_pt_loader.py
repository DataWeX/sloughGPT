"""Tests for the torch-free .pt checkpoint loader."""

import io
import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from domains.infrastructure import pt_loader


class _StorageStub:
    def __init__(self, key, typename="torch.FloatStorage"):
        self.key = key
        self.typename = typename


def _rebuild_tensor_v2_stub(*args):
    """Module-level callable; loader's find_class routes any name containing
    '_rebuild_tensor_v2' to its own _rebuild."""
    return args


class _TensorRecord:
    """Pickles as torch._utils._rebuild_tensor_v2(storage, offset, size, stride, req_grad)."""

    def __init__(self, storage, offset, size, stride=None, requires_grad=False):
        self.storage = storage
        self.offset = offset
        self.size = size
        self.stride = stride
        self.requires_grad = requires_grad

    def __reduce__(self):
        return (_rebuild_tensor_v2_stub,
                (self.storage, self.offset, self.size, self.stride, self.requires_grad))


class _TorchPickler(pickle.Pickler):
    def __init__(self, file, protocol=2):
        super().__init__(file, protocol=protocol)

    def persistent_id(self, obj):
        if isinstance(obj, _StorageStub):
            return ("storage", obj.typename, obj.key)
        return None


def _build_pt(path, payload):
    """Write a torch-style zip checkpoint. payload: OrderedDict of name -> numpy array."""
    data_arrays = {}
    record_payload = OrderedDict()
    idx = 0
    for name, arr in payload.items():
        if isinstance(arr, np.ndarray):
            key = str(idx)
            data_arrays[key] = np.ascontiguousarray(arr).tobytes()
            record_payload[name] = _TensorRecord(
                _StorageStub(key), 0, tuple(arr.shape), None, False
            )
            idx += 1
        else:
            record_payload[name] = arr

    buf = io.BytesIO()
    _TorchPickler(buf).dump(record_payload)

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("archive/data.pkl", buf.getvalue())
        for key, raw in data_arrays.items():
            z.writestr(f"data/{key}", raw)


@pytest.fixture
def simple_checkpoint(tmp_path):
    state = OrderedDict([
        ("model.layer1.weight", np.arange(6, dtype=np.float32).reshape(2, 3)),
        ("model.layer1.bias", np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)),
    ])
    path = tmp_path / "model.pt"
    _build_pt(path, state)
    return path, state


class TestLoadPtCheckpoint:
    def test_returns_matching_arrays(self, simple_checkpoint):
        path, state = simple_checkpoint
        result = pt_loader.load_pt_checkpoint(path)
        assert set(result.keys()) == {"model.layer1.weight", "model.layer1.bias"}
        assert np.array_equal(result["model.layer1.weight"], state["model.layer1.weight"])
        assert np.array_equal(result["model.layer1.bias"], state["model.layer1.bias"])

    def test_dtypes_preserved(self, tmp_path):
        state = OrderedDict([
            ("w.f32", np.array([[1, 2], [3, 4]], dtype=np.float32)),
            ("w.f16", np.array([1.5, 2.5], dtype=np.float16)),
            ("w.u8", np.array([10, 20, 30], dtype=np.uint8)),
            ("w.f64", np.array([1.0], dtype=np.float64)),
        ])
        path = tmp_path / "dtypes.pt"
        _build_pt(path, state)
        result = pt_loader.load_pt_checkpoint(path)
        assert result["w.f32"].dtype == np.float32
        assert result["w.f16"].dtype == np.float16
        assert result["w.u8"].dtype == np.uint8
        assert result["w.f64"].dtype == np.float64
        assert np.array_equal(result["w.f16"], state["w.f16"])

    def test_metadata_preserved(self, tmp_path):
        path = tmp_path / "meta.pt"
        _build_pt(path, OrderedDict([
            ("model.weight", np.array([1.0, 2.0], dtype=np.float32)),
            ("epoch", 7),
            ("tag", "hello"),
        ]))
        result = pt_loader.load_pt_checkpoint(path)
        assert result["model.weight"].tolist() == [1.0, 2.0]
        assert result["epoch"] == 7
        assert result["tag"] == "hello"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            pt_loader.load_pt_checkpoint(tmp_path / "ghost.pt")

    def test_no_pickle_raises(self, tmp_path):
        path = tmp_path / "empty.pt"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("data/0", b"\x00\x00\x80\x3f")
        with pytest.raises(ValueError):
            pt_loader.load_pt_checkpoint(path)

    def test_empty_tensor_shape(self, tmp_path):
        path = tmp_path / "scalar.pt"
        _build_pt(path, OrderedDict([("s.scalar", np.array(5.0, dtype=np.float32))]))
        result = pt_loader.load_pt_checkpoint(path)
        assert result["s.scalar"].item() == 5.0


class TestLoadPtStateDict:
    def test_flat_returns_arrays_only(self, simple_checkpoint):
        path, state = simple_checkpoint
        result = pt_loader.load_pt_state_dict(path)
        assert set(result.keys()) == {"model.layer1.weight", "model.layer1.bias"}
        assert all(isinstance(v, np.ndarray) for v in result.values())

    def test_load_pt_file_alias(self, simple_checkpoint):
        path, _ = simple_checkpoint
        a = pt_loader.load_pt_file(path)
        b = pt_loader.load_pt_state_dict(path)
        assert a.keys() == b.keys()
        assert all(np.array_equal(a[k], b[k]) for k in a)


class TestLoadPtBytes:
    def test_roundtrip(self, tmp_path):
        state = OrderedDict([("w.x", np.arange(4, dtype=np.float32))])
        path = tmp_path / "b.pt"
        _build_pt(path, state)
        data = path.read_bytes()
        result = pt_loader.load_pt_bytes(data)
        assert np.array_equal(result["w.x"], state["w.x"])


class TestParseHelpers:
    def test_extract_param_names_skips_global_module(self):
        state = OrderedDict([("model.weight", np.zeros((2,), dtype=np.float32))])
        buf = io.BytesIO()
        _TorchPickler(buf).dump(state)
        ops = pt_loader._parse_pickle_ops(buf.getvalue())
        names = pt_loader._extract_param_names(ops)
        assert "model.weight" in names
        assert not any(n.startswith("torch") for n in names)

    def test_parse_pickle_ops_structure(self):
        buf = io.BytesIO()
        pickle.dump({"a": 1}, buf)
        ops = pt_loader._parse_pickle_ops(buf.getvalue())
        assert all(len(op) == 3 for op in ops)
        assert ops[0][0] == "PROTO"
