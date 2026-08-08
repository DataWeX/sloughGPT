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

    def test_extract_param_names_sees_rebuild_marker(self):
        ops = [
            ("SHORT_BINUNICODE", "_rebuild_tensor_v2", 0),
            ("SHORT_BINUNICODE", "model.weight", 1),
        ]
        names = pt_loader._extract_param_names(ops)
        assert "model.weight" in names

    def test_extract_param_names_skips_dotless(self):
        ops = [("SHORT_BINUNICODE", "embed", 0)]
        assert pt_loader._extract_param_names(ops) == []


class TestUnpicklerInternals:
    def test_persistent_load_passthrough(self):
        up = pt_loader._PTUnpickler(io.BytesIO(b""), {})
        assert up.persistent_load(("other", 1)) == ("other", 1)

    def test_find_class_storage_returns_noop(self):
        up = pt_loader._PTUnpickler(io.BytesIO(b""), {})
        fn = up.find_class("torch", "FloatStorage")
        assert fn(1, 2) is None

    def test_find_class_delegates_to_super(self):
        up = pt_loader._PTUnpickler(io.BytesIO(b""), {})
        assert up.find_class("numpy", "ndarray") is np.ndarray

    def test_rebuild_non_storage_returns_as_is(self):
        up = pt_loader._PTUnpickler(io.BytesIO(b""), {})
        assert up._rebuild(("not-storage",), 0, (2,), None) == ("not-storage",)

    def test_rebuild_zero_bytes_defaults_f32(self):
        up = pt_loader._PTUnpickler(io.BytesIO(b""), {})
        arr = up._rebuild(("__storage__", "k", b""), 0, (3,), None)
        assert arr.dtype == np.float32

    def test_rebuild_offset_slices(self):
        up = pt_loader._PTUnpickler(io.BytesIO(b""), {})
        raw = np.arange(6, dtype=np.float64).tobytes()
        arr = up._rebuild(("__storage__", "k", raw), 2, (6,), None)
        assert arr.dtype == np.float64
        assert arr[0] == 2.0

    def test_rebuild_reshape_mismatch_keeps_flat(self):
        up = pt_loader._PTUnpickler(io.BytesIO(b""), {})
        arr = up._rebuild(("__storage__", "k", bytes(range(6))), 0, (4,), None)
        assert arr.shape == (6,)


class TestNestedCheckpoints:
    def test_plain_dict_model_state_dict_returned(self, tmp_path):
        payload = {"model_state_dict": {"w": np.array([5.0], dtype=np.float32)}}
        buf = io.BytesIO()
        _TorchPickler(buf).dump(payload)
        path = tmp_path / "msd.pt"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("archive/data.pkl", buf.getvalue())
        result = pt_loader.load_pt_checkpoint(path)
        assert "model_state_dict" in result

    def test_plain_dict_model_returned(self, tmp_path):
        payload = {"model": {"w": np.array([1.0], dtype=np.float32)}}
        buf = io.BytesIO()
        _TorchPickler(buf).dump(payload)
        path = tmp_path / "model.pt"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("archive/data.pkl", buf.getvalue())
        result = pt_loader.load_pt_checkpoint(path)
        assert "w" in result
        assert np.array_equal(result["w"], np.array([1.0], dtype=np.float32))

    def test_plain_dict_returned_as_is(self, tmp_path):
        buf = io.BytesIO()
        _TorchPickler(buf).dump({"tag": "x"})
        path = tmp_path / "plain.pt"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("archive/data.pkl", buf.getvalue())
        result = pt_loader.load_pt_checkpoint(path)
        assert result == {"tag": "x"}

    def test_top_level_tensor_wrapped_in_state_dict(self, tmp_path):
        data_arrays = {"0": np.arange(3, dtype=np.float32).tobytes()}
        record = _TensorRecord(_StorageStub("0"), 0, (3,), None, False)
        buf = io.BytesIO()
        _TorchPickler(buf).dump(record)
        path = tmp_path / "top.pt"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("archive/data.pkl", buf.getvalue())
            z.writestr("data/0", data_arrays["0"])
        result = pt_loader.load_pt_checkpoint(path)
        assert "state_dict" in result
        assert np.array_equal(result["state_dict"], np.arange(3, dtype=np.float32))

    def test_no_dot_tensor_name_kept_as_metadata(self, tmp_path):
        state = OrderedDict([
            ("model.weight", np.array([1.0, 2.0], dtype=np.float32)),
            ("embed", np.array([9.0, 8.0], dtype=np.float32)),
        ])
        path = tmp_path / "nodot.pt"
        _build_pt(path, state)
        result = pt_loader.load_pt_checkpoint(path)
        assert np.array_equal(result["embed"], np.array([9.0, 8.0], dtype=np.float32))

    def test_state_dict_nested_model(self, tmp_path):
        path = tmp_path / "nested_model.pt"
        _build_pt(path, OrderedDict([
            ("model", {"w": np.array([1.0, 2.0], dtype=np.float32)}),
            ("step", 3),
        ]))
        result = pt_loader.load_pt_state_dict(path)
        assert np.array_equal(result["w"], np.array([1.0, 2.0], dtype=np.float32))

    def test_state_dict_nested_model_state_dict(self, tmp_path):
        payload = {"model_state_dict": {"w": np.array([5.0], dtype=np.float32)}}
        buf = io.BytesIO()
        _TorchPickler(buf).dump(payload)
        path = tmp_path / "msd.pt"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("archive/data.pkl", buf.getvalue())
        result = pt_loader.load_pt_state_dict(path)
        assert np.array_equal(result["w"], np.array([5.0], dtype=np.float32))
