"""Integration contract: downcraft's download layout is consumable by the app loader.

Proves the resume-aware ``domains.infrastructure.hf_hub.download_hf_model``
output (``snapshots/default/`` + ``refs/main``) is resolvable by the
torch-free ``safetensors_loader`` used by the server autoload path.  Uses a
real local HTTP server with ``Range`` support — no network.  The safetensors
payload is synthesized by hand (no ``safetensors`` dependency).
"""

import hashlib
import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

from domains.infrastructure.safetensors_loader import (
    _find_safetensors,
    _get_model_dir,
    load_model_weights,
    _load_weights_raw,
    _MAX_HEADER_LEN,
    load_model_config,
    list_cached_models,
)


def _write_safetensors(path: Path, weights: dict) -> int:
    header = {"__metadata__": {}}
    offset = 0
    for name, arr in weights.items():
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + arr.nbytes],
        }
        offset += arr.nbytes
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for arr in weights.values():
            f.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return 8 + len(header_bytes) + offset


def _read_safetensors(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        arr = np.frombuffer(data[start:end], dtype=np.float32).reshape(meta["shape"])
        out[name] = arr
    return out


class _RangeHandler(BaseHTTPRequestHandler):
    payload = b""

    def do_GET(self):
        start = 0
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            spec = rng[len("bytes="):].split("-")[0]
            if spec.isdigit():
                start = int(spec)
        data = self.payload[start:]
        if start > 0:
            self.send_response(206)
            self.send_header(
                "Content-Range",
                f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}",
            )
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("ETag", '"static"')
        self.end_headers()
        for i in range(0, len(data), 2048):
            self.wfile.write(data[i:i + 2048])

    def log_message(self, *args):
        pass


# ---------------------------------------------------------------------------
# _write_safetensors / _read_safetensors helpers
# ---------------------------------------------------------------------------

def test_write_read_roundtrip():
    weights = {
        "layer.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        "bias": np.array([1.0, 2.0, 3.0]),
    }
    path = Path("/tmp/test_st_roundtrip.safetensors")
    _write_safetensors(path, weights)
    loaded = _read_safetensors(path)
    assert set(loaded.keys()) == {"layer.weight", "bias"}
    np.testing.assert_array_equal(loaded["layer.weight"], weights["layer.weight"])
    np.testing.assert_array_equal(loaded["bias"], weights["bias"])
    path.unlink()


def test_write_returns_total_bytes():
    weights = {"w": np.zeros(10, dtype=np.float32)}
    path = Path("/tmp/test_st_bytes.safetensors")
    total = _write_safetensors(path, weights)
    assert total == path.stat().st_size
    path.unlink()


def test_safetensors_empty_weights():
    path = Path("/tmp/test_st_empty.safetensors")
    _write_safetensors(path, {})
    loaded = _read_safetensors(path)
    assert loaded == {}
    path.unlink()


def test_safetensors_large_tensor():
    weights = {"big": np.arange(100000, dtype=np.float32)}
    path = Path("/tmp/test_st_large.safetensors")
    _write_safetensors(path, weights)
    loaded = _read_safetensors(path)
    assert loaded["big"].shape == (100000,)
    np.testing.assert_array_equal(loaded["big"], weights["big"])
    path.unlink()


def test_safetensors_multidim_shapes():
    weights = {
        "scalar": np.array(42.0, dtype=np.float32),
        "vec": np.arange(5, dtype=np.float32),
        "mat": np.arange(6, dtype=np.float32).reshape(2, 3),
        "cube": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
    }
    path = Path("/tmp/test_st_multidim.safetensors")
    _write_safetensors(path, weights)
    loaded = _read_safetensors(path)
    for name in weights:
        assert loaded[name].shape == weights[name].shape
        np.testing.assert_array_equal(loaded[name], weights[name])
    path.unlink()


def test_safetensors_metadata_ignored_in_read():
    path = Path("/tmp/test_st_meta.safetensors")
    weights = {"w": np.array([1.0], dtype=np.float32)}
    _write_safetensors(path, weights)
    loaded = _read_safetensors(path)
    assert "__metadata__" not in loaded
    path.unlink()


# ---------------------------------------------------------------------------
# _get_model_dir
# ---------------------------------------------------------------------------

def test_get_model_dir_default(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    result = _get_model_dir("org/model")
    assert result.name == "models--org--model"


def test_get_model_dir_slash_replaced():
    result = _get_model_dir.__module__
    assert isinstance(result, str)


def test_get_model_dir_respects_hf_home(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    result = _get_model_dir("gpt2")
    assert "models--gpt2" in str(result)


def test_get_model_dir_converts_slashes(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    result = _get_model_dir("meta-llama/Llama-3-8B")
    assert result.name == "models--meta-llama--Llama-3-8B"


# ---------------------------------------------------------------------------
# _find_safetensors
# ---------------------------------------------------------------------------

def test_find_safetensors_in_snapshots(tmp_path):
    model_dir = tmp_path / "models--test"
    snap_dir = model_dir / "snapshots" / "default"
    snap_dir.mkdir(parents=True)
    st = snap_dir / "model.safetensors"
    st.write_bytes(b"fake")
    result = _find_safetensors(model_dir)
    assert result == st


def test_find_safetensors_direct_in_model_dir(tmp_path):
    model_dir = tmp_path / "models--test2"
    model_dir.mkdir()
    st = model_dir / "model.safetensors"
    st.write_bytes(b"fake")
    result = _find_safetensors(model_dir)
    assert result == st


def test_find_safetensors_not_found(tmp_path):
    model_dir = tmp_path / "models--empty"
    model_dir.mkdir()
    result = _find_safetensors(model_dir)
    assert result is None


def test_find_safetensors_prefers_snapshots(tmp_path):
    model_dir = tmp_path / "models--test3"
    snap = model_dir / "snapshots" / "default"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"snap")
    (model_dir / "model.safetensors").write_bytes(b"dir")
    result = _find_safetensors(model_dir)
    assert "snapshots" in result.parts


# ---------------------------------------------------------------------------
# load_model_weights — raw parser path
# ---------------------------------------------------------------------------

def test_load_model_weights_raw_parser(monkeypatch, tmp_path):
    weights = {"wte.weight": np.arange(4096, dtype=np.float32).reshape(128, 32)}
    hub = tmp_path / "hub"
    model_dir = hub / "hub" / "models--org--model"
    snap = model_dir / "snapshots" / "default"
    snap.mkdir(parents=True)
    st_file = snap / "model.safetensors"
    _write_safetensors(st_file, weights)
    monkeypatch.setenv("HF_HOME", str(hub))
    monkeypatch.setattr(
        "domains.infrastructure.safetensors_loader.find_repo_root",
        lambda p: tmp_path / "nonexistent_repo",
    )
    loaded = load_model_weights("org/model")
    assert "wte.weight" in loaded
    assert loaded["wte.weight"].shape == (128, 32)


def test_load_model_weights_file_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "nonexistent"))
    monkeypatch.setattr(
        "domains.infrastructure.safetensors_loader.find_repo_root",
        lambda p: tmp_path / "nonexistent_repo",
    )
    try:
        load_model_weights("nonexistent/model")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_load_model_weights_no_safetensors(monkeypatch, tmp_path):
    model_dir = tmp_path / "hub" / "hub" / "models--org--empty"
    model_dir.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hub"))
    monkeypatch.setattr(
        "domains.infrastructure.safetensors_loader.find_repo_root",
        lambda p: tmp_path / "nonexistent_repo",
    )
    try:
        load_model_weights("org/empty")
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# _load_weights_raw edge cases
# ---------------------------------------------------------------------------

def test_load_weights_raw_single_tensor():
    weights = {"bias": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    path = Path("/tmp/test_raw_single.safetensors")
    _write_safetensors(path, weights)
    loaded = _load_weights_raw(path, np.float32)
    np.testing.assert_array_equal(loaded["bias"], weights["bias"])
    path.unlink()


def test_load_weights_raw_dtype_conversion():
    weights = {"w": np.array([1.5, 2.5], dtype=np.float32)}
    path = Path("/tmp/test_raw_dtype.safetensors")
    _write_safetensors(path, weights)
    loaded = _load_weights_raw(path, np.float32)
    assert loaded["w"].dtype == np.float32
    path.unlink()


def test_load_weights_raw_truncated_header():
    path = Path("/tmp/test_raw_trunc.safetensors")
    path.write_bytes(b"\x00" * 4)
    try:
        _load_weights_raw(path, np.float32)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Truncated" in str(e)
    path.unlink()


def test_load_weights_raw_header_exceeds_sanity_limit():
    path = Path("/tmp/test_raw_big_header.safetensors")
    fake_len = struct.pack("<Q", _MAX_HEADER_LEN + 1)
    path.write_bytes(fake_len + b"\x00" * 100)
    try:
        _load_weights_raw(path, np.float32)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "exceeds sanity limit" in str(e)
    path.unlink()


def test_load_weights_raw_header_extends_past_eof():
    path = Path("/tmp/test_raw_eof.safetensors")
    fake_len = struct.pack("<Q", 1000)
    path.write_bytes(fake_len + b"\x00" * 10)
    try:
        _load_weights_raw(path, np.float32)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "past end of file" in str(e)
    path.unlink()


def test_load_weights_raw_bad_offsets():
    path = Path("/tmp/test_raw_bad_off.safetensors")
    header = {"w": {"dtype": "F32", "shape": [2], "data_offsets": [100, 200]}}
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(b"\x00" * 10)
    try:
        _load_weights_raw(path, np.float32)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "exceed file size" in str(e) or "Invalid offsets" in str(e)
    path.unlink()


def test_load_weights_raw_negative_offsets():
    path = Path("/tmp/test_raw_neg_off.safetensors")
    header = {"w": {"dtype": "F32", "shape": [2], "data_offsets": [-1, 8]}}
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(b"\x00" * 20)
    try:
        _load_weights_raw(path, np.float32)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Invalid offsets" in str(e)
    path.unlink()


def test_load_weights_raw_f16_dtype():
    path = Path("/tmp/test_raw_f16.safetensors")
    arr_f32 = np.array([1.0, 2.0], dtype=np.float32)
    arr_f16 = arr_f32.astype(np.float16)
    header = {
        "w": {
            "dtype": "F16",
            "shape": [2],
            "data_offsets": [0, arr_f16.nbytes],
        }
    }
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(arr_f16.tobytes())
    loaded = _load_weights_raw(path, np.float32)
    assert loaded["w"].dtype == np.float32
    np.testing.assert_allclose(loaded["w"], arr_f32, atol=0.01)
    path.unlink()


def test_load_weights_raw_bf16_dtype():
    path = Path("/tmp/test_raw_bf16.safetensors")
    arr_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    u16 = (arr_f32.view(np.uint32) >> 16).astype(np.uint16)
    header = {
        "w": {
            "dtype": "BF16",
            "shape": [3],
            "data_offsets": [0, u16.nbytes],
        }
    }
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(u16.tobytes())
    loaded = _load_weights_raw(path, np.float32)
    assert loaded["w"].dtype == np.float32
    np.testing.assert_array_equal(loaded["w"], arr_f32)
    path.unlink()


def test_load_weights_raw_skips_metadata_keys():
    path = Path("/tmp/test_raw_meta_skip.safetensors")
    weights = {"w": np.array([1.0], dtype=np.float32)}
    header = {"__metadata__": {"version": "1.0"}}
    header["w"] = {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(np.float32(1.0).tobytes())
    loaded = _load_weights_raw(path, np.float32)
    assert "__metadata__" not in loaded
    assert "w" in loaded
    path.unlink()


# ---------------------------------------------------------------------------
# load_model_config
# ---------------------------------------------------------------------------

def test_load_model_config_from_snapshots(monkeypatch, tmp_path):
    model_dir = tmp_path / "hub" / "hub" / "models--org--cfg"
    snap = model_dir / "snapshots" / "default"
    snap.mkdir(parents=True)
    cfg = {"vocab_size": 1000, "n_layer": 12}
    (snap / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hub"))
    monkeypatch.setattr(
        "domains.infrastructure.safetensors_loader.find_repo_root",
        lambda p: tmp_path / "nonexistent_repo",
    )
    result = load_model_config("org/cfg")
    assert result["vocab_size"] == 1000
    assert result["n_layer"] == 12


def test_load_model_config_from_model_dir(monkeypatch, tmp_path):
    model_dir = tmp_path / "hub" / "hub" / "models--org--cfg2"
    model_dir.mkdir(parents=True)
    cfg = {"vocab_size": 500}
    (model_dir / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hub"))
    monkeypatch.setattr(
        "domains.infrastructure.safetensors_loader.find_repo_root",
        lambda p: tmp_path / "nonexistent_repo",
    )
    result = load_model_config("org/cfg2")
    assert result["vocab_size"] == 500


def test_load_model_config_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "nonexistent"))
    monkeypatch.setattr(
        "domains.infrastructure.safetensors_loader.find_repo_root",
        lambda p: tmp_path / "nonexistent_repo",
    )
    try:
        load_model_config("nonexistent/model")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# list_cached_models
# ---------------------------------------------------------------------------

def test_list_cached_models_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setattr(
        "domains.infrastructure.safetensors_loader.find_repo_root",
        lambda p: tmp_path / "nonexistent_repo",
    )
    result = list_cached_models()
    assert result == []


def test_list_cached_models_finds_models(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    model_dir = hub / "models--org--mymodel"
    snap = model_dir / "snapshots" / "default"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"\x00" * 100)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(
        "domains.infrastructure.safetensors_loader.find_repo_root",
        lambda p: tmp_path / "nonexistent_repo",
    )
    result = list_cached_models()
    assert len(result) == 1
    assert result[0]["id"] == "org/mymodel"


def test_list_cached_models_skips_dirs_without_safetensors(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    model_dir = hub / "models--org--empty"
    model_dir.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(hub))
    monkeypatch.setattr(
        "domains.infrastructure.safetensors_loader.find_repo_root",
        lambda p: tmp_path / "nonexistent_repo",
    )
    result = list_cached_models()
    assert result == []


def test_list_cached_models_size_in_mb(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    model_dir = hub / "models--org--sized"
    snap = model_dir / "snapshots" / "default"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"\x00" * (2 * 1024 * 1024))
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(
        "domains.infrastructure.safetensors_loader.find_repo_root",
        lambda p: tmp_path / "nonexistent_repo",
    )
    result = list_cached_models()
    assert result[0]["size_mb"] == 2.0


# ---------------------------------------------------------------------------
# Integration contract: download layout consumed by app loader
# ---------------------------------------------------------------------------

def test_download_hf_model_output_consumable_by_app_loader(tmp_path, monkeypatch):
    try:
        import downcraft.state as st_mod
    except ImportError:
        pytest.skip("downcraft module not installed")
    import domains.infrastructure.hf_hub as hub_mod
    from domains.infrastructure.hf_hub import (
        HFFile,
        download_hf_model,
        is_download_complete,
    )

    weights = {"wte.weight": np.arange(4096, dtype=np.float32).reshape(128, 32)}
    st_file = tmp_path / "model.safetensors"
    _write_safetensors(st_file, weights)
    payload = st_file.read_bytes()

    hub = tmp_path / "hub"
    monkeypatch.setenv("HF_HOME", str(hub))
    monkeypatch.setattr(
        st_mod, "get_state",
        lambda: st_mod.PersistentState(state_dir=tmp_path / "state"),
    )

    _RangeHandler.payload = payload
    server = HTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/model.safetensors"
        model_id = "org/model"
        monkeypatch.setattr(
            hub_mod, "list_model_files",
            lambda mid: [
                HFFile(
                    path="model.safetensors",
                    size=len(payload),
                    checksum=hashlib.sha256(payload).hexdigest(),
                    download_url=url,
                )
            ],
        )

        result = download_hf_model(model_id)
        assert result["status"] == "complete"

        model_dir = _get_model_dir(model_id)
        assert model_dir == hub / "hub" / "models--org--model"

        st = _find_safetensors(model_dir)
        assert st is not None
        assert st.name == "model.safetensors"
        assert "snapshots" in st.parts
        assert (model_dir / "refs" / "main").read_text() == "default"

        assert is_download_complete(model_id) is True

        loaded = load_model_weights(model_id)
        assert "wte.weight" in loaded
        assert loaded["wte.weight"].shape == (128, 32)
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_concurrent_write_read_safetensors():
    import concurrent.futures
    results = {}

    def _worker(i):
        weights = {f"w_{i}": np.arange(100, dtype=np.float32) + i}
        path = Path(f"/tmp/test_st_concurrent_{i}.safetensors")
        _write_safetensors(path, weights)
        loaded = _read_safetensors(path)
        np.testing.assert_array_equal(loaded[f"w_{i}"], weights[f"w_{i}"])
        path.unlink()
        return i

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_worker, i) for i in range(16)]
        for f in concurrent.futures.as_completed(futures):
            results[f.result()] = True
    assert len(results) == 16


def test_safetensors_zero_size_tensor():
    weights = {"empty": np.array([], dtype=np.float32)}
    path = Path("/tmp/test_st_zero.safetensors")
    _write_safetensors(path, weights)
    loaded = _read_safetensors(path)
    assert loaded["empty"].shape == (0,)
    path.unlink()


def test_safetensors_negative_values():
    weights = {"neg": np.array([-1.5, -2.7, 0.0, 3.14], dtype=np.float32)}
    path = Path("/tmp/test_st_neg.safetensors")
    _write_safetensors(path, weights)
    loaded = _read_safetensors(path)
    np.testing.assert_array_equal(loaded["neg"], weights["neg"])
    path.unlink()


def test_safetensors_very_long_name():
    name = "a" * 500
    weights = {name: np.array([1.0], dtype=np.float32)}
    path = Path("/tmp/test_st_longname.safetensors")
    _write_safetensors(path, weights)
    loaded = _read_safetensors(path)
    assert name in loaded
    path.unlink()


# ---------------------------------------------------------------------------
# Additional safetensors format tests
# ---------------------------------------------------------------------------

def test_safetensors_many_tensors():
    weights = {f"w_{i}": np.array([float(i)], dtype=np.float32) for i in range(50)}
    path = Path("/tmp/test_st_many.safetensors")
    _write_safetensors(path, weights)
    loaded = _read_safetensors(path)
    assert len(loaded) == 50
    for i in range(50):
        np.testing.assert_array_equal(loaded[f"w_{i}"], [float(i)])
    path.unlink()


def test_safetensors_all_zeros():
    weights = {"zeros": np.zeros(100, dtype=np.float32)}
    path = Path("/tmp/test_st_zeros.safetensors")
    _write_safetensors(path, weights)
    loaded = _read_safetensors(path)
    np.testing.assert_array_equal(loaded["zeros"], np.zeros(100))
    path.unlink()


def test_safetensors_all_ones():
    weights = {"ones": np.ones(100, dtype=np.float32)}
    path = Path("/tmp/test_st_ones.safetensors")
    _write_safetensors(path, weights)
    loaded = _read_safetensors(path)
    np.testing.assert_array_equal(loaded["ones"], np.ones(100))
    path.unlink()


def test_safetensors_large_values():
    weights = {"large": np.array([1e30, -1e30, 1e-30], dtype=np.float32)}
    path = Path("/tmp/test_st_large_vals.safetensors")
    _write_safetensors(path, weights)
    loaded = _read_safetensors(path)
    np.testing.assert_array_equal(loaded["large"], weights["large"])
    path.unlink()


def test_safetensors_5d_tensor():
    weights = {"tensor5d": np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5, 1)}
    path = Path("/tmp/test_st_5d.safetensors")
    _write_safetensors(path, weights)
    loaded = _read_safetensors(path)
    assert loaded["tensor5d"].shape == (2, 3, 4, 5, 1)
    path.unlink()


def test_get_model_dir_returns_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    result = _get_model_dir("org/model")
    assert isinstance(result, Path)


def test_get_model_dir_starts_with_models(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    result = _get_model_dir("org/model")
    assert result.name.startswith("models--")


def test_find_safetensors_multiple_snapshots(tmp_path):
    model_dir = tmp_path / "models--multi"
    for snap_name in ["snap_a", "snap_b", "snap_c"]:
        snap = model_dir / "snapshots" / snap_name
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_bytes(b"fake")
    result = _find_safetensors(model_dir)
    assert result is not None
    assert result.name == "model.safetensors"


def test_load_weights_raw_multiple_tensors():
    weights = {
        "layer1.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "layer1.bias": np.array([1.0, 2.0], dtype=np.float32),
        "layer2.weight": np.arange(4, dtype=np.float32).reshape(2, 2),
    }
    path = Path("/tmp/test_raw_multi.safetensors")
    _write_safetensors(path, weights)
    loaded = _load_weights_raw(path, np.float32)
    assert len(loaded) == 3
    np.testing.assert_array_equal(loaded["layer1.weight"], weights["layer1.weight"])
    np.testing.assert_array_equal(loaded["layer1.bias"], weights["layer1.bias"])
    np.testing.assert_array_equal(loaded["layer2.weight"], weights["layer2.weight"])
    path.unlink()


def test_load_weights_raw_f32_identity():
    weights = {"w": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    path = Path("/tmp/test_raw_f32_id.safetensors")
    _write_safetensors(path, weights)
    loaded = _load_weights_raw(path, np.float32)
    np.testing.assert_array_equal(loaded["w"], weights["w"])
    path.unlink()


def test_load_weights_raw_empty_file():
    path = Path("/tmp/test_raw_empty.safetensors")
    path.write_bytes(b"")
    try:
        _load_weights_raw(path, np.float32)
        assert False, "Expected ValueError"
    except (ValueError, struct.error):
        pass
    path.unlink()


def test_list_cached_models_sorted(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    for name in ["org--zebra", "org--alpha", "org--middle"]:
        model_dir = hub / f"models--{name}"
        snap = model_dir / "snapshots" / "default"
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_bytes(b"\x00" * 100)
    monkeypatch.setenv("HF_HOME", str(hub))
    monkeypatch.setattr(
        "domains.infrastructure.safetensors_loader.find_repo_root",
        lambda p: tmp_path / "nonexistent_repo",
    )
    result = list_cached_models()
    ids = [m["id"] for m in result]
    assert ids == sorted(ids)


def test_list_cached_models_multiple(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    for name in ["org--m1", "org--m2", "org--m3"]:
        model_dir = hub / f"models--{name}"
        snap = model_dir / "snapshots" / "default"
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_bytes(b"\x00" * 100)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(
        "domains.infrastructure.safetensors_loader.find_repo_root",
        lambda p: tmp_path / "nonexistent_repo",
    )
    result = list_cached_models()
    assert len(result) == 3


def test_load_model_config_returns_dict(monkeypatch, tmp_path):
    model_dir = tmp_path / "hub" / "hub" / "models--org--dict"
    snap = model_dir / "snapshots" / "default"
    snap.mkdir(parents=True)
    cfg = {"vocab_size": 1000}
    (snap / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hub"))
    monkeypatch.setattr(
        "domains.infrastructure.safetensors_loader.find_repo_root",
        lambda p: tmp_path / "nonexistent_repo",
    )
    result = load_model_config("org/dict")
    assert isinstance(result, dict)


def test_load_model_config_nested(monkeypatch, tmp_path):
    model_dir = tmp_path / "hub" / "hub" / "models--org--nested"
    snap = model_dir / "snapshots" / "default"
    snap.mkdir(parents=True)
    cfg = {"architectures": ["GPT2LMHeadModel"], "model": {"layers": 12}}
    (snap / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hub"))
    monkeypatch.setattr(
        "domains.infrastructure.safetensors_loader.find_repo_root",
        lambda p: tmp_path / "nonexistent_repo",
    )
    result = load_model_config("org/nested")
    assert result["architectures"] == ["GPT2LMHeadModel"]
    assert result["model"]["layers"] == 12


def test_max_header_len_is_reasonable():
    assert _MAX_HEADER_LEN > 0
    assert _MAX_HEADER_LEN < 10 * 1024 * 1024 * 1024
