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

from domains.infrastructure.safetensors_loader import (
    _find_safetensors,
    _get_model_dir,
    load_model_weights,
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


def test_download_hf_model_output_consumable_by_app_loader(tmp_path, monkeypatch):
    import domains.infrastructure.hf_hub as hub_mod
    import downcraft.state as st_mod
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

        # Layout agreement: downcraft writes to HF_HOME/hub/models--<id>,
        # the same dir the app loader resolves.
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
