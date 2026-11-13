"""Goal 17 hermetic LoRA ship journey.

Covers the full user path against the real training router:
  POST /training/lora-finetune → poll GET /training/jobs/{id}
  → POST /training/load-adapter → POST /training/unload-adapter

Fixture is a hand-built tiny GPT-2-style .slnc + tokenizer.json under tmp_path
(no HF download, no real weights). find_repo_root is mocked so model/dataset
path guards resolve inside the temp tree.
"""

from __future__ import annotations

import json
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.conftest import build_test_app

N_LAYER, N_EMBD, N_HEAD, N_INNER, VOCAB, NPOS = 1, 32, 2, 128, 128, 64
WEIGHT_STD = 0.02


def _entry_size(name: str, arr: np.ndarray) -> int:
    return 4 + len(name.encode()) + 8 + 4 + 4 + 4 * arr.ndim + 4 + 4


def _hf_weights(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)

    def r(*shape: int) -> np.ndarray:
        return (rng.randn(*shape) * WEIGHT_STD).astype(np.float32)

    w: dict[str, np.ndarray] = {}
    for i in range(N_LAYER):
        w[f"h.{i}.ln_1.weight"] = np.ones(N_EMBD, dtype=np.float32)
        w[f"h.{i}.ln_1.bias"] = np.zeros(N_EMBD, dtype=np.float32)
        w[f"h.{i}.attn.c_attn.weight"] = r(N_EMBD, 3 * N_EMBD)
        w[f"h.{i}.attn.c_attn.bias"] = r(3 * N_EMBD)
        w[f"h.{i}.attn.c_proj.weight"] = r(N_EMBD, N_EMBD)
        w[f"h.{i}.attn.c_proj.bias"] = r(N_EMBD)
        w[f"h.{i}.ln_2.weight"] = np.ones(N_EMBD, dtype=np.float32)
        w[f"h.{i}.ln_2.bias"] = np.zeros(N_EMBD, dtype=np.float32)
        w[f"h.{i}.mlp.c_fc.weight"] = r(N_EMBD, N_INNER)
        w[f"h.{i}.mlp.c_fc.bias"] = r(N_INNER)
        w[f"h.{i}.mlp.c_proj.weight"] = r(N_INNER, N_EMBD)
        w[f"h.{i}.mlp.c_proj.bias"] = r(N_EMBD)
    w["wte.weight"] = r(VOCAB, N_EMBD)
    w["wpe.weight"] = r(NPOS, N_EMBD)
    w["ln_f.weight"] = np.ones(N_EMBD, dtype=np.float32)
    w["ln_f.bias"] = np.zeros(N_EMBD, dtype=np.float32)
    return w


def build_tiny_slnc(path: Path) -> None:
    from domain.infrastructure._internal.slnc.spec import compute_header_size

    weights = _hf_weights()
    config = {
        "architectures": ["GPT2LMHeadModel"],
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
        "vocab_size": VOCAB,
        "n_inner": N_INNER,
        "n_positions": NPOS,
        "n_ctx": NPOS,
        "hidden_act": "gelu",
        "layer_norm_type": "layer_norm",
    }
    jb = json.dumps(config, sort_keys=True).encode()
    header_size = compute_header_size(jb)
    table_size = sum(_entry_size(n, a) for n, a in weights.items())
    data_start = header_size + table_size
    table = b""
    entries: list[np.ndarray] = []
    cur = data_start
    for name, arr in weights.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        nb = name.encode()
        ndim = arr.ndim
        crc = zlib.crc32(arr.tobytes()) & 0xFFFFFFFF
        table += struct.pack("<I", len(nb)) + nb
        table += struct.pack("<Q", cur)
        table += struct.pack("<I", arr.nbytes)
        table += struct.pack("<I", ndim)
        table += struct.pack(f"<{ndim}I", *arr.shape)
        table += struct.pack("<I", 0)
        table += struct.pack("<I", crc)
        cur += arr.nbytes
        entries.append(arr)
    assert len(table) == table_size
    meta = (
        struct.pack(
            "<10I",
            N_LAYER,
            N_EMBD,
            N_HEAD,
            N_INNER,
            VOCAB,
            NPOS,
            N_LAYER,
            NPOS,
            len(weights),
            data_start,
        )
        + b"\x00" * 24
    )
    pad = header_size - (4 + 4 + 4 + 64 + 4 + len(jb))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"SLNC")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", 0))
        f.write(meta)
        f.write(struct.pack("<I", len(jb)))
        f.write(jb)
        f.write(b"\x00" * pad)
        f.write(table)
        for arr in entries:
            f.write(arr.tobytes())


def write_tokenizer_json(path: Path, vocab_size: int = VOCAB) -> None:
    alphabet = [chr(i) for i in range(32, 127)] + ["\n"]
    alphabet = alphabet[:vocab_size]
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    for i in range(len(alphabet), vocab_size):
        vocab[f"<unused_{i}>"] = i
    tok = {
        "model": {"type": "BPE", "vocab": vocab, "merges": [], "eos_token_id": vocab_size - 1},
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "."}, "invert": False},
    }
    path.write_text(json.dumps(tok))


@pytest.fixture
def journey_env(tmp_path, monkeypatch):
    """Stage model + dataset under a mocked repo root and return a TestClient."""
    from apps.api.server.training.jobs import training_jobs
    from apps.api.server.training.router import router as training_router
    from domain.infrastructure.server_state import reset_server_state

    reset_server_state()
    training_jobs.clear()

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    slnc = models_dir / "tiny.slnc"
    build_tiny_slnc(slnc)
    write_tokenizer_json(models_dir / "tokenizer.json")

    ds_dir = tmp_path / "datasets" / "journey_ds"
    ds_dir.mkdir(parents=True, exist_ok=True)
    cycle = "abcdefgjkl"
    corpus = "".join(cycle[i % len(cycle)] for i in range(400))
    (ds_dir / "input.txt").write_text(corpus)

    out_dir = tmp_path / "adapters"
    out_dir.mkdir(parents=True, exist_ok=True)

    from apps.api.server.training import lora as lora_mod

    monkeypatch.setattr(lora_mod, "find_repo_root", lambda *a, **k: tmp_path)

    app = build_test_app(training_router)
    client = TestClient(app, raise_server_exceptions=False)

    yield {
        "client": client,
        "tmp_path": tmp_path,
        "slnc": slnc,
        "out_dir": out_dir,
        "dataset": "journey_ds",
        "corpus": corpus,
    }

    training_jobs.clear()
    reset_server_state()


def _poll_job(client: TestClient, job_id: str, timeout_s: float = 30.0) -> dict:
    deadline = time.time() + timeout_s
    last: dict = {}
    while time.time() < deadline:
        resp = client.get(f"/training/jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        last = resp.json()
        if last.get("status") in ("completed", "failed", "stopped"):
            return last
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {last}")


def test_lora_ship_journey(journey_env):
    """Train → poll → load adapter → unload, end-to-end via HTTP."""
    client = journey_env["client"]
    out_dir: Path = journey_env["out_dir"]
    slnc: Path = journey_env["slnc"]

    resp = client.post(
        "/training/lora-finetune",
        json={
            "model_path": str(slnc),
            "dataset": journey_env["dataset"],
            "name": "goal17-journey",
            "rank": 2,
            "alpha": 4.0,
            "epochs": 1,
            "batch_size": 8,
            "block_size": 16,
            "learning_rate": 1e-2,
            "warmup_steps": 0,
            "weight_decay": 0.0,
            "log_interval": 1,
            "output_dir": str(out_dir),
            "adapter_name": "journey_lora",
        },
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    job = _poll_job(client, job_id, timeout_s=60.0)
    assert job["status"] == "completed", job
    assert job.get("error") is None

    history = job.get("loss_history") or []
    assert history, f"loss_history empty: {job}"
    losses = [h["value"] for h in history if h.get("value") is not None]
    assert losses, job
    assert all(np.isfinite(v) for v in losses), losses[:5]
    assert job.get("rank") == 2
    assert job.get("alpha") == 4.0

    adapter_path = Path(job["checkpoint"])
    assert adapter_path.is_file(), adapter_path
    assert adapter_path.name == "journey_lora.npz"
    result = job.get("result") or {}
    assert result.get("adapter_path") == str(adapter_path)
    assert (result.get("total_steps") or 0) > 0

    # Load adapter into a real provider holding a fresh tiny model.
    from domain.inference._internal.slonet_provider import SloNetChatProvider
    from domain.infrastructure.server_state import get_server_state
    from domain.training._internal.lora import get_lora_parameters

    provider = SloNetChatProvider.from_slnc(str(slnc), model_id="tiny")
    get_server_state().model.set(provider)
    model = provider._model
    assert not get_lora_parameters(model)

    load_resp = client.post(
        "/training/load-adapter",
        json={"adapter_path": str(adapter_path), "merge": False},
    )
    assert load_resp.status_code == 200, load_resp.text
    load_body = load_resp.json()
    assert load_body["status"] == "loaded"
    assert load_body["rank"] == 2
    assert load_body["alpha"] == 4.0
    assert load_body["n_params"] > 0
    assert "W_q" in load_body["target_modules"]
    assert get_lora_parameters(model)

    unload_resp = client.post("/training/unload-adapter")
    assert unload_resp.status_code == 200, unload_resp.text
    assert unload_resp.json()["status"] == "unloaded"
    assert not get_lora_parameters(model)
