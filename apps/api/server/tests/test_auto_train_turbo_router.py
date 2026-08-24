from infrastructure.exception_handlers import register_app_error_handler
"""
Tests for the /auto-train/start-turbo + /auto-train/turbo/status endpoints.

The turbo trainer runs SloughGPTTrainer on a background daemon thread and
publishes live telemetry (global_step, total_steps, steps_per_sec, eta_s,
elapsed_s) into module-level ``_turbo_state``. Only the auto_train router is
registered; the real trainer is replaced with a fake whose ``train()`` invokes
the ``on_progress`` callback so the progress plumbing is exercised end to end.
"""

import threading
import time

import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routers.auto_train as mod
from routers.auto_train import router as auto_train_router

app = FastAPI()
register_app_error_handler(app)
app.include_router(auto_train_router)
client = TestClient(app)


_IDLE = {
    "status": "idle",
    "job_id": None,
    "global_step": 0,
    "total_steps": 0,
    "progress": 0.0,
    "loss": None,
    "learning_rate": None,
    "steps_per_sec": None,
    "eta_s": None,
    "elapsed_s": None,
    "result": None,
    "error": None,
}


@pytest.fixture(autouse=True)
def _reset_turbo(tmp_path):
    """Reset the module-level turbo state/events and redirect all storage dirs."""
    mod._turbo_state.update(_IDLE)
    mod._turbo_cancel_event = threading.Event()
    inst = mod._auto_train_instance
    with patch.object(inst, "REPO_ROOT", tmp_path), \
         patch.object(inst, "TURBO_DIR", tmp_path / "models" / "turbo-trained"), \
         patch.object(inst, "CHECKPOINTS_DIR", tmp_path / "models" / "auto-training"), \
         patch.object(inst, "LORA_DIR", tmp_path / "data" / "user_adapters"):
        yield


def _wait_for_status(desired, timeout=5.0):
    """Poll GET /auto-train/turbo/status until it reaches ``desired``."""
    deadline = time.time() + timeout
    body = None
    while time.time() < deadline:
        body = client.get("/auto-train/turbo/status").json()
        if body["status"] == desired:
            return body
        time.sleep(0.05)
    raise AssertionError(f"status never became {desired!r}; last={body!r}")


def _fake_train_result(blocker=None, error=None, cancel=False):
    def _train(**kwargs):
        on_progress = kwargs.get("on_progress")
        if on_progress is not None:
            on_progress({
                "global_step": 42,
                "total_steps": 500,
                "progress_percent": 8.4,
                "train_loss": 2.31,
                "learning_rate": 0.0003,
                "steps_per_sec": 4.25,
                "eta_s": 98,
                "elapsed_s": 20,
            })
        if cancel:
            kwargs.get("cancel_event", threading.Event()).set()
        if blocker is not None:
            blocker.wait(timeout=5)
        if error is not None:
            raise error
        return {"status": "ok", "final_loss": 0.5, "total_steps": 500,
                "model_path": "/tmp/fake.soul"}
    return _train


@patch("domains.training.train_pipeline.SloughGPTTrainer")
def test_turbo_status_idle(_):
    body = client.get("/auto-train/turbo/status").json()["data"]
    assert body["status"] == "idle"
    assert body["job_id"] is None
    assert body["global_step"] == 0


def test_start_turbo_missing_data():
    resp = client.post("/auto-train/start-turbo", json={})
    assert resp.status_code == 422
    body = resp.json()
    assert "error" in body


def test_start_turbo_rejects_when_running():
    mod._turbo_state["status"] = "running"
    mod._turbo_state["job_id"] = "turbo_123"
    resp = client.post("/auto-train/start-turbo", json={"data_path": "x.txt"})
    body = resp.json()
    assert "error" in body
    assert "already running" in body["error"]


@patch("domains.training.train_pipeline.SloughGPTTrainer")
def test_start_turbo_validates_epochs(MockTrainer):
    resp = client.post("/auto-train/start-turbo", json={"data_path": "x.txt", "epochs": 0})
    assert resp.status_code == 422
    MockTrainer.assert_not_called()


@patch("domains.training.train_pipeline.SloughGPTTrainer")
def test_turbo_progress_and_complete(MockTrainer):
    blocker = threading.Event()
    MockTrainer.return_value.train.side_effect = _fake_train_result(blocker=blocker)

    resp = client.post("/auto-train/start-turbo", json={"data_path": "x.txt"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "started"
    assert body["job_id"]

    running = _wait_for_status("running")
    assert running["job_id"] == body["job_id"]
    assert running["global_step"] == 42
    assert running["total_steps"] == 500
    assert running["progress"] == 8.4
    assert running["loss"] == 2.31
    assert running["learning_rate"] == 0.0003
    assert running["steps_per_sec"] == 4.25
    assert running["eta_s"] == 98
    assert running["elapsed_s"] == 20

    blocker.set()
    complete = _wait_for_status("complete")
    assert complete["progress"] == 100.0
    assert complete["result"]["final_loss"] == 0.5
    assert complete["result"]["total_steps"] == 500


@patch("domains.training.train_pipeline.SloughGPTTrainer")
def test_turbo_cancelled(MockTrainer):
    MockTrainer.return_value.train.side_effect = _fake_train_result(cancel=True)
    client.post("/auto-train/start-turbo", json={"data_path": "x.txt"})
    body = _wait_for_status("error")
    assert body["error"] == "Training cancelled"


@patch("domains.training.train_pipeline.SloughGPTTrainer")
def test_turbo_trainer_error(MockTrainer):
    MockTrainer.return_value.train.side_effect = _fake_train_result(error=RuntimeError("boom"))
    client.post("/auto-train/start-turbo", json={"data_path": "x.txt"})
    body = _wait_for_status("error")
    assert body["error"] == "boom"


def _fake_turbo_soul(tmp_path, name):
    """Create a listing-visible fake .soul + .meta.json sidecar in a temp dir."""
    soul = tmp_path / name
    soul.write_bytes(b"\x00" * 5000)
    (tmp_path / (name + ".meta.json")).write_text(
        '{"soul_name": "turbo-test", "final_train_loss": 0.42, '
        '"epochs_trained": 3, "metadata": {"steps": 120, "avg_loss": 0.42}}'
    )
    return soul


def test_checkpoints_includes_turbo_models(tmp_path):
    inst = mod._auto_train_instance
    ckpt_dir = tmp_path / "ckpt"
    turbo_dir = tmp_path / "turbo"
    ckpt_dir.mkdir()
    turbo_dir.mkdir()
    with patch.object(inst, "CHECKPOINTS_DIR", ckpt_dir), patch.object(inst, "TURBO_DIR", turbo_dir):
        _fake_turbo_soul(turbo_dir, "turbo_123.soul")
        _fake_turbo_soul(ckpt_dir, "reg_1.soul")
        body = client.get("/auto-train/checkpoints").json()
        names = [c["name"] for c in body["data"]]
        assert "turbo_123.soul" in names
        turbo = next(c for c in body["data"] if c["name"] == "turbo_123.soul")
        assert turbo["source"] == "turbo"
        assert turbo["loss"] == 0.42
        reg = next(c for c in body["data"] if c["name"] == "reg_1.soul")
        assert "source" not in reg


def test_checkpoint_info_resolves_turbo_dir(tmp_path):
    inst = mod._auto_train_instance
    with patch.object(inst, "CHECKPOINTS_DIR", tmp_path / "empty"), patch.object(inst, "TURBO_DIR", tmp_path):
        _fake_turbo_soul(tmp_path, "turbo_9.soul")
        resp = client.get("/auto-train/checkpoints/turbo_9.soul/info")
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "turbo_9.soul"


def test_checkpoint_delete_turbo(tmp_path):
    inst = mod._auto_train_instance
    with patch.object(inst, "CHECKPOINTS_DIR", tmp_path / "empty"), patch.object(inst, "TURBO_DIR", tmp_path):
        _fake_turbo_soul(tmp_path, "turbo_7.soul")
        (tmp_path / "empty").mkdir(exist_ok=True)
        resp = client.delete("/auto-train/checkpoints/turbo_7.soul")
        assert resp.status_code == 200
        assert "deleted" in resp.json()["message"]
        assert not (tmp_path / "turbo_7.soul").exists()


def test_find_checkpoint_prefers_checkpoints_then_turbo(tmp_path):
    inst = mod._auto_train_instance
    ckpt_dir = tmp_path / "ckpt"
    turbo_dir = tmp_path / "turbo"
    ckpt_dir.mkdir()
    turbo_dir.mkdir()
    with patch.object(inst, "CHECKPOINTS_DIR", ckpt_dir), patch.object(inst, "TURBO_DIR", turbo_dir):
        (ckpt_dir / "dup.soul").write_bytes(b"\x00" * 5000)
        (turbo_dir / "dup.soul").write_bytes(b"\x00" * 5000)
        (turbo_dir / "only_turbo.soul").write_bytes(b"\x00" * 5000)
        assert inst._find_checkpoint("dup.soul") == ckpt_dir / "dup.soul"
        assert inst._find_checkpoint("only_turbo") == turbo_dir / "only_turbo.soul"
        assert inst._find_checkpoint("missing") is None


def _write_slo(dirpath, name="sage", **fields):
    """Write a plain-text .slo personality profile into ``dirpath``."""
    path = dirpath / f"{name}.slo"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"SOUL {name}"]
    for key, value in fields.items():
        lines.append(f"{key.upper()} {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_soul_meta_reads_slo_profile(tmp_path):
    """Plain-text .slo personality profiles are parsed as checkpoint metadata."""
    inst = mod._auto_train_instance
    slo_path = _write_slo(
        inst.CHECKPOINTS_DIR,
        tagline="A quiet mind",
        description="Calm and reflective",
        lineage="slonet",
        basemodel="sloughgpt-native",
        system="You are calm.",
    )
    meta = mod._load_soul_meta(slo_path)
    assert meta["soul_name"] == "sage"
    assert meta["tagline"] == "A quiet mind"
    assert meta["lineage"] == "slonet"
    assert meta["base_model"] == "sloughgpt-native"
    assert meta["system_prompt"] == "You are calm."


def test_load_soul_from_slo_profile(tmp_path):
    """A .slo profile in the catalog resolves its metadata, not 'unknown'."""
    inst = mod._auto_train_instance
    _write_slo(inst.CHECKPOINTS_DIR, tagline="A quiet mind", lineage="slonet")
    info = inst._load_soul("sage.slo")
    assert info["soul"] == "sage"
    assert info["tagline"] == "A quiet mind"
    assert info["lineage"] == "slonet"
    assert info["model_type"] == "slonet"


def test_export_mobile_rejects_slo_metadata(tmp_path):
    """A .slo profile is metadata, not a loadable model — export-mobile must reject it."""
    inst = mod._auto_train_instance
    _write_slo(inst.CHECKPOINTS_DIR, tagline="A quiet mind")
    resp = client.get("/auto-train/checkpoints/sage.slo/export-mobile")
    assert resp.status_code == 404
    resp = client.get("/auto-train/checkpoints/sage.slo/download")
    assert resp.status_code == 200


def test_turbo_real_training_end_to_end(tmp_path):
    """Run a real tiny SloughGPTTrainer through the API; verify catalog + load.

    The trainer is NOT mocked here — a small corpus + tiny model config trains
    in well under a second, so the full chain (start -> status -> checkpoint in
    catalog with source=turbo -> load for chat) is exercised for real.
    """
    data = tmp_path / "data.txt"
    data.write_text("hello turbo world, this is a tiny training corpus. " * 20)

    resp = client.post("/auto-train/start-turbo", json={
        "data_path": str(data),
        "epochs": 1,
        "batch_size": 2,
        "block_size": 32,
        "n_embed": 32,
        "n_layer": 1,
        "n_head": 2,
        "learning_rate": 0.001,
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"

    complete = _wait_for_status("complete", timeout=60.0)
    assert complete["progress"] == 100.0
    result = complete["result"]
    assert result["total_steps"] >= 1
    assert isinstance(result["final_loss"], float)
    model_path = str(result["model_path"])
    assert model_path.endswith(".soul")

    ckpt_name = model_path.split("/")[-1]
    ckpt_body = client.get("/auto-train/checkpoints").json()
    names = [c["name"] for c in ckpt_body["data"]]
    assert ckpt_name in names
    turbo = next(c for c in ckpt_body["data"] if c["name"] == ckpt_name)
    assert turbo["source"] == "turbo"

    load = client.post(f"/auto-train/checkpoints/{ckpt_name}/load")
    assert load.status_code == 200
    load_body = load.json()
    assert load_body["status"] == "success"
    assert load_body["data"]["name"] == ckpt_name
