from infrastructure.exception_handlers import register_app_error_handler

"""
Tests for the /training/finetuned-models endpoints (list, load, delete).

Only registers the training router. Filesystem access is redirected to a
temporary dir via ``_finetuned_dir`` and the models controller is mocked.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from training import router as training_router

app = FastAPI()
register_app_error_handler(app)
app.include_router(training_router)
client = TestClient(app)

# Track the "real" temp dir so _finetuned_dir can be patched once and reused.
_FINETUNED = tempfile.mkdtemp(prefix="slough-finetuned-")


@pytest.fixture(autouse=True)
def _point_at_tmpdir():
    """Redirect the router's finetuned dir to a temp location and reset it."""
    import importlib

    _rt = importlib.import_module("training.router")
    base = Path(_FINETUNED)
    for child in base.iterdir():
        if child.is_dir():
            import shutil

            shutil.rmtree(child)
        else:
            child.unlink()
    _rt._finetuned_models_cache = None
    with patch.object(_rt, "_finetuned_dir", return_value=base):
        yield
    _rt._finetuned_models_cache = None


def _make_model(name="gpt2_finetune_a_1"):
    """Create a fake fine-tuned model dir with a config.json and a weight file."""
    base = Path(_FINETUNED)
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text('{"architectures": ["GPT2LMHeadModel"]}')
    (d / "model.safetensors").write_bytes(b"\x00" * (2 * 1024 * 1024))
    return d


# ── GET /training/finetuned-models ─────────────────────────────────────────


def test_list_finetuned_empty():
    resp = client.get("/training/finetuned-models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["models"] == []


def test_list_finetuned_returns_dirs():
    _make_model("gpt2_finetune_a_1")
    _make_model("qwen_ft_b_2")
    resp = client.get("/training/finetuned-models")
    assert resp.status_code == 200
    models = resp.json()["models"]
    names = {m["name"] for m in models}
    assert names == {"gpt2_finetune_a_1", "qwen_ft_b_2"}
    gpt2 = next(m for m in models if m["name"] == "gpt2_finetune_a_1")
    assert gpt2["model"] == "gpt2"
    assert gpt2["dataset"] == "finetune"
    assert gpt2["size_mb"] > 0
    assert gpt2["model_path"].endswith("gpt2_finetune_a_1")
    # Legacy keys consumed by the shell `finetuned` command table.
    assert gpt2["model_name"] == "gpt2_finetune_a_1"
    assert gpt2["size_bytes"] > 0
    assert gpt2["epochs"] == 0


def test_list_finetuned_newest_first():
    _make_model("older")
    import time

    time.sleep(0.01)
    _make_model("newer")
    models = client.get("/training/finetuned-models").json()["models"]
    assert models[0]["name"] == "newer"


def test_list_finetuned_prefers_metadata_over_dir_name():
    import json

    d = _make_model("legacy_weird_name")
    (d / "metadata.json").write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "dataset": "my_awesome_dataset",
                "final_loss": 0.42,
                "epochs": 3,
            }
        )
    )
    gpt2 = next(
        m
        for m in client.get("/training/finetuned-models").json()["models"]
        if m["name"] == "legacy_weird_name"
    )
    assert gpt2["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert gpt2["dataset"] == "my_awesome_dataset"
    assert gpt2["final_loss"] == 0.42
    assert gpt2["epochs"] == 3


def test_list_finetuned_falls_back_to_dir_parse_without_metadata():
    _make_model("gpt2_finetune_a_1")
    gpt2 = next(
        m
        for m in client.get("/training/finetuned-models").json()["models"]
        if m["name"] == "gpt2_finetune_a_1"
    )
    assert gpt2["model"] == "gpt2"
    assert gpt2["dataset"] == "finetune"
    assert gpt2["final_loss"] is None
    assert gpt2["epochs"] == 0


# ── POST /training/finetuned-models/{name}/load ────────────────────────────


def test_load_finetuned_delegates_to_controller():
    _make_model("gpt2_finetune_a_1")
    ctrl = MagicMock()
    ctrl.load_model_path.return_value = {
        "status": "loaded",
        "model_id": "gpt2_finetune_a_1",
        "type": "slonet",
        "device": "cpu",
        "model_path": str(Path(_FINETUNED) / "gpt2_finetune_a_1"),
    }
    with patch("controllers.models.get_models_controller", return_value=ctrl):
        resp = client.post("/training/finetuned-models/gpt2_finetune_a_1/load")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "loaded",
        "name": "gpt2_finetune_a_1",
        "model_path": str(Path(_FINETUNED) / "gpt2_finetune_a_1"),
        "model_id": "gpt2_finetune_a_1",
    }
    ctrl.load_model_path.assert_called_once()
    args, kwargs = ctrl.load_model_path.call_args
    assert args[1] == "cpu"
    assert str(Path(_FINETUNED) / "gpt2_finetune_a_1") in args[0]
    assert "device" not in kwargs


def test_load_finetuned_not_found():
    resp = client.post("/training/finetuned-models/does_not_exist/load")
    assert resp.status_code == 404


def test_load_finetuned_controller_error():
    _make_model("broken")
    ctrl = MagicMock()
    ctrl.load_model_path.return_value = {"status": "error", "error": "boom"}
    with patch("controllers.models.get_models_controller", return_value=ctrl):
        resp = client.post("/training/finetuned-models/broken/load")
    assert resp.status_code == 500
    assert "boom" in resp.json()["error"]


# ── DELETE /training/finetuned-models/{name} ───────────────────────────────


def test_delete_finetuned_removes_dir():
    d = _make_model("gpt2_finetune_a_1")
    resp = client.delete("/training/finetuned-models/gpt2_finetune_a_1")
    assert resp.status_code == 200
    assert resp.json() == {"status": "deleted", "name": "gpt2_finetune_a_1"}
    assert not d.exists()


def test_delete_finetuned_not_found():
    resp = client.delete("/training/finetuned-models/nope")
    assert resp.status_code == 404


def test_delete_finetuned_rejects_path_traversal():
    sibling = Path(tempfile.mkdtemp(prefix="slough-sibling-"))
    (sibling / "pwned.txt").write_text("x")
    resp = client.delete(f"/training/finetuned-models/..%2F{sibling.name}")
    # Traversal is rejected either by Starlette path normalization (404) or by
    # the router's guard (400). The guarantee: nothing outside the base is removed.
    assert resp.status_code in (400, 404)
    assert (sibling / "pwned.txt").exists()


# ── GET /training/jobs — internal fields never leak into responses ─────────


@pytest.fixture(autouse=True)
def _clean_training_jobs():
    """Reset the in-memory job registry between tests."""
    import training.jobs as tj

    tj.training_jobs.clear()
    yield
    tj.training_jobs.clear()


def test_list_jobs_serializes_with_internal_fields():
    """A job carrying a threading.Event cancel handle must serialize cleanly."""
    import threading

    import training.jobs as tj

    tj.training_jobs["job_1"] = {
        "status": "running",
        "model": "gpt2",
        "dataset": "shakespeare",
        "progress": 50,
        "_cancel_event": threading.Event(),
    }

    resp = client.get("/training/jobs")
    assert resp.status_code == 200
    jobs = resp.json()
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "running"
    assert job["status_message"].startswith("Training gpt2 on shakespeare, 50% done")
    assert "_cancel_event" not in job
    assert all(not k.startswith("_") for k in job)


def test_get_single_job_serializes_with_internal_fields():
    """GET /training/jobs/{id} must also strip internal (underscore) fields."""
    import threading

    import training.jobs as tj

    tj.training_jobs["job_2"] = {
        "status": "completed",
        "model": "gpt2",
        "dataset": "shakespeare",
        "checkpoint": "models/gpt2_shakespeare_trained.soul",
        "_cancel_event": threading.Event(),
    }

    resp = client.get("/training/jobs/job_2")
    assert resp.status_code == 200
    job = resp.json()
    assert job["status"] == "completed"
    assert "_cancel_event" not in job
