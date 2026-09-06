"""
Tests for audit-log instrumentation on privileged operations.

Verifies that privileged endpoints emit structured audit events via
``AuditLogger.log`` (event, resource, actor, detail/extra) without affecting
the endpoint's response, and that a failing audit logger never breaks the
underlying operation.
"""

import json
import struct
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.models import ModelsRouter
from apps.api.server.routers.souls import SoulsRouter
from apps.api.server.routers.auto_train import AutoTrainRouter
from apps.api.server.routers.datasets import DatasetsRouter
from apps.api.server.routers.kb import KBRouter
from apps.api.server.routers.agents import AgentsRouter
from apps.api.server.routers.multimodal import MultimodalRouter
from apps.api.server.infrastructure.exception_handlers import register_all_handlers
from apps.api.server.routers.config import ConfigRouter
from apps.api.server.routers.experiments import ExperimentsRouter
from apps.api.server.routers.user_adapters import UserAdaptersRouter
from apps.api.server.routers.lora_eval import LoraEvalRouter
from apps.api.server.routers.tokenizer import TokenizerRouter
from apps.api.server.routers.system import SystemRouter
from apps.api.server.routers.self_train import SelfTrainRouter


@pytest.fixture
def models_client():
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(ModelsRouter().router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def souls_client():
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(SoulsRouter().router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auto_train_client(tmp_path, monkeypatch):
    rtr = AutoTrainRouter()
    rtr.CHECKPOINTS_DIR = tmp_path / "checkpoints"
    rtr.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    import domains.training.service as _svc
    monkeypatch.setattr(_svc, "CHECKPOINTS_DIR", rtr.CHECKPOINTS_DIR)
    app = FastAPI()
    register_all_handlers(app)
    app.state.checkpoint_dir = rtr.CHECKPOINTS_DIR
    app.include_router(rtr.router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def datasets_client():
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(DatasetsRouter().router)
    return TestClient(app, raise_server_exceptions=False)


def _make_fake_sou(path, meta=None):
    meta = meta or {
        "stoi": {"a": 0, "b": 1},
        "itos": {0: "a", 1: "b"},
        "final_train_loss": 0.5,
        "total_steps": 10,
    }
    payload = json.dumps(meta).encode()
    blob = b"\x00" * 8 + struct.pack("<I", len(payload)) + payload + b"\x00" * 64
    path.write_bytes(blob)


class TestModelAudit:
    """POST /models/load and POST /models/unload emit audit events."""

    @patch("apps.api.server.routers.models.get_models_controller")
    @patch("infrastructure.auth.get_audit_logger")
    def test_load_model_logs_event(self, mock_logger, mock_ctrl, models_client):
        ctrl = MagicMock()
        ctrl.load_model.return_value = {"status": "loaded", "device": "cpu"}
        mock_ctrl.return_value = ctrl
        resp = models_client.post("/models/load", json={"model_id": "gpt2", "device": "cpu"})
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "model.load"
        assert kwargs["resource"] == "gpt2"
        assert kwargs["user"] == "anonymous"
        assert kwargs["extra"] == {"device": "cpu", "quantize": None}

    @patch("apps.api.server.routers.models.get_models_controller")
    @patch("infrastructure.auth.get_audit_logger")
    def test_unload_model_logs_event(self, mock_logger, mock_ctrl, models_client):
        ctrl = MagicMock()
        ctrl.unload_model.return_value = {"status": "unloaded"}
        ctrl._current_model = "gpt2"
        mock_ctrl.return_value = ctrl
        resp = models_client.post("/models/unload")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "model.unload"
        assert kwargs["resource"] == "gpt2"

    @patch("apps.api.server.routers.models.get_models_controller")
    @patch("infrastructure.auth.get_audit_logger", side_effect=RuntimeError("broken"))
    def test_audit_failure_does_not_break_load(self, mock_logger, mock_ctrl, models_client):
        ctrl = MagicMock()
        ctrl.load_model.return_value = {"status": "loaded", "device": "cpu"}
        mock_ctrl.return_value = ctrl
        resp = models_client.post("/models/load", json={"model_id": "gpt2", "device": "cpu"})
        assert resp.status_code == 200

    @patch("domains.infrastructure.quantization.walk_slo_linears", return_value={})
    @patch("domains.models.provider.get_provider")
    @patch("infrastructure.auth.get_audit_logger")
    def test_quantize_model_logs_event(self, mock_logger, mock_provider, mock_walk, models_client):
        provider = MagicMock()
        provider._model = MagicMock()
        provider._model_path = None
        provider._model_id = "gpt2"
        mock_provider.return_value = provider
        resp = models_client.post("/models/quantize", json={"bits": 8, "mode": "symmetric"})
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "model.quantize"
        assert kwargs["resource"] == "gpt2"
        assert kwargs["user"] == "anonymous"
        assert kwargs["detail"] == "bits=8 mode=symmetric"
        assert kwargs["extra"] == {"bits": 8, "mode": "symmetric", "layers_quantized": 0, "model_type": "slonet"}

    @patch("domains.infrastructure.quantization.walk_slo_linears", return_value={})
    @patch("domains.models.provider.get_provider")
    @patch("infrastructure.auth.get_audit_logger")
    def test_dequantize_model_logs_event(self, mock_logger, mock_provider, mock_walk, models_client):
        provider = MagicMock()
        provider._model = MagicMock()
        provider._quant_engine = MagicMock()
        provider._model_id = "gpt2"
        mock_provider.return_value = provider
        resp = models_client.post("/models/dequantize")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "model.dequantize"
        assert kwargs["resource"] == "gpt2"
        assert kwargs["detail"] == "model_type=slonet"
        assert kwargs["extra"] == {"layers_reset": 0}

    @patch("domains.models.provider.get_provider", return_value=None)
    @patch("domains.slolib.gpu.get_accelerator")
    @patch("infrastructure.auth.get_audit_logger")
    def test_set_precision_logs_event(self, mock_logger, mock_acc, mock_provider, models_client):
        acc = MagicMock()
        acc.name = "cpu"
        acc.device_type = "cpu"
        acc._fp16_mode = False
        mock_acc.return_value = acc
        resp = models_client.post("/models/precision", json={"mode": "auto"})
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "model.precision"
        assert kwargs["resource"] == "cpu"
        assert kwargs["extra"] == {"mode": "auto"}
        assert kwargs["detail"] == resp.json()["data"]["precision"]

    @patch("domains.infrastructure.download_manager.get_download_manager")
    @patch("infrastructure.auth.get_audit_logger")
    def test_start_download_logs_event(self, mock_logger, mock_mgr, models_client):
        mgr = MagicMock()
        mgr.is_cached.return_value = False
        mgr.is_downloading.return_value = False
        mgr.download = AsyncMock(return_value={"status": "in_progress"})
        mock_mgr.return_value = mgr
        resp = models_client.post("/models/download", json={"model_id": "gpt2", "total_bytes_hint": 1000})
        assert resp.status_code == 200
        assert resp.json()["data"]["model_id"] == "gpt2"
        logger = mock_logger.return_value
        assert logger.log.call_count >= 1
        first_call_args, first_call_kwargs = logger.log.call_args_list[0]
        assert first_call_args[0] == "model.download"
        assert first_call_kwargs["resource"] == "gpt2"
        assert first_call_kwargs["detail"] == "started"
        assert first_call_kwargs["extra"] == {"total_bytes_hint": 1000}

    @patch("domains.infrastructure.download_manager.get_download_manager")
    @patch("infrastructure.auth.get_audit_logger")
    def test_cancel_download_logs_event(self, mock_logger, mock_mgr, models_client):
        mgr = MagicMock()
        mgr.cancel.return_value = True
        mock_mgr.return_value = mgr
        resp = models_client.post("/models/download/gpt2/cancel")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "model.cancel"
        assert kwargs["resource"] == "gpt2"
        assert kwargs["detail"] == "cancelled"

    @patch("domains.infrastructure.download_manager.get_download_manager")
    @patch("infrastructure.auth.get_audit_logger")
    def test_cancel_missing_download_no_audit(self, mock_logger, mock_mgr, models_client):
        mgr = MagicMock()
        mgr.cancel.return_value = False
        mock_mgr.return_value = mgr
        resp = models_client.post("/models/download/gpt2/cancel")
        assert resp.status_code == 200
        assert resp.json()["message"] == "not_found"
        logger = mock_logger.return_value
        logger.log.assert_not_called()


class TestSoulAudit:
    """Soul switch and weight-snapshot operations emit audit events."""

    @patch("domains.inference.slo_manager.get_slo_manager")
    @patch("infrastructure.auth.get_audit_logger")
    def test_switch_soul_logs_event(self, mock_logger, mock_manager, souls_client):
        manager = MagicMock()
        manager.switch_soul.return_value = {"success": True}
        manager.get_soul.return_value = None
        mock_manager.return_value = manager
        resp = souls_client.post("/souls/switch", json={"name": "friendly"})
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "soul.switch"
        assert kwargs["resource"] == "friendly"
        assert kwargs["extra"] == {"checkpoint_name": ""}

    @patch("domains.context.managers.get_trait_config")
    @patch("infrastructure.auth.get_audit_logger")
    def test_save_weight_snapshot_logs_event(self, mock_logger, mock_config, souls_client):
        config = MagicMock()
        config.save_snapshot.return_value = "/tmp/snap.json"
        mock_config.return_value = config
        resp = souls_client.post("/souls/weights/snapshot/my-snap")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "weights.snapshot.save"
        assert kwargs["resource"] == "my-snap"

    @patch("domains.context.managers.get_trait_config")
    @patch("infrastructure.auth.get_audit_logger")
    def test_load_weight_snapshot_logs_event(self, mock_logger, mock_config, souls_client):
        config = MagicMock()
        config.load_snapshot.return_value = 5
        mock_config.return_value = config
        resp = souls_client.post("/souls/weights/snapshot/my-snap/load")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "weights.snapshot.load"
        assert kwargs["resource"] == "my-snap"
        assert kwargs["detail"] == "traits_loaded=5"

    @patch("domains.context.managers.get_trait_config")
    @patch("infrastructure.auth.get_audit_logger")
    def test_delete_weight_snapshot_logs_event(self, mock_logger, mock_config, souls_client):
        config = MagicMock()
        config.delete_snapshot.return_value = True
        mock_config.return_value = config
        resp = souls_client.delete("/souls/weights/snapshot/my-snap")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "weights.snapshot.delete"
        assert kwargs["resource"] == "my-snap"
        assert kwargs["detail"] == "deleted=True"

    @patch("domains.context.managers.get_trait_config")
    @patch("infrastructure.auth.get_audit_logger")
    def test_save_trait_weights_logs_event(self, mock_logger, mock_config, souls_client):
        config = MagicMock()
        mock_config.return_value = config
        resp = souls_client.post(
            "/souls/weights",
            json={"personality": {"warmth": 0.8, "curiosity": 0.5}, "emotion": {"empathy": 0.6}},
        )
        assert resp.status_code == 200
        config.set_many.assert_called_once()
        assert config.set_many.call_args[0][0] == {"warmth": 0.8, "curiosity": 0.5, "empathy": 0.6}
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "soul.weights.save"
        assert kwargs["resource"] == "traits"
        assert kwargs["detail"] == "traits_saved=3"
        assert kwargs["extra"] == {"groups": ["personality", "emotion"]}


class TestAutoTrainAudit:
    """Auto-train start and checkpoint ops emit audit events."""

    @patch("infrastructure.auth.get_audit_logger")
    def test_start_logs_event(self, mock_logger, auto_train_client):
        resp = auto_train_client.post(
            "/auto-train/start",
            json={"dataset_id": "missing-ds", "epochs": 5, "soul_name": "assistant"},
        )
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "training.start"
        assert kwargs["resource"] == "missing-ds"
        assert kwargs["detail"] == "fresh"
        assert kwargs["extra"]["method"] == "slonet"

    @patch("infrastructure.auth.get_audit_logger")
    def test_delete_checkpoint_logs_event(self, mock_logger, auto_train_client):
        ckpt_dir = auto_train_client.app.state.checkpoint_dir
        (ckpt_dir / "fake.soul").write_bytes(b"x" * 16)
        resp = auto_train_client.delete("/auto-train/checkpoints/fake.soul")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "training.checkpoint.delete"
        assert kwargs["resource"] == "fake.soul"
        assert kwargs["detail"] == "deleted"

    @patch("domains.training.slonet.import_from_sou")
    @patch("domains.models.provider.register_provider")
    @patch("infrastructure.auth.get_audit_logger")
    def test_load_checkpoint_logs_event(self, mock_logger, mock_register, mock_import, auto_train_client):
        ckpt_dir = auto_train_client.app.state.checkpoint_dir
        _make_fake_sou(ckpt_dir / "fake.soul")
        soul_net = MagicMock()
        soul_net.soul_signature.return_value = {"soul_name": "fake", "soul_traits": {}}
        soul_net.num_parameters.return_value = 42
        mock_import.return_value = soul_net
        resp = auto_train_client.post("/auto-train/checkpoints/fake.soul/load")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "training.checkpoint.load"
        assert kwargs["resource"] == "fake.soul"
        assert "vocab=" in kwargs["detail"]


class TestDatasetAudit:
    """DELETE /datasets/{id} emits an audit event."""

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    @patch("infrastructure.auth.get_audit_logger")
    def test_delete_dataset_logs_event(self, mock_logger, mock_ctrl, datasets_client):
        ctrl = MagicMock()
        ctrl.delete_dataset.return_value = True
        mock_ctrl.return_value = ctrl
        resp = datasets_client.delete("/datasets/ds1")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "dataset.delete"
        assert kwargs["resource"] == "ds1"

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    @patch("infrastructure.auth.get_audit_logger")
    def test_delete_missing_dataset_no_audit(self, mock_logger, mock_ctrl, datasets_client):
        ctrl = MagicMock()
        ctrl.delete_dataset.return_value = False
        mock_ctrl.return_value = ctrl
        resp = datasets_client.delete("/datasets/ds-missing")
        assert resp.status_code == 404
        logger = mock_logger.return_value
        logger.log.assert_not_called()


@pytest.fixture
def training_router_client():
    import importlib

    training_router_module = importlib.import_module("apps.api.server.training.router")
    from apps.api.server.training.jobs import training_jobs

    training_jobs.clear()
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(training_router_module.router)
    client = TestClient(app, raise_server_exceptions=False)
    yield client
    training_jobs.clear()


class TestTrainingRouterAudit:
    """Training job lifecycle + webhook ops emit audit events."""

    @patch("apps.api.server.training.router.get_training_executor")
    @patch("apps.api.server.training.router.get_training_controller")
    @patch("apps.api.server.training.execution.resolve_training_inputs")
    @patch("infrastructure.auth.get_audit_logger")
    def test_start_training_logs_event(
        self, mock_logger, mock_resolve, mock_ctrl, mock_executor, training_router_client
    ):
        mock_resolve.return_value = ("/tmp/input.txt", "out_stem", None, "dataset")
        mock_ctrl.return_value = MagicMock()
        mock_executor.return_value = MagicMock()
        resp = training_router_client.post(
            "/training/start",
            json={"dataset": "ds1", "model": "gpt2", "epochs": 3, "name": "my-job"},
        )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "training.start"
        assert kwargs["resource"] == "ds1"
        assert kwargs["detail"] == "char"
        assert kwargs["extra"] == {"job_id": job_id, "model": "gpt2", "epochs": 3, "source_kind": "dataset"}

    @patch("apps.api.server.training.router.get_training_executor")
    @patch("infrastructure.auth.get_audit_logger")
    def test_start_hf_training_logs_event(self, mock_logger, mock_executor, training_router_client, tmp_path):
        text_file = tmp_path / "input.txt"
        text_file.write_text("hello world\n")
        mock_executor.return_value = MagicMock()
        model_file = tmp_path / "model.slnc"
        model_file.write_bytes(b"\x00" * 16)
        from pathlib import Path as RealPath
        repo_root = RealPath(__file__).resolve().parents[2]
        ds_dir = repo_root / "datasets" / "audit_test_ds"
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / "input.txt").write_text("hello world\n")
        try:
            resp = training_router_client.post(
                "/training/lora-finetune",
                json={"model_path": str(model_file), "dataset": "audit_test_ds", "epochs": 1},
            )
            assert resp.status_code == 200, resp.text[:200]
        finally:
            import shutil
            shutil.rmtree(ds_dir, ignore_errors=True)
        job_id = resp.json()["job_id"]
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "training.start"
        assert kwargs["resource"] == "audit_test_ds"
        assert kwargs["detail"] == "lora"
        assert kwargs["extra"]["job_id"] == job_id
        assert kwargs["extra"]["model"] == "model"

    @patch("apps.api.server.training.router.get_training_executor")
    @patch("infrastructure.auth.get_audit_logger")
    def test_stop_training_job_logs_event(self, mock_logger, mock_executor, training_router_client):
        from apps.api.server.training.jobs import training_jobs

        training_jobs["job_1"] = {"status": "running", "id": "job_1"}
        mock_executor.return_value = MagicMock()
        resp = training_router_client.post("/training/jobs/job_1/stop")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "training.stop"
        assert kwargs["resource"] == "job_1"
        assert kwargs["detail"] == "from=running"

    @patch("infrastructure.auth.get_audit_logger")
    def test_delete_training_job_logs_event(self, mock_logger, training_router_client):
        from apps.api.server.training.jobs import training_jobs

        training_jobs["job_1"] = {"id": "job_1", "status": "completed"}
        resp = training_router_client.delete("/training/jobs/job_1")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "training.delete"
        assert kwargs["resource"] == "job_1"
        assert kwargs["detail"] == "deleted_files=0"

    @patch("infrastructure.auth.get_audit_logger")
    def test_delete_missing_training_job_no_audit(self, mock_logger, training_router_client):
        resp = training_router_client.delete("/training/jobs/job-missing")
        assert resp.status_code == 404
        logger = mock_logger.return_value
        logger.log.assert_not_called()

    @patch("infrastructure.auth.get_audit_logger")
    def test_register_webhook_logs_event(self, mock_logger, training_router_client):
        resp = training_router_client.post(
            "/training/webhooks",
            params={"url": "https://example.com/hook", "events": '["training.completed","training.failed"]'},
        )
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "training.webhook.register"
        assert kwargs["resource"] == "https://example.com/hook"
        assert kwargs["extra"]["events"] == ["training.completed", "training.failed"]
        assert kwargs["extra"]["webhook_id"] == resp.json()["id"]

    @patch("infrastructure.auth.get_audit_logger")
    def test_unregister_webhook_logs_event(self, mock_logger, training_router_client):
        from apps.api.server.training.webhooks import get_webhook_store

        webhook_id = get_webhook_store().register(
            url="https://example.com/hook",
            events=["training.completed"],
            secret=None,
            description="",
            headers=None,
        )
        resp = training_router_client.delete(f"/training/webhooks/{webhook_id}")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "training.webhook.delete"
        assert kwargs["resource"] == webhook_id

    @patch("infrastructure.auth.get_audit_logger")
    def test_unregister_missing_webhook_no_audit(self, mock_logger, training_router_client):
        resp = training_router_client.delete("/training/webhooks/webhook-missing")
        assert resp.status_code == 404
        logger = mock_logger.return_value
        logger.log.assert_not_called()


# ── Knowledge base ────────────────────────────────────────────────────


@pytest.fixture
def kb_client():
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(KBRouter().router)
    with patch("domains.cognitive.rag_service.get_rag_service", return_value=MagicMock()):
        yield TestClient(app, raise_server_exceptions=False)


class TestKnowledgeAudit:
    """Knowledge base mutations emit audit events."""

    @patch("apps.api.server.routers.kb.KBRouter._get_memory")
    @patch("infrastructure.auth.get_audit_logger")
    def test_add_logs_event(self, mock_logger, mock_memory, kb_client):
        memory = mock_memory.return_value
        memory.add_fact.return_value = True
        resp = kb_client.post(
            "/knowledge",
            json={"content": "SloNet is a numpy autograd engine", "topic": "tech", "source": "manual"},
        )
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "knowledge.add"
        assert kwargs["resource"] == "tech"
        assert kwargs["detail"] == "stored"
        assert kwargs["extra"]["source"] == "manual"

    @patch("apps.api.server.routers.kb.KBRouter._get_memory")
    @patch("infrastructure.auth.get_audit_logger")
    def test_add_duplicate_logs_detail(self, mock_logger, mock_memory, kb_client):
        memory = mock_memory.return_value
        memory.add_fact.return_value = False
        resp = kb_client.post("/knowledge", json={"content": "duplicate fact"})
        assert resp.status_code == 200
        logger = mock_logger.return_value
        args, kwargs = logger.log.call_args
        assert kwargs["detail"] == "duplicate"

    @patch("apps.api.server.routers.kb.KBRouter._get_memory")
    @patch("infrastructure.auth.get_audit_logger")
    def test_update_logs_event(self, mock_logger, mock_memory, kb_client):
        memory = mock_memory.return_value
        memory.list_all.return_value = [{"id": "fact_1", "content": "old", "topic": "tech", "source": "manual", "timestamp": 0.0}]
        memory.delete_by_id.return_value = True
        memory.add_fact.return_value = True
        resp = kb_client.patch("/knowledge/fact_1", json={"content": "new"})
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "knowledge.update"
        assert kwargs["resource"] == "fact_1"

    @patch("apps.api.server.routers.kb.KBRouter._get_memory")
    @patch("infrastructure.auth.get_audit_logger")
    def test_batch_ingest_logs_event(self, mock_logger, mock_memory, kb_client):
        memory = mock_memory.return_value
        memory.add_fact.return_value = True
        resp = kb_client.post("/knowledge/batch", json={"items": [{"content": "a"}, {"content": "b"}]})
        assert resp.status_code == 200
        logger = mock_logger.return_value
        args, kwargs = logger.log.call_args
        assert args[0] == "knowledge.add"
        assert kwargs["resource"] == "batch"
        assert kwargs["detail"] == "stored=2"

    @patch("apps.api.server.routers.kb.KBRouter._get_memory")
    @patch("infrastructure.auth.get_audit_logger")
    def test_batch_delete_logs_event(self, mock_logger, mock_memory, kb_client):
        memory = mock_memory.return_value
        memory.delete_by_id.return_value = True
        resp = kb_client.post("/knowledge/batch-delete", json={"ids": ["fact_1", "fact_2"]})
        assert resp.status_code == 200
        logger = mock_logger.return_value
        args, kwargs = logger.log.call_args
        assert args[0] == "knowledge.batch.delete"
        assert kwargs["detail"] == "deleted=2"

    @patch("apps.api.server.routers.kb.KBRouter._get_memory")
    @patch("infrastructure.auth.get_audit_logger")
    def test_delete_logs_event(self, mock_logger, mock_memory, kb_client):
        memory = mock_memory.return_value
        memory.delete_by_id.return_value = True
        resp = kb_client.delete("/knowledge/fact_1")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        args, kwargs = logger.log.call_args
        assert args[0] == "knowledge.delete"
        assert kwargs["resource"] == "fact_1"

    @patch("apps.api.server.routers.kb.KBRouter._get_memory")
    @patch("infrastructure.auth.get_audit_logger")
    def test_delete_missing_no_audit(self, mock_logger, mock_memory, kb_client):
        memory = mock_memory.return_value
        memory.delete_by_id.return_value = False
        resp = kb_client.delete("/knowledge/fact-missing")
        assert resp.status_code == 404
        logger = mock_logger.return_value
        logger.log.assert_not_called()


# ── Agents ────────────────────────────────────────────────────────────


@pytest.fixture
def agents_client():
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(AgentsRouter().router)
    return TestClient(app, raise_server_exceptions=False)


class TestAgentsAudit:
    """Agent CRUD + execution emit audit events."""

    @patch("domains.agents.system.get_agent_system")
    @patch("infrastructure.auth.get_audit_logger")
    def test_create_logs_event(self, mock_logger, mock_system, agents_client):
        system = mock_system.return_value
        system.get.return_value = None
        system.create.return_value = {
            "id": "researcher", "name": "Researcher", "description": "", "instructions": "",
            "tools": [], "avatar": "", "created_at": 0.0, "updated_at": 0.0,
        }
        resp = agents_client.post("/agents", json={"name": "Researcher"})
        assert resp.status_code == 201
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "agent.create"
        assert kwargs["resource"] == "researcher"
        assert kwargs["detail"] == "Researcher"

    @patch("domains.agents.system.get_agent_system")
    @patch("infrastructure.auth.get_audit_logger")
    def test_update_logs_event(self, mock_logger, mock_system, agents_client):
        system = mock_system.return_value
        system.update.return_value = {
            "id": "researcher", "name": "Renamed", "description": "", "instructions": "",
            "tools": [], "avatar": "", "created_at": 0.0, "updated_at": 0.0,
        }
        resp = agents_client.put("/agents/researcher", json={"name": "Renamed"})
        assert resp.status_code == 200
        logger = mock_logger.return_value
        args, kwargs = logger.log.call_args
        assert args[0] == "agent.update"
        assert kwargs["resource"] == "researcher"

    @patch("domains.agents.system.get_agent_system")
    @patch("infrastructure.auth.get_audit_logger")
    def test_delete_logs_event(self, mock_logger, mock_system, agents_client):
        system = mock_system.return_value
        system.delete.return_value = True
        resp = agents_client.delete("/agents/researcher")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        args, kwargs = logger.log.call_args
        assert args[0] == "agent.delete"
        assert kwargs["resource"] == "researcher"

    @patch("domains.agents.system.get_agent_system")
    @patch("infrastructure.auth.get_audit_logger")
    def test_execute_logs_event(self, mock_logger, mock_system, agents_client):
        system = mock_system.return_value
        system.execute = AsyncMock(return_value={"output": "done"})
        resp = agents_client.post("/agents/researcher/execute", json={"request": "research"})
        assert resp.status_code == 200
        logger = mock_logger.return_value
        args, kwargs = logger.log.call_args
        assert args[0] == "agent.execute"
        assert kwargs["resource"] == "researcher"

    @patch("domains.agents.system.get_agent_system")
    @patch("infrastructure.auth.get_audit_logger")
    def test_execute_error_no_audit(self, mock_logger, mock_system, agents_client):
        system = mock_system.return_value
        system.execute = AsyncMock(return_value={"error": "agent not found"})
        resp = agents_client.post("/agents/ghost/execute", json={"request": "hi"})
        assert resp.status_code == 404
        logger = mock_logger.return_value
        logger.log.assert_not_called()


# ── Multimodal ────────────────────────────────────────────────────────


@pytest.fixture
def multimodal_client():
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(MultimodalRouter().router)
    return TestClient(app, raise_server_exceptions=False)


class TestMultimodalAudit:
    """Multimodal checkpoint load, delete and reset emit audit events."""

    @patch("domains.training.video_trainer.VideoCaptionTrainer")
    @patch("domains.training.video_trainer.list_video_checkpoints")
    @patch("infrastructure.auth.get_audit_logger")
    def test_load_checkpoint_logs_event(self, mock_logger, mock_list, mock_trainer, multimodal_client):
        mock_list.return_value = [{"name": "video1", "path": "/tmp/video1.slnc"}]
        resp = multimodal_client.post("/multimodal/checkpoints/video1/load")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "multimodal.checkpoint.load"
        assert kwargs["resource"] == "video1"

    @patch("domains.training.video_trainer.list_video_checkpoints")
    @patch("infrastructure.auth.get_audit_logger")
    def test_load_missing_checkpoint_no_audit(self, mock_logger, mock_list, multimodal_client):
        mock_list.return_value = []
        resp = multimodal_client.post("/multimodal/checkpoints/nope/load")
        assert resp.status_code == 404
        logger = mock_logger.return_value
        logger.log.assert_not_called()

    @patch("apps.api.server.routers.multimodal.get_multimodal_manager")
    @patch("infrastructure.auth.get_audit_logger")
    def test_reset_logs_event(self, mock_logger, mock_mgr, multimodal_client):
        mgr = mock_mgr.return_value
        mgr._initialized = True
        resp = multimodal_client.post("/multimodal/reset")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "multimodal.reset"
        assert kwargs["resource"] == "all"


# ── Config ────────────────────────────────────────────────────────────


@pytest.fixture
def config_client():
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(ConfigRouter().router)
    return TestClient(app, raise_server_exceptions=False)


class TestConfigAudit:
    """PUT/PATCH /config/generation emits an audit event."""

    @patch("apps.api.server.routers.config.get_config_controller")
    @patch("infrastructure.auth.get_audit_logger")
    def test_update_generation_logs_event(self, mock_logger, mock_ctrl, config_client):
        ctrl = mock_ctrl.return_value
        ctrl.update_generation_config.return_value = {"temperature": 0.9}
        resp = config_client.patch("/config/generation", json={"temperature": 0.9})
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "config.generation.save"
        assert kwargs["resource"] == "generation"
        assert kwargs["detail"] == "{'temperature': 0.9}"
        assert kwargs["extra"] == {"temperature": "0.9"}

    @patch("apps.api.server.routers.config.get_config_controller")
    @patch("infrastructure.auth.get_audit_logger")
    def test_update_none_fields_logs_empty_detail(self, mock_logger, mock_ctrl, config_client):
        ctrl = mock_ctrl.return_value
        ctrl.update_generation_config.return_value = {}
        resp = config_client.patch("/config/generation", json={})
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "config.generation.save"
        assert kwargs["detail"] == "{}"


# ── Experiments ───────────────────────────────────────────────────────


@pytest.fixture
def experiments_client(tmp_path):
    rtr = ExperimentsRouter()
    rtr.EXPERIMENTS_DIR = tmp_path / "experiments"
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(rtr.router)
    return TestClient(app, raise_server_exceptions=False)


class TestExperimentsAudit:
    """Experiment create/delete emit audit events."""

    @patch("infrastructure.auth.get_audit_logger")
    def test_create_logs_event(self, mock_logger, experiments_client):
        resp = experiments_client.post("/experiments", json={"name": "exp1"})
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "experiment.create"
        assert kwargs["detail"] == "exp1"
        assert kwargs["resource"].startswith("exp1_")

    @patch("infrastructure.auth.get_audit_logger")
    def test_delete_logs_event(self, mock_logger, experiments_client):
        resp = experiments_client.post("/experiments", json={"name": "exp2"})
        exp_id = resp.json()["data"]["id"]
        resp = experiments_client.delete(f"/experiments/{exp_id}")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_with("experiment.delete", user="anonymous", resource=exp_id, detail="", extra=None)
        assert [c.args[0] for c in logger.log.call_args_list] == ["experiment.create", "experiment.delete"]


# ── User adapters ─────────────────────────────────────────────────────


@pytest.fixture
def user_adapters_client():
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(UserAdaptersRouter().router)
    return TestClient(app, raise_server_exceptions=False)


class TestUserAdaptersAudit:
    """Per-user LoRA adapter mutations emit audit events."""

    @patch("domains.feedback.get_per_user_lora")
    @patch("infrastructure.auth.get_audit_logger")
    def test_update_logs_event(self, mock_logger, mock_store, user_adapters_client):
        store = mock_store.return_value
        resp = user_adapters_client.post("/user-adapters/user1/update", json={"rating": "thumbs_up"})
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "adapter.update"
        assert kwargs["resource"] == "user1"
        assert kwargs["detail"] == "rating=thumbs_up"

    @patch("domains.feedback.get_per_user_lora")
    @patch("infrastructure.auth.get_audit_logger")
    def test_reset_logs_event(self, mock_logger, mock_store, user_adapters_client):
        store = mock_store.return_value
        resp = user_adapters_client.post("/user-adapters/user1/reset")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        args, kwargs = logger.log.call_args
        assert args[0] == "adapter.reset"
        assert kwargs["resource"] == "user1"

    @patch("domains.feedback.get_per_user_lora")
    @patch("infrastructure.auth.get_audit_logger")
    def test_merge_logs_event(self, mock_logger, mock_store, user_adapters_client):
        store = mock_store.return_value
        resp = user_adapters_client.post("/user-adapters/merge")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        args, kwargs = logger.log.call_args
        assert args[0] == "adapter.merge"
        assert kwargs["resource"] == "all"

    @patch("domains.feedback.get_per_user_lora")
    @patch("infrastructure.auth.get_audit_logger")
    def test_aggregate_best_logs_event(self, mock_logger, mock_store, user_adapters_client):
        store = mock_store.return_value
        store.aggregate_best_adapters.return_value = {
            "user_count": 3, "total_feedback": 10, "output_path": "/tmp/best.sou",
            "eval": {"delta": {"verdict": "better"}},
        }
        resp = user_adapters_client.post("/user-adapters/aggregate-best", json={"output_name": "best"})
        assert resp.status_code == 200
        logger = mock_logger.return_value
        args, kwargs = logger.log.call_args
        assert args[0] == "adapter.aggregate"
        assert kwargs["resource"] == "best"
        assert kwargs["extra"] == {"user_count": 3, "total_feedback": 10}

    @patch("domains.feedback.get_per_user_lora")
    @patch("infrastructure.auth.get_audit_logger")
    def test_delete_logs_event(self, mock_logger, mock_store, user_adapters_client):
        store = mock_store.return_value
        resp = user_adapters_client.delete("/user-adapters/user1")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        args, kwargs = logger.log.call_args
        assert args[0] == "adapter.delete"
        assert kwargs["resource"] == "user1"

    @patch("domains.feedback.get_per_user_lora")
    @patch("infrastructure.auth.get_audit_logger")
    def test_prune_logs_event(self, mock_logger, mock_store, user_adapters_client):
        store = mock_store.return_value
        store.prune_low_quality.return_value = ["user1", "user2"]
        resp = user_adapters_client.post("/user-adapters/prune", json={})
        assert resp.status_code == 200
        logger = mock_logger.return_value
        args, kwargs = logger.log.call_args
        assert args[0] == "adapter.prune"
        assert kwargs["resource"] == "all"
        assert kwargs["detail"] == "deleted=2"
        assert kwargs["extra"] == {"deleted_users": ["user1", "user2"]}


# ── LoRA eval ─────────────────────────────────────────────────────────


@pytest.fixture
def lora_eval_client():
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(LoraEvalRouter().router)
    return TestClient(app, raise_server_exceptions=False)


class TestLoraEvalAudit:
    """POST /lora-eval/aggregate emits an audit event."""

    @patch("domains.feedback.per_user_lora.get_per_user_lora")
    @patch("infrastructure.auth.get_audit_logger")
    def test_aggregate_logs_event(self, mock_logger, mock_store, lora_eval_client):
        store = mock_store.return_value
        store.aggregate_best_adapters.return_value = {
            "output_path": "/tmp/best.sou",
            "user_count": 1, "total_feedback": 3,
            "eval": {"delta": {"verdict": "better"}},
        }
        resp = lora_eval_client.post("/lora-eval/aggregate")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "adapter.eval.aggregate"
        assert kwargs["resource"] == "best_aggregated"
        assert kwargs["extra"] == {"user_count": 1, "total_feedback": 3}


# ── Tokenizer ─────────────────────────────────────────────────────────


@pytest.fixture
def tokenizer_client():
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(TokenizerRouter().router)
    return TestClient(app, raise_server_exceptions=False)


class TestTokenizerAudit:
    """POST /tokenizer/train emits an audit event."""

    @patch("apps.api.server.routers.tokenizer.get_tokenizer_manager")
    @patch("infrastructure.auth.get_audit_logger")
    def test_train_logs_event(self, mock_logger, mock_mgr, tokenizer_client):
        mgr = mock_mgr.return_value
        mgr.train.return_value = None
        mgr.stats.return_value = {"vocab_size": 64}
        resp = tokenizer_client.post(
            "/tokenizer/train",
            json={"vocab_size": 64, "texts": ["hello", "world"]},
        )
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "tokenizer.train"
        assert kwargs["resource"] == "bpe"
        assert kwargs["extra"] == {"vocab_size": 64, "corpus_size": 2}


# ── System executor ───────────────────────────────────────────────────


@pytest.fixture
def system_client():
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(SystemRouter().router)
    return TestClient(app, raise_server_exceptions=False)


class TestSystemAudit:
    """Executor purge/cancel emit audit events with the acting user."""

    @patch("domains.training.executor._instance")
    @patch("infrastructure.auth.get_audit_logger")
    def test_purge_executor_logs_event(self, mock_logger, mock_instance, system_client):
        mock_instance.purge_completed.return_value = 3
        resp = system_client.post("/system/executor/purge?max_age_s=3600")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "executor.purge"
        assert kwargs["resource"] == "executor"
        assert kwargs["detail"] == "purged=3 max_age_s=3600.0"
        assert kwargs["user"] == "anonymous"

    @patch("domains.training.executor._instance")
    @patch("infrastructure.auth.get_audit_logger")
    def test_cancel_executor_logs_event(self, mock_logger, mock_instance, system_client):
        mock_instance.cancel.return_value = True
        resp = system_client.post("/system/executor/job_1/cancel")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "executor.cancel"
        assert kwargs["resource"] == "job_1"
        assert kwargs["detail"] == "cancelled=True"
        assert kwargs["user"] == "anonymous"


# ── Self-train ────────────────────────────────────────────────────────


@pytest.fixture
def self_train_client():
    import state as server_state

    server_state._self_train_proc = None
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(SelfTrainRouter().router)
    client = TestClient(app, raise_server_exceptions=False)
    yield client
    server_state._self_train_proc = None


class TestSelfTrainAudit:
    """Self-training start/stop emit audit events."""

    @patch("apps.api.server.routers.self_train.subprocess.Popen")
    @patch("infrastructure.auth.get_audit_logger")
    def test_start_logs_event(self, mock_logger, mock_popen, self_train_client):
        proc = mock_popen.return_value
        proc.pid = 4242
        proc.poll.return_value = None
        resp = self_train_client.post("/self-train/start", json={"model": "gpt2", "temperature": 0.8})
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "started"
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "self_train.start"
        assert kwargs["resource"] == "gpt2"
        assert kwargs["detail"] == "pid=4242"
        assert kwargs["extra"] == {"temperature": 0.8, "forever": False}

    @patch("infrastructure.auth.get_audit_logger")
    def test_stop_logs_event(self, mock_logger, self_train_client):
        import state as server_state

        proc = MagicMock()
        proc.pid = 4242
        proc.poll.return_value = None
        server_state._self_train_proc = proc
        try:
            resp = self_train_client.post("/self-train/stop")
            assert resp.status_code == 200
            assert resp.json()["data"]["status"] == "stopped"
            logger = mock_logger.return_value
            logger.log.assert_called_once()
            args, kwargs = logger.log.call_args
            assert args[0] == "self_train.stop"
            assert kwargs["resource"] == "4242"
            assert kwargs["detail"] == "stopped"
        finally:
            server_state._self_train_proc = None

    @patch("infrastructure.auth.get_audit_logger")
    def test_stop_not_running_no_audit(self, mock_logger, self_train_client):
        import state as server_state

        server_state._self_train_proc = None
        resp = self_train_client.post("/self-train/stop")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "not_running"
        mock_logger.return_value.log.assert_not_called()


# ── Auto-train pause/resume/cancel ────────────────────────────────────


class TestAutoTrainControlAudit:
    """Auto-train pause/resume and from-sessions cancel emit audit events."""

    @patch("infrastructure.auth.get_audit_logger")
    def test_pause_logs_event(self, mock_logger, auto_train_client):
        import apps.api.server.routers.auto_train as at_module

        at_module._auto_train_pause_event = threading.Event()
        try:
            resp = auto_train_client.post("/auto-train/pause")
            assert resp.status_code == 200
            assert resp.json()["data"]["success"] is True
            logger = mock_logger.return_value
            logger.log.assert_called_once()
            args, kwargs = logger.log.call_args
            assert args[0] == "training.pause"
            assert "resource" in kwargs
        finally:
            at_module._auto_train_pause_event = None

    @patch("infrastructure.auth.get_audit_logger")
    def test_pause_no_active_training_no_audit(self, mock_logger, auto_train_client):
        import apps.api.server.routers.auto_train as at_module

        at_module._auto_train_pause_event = None
        try:
            resp = auto_train_client.post("/auto-train/pause")
            assert resp.status_code == 200
            assert resp.json()["data"]["success"] is False
            mock_logger.return_value.log.assert_not_called()
        finally:
            at_module._auto_train_pause_event = None

    @patch("infrastructure.auth.get_audit_logger")
    def test_resume_logs_event(self, mock_logger, auto_train_client):
        import apps.api.server.routers.auto_train as at_module

        evt = threading.Event()
        evt.set()
        at_module._auto_train_pause_event = evt
        try:
            resp = auto_train_client.post("/auto-train/resume")
            assert resp.status_code == 200
            assert resp.json()["data"]["success"] is True
            logger = mock_logger.return_value
            logger.log.assert_called_once()
            args, kwargs = logger.log.call_args
            assert args[0] == "training.resume"
            assert "resource" in kwargs
        finally:
            at_module._auto_train_pause_event = None

    @patch("infrastructure.auth.get_audit_logger")
    def test_cancel_from_sessions_logs_event(self, mock_logger, auto_train_client):
        import apps.api.server.routers.auto_train as at_module

        at_module._auto_train_cancel_event = threading.Event()
        try:
            resp = auto_train_client.get("/auto-train/from-sessions/cancel")
            assert resp.status_code == 200
            logger = mock_logger.return_value
            logger.log.assert_called_once()
            args, kwargs = logger.log.call_args
            assert args[0] == "training.stop"
            assert "resource" in kwargs
            assert kwargs["detail"] == "cancelled"
        finally:
            at_module._auto_train_cancel_event = None


# ── Datasets extra mutations ──────────────────────────────────────────


class TestDatasetMutationsAudit:
    """Dataset create/version/append/convert emit audit events."""

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    @patch("infrastructure.auth.get_audit_logger")
    def test_create_logs_event(self, mock_logger, mock_ctrl, datasets_client):
        ctrl = mock_ctrl.return_value
        ctrl.create_dataset.return_value = {"id": "ds1", "name": "ds1", "path": "/tmp/ds1"}
        resp = datasets_client.post("/datasets", json={"name": "ds1", "description": "test ds"})
        assert resp.status_code == 200
        logger = mock_logger.return_value
        logger.log.assert_called_once()
        args, kwargs = logger.log.call_args
        assert args[0] == "dataset.create"
        assert kwargs["resource"] == "ds1"
        assert kwargs["detail"] == "test ds"

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    @patch("infrastructure.auth.get_audit_logger")
    def test_update_logs_event(self, mock_logger, mock_ctrl, datasets_client):
        ctrl = mock_ctrl.return_value
        ctrl.update_dataset.return_value = {"id": "ds1", "name": "renamed", "path": "/tmp/ds1"}
        resp = datasets_client.patch("/datasets/ds1", json={"name": "renamed"})
        assert resp.status_code == 200
        logger = mock_logger.return_value
        args, kwargs = logger.log.call_args
        assert args[0] == "dataset.update"
        assert kwargs["resource"] == "ds1"

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    @patch("infrastructure.auth.get_audit_logger")
    def test_create_version_logs_event(self, mock_logger, mock_ctrl, datasets_client):
        ctrl = mock_ctrl.return_value
        ctrl.create_version_snapshot.return_value = "1714000000"
        resp = datasets_client.post("/datasets/ds1/versions")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        args, kwargs = logger.log.call_args
        assert args[0] == "dataset.version"
        assert kwargs["resource"] == "ds1"

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    @patch("infrastructure.auth.get_audit_logger")
    def test_append_data_logs_event(self, mock_logger, mock_ctrl, datasets_client):
        ctrl = mock_ctrl.return_value
        ctrl.add_data.return_value = 3
        resp = datasets_client.post("/datasets/ds1/data", json={"data": ["a", "b", "c"]})
        assert resp.status_code == 200
        logger = mock_logger.return_value
        args, kwargs = logger.log.call_args
        assert args[0] == "dataset.data.append"
        assert kwargs["detail"] == "rows=3"

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    @patch("infrastructure.auth.get_audit_logger")
    def test_convert_format_logs_event(self, mock_logger, mock_ctrl, tmp_path):
        rtr = DatasetsRouter()
        rtr._DATASETS_DIR = tmp_path / "datasets"
        ds_dir = rtr._DATASETS_DIR / "ds1"
        ds_dir.mkdir(parents=True)
        (ds_dir / "input.jsonl").write_text(json.dumps({"text": "hello"}) + "\n")
        app = FastAPI()
        app.include_router(rtr.router)
        client = TestClient(app, raise_server_exceptions=False)
        ctrl = mock_ctrl.return_value
        ctrl.list_datasets.return_value = [{"id": "ds1", "name": "ds1", "path": str(ds_dir)}]
        ctrl.create_dataset.return_value = {"id": "ds1-messages", "name": "ds1-messages", "path": str(ds_dir)}
        resp = client.post("/datasets/convert-to-messages?dataset_id=ds1")
        assert resp.status_code == 200
        logger = mock_logger.return_value
        args, kwargs = logger.log.call_args
        assert args[0] == "dataset.convert"
        assert kwargs["resource"] == "ds1"
        assert kwargs["extra"]["conversations"] == 1
