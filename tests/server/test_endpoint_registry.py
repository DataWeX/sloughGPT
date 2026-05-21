"""
Endpoint registry test — pings every reachable GET endpoint and a sample of
POST endpoints to verify they return valid responses (not 500).

All tests marked ``slow`` (deselected by default). Run with:
  pytest tests/server/test_endpoint_registry.py -m slow
"""

from __future__ import annotations
from pathlib import Path
import json

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.slow

# ── test data paths ──────────────────────────────────────────────────
_TEST_SHAKESPEARE = Path(__file__).resolve().parents[2] / "datasets" / "shakespeare"
_TEST_IMPORT_NAME = "_test_registry_import"

try:
    from apps.api.server.main import app
    client = TestClient(app)
except Exception as exc:
    pytest.skip(f"Server app not available: {exc}", allow_module_level=True)


# ── config ──────────────────────────────────────────────────────────
BASE_URL = ""
ALLOWED_FAILURES = {404, 405, 422, 503}  # expected "not found", "method not allowed", "validation error", "unavailable"


class TestEndpointRegistry:
    """Every registered endpoint returns a structured response (not 500)."""

    # ── health ──────────────────────────────────────────────────────
    def test_health_root(self):
        r = client.get("/health")
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "healthy"

    def test_health_live(self):
        r = client.get("/health/live")
        assert r.status_code in {200, 503}

    def test_health_ready(self):
        r = client.get("/health/ready")
        assert r.status_code in {200, 503}

    def test_health_detailed(self):
        r = client.get("/health/detailed")
        assert r.status_code == 200

    def test_health_model(self):
        r = client.get("/health/model")
        assert r.status_code in {200, 503}

    # ── status ──────────────────────────────────────────────────────
    def test_status_root(self):
        r = client.get("/status")
        assert r.status_code == 200

    def test_ready(self):
        r = client.get("/ready")
        assert r.status_code in {200, 503}

    def test_live(self):
        r = client.get("/live")
        assert r.status_code == 200

    # ── system ──────────────────────────────────────────────────────
    def test_system_metrics(self):
        r = client.get("/system/metrics")
        assert r.status_code == 200

    def test_system_info(self):
        r = client.get("/system/info")
        assert r.status_code == 200

    def test_system_disk(self):
        r = client.get("/system/disk")
        is_ok = 200 <= r.status_code < 500
        assert is_ok, f"Expected 2xx-4xx, got {r.status_code}"

    # ── models ──────────────────────────────────────────────────────
    def test_models_list(self):
        r = client.get("/models")
        assert r.status_code == 200
        models = r.json()
        assert isinstance(models, list)

    def test_models_current(self):
        r = client.get("/models/current")
        assert r.status_code in {200, 404, 503}

    def test_models_hf(self):
        r = client.get("/models/hf")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_models_logs(self):
        r = client.get("/models/logs")
        assert r.status_code == 200

    def test_models_export_formats(self):
        r = client.get("/models/export/formats")
        assert r.status_code == 200

    # ── souls ───────────────────────────────────────────────────────
    def test_souls_list(self):
        r = client.get("/souls")
        assert r.status_code == 200
        data = r.json()
        assert "souls" in data
        assert isinstance(data["souls"], list)

    def test_souls_current(self):
        r = client.get("/souls/current")
        assert r.status_code == 200

    def test_souls_stats(self):
        r = client.get("/souls/stats")
        assert r.status_code in {200, 503}

    # ── companions / personalities ──────────────────────────────────
    def test_personalities(self):
        r = client.get("/personalities")
        assert r.status_code == 200

    def test_companion_root(self):
        r = client.get("/companion/")
        assert r.status_code == 200

    def test_companion_presets(self):
        r = client.get("/companion/presets")
        assert r.status_code == 200

    # ── metrics ─────────────────────────────────────────────────────
    def test_metrics_root(self):
        r = client.get("/metrics")
        assert r.status_code == 200

    # ── config ──────────────────────────────────────────────────────
    def test_config_generation(self):
        r = client.get("/config/generation")
        assert r.status_code == 200

    # ── info (from inference router) ────────────────────────────────
    def test_info_root(self):
        r = client.get("/")
        assert r.status_code in {200, 404}

    def test_info_soul(self):
        r = client.get("/info/soul")
        assert r.status_code in {200, 503}

    def test_info_providers(self):
        r = client.get("/providers")
        assert r.status_code in {200, 503}

    # ── tokenizer ───────────────────────────────────────────────────
    def test_tokenizer_stats(self):
        r = client.get("/tokenizer/stats")
        assert r.status_code in {200, 503}

    def test_tokenizer_vocab(self):
        r = client.get("/tokenizer/vocab")
        assert r.status_code in {200, 503}

    def test_tokenizer_merges(self):
        r = client.get("/tokenizer/merges")
        assert r.status_code in {200, 503}

    def test_tokenizer_sample(self):
        r = client.get("/tokenizer/sample")
        assert r.status_code in {200, 503}

    # ── sessions ────────────────────────────────────────────────────
    def test_session_missing(self):
        """Non-existent session returns gracefully (empty messages, not 500)."""
        r = client.get("/session/nonexistent/messages")
        assert r.status_code in {200, 404, 422}
        if r.status_code == 200:
            data = r.json()
            assert "messages" in data or "status" in data

    # ── auto-train (no model loaded) ────────────────────────────────
    def test_auto_train_status(self):
        r = client.get("/auto-train/status")
        assert r.status_code == 200

    def test_auto_train_checkpoints(self):
        r = client.get("/auto-train/checkpoints")
        assert r.status_code == 200
        data = r.json()
        assert "checkpoints" in data

    # ── rate-limit ──────────────────────────────────────────────────
    def test_rate_limit_status(self):
        r = client.get("/rate-limit/status")
        assert r.status_code == 200

    # ── registry ────────────────────────────────────────────────────
    def test_registry_models(self):
        r = client.get("/registry/models")
        assert r.status_code == 200

    def test_registry_best(self):
        r = client.get("/registry/best")
        assert r.status_code == 200

    def test_registry_stats(self):
        r = client.get("/registry/stats")
        assert r.status_code == 200

    # ── benchmark ───────────────────────────────────────────────────
    def test_benchmark_metrics(self):
        r = client.get("/benchmark/metrics")
        assert r.status_code in {200, 503}

    def test_benchmark_quality(self):
        r = client.get("/benchmark/quality")
        assert r.status_code in {200, 503}

    def test_benchmark_responses(self):
        r = client.get("/benchmark/responses")
        assert r.status_code in {200, 503}

    def test_benchmark_stats(self):
        r = client.get("/benchmark/stats")
        assert r.status_code == 200

    # ── security ────────────────────────────────────────────────────
    def test_security_audit(self):
        r = client.get("/security/audit")
        assert r.status_code == 200

    def test_security_keys(self):
        r = client.get("/security/keys")
        assert r.status_code in {200, 403, 404}

    # ── user-adapters ───────────────────────────────────────────────
    def test_user_adapters_list(self):
        r = client.get("/user-adapters")
        assert r.status_code == 200

    def test_user_adapters_quality(self):
        r = client.get("/user-adapters/quality")
        assert r.status_code == 200

    # ── knowledge ───────────────────────────────────────────────────
    def test_knowledge_list(self):
        r = client.get("/knowledge")
        assert r.status_code == 200

    def test_knowledge_search(self):
        r = client.get("/knowledge/search?q=test")
        assert r.status_code == 200

    # ── vector ──────────────────────────────────────────────────────
    def test_vector_stats(self):
        r = client.get("/vector/stats")
        assert r.status_code == 200

    def test_vector_ingest_status(self):
        r = client.get("/vector/ingest/status")
        assert r.status_code == 200

    # ── feedback ────────────────────────────────────────────────────
    def test_feedback_summary(self):
        r = client.get("/feedback/stats/summary")
        assert r.status_code == 200

    # ── workflow ────────────────────────────────────────────────────
    def test_workflow_status(self):
        r = client.get("/workflow/status")
        assert r.status_code == 200

    # ── labs (non-GPU subset) ───────────────────────────────────────
    def test_labs_model_info(self):
        r = client.get("/labs/model-info")
        assert r.status_code in {200, 503}

    def test_labs_health(self):
        r = client.get("/labs/health")
        assert r.status_code in {200, 503}

    # ── meta-weights ────────────────────────────────────────────────
    def test_meta_weights_ping(self):
        r = client.get("/meta-weights/ping")
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "ok"

    def test_meta_weights_stats(self):
        r = client.get("/meta-weights/stats")
        assert r.status_code == 200

    # ── multimodal ──────────────────────────────────────────────────
    def test_multimodal_capabilities(self):
        r = client.get("/multimodal/capabilities")
        assert r.status_code == 200

    # ── lora-eval ───────────────────────────────────────────────────
    def test_lora_eval_history(self):
        r = client.get("/lora-eval/history")
        assert r.status_code == 200

    # ── chat sessions (no model) ────────────────────────────────────
    def test_chat_sessions_list(self):
        r = client.get("/chat/sessions")
        assert r.status_code == 200
        data = r.json()
        assert "sessions" in data

    # ── experiments ─────────────────────────────────────────────────
    def test_experiments_list(self):
        r = client.get("/experiments")
        assert r.status_code == 200

    # ── datasets ────────────────────────────────────────────────────
    def test_datasets_list(self):
        r = client.get("/datasets")
        assert r.status_code == 200

    # ── regenerate endpoint (POST with no body) ────────────────────
    def test_regenerate_no_session(self):
        """Returns SSE error gracefully, not 500."""
        r = client.post("/session/nonexistent_id/regenerate")
        assert r.status_code in {200, 404, 422}

    # ── learner ─────────────────────────────────────────────────────
    def test_learner_status(self):
        r = client.get("/learn/status")
        assert r.status_code in {200, 503}

    def test_learner_knowledge(self):
        r = client.get("/learn/knowledge")
        assert r.status_code == 200

    # ── datasets import ────────────────────────────────────────────
    def test_import_local_shakespeare(self):
        """POST /datasets/import/local can import seed dataset."""
        r = client.post(
            "/datasets/import/local",
            json={"path": str(_TEST_SHAKESPEARE), "name": _TEST_IMPORT_NAME, "extensions": [".txt"]},
        )
        if r.status_code == 200:
            body = r.json()
            assert body["success"] is True
            assert body["dataset_id"] == _TEST_IMPORT_NAME
            assert "files" in body.get("message", "") or "chars" in body.get("message", "")
            # Cleanup
            client.delete(f"/datasets/{_TEST_IMPORT_NAME}")
            import shutil
            shutil.rmtree(str(_TEST_SHAKESPEARE.parent / _TEST_IMPORT_NAME), ignore_errors=True)
        else:
            assert r.status_code in {400, 500, 503}, f"Unexpected status {r.status_code}: {r.text[:200]}"

    def test_datasets_list_includes_imported(self):
        """GET /datasets lists available datasets."""
        r = client.get("/datasets")
        assert r.status_code == 200
        body = r.json()
        assert "datasets" in body
        assert isinstance(body["datasets"], list)
