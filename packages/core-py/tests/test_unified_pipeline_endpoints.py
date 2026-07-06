"""Tests for unified pipeline endpoint wiring (schemas + routes).

Tests that:
- UnifiedStartRequest schema accepts all fields
- Router registers /training/unified-start and /training/unified-stream
- TrainingRunConfig skip_train is tracked
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure server modules are importable (from repo-root /apps/api/server)
_server_path = str(Path(__file__).resolve().parents[2] / "apps/api/server")
if _server_path not in sys.path:
    sys.path.insert(0, _server_path)

import pytest


class TestUnifiedStartRequest:
    """UnifiedStartRequest schema validation."""

    def test_defaults(self):
        from apps.api.server.training.schemas import UnifiedStartRequest

        req = UnifiedStartRequest()
        assert req.method == "auto"
        assert req.epochs == 3
        assert req.batch_size == 8
        assert req.learning_rate == 1e-4
        assert req.distill is False
        assert req.skip_generate is False
        assert req.skip_train is False

    def test_hf_config(self):
        from apps.api.server.training.schemas import UnifiedStartRequest

        req = UnifiedStartRequest(
            method="hf",
            hf_model_name="gpt2",
            data_path="datasets/shakespeare",
            epochs=5,
            use_lora=True,
            lora_rank=16,
        )
        assert req.method == "hf"
        assert req.hf_model_name == "gpt2"
        assert req.epochs == 5
        assert req.use_lora is True
        assert req.lora_rank == 16

    def test_serialization_roundtrip(self):
        from apps.api.server.training.schemas import UnifiedStartRequest

        req = UnifiedStartRequest(
            method="slonet",
            data_path="datasets/shakespeare",
            epochs=10,
            batch_size=32,
            learning_rate=1e-3,
            vocab_size=512,
            skip_generate=True,
            skip_distill=True,
        )
        d = req.model_dump()
        assert d["method"] == "slonet"
        assert d["epochs"] == 10
        assert d["skip_generate"] is True
        assert d["skip_distill"] is True
        assert d["skip_train"] is False

        # Roundtrip through JSON
        restored = UnifiedStartRequest(**json.loads(json.dumps(d)))
        assert restored.method == "slonet"
        assert restored.epochs == 10


class TestRouteRegistration:
    """Verify that the training router has the unified endpoints."""

    def test_routes_contain_unified_start(self):
        from apps.api.server.training.router import router

        routes = [r.path for r in router.routes]
        assert "/training/unified-start" in routes, f"Missing /training/unified-start in {routes}"

    def test_routes_contain_unified_stream(self):
        from apps.api.server.training.router import router

        routes = [r.path for r in router.routes]
        assert "/training/unified-stream" in routes, f"Missing /training/unified-stream in {routes}"

    def test_routes_contain_existing_endpoints(self):
        from apps.api.server.training.router import router

        routes = [r.path for r in router.routes]
        assert "/train" in routes
        assert "/training/jobs" in routes
        # Existing endpoints still present — no regressions


class TestUnifiedPipelineIntegration:
    """Test the pipeline creation from UnifiedStartRequest config."""

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_pipeline_from_request_config(self):
        from apps.api.server.training.schemas import UnifiedStartRequest
        from domains.training.unified_pipeline import UnifiedTrainingPipeline, UnifiedTrainingConfig
        from domains.training.sequence import TrainingRunConfig

        req = UnifiedStartRequest(
            method="hf",
            hf_model_name="gpt2",
            data_path="datasets/shakespeare",
            epochs=1,
            skip_generate=True,
            skip_distill=True,
            skip_train=True,
            skip_evaluate=True,
            skip_deploy=True,
        )
        config = UnifiedTrainingConfig(**req.model_dump())
        run_config = TrainingRunConfig(
            skip_generate=config.skip_generate,
            skip_distill=config.skip_distill,
            skip_train=config.skip_train,
            skip_evaluate=config.skip_evaluate,
            skip_deploy=config.skip_deploy,
        )
        pipeline = UnifiedTrainingPipeline(config, run_config=run_config)
        result = pipeline.run()
        assert result["status"] == "completed"
        assert all(
            pr["status"] == "skipped"
            for pr in result["phases"]
        ), "All phases should be skipped"
