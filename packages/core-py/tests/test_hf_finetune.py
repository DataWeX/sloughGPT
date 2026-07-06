"""Tests for HFFineTuner (HuggingFace fine-tuning pipeline).

Tests initialization, parameter handling, schema validation,
and endpoint route registration. Model-loading-heavy paths are
tested via integration tests in test_hf_finetune_integration.py.
"""

from __future__ import annotations

import pytest

pytest.importorskip("transformers", reason="transformers not installed")
pytest.importorskip("torch", reason="torch not installed")

from domains.training.hf_finetune import HFFineTuner


class TestHFFineTunerInit:
    """HFFineTuner initialization and parameter handling."""

    def test_default_device_is_cpu(self):
        tuner = HFFineTuner(model_name="gpt2", data_path="/tmp/test.txt")
        assert tuner.device == "cpu"

    def test_explicit_device_override(self):
        tuner = HFFineTuner(model_name="gpt2", data_path="/tmp/test.txt", device="cpu")
        assert tuner.device == "cpu"

    def test_default_lora_target_modules(self):
        tuner = HFFineTuner(model_name="gpt2", data_path="/tmp/test.txt")
        assert tuner.lora_target_modules == ["c_attn", "c_fc", "c_proj"]

    def test_custom_lora_target_modules(self):
        tuner = HFFineTuner(
            model_name="gpt2",
            data_path="/tmp/test.txt",
            lora_target_modules=["k_proj", "o_proj"],
        )
        assert tuner.lora_target_modules == ["k_proj", "o_proj"]

    def test_default_lora_disabled(self):
        tuner = HFFineTuner(model_name="gpt2", data_path="/tmp/test.txt")
        assert tuner.use_lora is False

    def test_default_epochs(self):
        tuner = HFFineTuner(model_name="gpt2", data_path="/tmp/test.txt")
        assert tuner.epochs == 3

    def test_default_batch_size(self):
        tuner = HFFineTuner(model_name="gpt2", data_path="/tmp/test.txt")
        assert tuner.batch_size == 4

    def test_default_learning_rate(self):
        tuner = HFFineTuner(model_name="gpt2", data_path="/tmp/test.txt")
        assert tuner.learning_rate == 2e-4

    def test_default_max_seq_length(self):
        tuner = HFFineTuner(model_name="gpt2", data_path="/tmp/test.txt")
        assert tuner.max_seq_length == 512

    def test_custom_params(self):
        tuner = HFFineTuner(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            data_path="/tmp/dataset.txt",
            use_lora=True,
            lora_rank=16,
            epochs=5,
            batch_size=2,
            learning_rate=1e-4,
            max_seq_length=256,
            warmup_steps=50,
            weight_decay=0.05,
        )
        assert tuner.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert tuner.data_path == "/tmp/dataset.txt"
        assert tuner.use_lora is True
        assert tuner.lora_rank == 16
        assert tuner.epochs == 5
        assert tuner.batch_size == 2
        assert tuner.learning_rate == 1e-4
        assert tuner.max_seq_length == 256
        assert tuner.warmup_steps == 50
        assert tuner.weight_decay == 0.05


try:
    from training.schemas import HFTrainingRequest as _HFTrainingRequest  # noqa: F401
    _SCHEMA_IMPORTABLE = True
except ImportError:
    _SCHEMA_IMPORTABLE = False


class TestHFTrainingRequestSchema:
    """Test that the HFTrainingRequest schema validates correctly."""

    def test_importable(self):
        if not _SCHEMA_IMPORTABLE:
            pytest.skip("server module not on path")
        from training.schemas import HFTrainingRequest
        assert HFTrainingRequest is not None

    def test_defaults(self):
        if not _SCHEMA_IMPORTABLE:
            pytest.skip("server module not on path")
        from training.schemas import HFTrainingRequest
        req = HFTrainingRequest(model="gpt2", dataset="shakespeare")
        assert req.model == "gpt2"
        assert req.dataset == "shakespeare"
        assert req.epochs == 3
        assert req.batch_size == 4
        assert req.use_lora is False

    def test_optional_device(self):
        if not _SCHEMA_IMPORTABLE:
            pytest.skip("server module not on path")
        from training.schemas import HFTrainingRequest
        req = HFTrainingRequest(model="gpt2", dataset="test")
        assert req.device is None

    def test_device_cpu(self):
        if not _SCHEMA_IMPORTABLE:
            pytest.skip("server module not on path")
        from training.schemas import HFTrainingRequest
        req = HFTrainingRequest(model="gpt2", dataset="test", device="cpu")
        assert req.device == "cpu"

    def test_all_fields(self):
        if not _SCHEMA_IMPORTABLE:
            pytest.skip("server module not on path")
        from training.schemas import HFTrainingRequest
        req = HFTrainingRequest(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            dataset="shakespeare",
            epochs=5,
            batch_size=2,
            learning_rate=1e-4,
            use_lora=True,
            lora_rank=16,
            lora_alpha=32,
            max_seq_length=256,
            warmup_steps=50,
            weight_decay=0.05,
            gradient_accumulation_steps=2,
            device="cpu",
        )
        assert req.model == "Qwen/Qwen2.5-0.5B-Instruct"
        assert req.epochs == 5
        assert req.batch_size == 2
        assert req.use_lora is True
        assert req.lora_rank == 16
        assert req.lora_alpha == 32
        assert req.max_seq_length == 256
        assert req.warmup_steps == 50
        assert req.weight_decay == 0.05
        assert req.gradient_accumulation_steps == 2


try:
    from training.router import router as _router  # noqa: F401
    _ROUTER_IMPORTABLE = True
except ImportError:
    _ROUTER_IMPORTABLE = False


class TestHFBackendRoutes:
    """Verify HF fine-tune routes are registered on the router."""

    def test_hf_start_route_registered(self):
        if not _ROUTER_IMPORTABLE:
            pytest.skip("server module not on path")
        from training.router import router
        routes = [r.path for r in router.routes]
        assert "/training/hf-start" in routes

    def test_hf_jobs_list_route_registered(self):
        if not _ROUTER_IMPORTABLE:
            pytest.skip("server module not on path")
        from training.router import router
        routes = [r.path for r in router.routes]
        assert "/training/jobs" in routes
