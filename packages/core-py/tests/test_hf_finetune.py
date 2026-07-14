"""Tests for HFFineTuner (HuggingFace fine-tuning pipeline).

Tests initialization, parameter handling, schema validation,
and endpoint route registration. Model-loading-heavy paths are
tested via integration tests in test_hf_finetune_integration.py.
"""

from __future__ import annotations

import pytest


try:
    import transformers  # noqa: F401
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from training.schemas import HFTrainingRequest  # noqa: F401
    _SCHEMA_IMPORTABLE = True
except ImportError:
    _SCHEMA_IMPORTABLE = False

try:
    from training.router import router as _router  # noqa: F401
    _ROUTER_IMPORTABLE = True
except ImportError:
    _ROUTER_IMPORTABLE = False


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
class TestHFFineTunerInit:
    """HFFineTuner initialization and parameter handling."""

    def test_default_device_is_cpu(self):
        from domains.training.hf_finetune import HFFineTuner
        tuner = HFFineTuner(model_name="gpt2", data_path="/tmp/test.txt")
        assert tuner.device == "cpu"

    def test_explicit_device_override(self):
        from domains.training.hf_finetune import HFFineTuner
        tuner = HFFineTuner(model_name="gpt2", data_path="/tmp/test.txt", device="cpu")
        assert tuner.device == "cpu"

    def test_default_lora_target_modules(self):
        from domains.training.hf_finetune import HFFineTuner
        tuner = HFFineTuner(model_name="gpt2", data_path="/tmp/test.txt")
        assert tuner.lora_target_modules == ["c_attn", "c_fc", "c_proj"]

    def test_custom_lora_target_modules(self):
        from domains.training.hf_finetune import HFFineTuner
        tuner = HFFineTuner(
            model_name="gpt2",
            data_path="/tmp/test.txt",
            lora_target_modules=["k_proj", "o_proj"],
        )
        assert tuner.lora_target_modules == ["k_proj", "o_proj"]

    def test_default_lora_disabled(self):
        from domains.training.hf_finetune import HFFineTuner
        tuner = HFFineTuner(model_name="gpt2", data_path="/tmp/test.txt")
        assert tuner.use_lora is False

    def test_default_epochs(self):
        from domains.training.hf_finetune import HFFineTuner
        tuner = HFFineTuner(model_name="gpt2", data_path="/tmp/test.txt")
        assert tuner.epochs == 3

    def test_default_batch_size(self):
        from domains.training.hf_finetune import HFFineTuner
        tuner = HFFineTuner(model_name="gpt2", data_path="/tmp/test.txt")
        assert tuner.batch_size == 4

    def test_default_learning_rate(self):
        from domains.training.hf_finetune import HFFineTuner
        tuner = HFFineTuner(model_name="gpt2", data_path="/tmp/test.txt")
        assert tuner.learning_rate == 2e-4

    def test_default_max_seq_length(self):
        from domains.training.hf_finetune import HFFineTuner
        tuner = HFFineTuner(model_name="gpt2", data_path="/tmp/test.txt")
        assert tuner.max_seq_length == 512

    def test_custom_params(self):
        from domains.training.hf_finetune import HFFineTuner
        tuner = HFFineTuner(
            model_name="gpt2",
            data_path="/tmp/test.txt",
            epochs=5,
            batch_size=8,
            learning_rate=1e-4,
            max_seq_length=256,
            use_lora=True,
            lora_rank=16,
            lora_alpha=32,
        )
        assert tuner.epochs == 5
        assert tuner.batch_size == 8
        assert tuner.learning_rate == 1e-4
        assert tuner.max_seq_length == 256
        assert tuner.use_lora is True
        assert tuner.lora_rank == 16
        assert tuner.lora_alpha == 32


@pytest.mark.skipif(not _SCHEMA_IMPORTABLE, reason="HFTrainingRequest schema not on path")
class TestHFTrainingRequestSchema:
    """HFTrainingRequest schema validation."""

    def test_defaults(self):
        from training.schemas import HFTrainingRequest
        req = HFTrainingRequest(model="gpt2", dataset="shakespeare")
        assert req.model == "gpt2"
        assert req.dataset == "shakespeare"
        assert req.epochs == 3
        assert req.batch_size == 4
        assert req.learning_rate == 2e-4
        assert req.use_lora is False
        assert req.lora_rank == 8
        assert req.lora_alpha == 16

    def test_optional_device(self):
        from training.schemas import HFTrainingRequest
        req = HFTrainingRequest(model="gpt2", dataset="test")
        assert req.device is None

    def test_device_cpu(self):
        from training.schemas import HFTrainingRequest
        req = HFTrainingRequest(model="gpt2", dataset="test", device="cpu")
        assert req.device == "cpu"

    def test_all_fields(self):
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


@pytest.mark.skipif(not _ROUTER_IMPORTABLE, reason="server module not on path")
class TestHFBackendRoutes:
    """Verify HF fine-tune routes are registered on the router."""

    def test_hf_start_route_registered(self):
        from training.router import router
        routes = [r.path for r in router.routes]
        assert "/training/hf-start" in routes

    def test_hf_jobs_list_route_registered(self):
        from training.router import router
        routes = [r.path for r in router.routes]
        assert "/training/jobs" in routes

    def test_merge_lora_route_registered(self):
        from training.router import router
        routes = [r.path for r in router.routes]
        assert "/training/merge-lora" in routes


class TestMergeLoRAFunction:
    """Tests for merge_lora_adapter function (no model loading)."""

    def test_merge_requires_peft(self):
        """merge_lora_adapter raises ImportError when peft is missing."""
        try:
            import peft  # noqa: F401
            pytest.skip("peft installed — can't test ImportError path")
        except ImportError:
            pass
        from domains.training.hf_finetune import merge_lora_adapter
        with pytest.raises(ImportError, match="peft"):
            merge_lora_adapter("/nonexistent/adapter")

    def test_merge_nonexistent_path(self):
        """merge_lora_adapter raises FileNotFoundError for missing adapter."""
        try:
            import peft  # noqa: F401
        except ImportError:
            pytest.skip("peft not installed — can't reach FileNotFoundError path")
        from domains.training.hf_finetune import merge_lora_adapter
        with pytest.raises(FileNotFoundError):
            merge_lora_adapter("/nonexistent/path/to/adapter")

    def test_merge_requires_hub_id(self):
        """merge_lora_adapter raises ValueError when push_to_hub without hub_model_id."""
        import tempfile, json, os
        from domains.training.hf_finetune import merge_lora_adapter
        tmpdir = tempfile.mkdtemp()
        adapter_dir = os.path.join(tmpdir, "adapter")
        os.makedirs(adapter_dir)
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
            json.dump({"base_model_name_or_path": "gpt2"}, f)
        try:
            with pytest.raises((ImportError, ValueError, KeyError)):
                merge_lora_adapter(adapter_dir, push_to_hub=True, hub_model_id=None)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
