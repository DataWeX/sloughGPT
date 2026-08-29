"""Tests for domains.training.performance — TrainingOptimizations, InferenceOptimizations, PerformanceConfig."""

from domains.training.performance import (
    TrainingOptimizations, InferenceOptimizations, PerformanceConfig,
)


class TestTrainingOptimizations:
    def test_defaults(self):
        to = TrainingOptimizations()
        assert to.use_compile is True
        assert to.compile_mode == "reduce-overhead"
        assert to.dataloader_workers == 4
        assert to.use_flash_attention is True
        assert to.gradient_checkpointing is True

    def test_custom(self):
        to = TrainingOptimizations(use_compile=False, dataloader_workers=0)
        assert to.use_compile is False
        assert to.dataloader_workers == 0


class TestInferenceOptimizations:
    def test_defaults(self):
        io = InferenceOptimizations()
        assert io.use_compile is True
        assert io.use_cuda_graphs is True
        assert io.max_batch_size == 32
        assert io.use_kv_cache is True

    def test_custom(self):
        io = InferenceOptimizations(max_batch_size=1, use_kv_cache=False)
        assert io.max_batch_size == 1
        assert io.use_kv_cache is False


class TestPerformanceConfig:
    def test_defaults(self):
        pc = PerformanceConfig()
        assert pc.device in ("cpu", "cuda", "mps")
        assert isinstance(pc.training, TrainingOptimizations)
        assert isinstance(pc.inference, InferenceOptimizations)

    def test_custom_device(self):
        pc = PerformanceConfig(device="cpu")
        assert pc.device == "cpu"
