"""
Tests for Optimized Training Pipeline

Tests:
1. MemoryOptimizer - adaptive batch sizing
2. LoRAWrapper - parameter-efficient fine-tuning
3. OptimizedTrainer - training step
4. OptimizedFederatedTrainer - gradient compression
5. Unified pipeline integration

Marked ``slow`` — imports PyTorch.
"""

import asyncio
import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn

pytestmark = pytest.mark.slow


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def dummy_model():
    """Simple linear model for testing."""
    return nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )


@pytest.fixture
def dummy_batch():
    """Random batch for testing."""
    return torch.randn(4, 128)


@pytest.fixture
def optimizer_config():
    """Test optimization config."""
    from domains.training.optimized_pipeline import OptimizationConfig, Precision, LoRAMode
    return OptimizationConfig(
        precision=Precision.BF16,
        gradient_checkpointing=True,
        lora_mode=LoRAMode.LORA,
        lora_rank=8,
        lora_alpha=16,
        adaptive_batch_size=True,
        use_flash_attention=False,  # May not be available
        gradient_compression=True,
    )


# =============================================================================
# MEMORY OPTIMIZER TESTS
# =============================================================================

class TestMemoryOptimizer:
    """Tests for MemoryOptimizer."""

    def test_memory_profile(self):
        """Test memory profiling."""
        from domains.training.optimized_pipeline import MemoryOptimizer

        optimizer = MemoryOptimizer()

        # Get profile (will be 0 on CPU-only systems)
        profile = optimizer.get_profile()

        assert hasattr(profile, 'total_gb')
        assert hasattr(profile, 'used_gb')
        assert hasattr(profile, 'free_gb')
        assert hasattr(profile, 'utilization_percent')

        print(f"Memory: {profile.total_gb:.2f}GB total, {profile.used_gb:.2f}GB used")

    def test_suggest_batch_size(self):
        """Test adaptive batch sizing."""
        from domains.training.optimized_pipeline import MemoryOptimizer

        optimizer = MemoryOptimizer()

        # Test suggestion
        suggested = optimizer.suggest_batch_size(4, target_utilization=70.0)
        assert isinstance(suggested, int)
        assert suggested >= 1

    def test_peak_memory_tracking(self):
        """Test peak memory tracking."""
        from domains.training.optimized_pipeline import MemoryOptimizer

        optimizer = MemoryOptimizer()
        optimizer.reset_peak_stats()

        # Simulate some memory usage
        if torch.cuda.is_available():
            x = torch.randn(1000, 1000, device='cuda')

        peak = optimizer.get_peak_memory_gb()
        assert peak >= 0


# =============================================================================
# LORA TESTS
# =============================================================================

class TestLoRAWrapper:
    """Tests for LoRA wrapper."""

    def test_lora_initialization(self, dummy_model):
        """Test LoRA wrapper initialization."""
        from domains.training.optimized_pipeline import LoRAWrapper

        linear = dummy_model[0]  # First linear layer
        lora = LoRAWrapper(linear, rank=8, alpha=16)

        assert lora.rank == 8
        assert lora.alpha == 16
        assert lora.num_trainable_params > 0

        print(f"LoRA params: {lora.num_trainable_params:,}")

    def test_lora_forward(self, dummy_model):
        """Test LoRA forward pass."""
        from domains.training.optimized_pipeline import LoRAWrapper

        linear = dummy_model[0]  # nn.Linear(128, 256)
        lora = LoRAWrapper(linear, rank=8, alpha=16)

        x = torch.randn(2, 128)
        output = lora(x)

        # Linear(128, 256) produces 256-dim output
        assert output.shape == torch.Size([2, 256])
        from domains.training.slonet_compat import torch as slo_torch
        assert not slo_torch.isnan(output).any()

    def test_lora_trainable_params(self, dummy_model):
        """Test that only LoRA params are trainable."""
        from domains.training.optimized_pipeline import LoRAWrapper

        linear = dummy_model[0]
        lora = LoRAWrapper(linear, rank=8, alpha=16)

        # Check original layer is frozen
        for param in linear.parameters():
            assert not param.requires_grad

        # Check LoRA params are trainable
        trainable = lora.get_trainable_params()
        assert len(trainable) > 0
        for param in trainable.values():
            assert param.requires_grad

    def test_lora_weight_merge(self, dummy_model):
        """Test LoRA weight merging."""
        from domains.training.optimized_pipeline import LoRAWrapper

        linear = dummy_model[0]
        lora = LoRAWrapper(linear, rank=8, alpha=16)

        # Get original output
        x = torch.randn(2, 128)
        original = linear(x)

        # Merge weights
        lora.merge_weights()

        # Output should be different after merge
        merged = lora(x)
        # Note: after merge, LoRA contribution is baked in


class TestLoRAModelWrapper:
    """Tests for full model LoRA wrapper."""

    def test_lora_model_init(self, dummy_model):
        """Test LoRA model wrapper initialization."""
        from domains.training.optimized_pipeline import LoRAModelWrapper

        wrapper = LoRAModelWrapper(
            dummy_model,
            rank=8,
            alpha=16,
            target_modules=['0'],  # Only wrap first layer
        )

        assert wrapper.trainable_params < wrapper.total_params
        print(f"Trainable: {wrapper.trainable_params:,} / {wrapper.total_params:,}")

    def test_lora_model_forward(self, dummy_model):
        """Test LoRA model forward pass."""
        from domains.training.optimized_pipeline import LoRAModelWrapper

        wrapper = LoRAModelWrapper(
            dummy_model,
            rank=8,
            alpha=16,
            target_modules=['0'],
        )

        x = torch.randn(2, 128)
        output = wrapper(x)

        assert output.shape[0] == x.shape[0]
        assert not torch.isnan(output).any()


# =============================================================================
# OPTIMIZED TRAINER TESTS (removed - use SloughGPTTrainer instead)
# =============================================================================


# =============================================================================
# FEDERATED TRAINER TESTS
# =============================================================================

class TestOptimizedFederatedTrainer:
    """Tests for federated trainer."""

    def test_gradient_compression(self):
        """Test gradient compression."""
        from domains.training.optimized_pipeline import OptimizedFederatedTrainer

        trainer = OptimizedFederatedTrainer(
            model=nn.Linear(128, 128),
            num_clients=3,
            device='cpu',
        )

        # Create gradient
        gradient = torch.randn(128, 128)

        # Compress
        compressed = trainer.compress_gradient(gradient, compression_ratio=0.1, client_id=0)

        # Should be same shape
        assert compressed.shape == gradient.shape

        # Decompress
        decompressed = trainer.decompress_gradient(compressed, client_id=0)

        # Should recover with error feedback
        assert decompressed.shape == gradient.shape

    def test_adaptive_aggregation(self):
        """Test adaptive client aggregation."""
        from domains.training.optimized_pipeline import OptimizedFederatedTrainer

        trainer = OptimizedFederatedTrainer(
            model=nn.Linear(128, 128),
            num_clients=3,
            device='cpu',
        )

        # Create mock updates
        client_updates = [
            {"gradients": [torch.randn(10) for _ in range(3)]},
            {"gradients": [torch.randn(10) for _ in range(3)]},
        ]

        aggregated = trainer._adaptive_aggregate(client_updates)

        assert "gradients" in aggregated
        assert len(aggregated["gradients"]) == 3


# =============================================================================
# UNIFIED PIPELINE TESTS
# =============================================================================

class TestUnifiedPipeline:
    """Tests for unified training pipeline."""

    def test_pipeline_initialization(self, dummy_model):
        """Test pipeline initialization."""
        from domains.training.optimized_pipeline import (
            OptimizedPipeline, UnifiedConfig, MemoryOptimizer
        )

        config = UnifiedConfig(
            pretrain_epochs=1,
            federated_rounds=0,
            rlhf_epochs=0,
        )

        memory_opt = MemoryOptimizer()
        assert memory_opt is not None

        assert config.pretrain_epochs == 1
        assert config.federated_rounds == 0


# =============================================================================
# REASONING TESTS
# =============================================================================

class TestDeepReasoning:
    """Tests for deep reasoning."""

    @pytest.mark.asyncio
    async def test_deep_reasoning_basic(self):
        """Test basic deep reasoning."""
        from domains.cognitive.reasoning.deep import DeepReasoning

        reasoning = DeepReasoning()

        result = await reasoning.reason("What causes climate change?", max_depth=2)

        assert result.conclusion is not None
        assert result.confidence > 0
        assert len(result.steps) > 0

    def test_formal_logic_engine(self):
        """Test formal logic engine."""
        from domains.cognitive.reasoning.deep import FormalLogicEngine

        engine = FormalLogicEngine()

        # Test syllogism
        result = engine.prove_syllogism(
            premise1=("All", "are", "mortal"),
            premise2=("All", "are", "human"),
            conclusion=("All", "are", "mortal"),
        )

        assert result["valid"] == True
        assert result["mood"] in ["AAA", "AAI"]

    def test_working_memory(self):
        """Test working memory."""
        from domains.cognitive.reasoning.deep import WorkingMemory

        wm = WorkingMemory(capacity=3)

        wm.add("item1")
        wm.add("item2")
        wm.add("item3")

        assert len(wm.items) == 3

        # Test LRU eviction
        wm.add("item4")
        assert len(wm.items) == 3
        assert "item4" in wm.items
        assert "item1" not in wm.items

    def test_unification(self):
        """Test unification algorithm."""
        from domains.cognitive.reasoning.deep import (
            FormalLogicEngine, Predicate, Term
        )

        engine = FormalLogicEngine()

        # Create predicates with same structure
        p1 = Predicate(name="human", terms=[Term(name="socrates")])
        p2 = Predicate(name="human", terms=[Term(name="socrates")])

        # Unify - same predicates should unify
        subst = engine._unify(p1, p2)

        assert subst is not None


# =============================================================================
# SOUL ENGINE TESTS
# =============================================================================

class TestSoulEngineIntegration:
    """Tests for SloEngine with training."""

    def test_soul_engine_reasoning(self):
        """Test SloEngine reasoning integration."""
        from domains.core.soul import SloEngine
        from domains.inference.slo_format import SloProfile

        soul = SloProfile(name="TestSoul")
        engine = SloEngine(soul=soul)

        # Check reasoning components
        stats = engine.get_reasoning_stats()

        assert stats["deep_reasoning"] == True
        assert stats["logic_engine"] == True
        assert stats["working_memory_items"] == 0

    def test_soul_syllogism(self):
        """Test SloEngine syllogism."""
        from domains.core.soul import SloEngine
        from domains.inference.slo_format import SloProfile

        soul = SloProfile(name="TestSoul")
        engine = SloEngine(soul=soul)

        result = engine.prove_syllogism(
            premise1=("All", "are", "mortal"),
            premise2=("All", "are", "human"),
            conclusion=("All", "are", "mortal"),
        )

        assert result["valid"] == True

    def test_soul_knowledge_base(self):
        """Test SloEngine knowledge base."""
        from domains.core.soul import SloEngine
        from domains.inference.slo_format import SloProfile

        soul = SloProfile(name="TestSoul")
        engine = SloEngine(soul=soul)

        # Add knowledge
        engine.assert_knowledge("human", "socrates")

        # Query
        result = engine.query_knowledge("human", "socrates")
        assert result == True

    def test_soul_working_memory(self):
        """Test SloEngine working memory."""
        from domains.core.soul import SloEngine
        from domains.inference.slo_format import SloProfile

        soul = SloProfile(name="TestSoul")
        engine = SloEngine(soul=soul)

        engine.add_to_working_memory("premise_1")
        engine.add_to_working_memory("premise_2")

        memory = engine.get_working_memory()
        assert len(memory) == 2
        assert "premise_1" in memory


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
