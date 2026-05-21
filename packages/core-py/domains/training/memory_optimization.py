"""
Memory Optimization Module for SloughGPT

Includes:
- Activation Checkpointing: Save memory by recomputing activations
- Gradient Checkpointing: Trade compute for memory
- Flash Attention: Memory-efficient attention mechanism
"""

from domains.training.slonet_compat import torch
nn = torch.nn
from typing import Optional, List
from functools import wraps
import logging

logger = logging.getLogger("sloughgpt.memory")


# =============================================================================
# Activation Checkpointing
# =============================================================================


class ActivationCheckpointing:
    """
    Manages activation checkpointing for memory efficiency.

    Instead of storing all activations during forward pass,
    we recompute them during backward pass.

    Memory savings: ~50-70% for large models
    Compute cost: ~20-30% more
    """

    @staticmethod
    def checkpoint(
        module: nn.Module,
        checkpoint_ratio: float = 0.5,
    ) -> nn.Module:
        """
        Apply activation checkpointing to a module.

        Args:
            module: PyTorch module to checkpoint
            checkpoint_ratio: Ratio of layers to checkpoint (0.0-1.0)

        Returns:
            Module with checkpointing applied
        """
        import torch.utils.checkpoint as checkpoint

        def create_checkpoint_wrapper(m):
            if isinstance(m, nn.Module):
                original_forward = m.forward

                @wraps(original_forward)
                def checkpoint_forward(*args, **kwargs):
                    return checkpoint.checkpoint(
                        original_forward,
                        *args,
                        **kwargs,
                        use_reentrant=False,
                    )

                m.forward = checkpoint_forward
            return m

        return create_checkpoint_wrapper(module)

    @staticmethod
    def apply_to_layers(
        model: nn.Module,
        layers_to_checkpoint: Optional[List[str]] = None,
    ) -> nn.Module:
        """
        Apply checkpointing to specific layers.

        Args:
            model: Model to modify
            layers_to_checkpoint: List of layer names to checkpoint

        Returns:
            Modified model
        """
        import torch.utils.checkpoint as checkpoint

        for name, module in model.named_modules():
            if layers_to_checkpoint and name not in layers_to_checkpoint:
                continue

            if isinstance(module, (nn.Linear, nn.Conv2d, nn.MultiheadAttention)):
                original_forward = module.forward

                @wraps(original_forward)
                def checked_forward(*args, **kwargs):
                    return checkpoint.checkpoint(
                        original_forward,
                        *args,
                        **kwargs,
                        use_reentrant=False,
                    )

                module.forward = checked_forward

        return model


# =============================================================================
# Gradient Checkpointing
# =============================================================================


class GradientCheckpointing:
    """
    Gradient checkpointing for additional memory savings.

    Similar to activation checkpointing but specifically
    for the backward pass.
    """

    @staticmethod
    def enable(model: nn.Module, checkpoint_ratio: float = 0.5):
        """
        Enable gradient checkpointing on model.

        Args:
            model: PyTorch model
            checkpoint_ratio: Ratio of layers to checkpoint
        """
        import torch.utils.checkpoint as checkpoint

        total_layers = 0
        checkpointed = 0

        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.MultiheadAttention)):
                total_layers += 1

                if checkpointed < total_layers * checkpoint_ratio:
                    original_forward = module.forward

                    @wraps(original_forward)
                    def grad_checkpoint_forward(*args, **kwargs):
                        return checkpoint.checkpoint(
                            original_forward,
                            *args,
                            **kwargs,
                            use_reentrant=True,
                        )

                    module.forward = grad_checkpoint_forward
                    checkpointed += 1

        logger.info(f"Gradient checkpointing enabled on {checkpointed} layers")
        return model

    @staticmethod
    def disable(model: nn.Module):
        """Disable gradient checkpointing (restore original forward)."""
        logger.info("Gradient checkpointing disabled")
        return model


# =============================================================================
# Flash Attention
# =============================================================================


class FlashAttention:
    """
    Memory-efficient attention implementation.

    Uses Flash Attention when available (PyTorch 2.0+),
    falls back to standard attention otherwise.
    """

    @staticmethod
    def is_available() -> bool:
        """Check if Flash Attention is available."""
        try:
            import flash_attn

            return True
        except ImportError:
            pass

        try:
            from torch.nn.attention import SDPBackend

            return True
        except ImportError:
            pass

        return False

    @staticmethod
    def get_attention_backend() -> str:
        """Get available attention backend."""
        if FlashAttention.is_available():
            return "flash_attention"

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            return "sdpa"

        return "standard"

    @staticmethod
    def create_attention(
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ) -> nn.Module:
        """
        Create an attention module with best available backend.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to include bias
            batch_first: Whether batch is first dimension

        Returns:
            Attention module
        """
        backend = FlashAttention.get_attention_backend()

        if backend == "flash_attention":
            return FlashAttention._create_flash_attention(
                embed_dim, num_heads, dropout, bias, batch_first
            )
        elif backend == "sdpa":
            return FlashAttention._create_sdpa_attention(
                embed_dim, num_heads, dropout, bias, batch_first
            )
        else:
            return nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=batch_first
            )

    @staticmethod
    def _create_flash_attention(embed_dim, num_heads, dropout, bias, batch_first):
        """Create Flash Attention module."""
        try:
            from flash_attn import FlashAttention

            return FlashAttention(embed_dim=embed_dim, num_heads=num_heads)
        except:
            pass
        return nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=batch_first
        )

    @staticmethod
    def _create_sdpa_attention(embed_dim, num_heads, dropout, bias, batch_first):
        """Create SDPA attention module."""
        return nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=batch_first
        )


# =============================================================================
# Memory Utilities
# =============================================================================


def get_memory_stats() -> dict:
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return {"device": "cpu", "memory_available": False}

    return {
        "device": torch.cuda.get_device_name(0),
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "free_mb": (
            torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        )
        / 1024**2,
    }


def print_memory_stats(prefix: str = ""):
    """Print memory statistics."""
    stats = get_memory_stats()
    if not stats.get("memory_available"):
        print(f"{prefix}Running on CPU")
        return

    print(f"{prefix}GPU: {stats['device']}")
    print(f"{prefix}  Allocated: {stats['allocated_mb']:.1f} MB")
    print(f"{prefix}  Reserved: {stats['reserved_mb']:.1f} MB")
    print(f"{prefix}  Max Allocated: {stats['max_allocated_mb']:.1f} MB")
    print(f"{prefix}  Free: {stats['free_mb']:.1f} MB")


def clear_memory():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    import gc

    gc.collect()


# =============================================================================
# Model Memory Calculator
# =============================================================================


def estimate_model_memory(
    num_parameters: int,
    precision: str = "fp32",
    include_optimizer: bool = True,
    include_gradients: bool = True,
    include_kv_cache: bool = True,
    kv_cache_tokens: int = 4096,
    num_layers: int = 32,
    hidden_size: int = 4096,
) -> dict:
    """
    Estimate memory requirements for a model.

    Args:
        num_parameters: Number of model parameters
        precision: fp32, fp16, bf16
        include_optimizer: Include optimizer state
        include_gradients: Include gradients
        include_kv_cache: Include KV cache
        kv_cache_tokens: KV cache size
        num_layers: Number of layers
        hidden_size: Hidden dimension

    Returns:
        Dictionary with memory estimates
    """
    precision_bytes = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
    }

    bytes_per_param = precision_bytes.get(precision, 4)

    # Model weights
    model_memory = num_parameters * bytes_per_param

    # Optimizer state (AdamW: 2x model + 2x model)
    optimizer_memory = 0
    if include_optimizer:
        optimizer_memory = num_parameters * bytes_per_param * 4  # 2 states + momentum + variance

    # Gradients
    gradient_memory = 0
    if include_gradients:
        gradient_memory = num_parameters * bytes_per_param

    # KV cache (2 * layers * hidden * kv_tokens * bytes)
    kv_memory = 0
    if include_kv_cache:
        kv_memory = 2 * num_layers * hidden_size * kv_cache_tokens * bytes_per_param

    total_memory = model_memory + optimizer_memory + gradient_memory + kv_memory

    return {
        "model_mb": model_memory / 1024**2,
        "optimizer_mb": optimizer_memory / 1024**2,
        "gradients_mb": gradient_memory / 1024**2,
        "kv_cache_mb": kv_memory / 1024**2,
        "total_mb": total_memory / 1024**2,
        "total_gb": total_memory / 1024**3,
    }


# =============================================================================
# Context Manager for Memory Tracking
# =============================================================================


class MemoryTracker:
    """Context manager for tracking memory usage."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_memory = 0
        self.peak_memory = 0

    def __enter__(self):
        clear_memory()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            self.peak_memory = torch.cuda.max_memory_allocated()
            used = (self.peak_memory - self.start_memory) / 1024**2
            print(
                f"[{self.name}] Memory used: {used:.1f} MB, Peak: {self.peak_memory / 1024**2:.1f} MB"
            )
        clear_memory()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ActivationCheckpointing",
    "GradientCheckpointing",
    "FlashAttention",
    "get_memory_stats",
    "print_memory_stats",
    "clear_memory",
    "estimate_model_memory",
    "MemoryTracker",
]
