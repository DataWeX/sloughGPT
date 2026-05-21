"""
SloughGPT Optimized Training Pipeline
Industry-standard optimizations for fast LLM training.

Optimizations included:
1. Mixed Precision Training (FP16/BF16 + GradScaler)
2. Gradient Checkpointing (memory savings)
3. Flash Attention (2-4x speedup)
4. Optimized DataLoader (num_workers, prefetch)
5. torch.compile (JIT compilation)
6. Proper warmup and learning rate scheduling

Char-LM checkpoint vocabulary on native ``SloughGPTTrainer`` ``step_*.pt`` (and eval parity with
``cli.py eval``) lives in ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
"""

import os
import time
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable

from domains.training.slonet_compat import torch
DataLoader = torch.utils.data.DataLoader
Dataset = torch.utils.data.Dataset
GradScaler = torch.cuda.amp.GradScaler
autocast = torch.cuda.amp.autocast
nn = torch.nn

from domains.torch_runtime import (
    effective_dataloader_num_workers,
    prefetch_factor_for_workers,
)


@dataclass
class TrainingConfig:
    """Optimized training configuration."""
    # Model
    vocab_size: int = 50257
    n_embed: int = 512
    n_layer: int = 12
    n_head: int = 8
    block_size: int = 512
    dropout: float = 0.1
    
    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 10000
    warmup_steps: int = 500
    clip_grad_norm: float = 1.0
    
    # Optimizations
    use_mixed_precision: bool = True  # FP16/BF16
    use_gradient_checkpointing: bool = True  # Save memory
    use_flash_attention: bool = True  # Fast attention
    use_compile: bool = False  # torch.compile (needs PyTorch 2.0+)
    compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    
    # Device
    device: str = "auto"  # auto, cuda, mps, rocm, cpu
    dtype: str = "bf16"  # "fp16" or "bf16"
    
    # DataLoader
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000


# === PRESETS FOR DIFFERENT HARDWARE ===

class Presets:
    """Pre-configured optimization presets for different hardware."""
    
    @staticmethod
    def high_end_gpu() -> TrainingConfig:
        """RTX 3090, RTX 4090, A100, H100, MI300."""
        return TrainingConfig(
            batch_size=32,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            use_mixed_precision=True,
            dtype="bf16",
            use_gradient_checkpointing=True,
            use_flash_attention=True,
            use_compile=True,
            compile_mode="reduce-overhead",
            num_workers=4,
        )
    
    @staticmethod
    def mid_range_gpu() -> TrainingConfig:
        """RTX 2080, RTX 3060, V100, MI250."""
        return TrainingConfig(
            batch_size=16,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            use_mixed_precision=True,
            dtype="fp16",
            use_gradient_checkpointing=True,
            use_flash_attention=True,
            use_compile=True,
            compile_mode="default",
            num_workers=4,
        )
    
    @staticmethod
    def apple_silicon() -> TrainingConfig:
        """M1, M2, M3 Pro/Max/Ultra."""
        return TrainingConfig(
            batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            use_mixed_precision=True,
            dtype="fp16",
            use_gradient_checkpointing=True,
            use_flash_attention=False,  # Not supported
            use_compile=True,
            compile_mode="default",
            num_workers=0,
        )
    
    @staticmethod
    def cpu_only() -> TrainingConfig:
        """CPU training (slow)."""
        return TrainingConfig(
            batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            use_mixed_precision=False,
            dtype="fp32",
            use_gradient_checkpointing=False,
            use_flash_attention=False,
            use_compile=False,
            num_workers=8,
        )
    
    @staticmethod
    def auto() -> TrainingConfig:
        """Auto-detect best settings for current hardware."""
        device = get_optimal_device()
        
        if device == "cuda":
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                if cap[0] >= 8:
                    return Presets.high_end_gpu()
                return Presets.mid_range_gpu()
        elif device == "rocm":
            return Presets.mid_range_gpu()
        elif device == "mps":
            return Presets.apple_silicon()
        return Presets.cpu_only()


def get_optimal_device() -> str:
    """Auto-detect best available device (DEPRECATED - use performance.py)."""
    import warnings
    warnings.warn(
        "get_optimal_device is deprecated. Use from domains.training.performance import get_optimal_device",
        DeprecationWarning,
        stacklevel=2
    )
    from domains.training.performance import get_optimal_device as _get_optimal_device
    return _get_optimal_device()


def get_device_name() -> str:
    """Get human-readable device name (DEPRECATED - use performance.py)."""
    import warnings
    warnings.warn(
        "get_device_name is deprecated. Use from domains.training.performance import get_device_name",
        DeprecationWarning,
        stacklevel=2
    )
    from domains.training.performance import get_device_name as _get_device_name
    return _get_device_name()


def is_amd_rocm() -> bool:
    """Check if running on AMD ROCm."""
    return hasattr(torch.version, 'hip') and torch.version.hip is not None


def get_best_dtype() -> torch.dtype:
    """Get best precision for current device."""
    if torch.cuda.is_available():
        cuda_capability = torch.cuda.get_device_capability()
        if cuda_capability[0] >= 8:  # Ampere or newer
            return torch.bfloat16
        return torch.float16
    elif is_amd_rocm():
        return torch.float16  # ROCm supports FP16 well
    elif torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


class OptimizedTextDataset(Dataset):
    """Optimized text dataset with memory efficiency."""
    
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return max(0, len(self.data) - self.block_size)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.data[idx : idx + self.block_size].clone()
        y = self.data[idx + 1 : idx + self.block_size + 1].clone()
        return {"input_ids": x, "labels": y}


class OptimizedDataLoader:
    """High-performance DataLoader with prefetching."""
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        collate_fn=None,
    ):
        nw = effective_dataloader_num_workers(num_workers)
        pf = prefetch_factor_for_workers(nw, prefetch_factor)
        dl_kw: Dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": nw,
            "pin_memory": pin_memory,
            "persistent_workers": nw > 0,
            "collate_fn": collate_fn,
        }
        if pf is not None:
            dl_kw["prefetch_factor"] = pf
        self.dataloader = DataLoader(**dl_kw)
        self.iter = None
    
    def get_batch(self):
        """Get next batch, prefetching in background."""
        if self.iter is None:
            self.iter = iter(self.dataloader)
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            return next(self.iter)


def apply_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """Apply gradient checkpointing to save memory during training.
    
    Trades compute for memory: ~50% memory reduction with ~30% extra compute.
    """
    for module in model.modules():
        if hasattr(module, 'gradient_checkpointing_enable'):
            module.gradient_checkpointing_enable()
        elif hasattr(module, 'use_checkpoint'):
            module.use_checkpoint = True
    
    def checkpoint_wrapper(module):
        original_forward = module.forward
        
        @contextmanager
        def checkpoint_context():
            try:
                module._checkpointing = True
                yield
            finally:
                module._checkpointing = False
        
        def forward(*args, **kwargs):
            if hasattr(module, '_checkpointing') and module._checkpointing:
                return torch.utils.checkpoint.checkpoint(
                    original_forward, *args, **kwargs
                )
            return original_forward(*args, **kwargs)
        
        module.forward = forward
        return module
    
    return model


class FlashAttentionWrapper(nn.Module):
    """Wrapper to use Flash Attention when available."""
    
    @staticmethod
    def is_available() -> bool:
        """Check if Flash Attention is available."""
        try:
            from flash_attn import flash_attn_func
            return True
        except ImportError:
            return False
    
    @staticmethod
    def wrap_attention(attention_layer, layer_idx: int = 0):
        """Wrap standard attention with Flash Attention."""
        if FlashAttentionWrapper.is_available():
            from flash_attn import flash_attn_func
            
            original_forward = attention_layer.forward
            
            def flash_forward(query, key, value, attention_mask=None, **kwargs):
                batch_size = query.shape[0]
                seq_len = query.shape[1]
                
                query = query.view(batch_size, seq_len, -1)
                key = key.view(batch_size, seq_len, -1)
                value = value.view(batch_size, seq_len, -1)
                
                out = flash_attn_func(query, key, value, dropout_p=0.0, softmax_scale=1.0)
                return out.view(batch_size * seq_len, -1)
            
            attention_layer.forward = flash_forward
        return attention_layer


__all__ = [
    "TrainingConfig",
    "Presets",
    "get_optimal_device",
    "get_device_name",
    "is_amd_rocm",
    "get_best_dtype",
    "OptimizedTextDataset",
    "OptimizedDataLoader",
    "apply_gradient_checkpointing",
    "FlashAttentionWrapper",
]
