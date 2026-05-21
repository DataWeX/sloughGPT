"""
Performance Optimization Utilities
KV cache, batching, and inference optimizations.
"""

try:
    import torch
except ImportError:
    from domains.training.slonet_compat import torch
try:
    import torch.nn as nn
except ImportError:
    nn = torch.nn
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class OptimizationConfig:
    """Configuration for inference optimizations."""
    use_flash_attention: bool = False
    use_kv_cache: bool = True
    max_batch_size: int = 32
    prefill_chunk_size: int = 512
    use_speculative: bool = False
    speculative_tokens: int = 4


from domains.inference.kv_cache import KVCache as _UnifiedKVCache


class KVCacheOptimizer:
    """Optimized KV cache with pre-allocation and efficient memory management.

    Backed by the unified ``domains.inference.kv_cache.KVCache``.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_length: int = 4096,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self._cache = _UnifiedKVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_length=max_length,
            dtype=dtype,
            device=device,
        )
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_length = max_length
        self.dtype = dtype
        self.device = device
        self.key_cache = self._cache.key_cache
        self.value_cache = self._cache.value_cache
        self.current_length = 0

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        position: int,
    ):
        """Update cache at specific position."""
        self._cache.update(layer_idx, key, value, position=position)
        self.current_length = max(self.current_length, position + key.shape[2])

    def get(
        self,
        layer_idx: int,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached keys/values from start to end."""
        if end is None:
            end = self._cache.current_lengths[layer_idx]
        return self._cache.get(layer_idx, start, end)

    def reset(self):
        """Reset cache."""
        self._cache.reset()
        self.current_length = 0

    def get_allocated_memory_mb(self) -> float:
        """Get memory allocated for cache in MB."""
        return self._cache.memory_used_mb()


class AttentionMask:
    """Optimized attention mask operations."""
    
    @staticmethod
    def create_causal_mask(
        seq_len: int,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        return mask.masked_fill(mask, float("-inf"))
    
    @staticmethod
    def create_padder_mask(
        input_ids: torch.Tensor,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """Create mask for padded sequences."""
        return input_ids != pad_token_id


class BatchProcessor:
    """Efficient batch processing for inference."""
    
    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
    
    def pad_to_batch(
        self,
        input_ids_list: List[torch.Tensor],
        pad_token_id: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad sequences to same length for batched inference."""
        max_len = max(ids.shape[1] for ids in input_ids_list)
        
        batch_tensor = torch.full(
            (len(input_ids_list), max_len),
            pad_token_id,
            dtype=torch.long
        )
        attention_mask = torch.zeros(
            (len(input_ids_list), max_len),
            dtype=torch.long
        )
        
        for i, ids in enumerate(input_ids_list):
            seq_len = ids.shape[1]
            batch_tensor[i, :seq_len] = ids
            attention_mask[i, :seq_len] = 1
        
        return batch_tensor, attention_mask
    
    def split_by_length(
        self,
        input_ids: torch.Tensor,
        max_length: int = 512,
    ) -> List[torch.Tensor]:
        """Split long sequences into chunks."""
        batch_size, seq_len = input_ids.shape
        chunks = []
        
        for b in range(batch_size):
            for i in range(0, seq_len, max_length):
                chunk = input_ids[b:b+1, i:i+max_length]
                chunks.append(chunk)
        
        return chunks


class SpeculativeDecoder:
    """Speculative decoding for faster inference."""
    
    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        speculative_tokens: int = 4,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.speculative_tokens = speculative_tokens
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate with speculative decoding."""
        draft_model = self.draft_model
        target_model = self.target_model
        
        for _ in range(max_new_tokens):
            draft_tokens = input_ids
            for _ in range(self.speculative_tokens):
                draft_output = draft_model(draft_tokens)
                draft_logits = draft_output.logits[:, -1, :] / temperature
                draft_next = torch.argmax(draft_logits, dim=-1, keepdim=True)
                draft_tokens = torch.cat([draft_tokens, draft_next], dim=1)
            
            target_output = target_model(draft_tokens)
            target_logits = target_output.logits[:, -self.speculative_tokens - 1:, :] / temperature
            
            for i in range(self.speculative_tokens):
                idx = self.speculative_tokens - i - 1
                draft_token = draft_tokens[:, -(i + 1)]
                target_prob = torch.softmax(target_logits[:, idx, :], dim=-1)
                target_token = torch.argmax(target_prob, dim=-1)
                
                if not torch.allclose(draft_token, target_token):
                    input_ids = draft_tokens[:, :-(i + 1)]
                    break
            else:
                input_ids = draft_tokens
        
        return input_ids


def estimate_inference_memory(
    num_parameters: int,
    precision: str = "fp16",
    kv_cache_multiplier: float = 1.5,
) -> Dict[str, float]:
    """Estimate memory required for inference."""
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
    }
    
    bytes_per = bytes_per_param.get(precision, 2)
    model_memory = (num_parameters * bytes_per) / (1024 ** 3)
    total_memory = model_memory * kv_cache_multiplier
    
    return {
        "model_memory_gb": model_memory,
        "kv_cache_memory_gb": model_memory * (kv_cache_multiplier - 1),
        "total_memory_gb": total_memory,
        "precision": precision,
        "num_parameters": num_parameters,
    }


def optimize_model_for_inference(
    model: nn.Module,
    use_quantization: bool = False,
    precision: str = "fp16",
) -> nn.Module:
    """Apply inference optimizations to model."""
    model.eval()
    
    if use_quantization:
        if precision == "int8":
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
        elif precision in ("fp16", "bf16"):
            model = model.half()
    
    return model


__all__ = [
    "OptimizationConfig",
    "KVCacheOptimizer",
    "AttentionMask",
    "BatchProcessor",
    "SpeculativeDecoder",
    "estimate_inference_memory",
    "optimize_model_for_inference",
]
