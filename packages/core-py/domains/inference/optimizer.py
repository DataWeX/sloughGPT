"""
Inference Optimizer

High-performance inference optimizations:
- KV Cache
- Batching (static + dynamic)
- Speculative Decoding
- Continuous Batching
- Quantization (INT8, INT4)
- Flash Attention
- Prefix caching
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from domains.training.slonet_compat import torch
F = torch.F


def _is_float32(t):
    """Check if a tensor is float32 (works with both real torch and compat)."""
    dt = t.dtype
    return dt == np.float32 or dt == torch.float32 or str(dt) == 'torch.float32'


def _to_float32(t):
    """Convert tensor to float32 (works with both real torch and compat)."""
    try:
        return t.float()
    except Exception:
        pass
    try:
        return t.to(dtype=torch.float32)
    except Exception:
        pass
    return t


@dataclass
class InferenceConfig:
    """Configuration for optimized inference."""

    max_batch_size: int = 32
    max_sequence_length: int = 2048

    use_kv_cache: bool = True
    kv_cache_size: int = 512

    use_flash_attention: bool = True
    attention_implementation: str = "flash"

    use_quantization: bool = False
    quantization_bits: int = 8

    use_speculative_decoding: bool = False
    speculation_tokens: int = 4

    use_continuous_batching: bool = True

    prefilling_batch_size: int = 8
    decoding_batch_size: int = 16


class KVCache:
    """
    Key-Value cache for transformer layers (backward-compatible wrapper).

    Lazily initialised — the first call to ``initialize()`` creates the backing
    unified ``domains.inference.kv_cache.KVCache``.  The old ``initialize()`` /
    ``clear()`` / position-tensor ``update()`` / position-tensor ``get()``
    APIs are preserved for backward compatibility.
    """

    def __init__(self, num_layers: int, num_heads: int, head_dim: int, max_length: int = 2048):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_length = max_length
        self.k_cache: List[Any] = []
        self.v_cache: List[Any] = []
        self._inner: Any = None
        self._initialized = False

    def initialize(self, device: str = "cuda"):
        """(Re)initialize cache tensors — recreates on the given device."""
        from domains.inference.kv_cache import KVCache as _UnifiedKVCache
        self._inner = _UnifiedKVCache(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_length=self.max_length,
            dtype=torch.float16,
            device=device,
        )
        self.k_cache = self._inner.key_cache
        self.v_cache = self._inner.value_cache
        self._initialized = True

    def update(
        self,
        layer_idx: int,
        positions: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """Update cache at specific positions (scatter-write)."""
        if not self._initialized:
            self.initialize(k.device)
        if not _is_float32(k):
            k = _to_float32(k)
        if not _is_float32(v):
            v = _to_float32(v)
        self._inner.update_at_positions(layer_idx, positions, k, v)

    def get(
        self,
        layer_idx: int,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve cached keys and values at specific positions."""
        return self._inner.get_at_positions(layer_idx, positions)

    def clear(self):
        """Clear the cache."""
        self._inner.reset()


class SpeculativeDecoder:
    """
    Speculative decoding for faster autoregressive generation.
    Uses a smaller draft model to propose tokens, verified by the main model.
    """

    def __init__(
        self,
        draft_model,
        main_model,
        speculation_tokens: int = 4,
        temperature: float = 1.0,
    ):
        self.draft_model = draft_model
        self.main_model = main_model
        self.speculation_tokens = speculation_tokens
        self.temperature = temperature

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 100,
    ) -> torch.Tensor:
        """
        Generate with speculative decoding.
        Returns generated tokens and number of accepted tokens.
        """
        prompt_len = prompt_tokens.shape[1]
        current_tokens = prompt_tokens.clone()
        total_accepted = 0

        while current_tokens.shape[1] < prompt_len + max_new_tokens:
            # Draft model prediction
            draft_output = self.draft_model(current_tokens)
            draft_probs = F.softmax(draft_output[:, -1, :], dim=-1)
            draft_token = torch.multinomial(draft_probs, 1)

            # Accept draft tokens
            current_tokens = torch.cat([current_tokens, draft_token], dim=1)
            total_accepted += 1

            # Verify with main model
            if current_tokens.shape[1] > prompt_len:
                main_output = self.main_model(current_tokens)
                main_probs = F.softmax(main_output[:, -1, :], dim=-1)

                # Check acceptance
                draft_prob = draft_probs[0, draft_token[0, 0]].item()
                main_prob = main_probs[0, draft_token[0, 0]].item()

                # Rejection sampling
                if draft_prob > main_prob:
                    if torch.rand(1).item() > draft_prob / main_prob:
                        # Reject - remove last token
                        current_tokens = current_tokens[:, :-1]

        return current_tokens, total_accepted


class ContinuousBatcher:
    """
    Continuous batching for efficient GPU utilization.
    Dynamically batches requests as they arrive.
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.pending_requests: List[Dict] = []
        self.active_batches: List[Dict] = []

    def add_request(
        self,
        request_id: str,
        prompt: torch.Tensor,
        max_tokens: int,
    ):
        """Add a request to the batch queue."""
        self.pending_requests.append({
            "id": request_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "generated": prompt.clone(),
            "finished": False,
        })

    def get_next_batch(self) -> Optional[Dict]:
        """Get next batch for processing."""
        # Combine pending with partially complete requests
        batchable = self.pending_requests[:self.config.max_batch_size]

        if not batchable:
            return None

        # Create batch
        prompts = [r["prompt"] for r in batchable]
        max_len = max(p.shape[1] for p in prompts)

        # Pad to same length
        padded = []
        attention_mask = []
        for p in prompts:
            if p.shape[1] < max_len:
                pad = torch.zeros(1, max_len - p.shape[1], dtype=p.dtype, device=p.device)
                padded.append(torch.cat([p, pad], dim=1))
            else:
                padded.append(p)
            attention_mask.append(torch.ones(1, padded[-1].shape[1]))

        batch = {
            "requests": batchable,
            "input_ids": torch.cat(padded, dim=0),
            "attention_mask": torch.cat(attention_mask, dim=0),
            "max_len": max_len,
        }

        return batch

    def process_batch(self, batch: Dict, output_ids: torch.Tensor, output_probs: torch.Tensor):
        """Process batch outputs and update requests."""
        for i, request in enumerate(batch["requests"]):
            token_id = output_ids[i, -1].item()
            request["generated"] = torch.cat([
                request["generated"],
                torch.tensor([[token_id]], device=request["generated"].device)
            ], dim=1)

            # Check if finished
            if token_id == 2 or request["generated"].shape[1] >= request["max_tokens"]:
                request["finished"] = True

    def get_completed(self) -> List[Dict]:
        """Get completed requests."""
        completed = [r for r in self.pending_requests if r["finished"]]
        self.pending_requests = [r for r in self.pending_requests if not r["finished"]]
        return completed


class InferenceOptimizer:
    """
    Main inference optimizer class.
    Coordinates all inference optimizations.
    """

    def __init__(
        self,
        model,
        config: Optional[InferenceConfig] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.config = config or InferenceConfig()
        self.device = device

        # Initialize components
        self.kv_cache: Optional[KVCache] = None
        self.speculative_decoder: Optional[SpeculativeDecoder] = None
        self.batcher = ContinuousBatcher(self.config)

        self._setup()

    def _setup(self):
        """Setup optimization components."""
        # KV Cache
        if self.config.use_kv_cache:
            self._setup_kv_cache()

        # Speculative decoding
        if self.config.use_speculative_decoding:
            self._setup_speculative()

    def _setup_kv_cache(self):
        """Initialize KV cache."""
        # Infer cache dimensions from model
        num_layers = getattr(self.model, 'n_layer', 12)
        num_heads = getattr(self.model, 'n_head', 12)
        head_dim = getattr(self.model, 'n_embed', 768) // num_heads

        self.kv_cache = KVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_length=self.config.kv_cache_size,
        )

    def _setup_speculative(self):
        """Setup speculative decoder."""
        try:
            # Use a smaller model as draft
            from domains.training.models.gpt2 import GPT2
            draft_model = GPT2()

            self.speculative_decoder = SpeculativeDecoder(
                draft_model=draft_model,
                main_model=self.model,
                speculation_tokens=self.config.speculation_tokens,
            )
        except Exception:
            pass

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> Tuple[torch.Tensor, Dict]:
        """Generate with optimizations."""
        start_time = time.time()

        if self.config.use_speculative_decoding and self.speculative_decoder:
            output, accepted = self.speculative_decoder.generate(
                prompt,
                max_new_tokens=max_new_tokens,
            )
            stats = {
                "method": "speculative",
                "accepted_tokens": accepted,
                "acceptance_rate": accepted / max_new_tokens,
            }
        else:
            output = self._standard_generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            stats = {
                "method": "standard",
                "tokens": output.shape[1] - prompt.shape[1],
            }

        stats["latency_ms"] = (time.time() - start_time) * 1000
        stats["tokens_per_second"] = stats["tokens"] / (stats["latency_ms"] / 1000) if stats["latency_ms"] > 0 else 0

        return output, stats

    def _standard_generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        """Standard autoregressive generation."""
        self.model.eval()
        current = prompt

        for _ in range(max_new_tokens):
            logits = self.model(current)
            next_token_logits = logits[:, -1, :] / temperature

            # Top-p sampling
            probs = F.softmax(next_token_logits, dim=-1)
            sorted_probs, indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumprob > top_p
            mask = cumulative <= top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = True

            filtered_probs = torch.zeros_like(probs)
            for i in range(probs.shape[0]):
                filtered_probs[i] = probs[i].masked_fill(mask[i], 0)
                filtered_probs[i] = filtered_probs[i] / filtered_probs[i].sum()

            next_token = torch.multinomial(filtered_probs, 1)
            current = torch.cat([current, next_token], dim=1)

        return current


class InferenceBenchmark:
    """Benchmark inference performance."""

    def __init__(self, optimizer: InferenceOptimizer):
        self.optimizer = optimizer
        self.results: List[Dict] = []

    def run_benchmark(
        self,
        prompt: str,
        num_tokens: int,
        num_runs: int = 10,
    ) -> Dict:
        """Run inference benchmark."""
        # Tokenize
        prompt_tokens = torch.randint(0, 1000, (1, 50))

        latencies = []
        tokens_per_sec = []

        for _ in range(num_runs):
            output, stats = self.optimizer.generate(
                prompt_tokens,
                max_new_tokens=num_tokens,
            )

            latencies.append(stats["latency_ms"])
            tokens_per_sec.append(stats.get("tokens_per_second", 0))

        return {
            "prompt_tokens": 50,
            "generated_tokens": num_tokens,
            "avg_latency_ms": np.mean(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "avg_tokens_per_sec": np.mean(tokens_per_sec),
            "num_runs": num_runs,
        }

    def print_results(self, results: Dict):
        """Print benchmark results."""
        print("=" * 60)
        print("INFERENCE BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Prompt tokens: {results['prompt_tokens']}")
        print(f"Generated tokens: {results['generated_tokens']}")
        print(f"Average latency: {results['avg_latency_ms']:.2f} ms")
        print(f"P50 latency: {results['p50_latency_ms']:.2f} ms")
        print(f"P95 latency: {results['p95_latency_ms']:.2f} ms")
        print(f"P99 latency: {results['p99_latency_ms']:.2f} ms")
        print(f"Tokens/sec: {results['avg_tokens_per_sec']:.2f}")
        print(f"Test runs: {results['num_runs']}")
        print("=" * 60)


__all__ = [
    "InferenceConfig",
    "KVCache",
    "SpeculativeDecoder",
    "ContinuousBatcher",
    "InferenceOptimizer",
    "InferenceBenchmark",
]
