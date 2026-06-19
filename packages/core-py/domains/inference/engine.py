"""
Production Inference Engine
High-performance local model inference with KV cache, batching, and memory optimization.
"""

import asyncio
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, AsyncIterator
from collections import deque
import logging

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    from domains.training.slonet_compat import torch  # type: ignore[no-redef]
    F = torch.F

from domains.errors import require_non_empty_prompt

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """A single generation request."""

    id: str
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.0
    stop_tokens: List[int] = field(default_factory=lambda: [])

    generated_text: str = ""
    tokens: List[int] = field(default_factory=list)
    finished: bool = False
    error: Optional[str] = None


@dataclass
class BatchedRequest:
    """A request ready for batched inference."""

    request: GenerationRequest
    input_ids: torch.Tensor
    position: int = 0


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    do_sample: bool = True
    seed: Optional[int] = None


class KVCache:
    """Key-Value cache for transformer layers (backward-compatible wrapper).

    Dynamically grows via ``torch.cat``, matching the original API.
    The internal unified cache is used when dimensions are explicitly configured.
    """

    def __init__(self, num_layers: int, dtype: torch.dtype = torch.float16):
        self.num_layers = num_layers
        self.dtype = dtype
        self.key_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self.value_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self.max_length = 0
        self._unified = None

    def _ensure_unified(self, key: torch.Tensor):
        """Lazily initialise the unified cache from the first update's shape."""
        if self._unified is None:
            _, num_heads, _, head_dim = key.shape
            from domains.inference.kv_cache import KVCache as _KVCache
            self._unified = _KVCache(
                num_layers=self.num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                max_length=max(self.max_length, 4096),
                dtype=self.dtype,
                device=str(key.device),
            )
            # Copy any existing data
            for i in range(self.num_layers):
                if self.key_cache[i] is not None:
                    self._unified.update(i, self.key_cache[i], self.value_cache[i], position=0)

    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """Update cache for a specific layer."""
        self._ensure_unified(key)
        self._unified.update(layer_idx, key, value, position=self.max_length)
        self.max_length = max(self.max_length, self.max_length + key.shape[2])
        self.key_cache[layer_idx] = self._unified.key_cache[layer_idx]
        self.value_cache[layer_idx] = self._unified.value_cache[layer_idx]

    def get(self, layer_idx: int) -> tuple:
        """Get cached key-value pair."""
        if self._unified is not None:
            return self._unified.get(layer_idx, 0, self._unified.current_lengths[layer_idx])
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reset(self):
        """Reset all caches."""
        if self._unified is not None:
            self._unified.reset()
        self.key_cache = [None] * self.num_layers
        self.value_cache = [None] * self.num_layers
        self.max_length = 0
        self._unified = None

    def get_full(self, layer_idx: int) -> tuple:
        """Alias for ``get()`` — returns all cached positions."""
        return self.get(layer_idx)


class InferenceEngine:
    """
    Production-grade inference engine with:
    - KV caching
    - Continuous batching
    - Memory optimization
    - Streaming generation
    - Speculative decoding (optional)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_batch_size: int = 32,
        max_sequence_length: int = 4096,
        use_cache: bool = True,
        compile_mode: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.use_cache = use_cache
        self._is_mps = self.device.type == "mps"

        self.model.eval()
        self.model.to(self.device)

        self._compiled_forward = None
        if compile_mode and hasattr(torch, "compile"):
            try:
                self._compiled_forward = torch.compile(self.model, mode=compile_mode)
                print(f"Model compiled with mode: {compile_mode}")
            except Exception as e:
                print(f"Compilation failed: {e}")

        self._lock = threading.Lock()
        self._active_requests: Dict[str, GenerationRequest] = {}
        self._pending_queue: deque = deque()
        self._cache: Optional[KVCache] = None

        if self.use_cache and hasattr(model, "config"):
            num_layers = getattr(model.config, "num_hidden_layers", 12)
            self._cache = KVCache(num_layers)

        self._stats = {
            "requests_processed": 0,
            "tokens_generated": 0,
            "total_time": 0.0,
        }

        # LoRA adapter (optional)
        self._lora_adapter = None

    def set_lora_adapter(self, adapter):
        """Set LoRA adapter for personalization."""
        self._lora_adapter = adapter

    def _apply_lora_to_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adapter adjustment to logits (if compatible)."""
        if self._lora_adapter is None:
            return logits

        try:
            import numpy as np

            if not (hasattr(self._lora_adapter, "W_b") and hasattr(self._lora_adapter, "W_a")):
                return logits

            W_a = self._lora_adapter.W_a
            W_b = self._lora_adapter.W_b
            alpha = getattr(self._lora_adapter, "alpha", 16)
            rank = getattr(self._lora_adapter, "rank", 8)
            feedback_count = getattr(self._lora_adapter, "feedback_count", 1)

            # Convert to torch tensors if numpy
            if isinstance(W_a, np.ndarray):
                W_a = torch.from_numpy(W_a).to(logits.device, dtype=logits.dtype)
            if isinstance(W_b, np.ndarray):
                W_b = torch.from_numpy(W_b).to(logits.device, dtype=logits.dtype)

            # Check dimension compatibility
            lora_dim = W_b.shape[0]  # Should be model hidden dim
            logits_dim = logits.shape[-1]

            if lora_dim != logits_dim:
                # Dimensions don't match - skip LoRA for this generation
                return logits

            # Compute LoRA bias as mean of W_b @ W_a
            lora_matrix = torch.matmul(W_b, W_a)  # (dim, dim)
            lora_bias = lora_matrix.mean(dim=1)  # (dim,)

            # Scale based on alpha/rank and feedback confidence
            scale = (alpha / rank) * min(1.0, feedback_count / 10.0) * 0.05

            # Apply bias
            logits = logits + lora_bias * scale

        except Exception:
            pass

        return logits

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, return_tensors="pt").squeeze().tolist()

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _apply_repetition_penalty(
        self, logits: torch.Tensor, prev_tokens: torch.Tensor, penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to logits (no in-place mutation)."""
        if penalty == 1.0:
            return logits

        logits = logits.clone()
        for token_id in prev_tokens.unique():
            mask = logits[token_id] > 0
            logits[token_id] = torch.where(
                mask, logits[token_id] * penalty, logits[token_id] / penalty
            )

        return logits

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> int:
        """Sample a single token from logits."""
        if temperature == 0:
            return logits.argmax().item()

        logits = logits / temperature

        if top_k > 0:
            top_k_val = min(top_k, logits.numel())
            values, indices = torch.topk(logits, top_k_val)
            logits = torch.full_like(logits, float("-inf"))
            logits[indices] = values

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-10)
        return torch.multinomial(probs.unsqueeze(0), num_samples=1).item()

    def generate_single(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
    ) -> str:
        """Generate text for a single prompt (synchronous)."""
        prompt = require_non_empty_prompt(prompt)
        start_time = time.time()

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated = []
        past_key_values = None

        with torch.no_grad():
            for step in range(max_new_tokens):
                if past_key_values is not None:
                    model_inp = input_ids[:, -1:]
                else:
                    model_inp = input_ids

                try:
                    outputs = self.model(model_inp, past_key_values=past_key_values, use_cache=self.use_cache)
                except TypeError:
                    outputs = self.model(model_inp)

                if hasattr(outputs, "logits"):
                    logits = outputs.logits[:, -1, :].squeeze(0)
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs[:, -1, :].squeeze(0)
                else:
                    logits = outputs[0][:, -1, :].squeeze(0)

                if self.use_cache and hasattr(outputs, "past_key_values"):
                    past_key_values = outputs.past_key_values

                logits = self._apply_lora_to_logits(logits)

                if repetition_penalty != 1.0:
                    logits = self._apply_repetition_penalty(logits, input_ids[0], repetition_penalty)

                next_token = self._sample_token(logits, temperature, top_k, top_p)

                if next_token == self.tokenizer.eos_token_id:
                    break

                generated.append(next_token)
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[next_token]], device=self.device)], dim=1
                )

        result = self.decode(generated)

        self._stats["requests_processed"] += 1
        self._stats["tokens_generated"] += len(generated)
        self._stats["total_time"] += time.time() - start_time

        # Explicit KV cache cleanup + MPS memory release
        del input_ids, generated
        if self._is_mps:
            try:
                import gc
                gc.collect()
                torch.mps.empty_cache()
            except Exception:
                pass

        return result

    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Generate text with streaming (async)."""
        prompt = require_non_empty_prompt(prompt)

        # Check MPS memory before starting — auto-clear if near capacity
        if self._is_mps:
            try:
                from domains.infrastructure.mps_monitor import get_mps_monitor
                monitor = get_mps_monitor()
                usage = monitor.get_usage()
                if usage > 0.25:
                    logger.warning("MPS at %.0f%% — clearing cache before generation", usage * 100)
                    monitor._clear_mps_cache()
            except Exception:
                pass

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated = []
        past_key_values = None
        loop = asyncio.get_running_loop()

        with torch.no_grad():
            for step in range(max_new_tokens):
                # Check MPS mid-generation every 6 tokens — clear cache preventively
                if self._is_mps and step > 0 and step % 6 == 0:
                    try:
                        from domains.infrastructure.mps_monitor import get_mps_monitor
                        monitor = get_mps_monitor()
                        if not monitor.check_mid_generation():
                            logger.warning("MPS near capacity at token %d — yielding remaining tokens", step)
                            for t in generated:
                                yield self.decode([t])
                            return
                    except Exception:
                        pass

                model_inp = input_ids[:, -1:] if past_key_values is not None else input_ids

                def _forward(m_inp=model_inp, pkv=past_key_values):
                    try:
                        return self.model(m_inp, past_key_values=pkv, use_cache=self.use_cache)
                    except TypeError:
                        return self.model(m_inp)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            import gc
                            gc.collect()
                            if torch.backends.mps.is_available():
                                torch.mps.empty_cache()
                            from domains.infrastructure.mps_monitor import get_mps_monitor
                            get_mps_monitor().force_cpu()
                        raise

                outputs = await loop.run_in_executor(None, _forward)

                if hasattr(outputs, "logits"):
                    logits = outputs.logits[:, -1, :].squeeze(0)
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs[:, -1, :].squeeze(0)
                else:
                    logits = outputs[0][:, -1, :].squeeze(0)

                if self.use_cache and hasattr(outputs, "past_key_values"):
                    past_key_values = outputs.past_key_values

                logits = self._apply_lora_to_logits(logits)

                if repetition_penalty != 1.0:
                    logits = self._apply_repetition_penalty(logits, input_ids[0], repetition_penalty)

                next_token = await loop.run_in_executor(
                    None, lambda l=logits: self._sample_token(l, temperature, top_k, top_p)
                )

                if next_token == self.tokenizer.eos_token_id:
                    break

                generated.append(next_token)
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[next_token]], device=self.device)], dim=1
                )

                token_text = self.decode([next_token])
                yield token_text

        self._stats["requests_processed"] += 1
        self._stats["tokens_generated"] += len(generated)

        # Aggressive memory cleanup after every generation
        try:
            del input_ids, generated, past_key_values, outputs, logits, next_token
        except Exception:
            pass
        if self._is_mps:
            try:
                import gc
                gc.collect()
                torch.mps.empty_cache()
            except Exception:
                pass

    async def generate_batch(
        self,
        requests: List[GenerationRequest],
    ) -> Dict[str, str]:
        """Process multiple requests in parallel."""
        if len(requests) > self.max_batch_size:
            requests = requests[: self.max_batch_size]

        async def _run_one(request: GenerationRequest) -> tuple[str, str]:
            try:
                full_text = ""
                async for token in self.generate_stream(
                    request.prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                ):
                    full_text += token

                request.generated_text = full_text
                request.finished = True
                return request.id, full_text
            except Exception as e:
                request.error = str(e)
                request.finished = True
                return request.id, ""

        tasks = [_run_one(r) for r in requests]
        results_list = await asyncio.gather(*tasks)
        return dict(results_list)

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return {
            **self._stats,
            "avg_time_per_request": (
                self._stats["total_time"] / self._stats["requests_processed"]
                if self._stats["requests_processed"] > 0
                else 0
            ),
            "avg_tokens_per_request": (
                self._stats["tokens_generated"] / self._stats["requests_processed"]
                if self._stats["requests_processed"] > 0
                else 0
            ),
            "active_requests": len(self._active_requests),
            "pending_requests": len(self._pending_queue),
        }

    def reset_stats(self):
        """Reset statistics."""
        self._stats = {
            "requests_processed": 0,
            "tokens_generated": 0,
            "total_time": 0.0,
        }


def create_engine(
    model_name: str = "gpt2",
    device: str = "auto",
    max_batch_size: int = 32,
) -> InferenceEngine:
    """Create an inference engine with a local model."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    print(f"Loading {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure we have a distinct pad token. The default HF tokenizers set ``pad_token``
    # to ``eos_token`` which triggers a warning in the transformers library because
    # the model cannot infer an attention mask correctly. If ``pad_token`` is ``None``
    # we first fall back to ``eos_token`` (maintains compatibility with older code),
    # then we replace it with a unique token ``<|pad|>`` when it matches the EOS token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token == tokenizer.eos_token:
        # Add a dedicated pad token to silence the warning and give models a proper mask.
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        tokenizer.pad_token = "<|pad|>"


    model = AutoModelForCausalLM.from_pretrained(model_name)

    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_batch_size=max_batch_size,
    )

    print(f"Engine ready: {engine.get_stats()}")
    return engine


__all__ = [
    "InferenceEngine",
    "KVCache",
    "GenerationRequest",
    "BatchedRequest",
    "create_engine",
]
