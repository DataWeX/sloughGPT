#!/usr/bin/env python3
"""
SloughGPT Inference Engine (DEPRECATED)

.. deprecated::
    This module is deprecated. Use ``domains.inference.engine.InferenceEngine`` instead.

Supports streaming, batch processing, and advanced sampling.

Optimizations:
- Vectorized repetition penalty (O(1) instead of O(n))
- Efficient top-k/top-p filtering
- KV cache support
- torch.compile support
- Continuous batching

Canonical location: ``domains.inference.engine.InferenceEngine``
"""

import warnings
warnings.warn(
    "domains.training.inference_engine is deprecated. "
    "Use domains.inference.engine.InferenceEngine instead.",
    DeprecationWarning,
    stacklevel=2
)

from domains.training.slonet_compat import torch
F = torch.F
from typing import List, Optional, Iterator, Dict, Tuple
from dataclasses import dataclass, field

from domains.errors import require_non_empty_prompt
import warnings


warnings.warn(
    "domains.training.inference_engine is deprecated. Use domains.inference.engine instead.",
    DeprecationWarning,
    stacklevel=2,
)


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


def _apply_repetition_penalty_fast(
    logits: torch.Tensor,
    prev_tokens: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Fast vectorized repetition penalty (O(1) per batch instead of O(n))."""
    if penalty == 1.0 or prev_tokens.numel() == 0:
        return logits

    unique_tokens, inverse = torch.unique(prev_tokens, return_inverse=True)
    token_counts = torch.bincount(inverse, minlength=len(unique_tokens))

    mask = token_counts > 0
    penalized = unique_tokens[mask]

    if penalized.numel() == 0:
        return logits

    penalties = torch.ones(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    penalties[penalized] = penalty

    pos_mask = logits > 0
    logits = torch.where(pos_mask, logits * penalties, logits / penalties)
    return logits


def _sample_token_fast(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Fast token sampling with fused operations."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        v, _ = torch.topk(logits, top_k, dim=-1)
        threshold = v[:, [-1]]
        logits = torch.where(logits < threshold, torch.tensor(float("-inf"), device=logits.device), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumsum > top_p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False
        indices_to_remove = mask.scatter(1, sorted_indices, mask)
        logits = torch.where(indices_to_remove, torch.tensor(float("-inf"), device=logits.device), logits)

    probs = F.softmax(logits, dim=-1)
    probs = torch.clamp(probs, min=1e-10)
    return torch.multinomial(probs, num_samples=1)


class InferenceEngine:
    """
    High-performance inference engine for SloughGPT.

    Features:
    - Streaming generation
    - Batch processing
    - Advanced sampling (temperature, top-k, top-p)
    - Repetition penalty (vectorized)
    - torch.compile support
    - KV cache support
    """

    def __init__(
        self,
        model,
        stoi: Dict[str, int],
        itos: Dict[int, str],
        device: str = "cpu",
        use_compile: bool = False,
    ):
        self.model = model
        self.stoi = stoi
        self.itos = itos
        self.device = device
        self.vocab_size = len(stoi)
        self.model.eval()

        self._compiled_model = None
        if use_compile and hasattr(torch, "compile") and device == "cuda":
            try:
                self._compiled_model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                pass

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to tensor."""
        ids = [self.stoi.get(c, 0) for c in text]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def decode(self, ids: torch.Tensor) -> str:
        """Decode tensor to text."""
        return "".join([self.itos.get(i, "?") for i in ids[0].tolist()])

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate text from prompt (non-streaming)."""
        if config is None:
            config = GenerationConfig()

        prompt = require_non_empty_prompt(prompt)

        if config.seed is not None:
            torch.manual_seed(config.seed)

        self.model.eval()
        idx = self.encode(prompt)
        prev_tokens = idx.clone()

        model = self._compiled_model if self._compiled_model else self.model

        for _ in range(config.max_new_tokens):
            logits, _ = model(idx)
            logits = logits[:, -1, :]

            if config.repetition_penalty != 1.0:
                logits = _apply_repetition_penalty_fast(logits, prev_tokens, config.repetition_penalty)

            next_token = _sample_token_fast(logits, config.temperature, config.top_k, config.top_p)
            idx = torch.cat([idx, next_token], dim=1)
            prev_tokens = torch.cat([prev_tokens, next_token], dim=1)

            if next_token.item() == self.stoi.get("\n", -1):
                break

        return self.decode(idx)

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Generate text with streaming output."""
        if config is None:
            config = GenerationConfig()

        prompt = require_non_empty_prompt(prompt)

        if config.seed is not None:
            torch.manual_seed(config.seed)

        self.model.eval()
        idx = self.encode(prompt)
        prev_tokens = idx.clone()

        model = self._compiled_model if self._compiled_model else self.model

        for _ in range(config.max_new_tokens):
            logits, _ = model(idx)
            logits = logits[:, -1, :]

            if config.repetition_penalty != 1.0:
                logits = _apply_repetition_penalty_fast(logits, prev_tokens, config.repetition_penalty)

            next_token = _sample_token_fast(logits, config.temperature, config.top_k, config.top_p)
            idx = torch.cat([idx, next_token], dim=1)
            prev_tokens = torch.cat([prev_tokens, next_token], dim=1)

            new_char = self.itos.get(next_token.item(), "?")
            yield new_char

            if next_token.item() == self.stoi.get("\n", -1):
                break

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        """Generate text for multiple prompts at once (batched)."""
        if config is None:
            config = GenerationConfig()

        if config.seed is not None:
            torch.manual_seed(config.seed)

        self.model.eval()

        max_len = max(len(p) for p in prompts)
        input_ids = []
        for p in prompts:
            ids = [self.stoi.get(c, 0) for c in p]
            ids.extend([0] * (max_len - len(ids)))
            input_ids.append(ids)

        idx = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        batch_size = idx.shape[0]

        prev_tokens = idx.clone()
        all_prev = [idx[i] for i in range(batch_size)]

        model = self._compiled_model if self._compiled_model else self.model

        for _ in range(config.max_new_tokens):
            logits, _ = model(idx)
            logits = logits[:, -1, :]

            if config.temperature > 0:
                logits = logits / config.temperature

            if config.top_k > 0:
                top_k = min(config.top_k, logits.shape[-1])
                for i in range(batch_size):
                    v, idx_topk = torch.topk(logits[i], top_k)
                    logits[i] = torch.full_like(logits[i], float("-inf"))
                    logits[i, idx_topk] = v

            if config.do_sample:
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = logits.argmax(dim=-1)

            idx = torch.cat([idx, next_tokens.unsqueeze(1)], dim=1)

            for i in range(batch_size):
                all_prev[i] = torch.cat([all_prev[i], next_tokens[i].unsqueeze(0)])

        results = []
        for i in range(len(prompts)):
            text = "".join([self.itos.get(j, "?") for j in idx[i].tolist()])
            results.append(text)

        return results


def create_quantized_engine(
    checkpoint_path: str,
    device: str = "cpu",
    quantize: str = "int8",
) -> InferenceEngine:
    """Create a quantized inference engine for faster inference.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to run on
        quantize: Quantization type ("int8", "int4", "fp16", "bf16")

    Returns:
        Quantized InferenceEngine
    """
    from domains.models import SloughGPTModel

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    vocab_size = checkpoint.get("vocab_size", 65)
    n_embed = checkpoint.get("n_embed", 128)
    n_layer = checkpoint.get("n_layer", 4)
    n_head = checkpoint.get("n_head", 4)
    block_size = checkpoint.get("block_size", 64)

    model = SloughGPTModel(
        vocab_size=vocab_size,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
    )

    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        model.load_state_dict(checkpoint["model"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    stoi = checkpoint.get("stoi", {})
    itos = checkpoint.get("itos", {})

    if itos and isinstance(next(iter(itos.keys())), str):
        itos = {int(k): v for k, v in itos.items()}

    if quantize in ("int8", "int4"):
        import torch.quantization as tq

        if quantize == "int4":
            dtype = torch.quint4x2
        else:
            dtype = torch.quint8

        model = tq.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=dtype,
        )

    model = model.to(device)

    engine = InferenceEngine(model, stoi, itos, device, use_compile=(device == "cuda"))
    return engine


def load_model_for_inference(checkpoint_path: str, device: str = "cpu") -> InferenceEngine:
    """Load a trained model for inference."""
    from domains.models import SloughGPTModel

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    if "training_info" in checkpoint:
        info = checkpoint["training_info"]
        vocab_size = info.get("vocab_size", 65)
        n_embed = info.get("n_embed", 128)
        n_layer = info.get("n_layer", 4)
        n_head = info.get("n_head", 4)
        block_size = info.get("block_size", 64)
    else:
        vocab_size = checkpoint.get("vocab_size", 65)
        n_embed = checkpoint.get("n_embed", 128)
        n_layer = checkpoint.get("n_layer", 4)
        n_head = checkpoint.get("n_head", 4)
        block_size = checkpoint.get("block_size", 64)

    model = SloughGPTModel(
        vocab_size=vocab_size,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
    )

    # Load state dict
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        model.load_state_dict(checkpoint["model"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)

    stoi = checkpoint.get("stoi", {})
    itos = checkpoint.get("itos", {})

    # Convert itos keys to int if needed
    if itos and isinstance(next(iter(itos.keys())), str):
        itos = {int(k): v for k, v in itos.items()}

    return InferenceEngine(model, stoi, itos, device)


if __name__ == "__main__":
    # Example usage
    print("Loading model...")
    engine = load_model_for_inference("models/sloughgpt.pt")

    # Single generation
    print("\n=== Single Generation ===")
    config = GenerationConfig(
        max_new_tokens=100,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )
    result = engine.generate("First", config)
    print(result)

    # Streaming generation
    print("\n=== Streaming Generation ===")
    for char in engine.generate_stream("First", config):
        print(char, end="", flush=True)
    print()

    # Batch generation
    print("\n=== Batch Generation ===")
    prompts = ["First", "Second", "Hello"]
    results = engine.generate_batch(prompts, config)
    for i, r in enumerate(results):
        print(f"{prompts[i]}: {r[:50]}...")
