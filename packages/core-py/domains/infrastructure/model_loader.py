"""
Safe HuggingFace model loader — handles platform-specific quirks.

Apple Silicon MPS does not support BFloat16. This module detects the
platform and forces float32 when on MPS, then provides a uniform
load_model() that returns (model, tokenizer).

After loading, the model is verified for integrity:
  - No NaN / Inf parameter values (partial download detection)
  - Forward pass smoke test with dummy input → non-NaN logits

Raises RuntimeError if any check fails.

Design: uses ml_types for dtypes and device detection. Torch is only
imported when actually loading HF models (torch.nn.Module instances).
SloNet models bypass torch entirely.
"""

import logging
from typing import Optional

from domains.infrastructure.ml_types import (
    float32 as ml_float32,
    _mps_available as mps_available,
    _cuda_available as cuda_available,
    isnan as ml_isnan,
    isinf as ml_isinf,
    auto_device,
)

logger = logging.getLogger("man.infrastructure.model_loader")


def _torch_available() -> bool:
    """Check if PyTorch is installed."""
    try:
        from domains.training.slonet_compat import torch
        return True
    except ImportError:
        return False


def verify_model_integrity(model, model_id: str, tokenizer) -> None:
    """
    Run integrity checks on a loaded model to catch partial/corrupt downloads.

    Checks:
      1. No parameter is NaN or Inf (corrupt weights).
      2. Forward pass with a dummy input produces finite logits.

    Raises RuntimeError with a descriptive message on failure.
    """
    logger.info("Verifying integrity of %s ...", model_id)

    # ── check 1: no NaN / Inf weights ──────────────────────────────────
    nan_params = []
    inf_params = []
    for name, param in model.named_parameters():
        # Works for both torch tensors and numpy arrays
        param_np = param.detach().cpu().numpy() if hasattr(param, 'detach') else param
        if ml_isnan(param_np).any():
            nan_params.append(name)
        if ml_isinf(param_np).any():
            inf_params.append(name)

    if nan_params:
        raise RuntimeError(
            f"Model {model_id} has NaN weights in {len(nan_params)} parameters "
            f"(e.g. {nan_params[0]}). The download may be corrupt."
        )
    if inf_params:
        raise RuntimeError(
            f"Model {model_id} has Inf weights in {len(inf_params)} parameters "
            f"(e.g. {inf_params[0]}). The download may be corrupt."
        )

    # ── check 2: forward-pass smoke test ───────────────────────────────
    try:
        import torch

        device = next(model.parameters()).device
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        dummy_input = torch.tensor([[pad_id]], device=device)
        with torch.no_grad():
            output = model(dummy_input)
        logits = output.logits if hasattr(output, "logits") else output[0]

        logits_np = logits.detach().cpu().numpy()
        if ml_isnan(logits_np).any():
            raise RuntimeError(
                f"Model {model_id} produced NaN logits on a forward pass. "
                f"The download may be incomplete."
            )
        if ml_isinf(logits_np).any():
            raise RuntimeError(
                f"Model {model_id} produced Inf logits on a forward pass. "
                f"The download may be incomplete."
            )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Forward-pass smoke test for {model_id} failed: {e}"
        ) from e

    logger.info("Integrity check passed for %s", model_id)


def load_hf_model(model_id: str, device: Optional[str] = None):
    """
    Load a HuggingFace AutoModelForCausalLM with its tokenizer.

    Device resolution:
      - "auto" → mps > cuda > cpu
      - "mps" / "cuda" / "cpu" → explicit
      - None → same as "auto"

    Returns (model, tokenizer, resolved_device).

    Uses float32 on CPU, float16 on MPS/CUDA (BFloat16 not supported on MPS).

    Raises RuntimeError if the model fails integrity checks (partial download).
    """
    import os
    from pathlib import Path

    if device is None or device == "auto":
        resolved = auto_device()
    else:
        resolved = device

    # ── check HF cache before downloading ──────────────────────────────
    cache_id = model_id.replace("/", "--")
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    cache_dir = Path(hf_home) / "hub" / f"models--{cache_id}"
    if cache_dir.exists():
        logger.info("%s found in local cache (%s)", model_id, cache_dir)
    else:
        logger.info("%s not cached — downloading from HuggingFace", model_id)

    use_fp16 = resolved in ("mps", "cuda")
    dtype_str = "float16" if use_fp16 else "float32"
    logger.info("Loading %s → %s (%s)", model_id, resolved, dtype_str)

    # Try torch first (for full HF model support)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Monkey-patch Mistral regex check that tries remote even in offline mode
        try:
            import transformers.tokenization_utils_base as _tub
            _tub._patch_mistral_regex = lambda cls, name: cls
        except Exception:
            pass

        dtype = torch.float16 if use_fp16 else torch.float32

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                local_files_only=True,
                device_map="cpu" if resolved == "cpu" else None,
            )
            logger.info("%s loaded from local cache", model_id)
        except OSError:
            logger.info("%s not in cache — downloading from HuggingFace", model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                local_files_only=False,
                device_map="cpu" if resolved == "cpu" else None,
            )
            logger.info("%s downloaded successfully", model_id)

        if resolved == "mps":
            model = model.to("mps")
        elif resolved == "cuda":
            model = model.to("cuda")
        else:
            model = model.cpu()

        model.eval()
        verify_model_integrity(model, model_id, tokenizer)
        return model, tokenizer, resolved

    except ImportError:
        # No torch — use safetensors for weight loading
        logger.info("torch not available — using safetensors loader")
        from domains.infrastructure.safetensors_loader import load_model_weights, load_model_config

        weights = load_model_weights(model_id)
        config = load_model_config(model_id)

        # Return numpy weights dict and config (no model object)
        return weights, config, resolved
