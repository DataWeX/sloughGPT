"""
Safe HuggingFace model loader — handles platform-specific quirks.

Apple Silicon MPS does not support BFloat16. This module detects the
platform and forces float32 when on MPS, then provides a uniform
load_model() that returns (model, tokenizer).

After loading, the model is verified for integrity:
  - No NaN / Inf parameter values (partial download detection)
  - Forward pass smoke test with dummy input → non-NaN logits

Raises RuntimeError if any check fails.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def _mps_available() -> bool:
    try:
        return torch.backends.mps.is_available()
    except Exception:
        return False


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
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
    zero_params = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
        if torch.isinf(param).any():
            inf_params.append(name)
        if (param == 0).all():
            zero_params.append(name)

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
        device = next(model.parameters()).device
        # Use pad_token_id if available, otherwise fallback to 0
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        dummy_input = torch.tensor([[pad_id]], device=device)
        with torch.no_grad():
            output = model(dummy_input)
        logits = output.logits if hasattr(output, "logits") else output[0]

        if torch.isnan(logits).any():
            raise RuntimeError(
                f"Model {model_id} produced NaN logits on a forward pass. "
                f"The download may be incomplete."
            )
        if torch.isinf(logits).any():
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

    On MPS, always forces torch.float32 (BFloat16 not supported).

    Raises RuntimeError if the model fails integrity checks (partial download).
    """
    import os
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None or device == "auto":
        if _mps_available():
            resolved = "mps"
        elif _cuda_available():
            resolved = "cuda"
        else:
            resolved = "cpu"
    else:
        resolved = device

    # ── check HF cache before downloading ──────────────────────────────
    cache_id = model_id.replace("/", "--")
    # Check project-local cache first, then global cache
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    cache_dir = Path(hf_home) / "hub" / f"models--{cache_id}"
    if cache_dir.exists():
        logger.info("%s found in local cache (%s)", model_id, cache_dir)
    else:
        logger.info("%s not cached — downloading from HuggingFace", model_id)

    logger.info("Loading %s → %s (float32)", model_id, resolved)

    # Monkey-patch Mistral regex check that tries remote even in offline mode
    try:
        import transformers.tokenization_utils_base as _tub
        _tub._patch_mistral_regex = lambda cls, name: cls
    except Exception:
        pass

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        logger.info("%s loaded from local cache", model_id)
    except OSError:
        logger.info("%s not in cache — downloading from HuggingFace", model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            local_files_only=False,
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

    logger.info("Model %s → %s (device=%s)", model_id, resolved, next(model.parameters()).device)

    return model, tokenizer, resolved
