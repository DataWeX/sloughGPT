"""
HF model loader for subprocess inference workers.

Provides ``hf_model_loader`` — a module-level function that loads a HuggingFace
model and tokenizer in the worker subprocess, returning ``(model, tokenizer)``
as required by the ``ModelWorkerProcess`` protocol.

Usage::

    worker = ModelWorkerProcess(
        model_cls_path="domains.infrastructure.hf_model_worker.hf_model_loader",
        model_kwargs={"model_id": "gpt2"},
    )
"""

import logging
from typing import Any

logger = logging.getLogger("man.infrastructure.hf_model_worker")


def hf_model_loader(
    model_id: str,
    device: str = "cpu",
) -> tuple[Any, Any]:
    """Load an HF model + tokenizer in the worker subprocess.

    Uses the safe model loader first. If it returns safetensors weights
    (dict), falls back to raw ``transformers`` which returns a live model
    object suitable for ``.generate()`` and a callable tokenizer.

    Returns:
        ``(model, tokenizer)`` tuple as expected by ``ModelWorkerProcess``.
    """
    resolved_device = _resolve_device(device)

    from domains.infrastructure.model_loader import load_hf_model as safe_load
    model, tokenizer, actual_device = safe_load(model_id, resolved_device)

    if isinstance(model, dict):
        # Safe loader returned safetensors weights dict — load real model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map=device,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("hf_model_loader[%s]: loaded (device=%s)", model_id, resolved_device)
    return model, tokenizer


def _resolve_device(device: str) -> str:
    """Resolve device string, defaulting to CPU for stability."""
    if device == "auto":
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    return device
