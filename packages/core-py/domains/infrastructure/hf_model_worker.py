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

logger = logging.getLogger("slo.infrastructure.hf_model_worker")


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

    from domains.infrastructure.model_loader import get_model_loader
    result = get_model_loader().load(model_id, device=resolved_device, verify=False)
    model = result.model
    tokenizer = result.tokenizer

    if model is None:
        # Safe loader returned safetensors weights dict — load real model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="auto",
            device_map=device,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    logger.info("hf_model_loader[%s]: loaded (device=%s)", model_id, resolved_device,
        extra={"tag": "INFRA"})
    return model, tokenizer


def _resolve_device(device: str) -> str:
    """Resolve device string, defaulting to CPU for stability."""
    if device == "auto":
        try:
            from domains.infrastructure.ml_types import auto_device
            return auto_device()
        except Exception:
            pass
        return "cpu"
    return device
