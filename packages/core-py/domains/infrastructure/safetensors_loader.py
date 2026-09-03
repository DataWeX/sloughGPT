"""Safetensors loader — thin wrapper that auto-converts to .slnc.

All model resolution utilities live in model_resolver.py.
All .slnc parsing lives in slnc/parser.py.
This module just orchestrates: resolve → auto-convert → load via SLNC.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Dict

import numpy as np

from domains.infrastructure.model_resolver import (
    get_model_dir,
    find_safetensors,
    load_model_config,
)

logger = logging.getLogger("slo.infrastructure.safetensors_loader")


def load_model_weights(
    model_id: str,
    device: str = "cpu",
    dtype: np.dtype = np.float32,
) -> Dict[str, np.ndarray]:
    """
    Load model weights via SLNC (mmap, zero-copy). Auto-converts from safetensors.

    Args:
        model_id: HuggingFace model ID (e.g. "gpt2", "Qwen/Qwen2.5-0.5B-Instruct")
        device: Ignored for numpy (always CPU)
        dtype: Target dtype for weights

    Returns:
        Dict mapping parameter names to numpy arrays

    Raises:
        FileNotFoundError: If model not found in cache
    """
    model_dir = get_model_dir(model_id)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model {model_id} not found in cache: {model_dir}")

    safetensors_path = find_safetensors(model_dir)
    if safetensors_path is None:
        raise FileNotFoundError(f"No model weights found for {model_id}")

    slnc_path = safetensors_path.with_suffix(".slnc")
    if not slnc_path.exists():
        _auto_convert(safetensors_path, slnc_path, model_id)

    return _load_from_slnc(slnc_path, dtype)


def _auto_convert(st_path: Path, slnc_path: Path, model_id: str) -> None:
    """Convert safetensors to .slnc on first load."""
    from domains.infrastructure.slnc.compiler import SLNCCompiler
    logger.info("Converting %s → .slnc", st_path.name, extra={"tag": "INFRA"})
    SLNCCompiler().compile(model_id, str(slnc_path))
    logger.info("Converted to .slnc: %s (%.1f MB)", slnc_path.name,
                slnc_path.stat().st_size / 1e6, extra={"tag": "INFRA"})


def _load_from_slnc(slnc_path: Path, dtype: np.dtype) -> Dict[str, np.ndarray]:
    """Load weights from .slnc memory-mapped format."""
    from domains.infrastructure.slnc.parser import SLNCParser

    logger.info("Loading from .slnc: %s", slnc_path.name, extra={"tag": "INFRA"})
    parser = SLNCParser(str(slnc_path))
    weights = parser.get_weights_dict_parallel()

    result = {}
    for key, arr in weights.items():
        result[key] = arr.astype(dtype) if arr.dtype != dtype else arr

    logger.info("Loaded %d parameters from .slnc: %s", len(result), slnc_path.name, extra={"tag": "INFRA"})
    return result
