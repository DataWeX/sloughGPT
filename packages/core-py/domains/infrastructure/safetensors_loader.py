"""
Torch-free model loading using safetensors + numpy.

Loads HuggingFace model weights directly from .safetensors files
without requiring PyTorch. Weights are loaded as numpy arrays.

Usage:
    from domains.infrastructure.safetensors_loader import load_model_weights
    weights = load_model_weights("gpt2")
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("man.infrastructure.safetensors_loader")


def _get_model_dir(model_id: str) -> Path:
    """Resolve HuggingFace cache directory for a model."""
    import os
    cache_id = model_id.replace("/", "--")
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    return Path(hf_home) / "hub" / f"models--{cache_id}"


def _find_safetensors(model_dir: Path) -> Optional[Path]:
    """Find the safetensors file in a model directory."""
    # Check snapshots directory
    snapshots = model_dir / "snapshots"
    if snapshots.exists():
        for snapshot in snapshots.iterdir():
            st = snapshot / "model.safetensors"
            if st.exists():
                return st
    # Check model dir directly
    st = model_dir / "model.safetensors"
    if st.exists():
        return st
    return None


def load_model_weights(
    model_id: str,
    device: str = "cpu",
    dtype: np.dtype = np.float32,
) -> Dict[str, np.ndarray]:
    """
    Load model weights from safetensors file as numpy arrays.

    Args:
        model_id: HuggingFace model ID (e.g. "gpt2", "Qwen/Qwen2.5-0.5B-Instruct")
        device: Ignored for numpy (always CPU)
        dtype: Target dtype for weights

    Returns:
        Dict mapping parameter names to numpy arrays

    Raises:
        FileNotFoundError: If model not found in cache
        ValueError: If no safetensors file found
    """
    from safetensors import safe_open

    model_dir = _get_model_dir(model_id)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model {model_id} not found in cache: {model_dir}")

    safetensors_path = _find_safetensors(model_dir)
    if safetensors_path is None:
        raise ValueError(f"No .safetensors file found for {model_id}")

    logger.info("Loading %s from %s", model_id, safetensors_path.name)

    weights = {}
    with safe_open(str(safetensors_path), framework="numpy") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key).astype(dtype)

    logger.info("Loaded %d parameters from %s", len(weights), model_id)
    return weights


def load_model_config(model_id: str) -> Dict[str, Any]:
    """
    Load model config.json from HuggingFace cache.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Dict with model configuration (vocab_size, n_layer, n_head, etc.)
    """
    model_dir = _get_model_dir(model_id)
    snapshots = model_dir / "snapshots"

    config_path = None
    if snapshots.exists():
        for snapshot in snapshots.iterdir():
            candidate = snapshot / "config.json"
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None:
        config_path = model_dir / "config.json"

    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"No config.json found for {model_id}")

    with open(config_path) as f:
        return json.load(f)


def list_cached_models() -> list:
    """List all models cached locally with safetensors files."""
    import os
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    hub = Path(hf_home) / "hub"

    models = []
    for model_dir in hub.glob("models--*"):
        st = _find_safetensors(model_dir)
        if st is not None:
            model_name = model_dir.name.replace("models--", "").replace("--", "/")
            size_mb = st.stat().st_size / (1024 * 1024)
            models.append({
                "id": model_name,
                "path": str(st),
                "size_mb": round(size_mb, 1),
            })

    return sorted(models, key=lambda x: x["id"])


__all__ = [
    "load_model_weights",
    "load_model_config",
    "list_cached_models",
]
