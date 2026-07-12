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

    # Check for .slnc cache (2.2x faster load via mmap)
    slnc_path = safetensors_path.with_suffix(".slnc")
    if slnc_path.exists():
        return _load_from_slnc(slnc_path, dtype)

    logger.info("Loading %s from %s", model_id, safetensors_path.name)

    weights = {}
    with safe_open(str(safetensors_path), framework="numpy") as f:
        for key in f.keys():
            try:
                tensor = f.get_tensor(key)
                weights[key] = tensor.astype(dtype)
            except TypeError:
                # bfloat16 not supported by numpy — load as uint16 and convert
                tensor = f.get_slice(key)
                arr = tensor.get_all()
                if hasattr(arr, 'dtype') and arr.dtype.name == 'bfloat16':
                    # bfloat16 → float32: shift uint16 left by 16
                    raw = np.asarray(arr).view(np.uint16).astype(np.uint32) << 16
                    weights[key] = raw.view(np.float32)
                else:
                    weights[key] = np.asarray(arr).astype(dtype)

    logger.info("Loaded %d parameters from %s", len(weights), model_id)

    # Auto-convert to .slnc for faster future loads
    _try_convert_to_slnc(model_id, safetensors_path, weights)

    return weights


def _load_from_slnc(slnc_path: Path, dtype: np.dtype) -> Dict[str, np.ndarray]:
    """Load weights from .slnc memory-mapped format."""
    from domains.infrastructure.slnc.parser import SLNCParser

    logger.info("Loading from .slnc cache: %s (memory-mapped)", slnc_path.name)
    parser = SLNCParser(str(slnc_path))
    weights = parser.get_weights_dict()

    # Apply dtype conversion if needed
    result = {}
    for key, arr in weights.items():
        result[key] = arr.astype(dtype) if arr.dtype != dtype else arr

    logger.info("Loaded %d parameters from .slnc: %s", len(result), slnc_path.name)
    return result


def _try_convert_to_slnc(
    model_id: str,
    safetensors_path: Path,
    weights: Dict[str, np.ndarray],
) -> None:
    """Attempt to convert safetensors weights to .slnc format.

    This is a background optimization — failures are silently ignored.
    """
    try:
        from domains.infrastructure.slnc.compiler import SLNCCompiler

        slnc_path = safetensors_path.with_suffix(".slnc")
        config = load_model_config(model_id)

        logger.info("Auto-converting %s to .slnc format...", model_id)
        compiler = SLNCCompiler()
        compiler.compile_from_dict(config, weights, str(slnc_path))
        logger.info("SLNC conversion complete: %s (%.1f MB)",
                     slnc_path.name, slnc_path.stat().st_size / 1024 / 1024)
    except Exception as e:
        # Non-critical — just log and continue
        logger.debug("SLNC auto-conversion skipped for %s: %s", model_id, e)


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
