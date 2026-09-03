"""
Model loading using safetensors + numpy.

Loads HuggingFace model weights directly from .safetensors files.
Weights are loaded as numpy arrays.

Usage:
    from domains.infrastructure.safetensors_loader import load_model_weights
    weights = load_model_weights("gpt2")
"""

import json
import logging
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from domains.shared import find_repo_root

logger = logging.getLogger("slo.infrastructure.safetensors_loader")


def _get_model_dir(model_id: str) -> Path:
    """Resolve HuggingFace cache directory for a model.

    Searches the standard HF cache first, then the project-local
    cache (models/hf-cache/hub/) mirroring MorphTokenizer.from_pretrained.
    """
    import os
    cache_id = model_id.replace("/", "--")
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    candidates = [
        Path(hf_home) / "hub" / f"models--{cache_id}",
        find_repo_root(Path(__file__).resolve()) / "models" / "hf-cache" / "hub" / f"models--{cache_id}",
        Path("models/hf-cache/hub") / f"models--{cache_id}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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

    logger.info("Loading %s from %s", model_id, safetensors_path.name, extra={"tag": "INFRA"})

    weights: Dict[str, np.ndarray] = {}
    try:
        from safetensors import safe_open
    except ImportError:
        logger.info("safetensors package not installed — using built-in raw parser",
                    extra={"tag": "INFRA"})
        try:
            return _load_weights_raw(safetensors_path, dtype)
        except (ValueError, json.JSONDecodeError, struct.error) as e:
            logger.error("Failed to load corrupted safetensors file %s: %s",
                         safetensors_path.name, e, extra={"tag": "INFRA"})
            raise ValueError(f"Corrupted safetensors file {safetensors_path.name}: {e}") from e

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

    logger.info("Loaded %d parameters from %s", len(weights), model_id, extra={"tag": "INFRA"})

    # Auto-convert to .slnc for faster future loads
    _try_convert_to_slnc(model_id, safetensors_path, weights)

    return weights


_MAX_HEADER_LEN = 100 * 1024 * 1024  # 100 MB sanity limit for header length


def _load_weights_raw(path: Path, dtype: np.dtype) -> Dict[str, np.ndarray]:
    """Read a .safetensors file with the built-in raw parser.

    Walks the binary format directly (8-byte header length + JSON header +
    per-tensor data offsets) so no ``safetensors`` package is required.
    Handles F32/F16/BF16; any other dtype is read as float32.

    Args:
        path: Path to the .safetensors file.
        dtype: Target dtype for the returned arrays.

    Returns:
        Dict mapping parameter names to numpy arrays.

    Raises:
        ValueError: If the file is corrupted or contains out-of-bounds offsets.
    """
    import struct

    weights: Dict[str, np.ndarray] = {}
    file_size = path.stat().st_size
    with open(path, "rb") as f:
        raw_header_len = f.read(8)
        if len(raw_header_len) < 8:
            raise ValueError(f"Truncated safetensors header in {path.name}")
        header_len = struct.unpack("<Q", raw_header_len)[0]

        if header_len > _MAX_HEADER_LEN:
            raise ValueError(
                f"Header length {header_len} exceeds sanity limit "
                f"({_MAX_HEADER_LEN}) in {path.name}"
            )
        if 8 + header_len > file_size:
            raise ValueError(
                f"Header extends past end of file ({8 + header_len} > {file_size}) "
                f"in {path.name}"
            )

        header_bytes = f.read(header_len)
        if len(header_bytes) < header_len:
            raise ValueError(f"Truncated header JSON in {path.name}")
        header = json.loads(header_bytes)

        for key, info in header.items():
            if key.startswith("__"):
                continue
            start, end = info["data_offsets"]
            if start < 0 or end < start:
                raise ValueError(
                    f"Invalid offsets [{start}, {end}] for tensor '{key}' "
                    f"in {path.name}"
                )
            if 8 + header_len + end > file_size:
                raise ValueError(
                    f"Tensor '{key}' offsets [{start}, {end}] exceed file size "
                    f"({file_size}) in {path.name}"
                )
            f.seek(8 + header_len + start)
            raw = f.read(end - start)
            shape = info["shape"]
            dtype_str = info.get("dtype", "F32")
            if dtype_str == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16)
                f32 = np.zeros(len(u16), dtype=np.float32)
                f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
                arr = f32.reshape(shape)
            elif dtype_str == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
            elif dtype_str == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            else:
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            weights[key] = arr.astype(dtype)
    return weights


def _load_from_slnc(slnc_path: Path, dtype: np.dtype) -> Dict[str, np.ndarray]:
    """Load weights from .slnc memory-mapped format."""
    from domains.infrastructure.slnc.parser import SLNCParser

    logger.info("Loading from .slnc cache: %s (memory-mapped)", slnc_path.name, extra={"tag": "INFRA"})
    parser = SLNCParser(str(slnc_path))
    weights = parser.get_weights_dict()

    # Apply dtype conversion if needed
    result = {}
    for key, arr in weights.items():
        result[key] = arr.astype(dtype) if arr.dtype != dtype else arr

    logger.info("Loaded %d parameters from .slnc: %s", len(result), slnc_path.name, extra={"tag": "INFRA"})
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

        logger.info("Auto-converting %s to .slnc format...", model_id, extra={"tag": "INFRA"})
        compiler = SLNCCompiler()
        compiler.compile_from_dict(config, weights, str(slnc_path))
        logger.info("SLNC conversion complete: %s (%.1f MB)",
                     slnc_path.name, slnc_path.stat().st_size / 1024 / 1024, extra={"tag": "INFRA"})
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
    """List all models cached locally with safetensors files.

    Scans both the standard HF cache (HF_HOME/hub) and the project-local
    cache (models/hf-cache/hub/), deduplicating by model id.
    """
    import os
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    hub_dirs = [
        Path(hf_home) / "hub",
        find_repo_root(Path(__file__).resolve()) / "models" / "hf-cache" / "hub",
    ]

    models = {}
    for hub in hub_dirs:
        if not hub.exists():
            continue
        for model_dir in hub.glob("models--*"):
            st = _find_safetensors(model_dir)
            if st is not None:
                model_name = model_dir.name.replace("models--", "").replace("--", "/")
                if model_name in models:
                    continue
                size_mb = st.stat().st_size / (1024 * 1024)
                models[model_name] = {
                    "id": model_name,
                    "path": str(st),
                    "size_mb": round(size_mb, 1),
                }

    return sorted(models.values(), key=lambda x: x["id"])


__all__ = [
    "load_model_weights",
    "load_model_config",
    "list_cached_models",
]
