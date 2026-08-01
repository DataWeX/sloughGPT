"""
Shared checkpoint helpers for SloughGPT training and inference.

Used by ``train_sloughgpt.py``, ``SloughGPTTrainer`` (inference + ``train(resume=True)``),
and CLI load paths so hyperparameters and ``state_dict`` extraction stay consistent.

Char-LM bundles may include ``stoi`` / ``itos`` / ``chars`` (see
``tokenizer_maps_from_bundle``). For semantics and eval caveats, see
``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from domains.training.slonet import load_checkpoint_npz as _load_npz
from domains.training.slonet import save_checkpoint_npz as _save_npz

try:
    from domains.models import SloughGPTModel
except (ImportError, ModuleNotFoundError):
    SloughGPTModel = None  # type: ignore[assignment,misc]

KEY_MODEL_STATE = "model_state_dict"
KEY_MODEL_LEGACY = "model"
KEY_TRAINING_INFO = "training_info"

# Re-export standalone NPZ helpers
save_checkpoint_npz = _save_npz
load_checkpoint_npz = _load_npz


def torch_load_checkpoint(
    path: str,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load a ``.pt`` checkpoint as a dict with numpy arrays (torch-free)."""
    from domains.infrastructure.pt_loader import load_pt_file
    raw = load_pt_file(path, map_location=map_location)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected checkpoint at {path!r} to load to dict, got {type(raw).__name__}")
    return _to_numpy_dict(raw)


def normalize_raw_checkpoint(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrap a flat parameter ``state_dict`` in a standard bundle, or pass through
    bundled checkpoints (``model_state_dict``, ``model``, ``training_info``, …).
    """
    if KEY_MODEL_STATE in raw or KEY_MODEL_LEGACY in raw:
        return raw
    if any(
        isinstance(k, str) and (".weight" in k or ".bias" in k or "tok_emb" in k)
        for k in raw.keys()
    ):
        return {KEY_MODEL_STATE: raw, KEY_TRAINING_INFO: {}}
    return raw


def extract_state_dict(bundle: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Return the state dict as numpy arrays from a normalized bundle."""
    if KEY_MODEL_LEGACY in bundle and isinstance(bundle[KEY_MODEL_LEGACY], dict):
        state = bundle[KEY_MODEL_LEGACY]
    elif KEY_MODEL_STATE in bundle:
        state = bundle[KEY_MODEL_STATE]
    else:
        state = bundle
    if not isinstance(state, dict):
        raise ValueError("Model state must be a dict")
    return _to_numpy_dict(state)


def _to_numpy_dict(d: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Recursively convert torch tensors in a dict to numpy arrays."""
    result = {}
    for k, v in d.items():
        if hasattr(v, 'cpu'):
            arr = v.cpu().numpy()
            result[k] = arr.astype(arr.dtype)
        elif isinstance(v, np.ndarray):
            result[k] = v.copy() if v.dtype != np.float32 else v
        elif isinstance(v, dict):
            result[k] = _to_numpy_dict(v)
        else:
            result[k] = v
    return result


def resolve_sloughgpt_hyperparams(
    bundle: Dict[str, Any],
    *,
    fallback_vocab_size: int,
    fallback_n_embed: int,
    fallback_n_layer: int,
    fallback_n_head: int,
    fallback_block_size: int,
    fallback_dropout: float = 0.1,
) -> Dict[str, Any]:
    """Merge ``training_info`` / top-level keys with fallbacks for ``SloughGPTModel`` kwargs."""
    info: Dict[str, Any] = bundle.get(KEY_TRAINING_INFO) or {}
    if isinstance(bundle.get("config"), dict):
        cfg = bundle["config"]
        info = {**cfg, **info}

    if "chars" in bundle:
        vocab_size = len(bundle["chars"])
    else:
        vocab_size = int(
            info.get("vocab_size", bundle.get("vocab_size", fallback_vocab_size))
        )
    if vocab_size <= 0:
        vocab_size = int(fallback_vocab_size)

    return {
        "vocab_size": vocab_size,
        "n_embed": int(info.get("n_embed", bundle.get("n_embed", fallback_n_embed))),
        "n_layer": int(info.get("n_layer", bundle.get("n_layer", fallback_n_layer))),
        "n_head": int(info.get("n_head", bundle.get("n_head", fallback_n_head))),
        "block_size": int(
            info.get("block_size", bundle.get("block_size", fallback_block_size))
        ),
        "dropout": float(info.get("dropout", bundle.get("dropout", fallback_dropout))),
    }


def load_sloughgpt_from_checkpoint(
    bundle: Dict[str, Any],
    *,
    device: str = "cpu",
    strict: bool = True,
    fallback_vocab_size: int = 256,
    fallback_n_embed: int = 256,
    fallback_n_layer: int = 6,
    fallback_n_head: int = 8,
    fallback_block_size: int = 128,
    fallback_dropout: float = 0.1,
) -> Tuple[SloughGPTModel, Dict[str, Any]]:
    """
    Build ``SloughGPTModel``, load weights, move to ``device``.

    Returns:
        ``(model, hyperparams_dict)`` — hyperparams are the resolved constructor kwargs.

    Raises:
        RuntimeError: If ``SloughGPTModel`` cannot be imported.
    """
    if SloughGPTModel is None:
        raise RuntimeError(
            "SloughGPTModel is not available — "
            "ensure domains.models is importable."
        )
    bundle = normalize_raw_checkpoint(bundle)
    hp = resolve_sloughgpt_hyperparams(
        bundle,
        fallback_vocab_size=fallback_vocab_size,
        fallback_n_embed=fallback_n_embed,
        fallback_n_layer=fallback_n_layer,
        fallback_n_head=fallback_n_head,
        fallback_block_size=fallback_block_size,
        fallback_dropout=fallback_dropout,
    )
    model = SloughGPTModel(
        vocab_size=hp["vocab_size"],
        n_embed=hp["n_embed"],
        n_layer=hp["n_layer"],
        n_head=hp["n_head"],
        block_size=hp["block_size"],
        dropout=hp["dropout"],
    )
    model.load_state_dict(extract_state_dict(bundle), strict=strict)
    model = model.to(device)
    return model, hp


def tokenizer_maps_from_bundle(
    bundle: Dict[str, Any],
) -> Tuple[Optional[Dict[str, int]], Optional[Dict[int, str]]]:
    """Return ``(stoi, itos)`` if present on the checkpoint bundle."""
    bundle = normalize_raw_checkpoint(bundle)
    return bundle.get("stoi"), bundle.get("itos")
