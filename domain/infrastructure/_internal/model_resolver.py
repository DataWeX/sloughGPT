"""Model resolution utilities — find models in HuggingFace cache."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from domain.shared import find_repo_root

logger = logging.getLogger("slo.infrastructure.model_resolver")

# Payload files that prove a cache entry holds real weights. `blobs/`
# (non-empty) is checked separately — HF stores downloaded weight blobs there.
_WEIGHT_PATTERNS = ("*.safetensors", "*.slnc", "*.soul")


def _dir_has_weights(model_dir: Path) -> bool:
    """True when *model_dir* holds real model payload, not a refs-only stub.

    HF cache entries can be empty refs-only dirs (refs/main, zero blobs)
    that nonetheless `exists()` — such a stub must never shadow an intact
    candidate later in the search order (card c158a0e3).
    """
    if not model_dir.is_dir():
        return False
    blobs = model_dir / "blobs"
    if blobs.is_dir() and any(blobs.iterdir()):
        return True
    scopes = [model_dir]
    snapshots = model_dir / "snapshots"
    if snapshots.is_dir():
        scopes.extend(p for p in snapshots.iterdir() if p.is_dir())
    for scope in scopes:
        for pattern in _WEIGHT_PATTERNS:
            if any(scope.glob(pattern)):
                return True
    return False


def get_model_dir(model_id: str) -> Path:
    """Resolve HuggingFace cache directory for a model.

    Searches the standard HF cache first, then the project-local
    cache (models/hf-cache/hub/) mirroring MorphTokenizer.from_pretrained.
    A candidate that actually contains weights wins over an empty
    refs-only stub anywhere earlier in the order; the first existing
    candidate remains the fallback when no candidate has weights.
    """
    cache_id = model_id.replace("/", "--")
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    candidates = [
        Path(hf_home) / "hub" / f"models--{cache_id}",
        find_repo_root(Path(__file__).resolve())
        / "models"
        / "hf-cache"
        / "hub"
        / f"models--{cache_id}",
        Path("models/hf-cache/hub") / f"models--{cache_id}",
    ]

    for candidate in candidates:
        if candidate.exists() and _dir_has_weights(candidate):
            stubs = [
                c
                for c in candidates[: candidates.index(candidate)]
                if c.exists() and not _dir_has_weights(c)
            ]
            if stubs:
                logger.info(
                    "get_model_dir(%s): chose %s over weightless earlier candidate(s) %s",
                    model_id,
                    candidate,
                    stubs,
                )
            return candidate
    for candidate in candidates:
        if candidate.exists():
            logger.warning(
                "get_model_dir(%s): no candidate holds weights; using %s anyway",
                model_id,
                candidate,
            )
            return candidate
    return candidates[0]


def find_safetensors(model_dir: Path) -> Path | None:
    """Find the safetensors file in a model directory."""
    snapshots = model_dir / "snapshots"
    if snapshots.exists():
        for snapshot in snapshots.iterdir():
            st = snapshot / "model.safetensors"
            if st.exists():
                return st
    st = model_dir / "model.safetensors"
    if st.exists():
        return st
    return None


def load_model_config(model_id: str) -> dict[str, Any]:
    """Load model config.json from HuggingFace cache."""
    model_dir = get_model_dir(model_id)
    config_path = None
    snapshots = model_dir / "snapshots"
    if snapshots.exists():
        for snap in snapshots.iterdir():
            c = snap / "config.json"
            if c.exists():
                config_path = c
                break
    if config_path is None:
        config_path = model_dir / "config.json"
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"No config.json for {model_id}")
    with open(config_path) as f:
        return json.load(f)
