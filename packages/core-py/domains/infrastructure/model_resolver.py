"""Model resolution utilities — find models in HuggingFace cache."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from domains.shared import find_repo_root


def get_model_dir(model_id: str) -> Path:
    """Resolve HuggingFace cache directory for a model.

    Searches the standard HF cache first, then the project-local
    cache (models/hf-cache/hub/) mirroring MorphTokenizer.from_pretrained.
    """
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


def find_safetensors(model_dir: Path) -> Optional[Path]:
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


def load_model_config(model_id: str) -> Dict[str, Any]:
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
