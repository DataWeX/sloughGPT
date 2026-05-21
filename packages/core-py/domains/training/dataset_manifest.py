"""
SloughGPT Standard v1 — resolve training text path from a dataset manifest JSON.

The char-level trainer expects a single UTF-8 text file. The manifest may point to it via
``splits.train`` (relative to the manifest file's directory) or default ``input.txt`` sibling.

Checkpoint vocabulary on native ``step_*.pt`` (separate from manifest JSON): see
``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ManifestError(ValueError):
    """Invalid or unusable dataset manifest for training."""


def load_manifest(path: str | Path) -> Dict[str, Any]:
    """Load and minimally validate a v1 dataset manifest."""
    p = Path(path)
    if not p.is_file():
        raise ManifestError(f"Manifest not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ManifestError(f"Invalid JSON manifest: {e}") from e
    ver = data.get("schema_version")
    if ver != "1.0":
        raise ManifestError(f"Unsupported schema_version {ver!r}; expected '1.0'")
    for key in ("dataset_id", "version", "domain", "pii_policy", "sources"):
        if key not in data:
            raise ManifestError(f"Manifest missing required field: {key}")
    if not isinstance(data.get("sources"), list) or not data["sources"]:
        raise ManifestError("Manifest 'sources' must be a non-empty list.")
    return data


def _glob_train_files(pattern: str, base_dir: Path) -> List[Path]:
    if any(ch in pattern for ch in "*?["):
        parts = pattern.replace("\\", "/").split("/")
        if not parts:
            return []
        if "**" in pattern:
            return sorted(base_dir.glob(pattern))
        # pathlib glob: relative to base_dir
        return sorted(base_dir.glob(pattern))
    return [base_dir / pattern]


def resolve_training_data_path(manifest_path: str | Path) -> Tuple[Path, Dict[str, Any]]:
    """
    Return (path_to_utf8_text_file, manifest_dict).

    Resolution order:
    1. ``splits.train`` — single existing file, or if glob matches exactly one ``.txt`` file.
    2. ``input.txt`` in the same directory as the manifest.
    """
    mp = Path(manifest_path).expanduser()
    if not mp.is_file():
        raise ManifestError(f"Manifest path is not a file: {mp}")
    manifest = load_manifest(mp)
    base_dir = mp.parent

    splits = manifest.get("splits") or {}
    train = splits.get("train")
    if train:
        train_s = str(train).strip()
        candidates = _glob_train_files(train_s, base_dir)
        existing = [c for c in candidates if c.is_file()]
        if len(existing) == 1:
            return existing[0], manifest
        if len(existing) > 1:
            txt_only = [c for c in existing if c.suffix.lower() == ".txt"]
            if len(txt_only) == 1:
                return txt_only[0], manifest
            raise ManifestError(
                f"splits.train pattern {train_s!r} matched {len(existing)} files; "
                "concatenate to a single .txt or narrow the glob."
            )
        raise ManifestError(
            f"splits.train {train_s!r} did not resolve to an existing file under {base_dir}"
        )

    default_txt = base_dir / "input.txt"
    if default_txt.is_file():
        return default_txt, manifest

    raise ManifestError(
        f"No training text file found: set splits.train in the manifest or place input.txt next to {mp.name}"
    )
