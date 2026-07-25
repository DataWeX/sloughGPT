"""
Model Protector — prevents accidental deletion of local model files.

After a model is downloaded or converted to .slnc, this module:
  1. Sets file permissions to read-only (chmod444)
  2. Drops a .nomodeldelete marker (cleanup tools like BleachBit respect this)
  3. Tracks protected files in a manifest for integrity checks

Usage:
    from domains.infrastructure.model_protector import protect_model, check_model, unprotect_model

    # After download/conversion:
    protect_model("gpt2", ["/path/to/model.slnc"])

    # Before loading:
    missing = check_model("gpt2")
    if missing:
        print(f"Model files deleted: {missing}")

    # To allow deletion:
    unprotect_model("gpt2")
"""

import json
import logging
import os
import stat
from pathlib import Path
from typing import Optional

logger = logging.getLogger("slo.infrastructure.model_protector")

_MARKER_FILENAME = ".nomodeldelete"
_MANIFEST_FILENAME = ".sloughgpt-protected"


def _get_model_dir(model_id: str) -> Path:
    """Resolve HuggingFace cache directory for a model."""
    cache_id = model_id.replace("/", "--")
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    return Path(hf_home) / "hub" / f"models--{cache_id}"


def _get_protected_dir() -> Path:
    """Directory for our own model cache (non-HF)."""
    return Path.home() / ".cache" / "sloughgpt" / "models"


def _write_manifest(model_dir: Path, files: list[Path]) -> None:
    """Write a manifest of protected files for integrity checks."""
    manifest_path = model_dir / _MANIFEST_FILENAME
    manifest = {
        "model_dir": str(model_dir),
        "protected_files": [
            {
                "path": str(f),
                "size": f.stat().st_size if f.exists() else 0,
                "mode": oct(f.stat().st_mode) if f.exists() else None,
            }
            for f in files
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    # Make manifest read-only too
    try:
        os.chmod(manifest_path, stat.S_IRUSR | stat.S_IRGRP)
    except OSError:
        pass


def _read_manifest(model_dir: Path) -> Optional[dict]:
    """Read the protection manifest."""
    manifest_path = model_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def protect_model(model_id: str, file_paths: list[str | Path] | None = None) -> dict:
    """Protect a model's files from accidental deletion.

    Args:
        model_id: HuggingFace model ID (e.g. "gpt2", "Qwen/Qwen2.5-0.5B-Instruct")
        file_paths: Specific files to protect. If None, protects all .slnc/.safetensors
                    in the model's cache directory.

    Returns:
        dict with "protected" (list of files) and "errors" (list of failures)
    """
    model_dir = _get_model_dir(model_id)

    if file_paths is None:
        # Auto-discover weight files
        file_paths = []
        for ext in ("*.slnc", "*.safetensors", "*.bin"):
            file_paths.extend(model_dir.rglob(ext))
        # Also protect tokenizer files
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                      "vocab.json", "merges.txt", "config.json"):
            p = model_dir / name
            if p.exists():
                file_paths.append(p)

    protected = []
    errors = []

    for fp in file_paths:
        fpath = Path(fp)
        if not fpath.exists():
            continue
        try:
            # Set read-only: owner r, group r, other r (444)
            current = fpath.stat().st_mode
            read_only = current & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            os.chmod(fpath, read_only)
            protected.append(str(fpath))
        except OSError as e:
            errors.append({"file": str(fpath), "error": str(e)})

    # Drop marker file
    marker = model_dir / _MARKER_FILENAME
    try:
        if not marker.exists():
            marker.write_text(
                "# This directory contains AI model weights for SloughGPT.\n"
                "# Do not delete — re-downloading costs bandwidth.\n"
                "# To remove: run `sloughgpt model unprotect <model_id>`\n"
            )
            os.chmod(marker, stat.S_IRUSR | stat.S_IRGRP)
    except OSError as e:
        errors.append({"file": str(marker), "error": str(e)})

    # Write manifest
    try:
        _write_manifest(model_dir, [Path(f) for f in protected])
    except OSError as e:
        errors.append({"file": str(model_dir / _MANIFEST_FILENAME), "error": str(e)})

    if protected:
        logger.info(
            "Protected %d files for model '%s' (chmod444 + manifest)",
            len(protected), model_id,
        )

    return {"protected": protected, "errors": errors}


def unprotect_model(model_id: str) -> dict:
    """Remove protection from a model's files (allows deletion).

    Returns:
        dict with "unprotected" count and "errors"
    """
    model_dir = _get_model_dir(model_id)

    unprotected = []
    errors = []

    manifest = _read_manifest(model_dir)
    if manifest:
        for entry in manifest.get("protected_files", []):
            fpath = Path(entry["path"])
            if fpath.exists():
                try:
                    # Restore owner write permission
                    current = fpath.stat().st_mode
                    os.chmod(fpath, current | stat.S_IWUSR)
                    unprotected.append(str(fpath))
                except OSError as e:
                    errors.append({"file": str(fpath), "error": str(e)})

    # Remove marker
    marker = model_dir / _MARKER_FILENAME
    if marker.exists():
        try:
            os.chmod(marker, stat.S_IWUSR | stat.S_IRUSR)
            marker.unlink()
        except OSError as e:
            errors.append({"file": str(marker), "error": str(e)})

    # Remove manifest
    manifest_path = model_dir / _MANIFEST_FILENAME
    if manifest_path.exists():
        try:
            os.chmod(manifest_path, stat.S_IWUSR | stat.S_IRUSR)
            manifest_path.unlink()
        except OSError as e:
            errors.append({"file": str(manifest_path), "error": str(e)})

    if unprotected:
        logger.info("Unprotected %d files for model '%s'", len(unprotected), model_id)

    return {"unprotected": len(unprotected), "errors": errors}


def check_model(model_id: str) -> list[str]:
    """Check if a model's protected files still exist.

    Returns:
        List of missing file paths (empty = all OK)
    """
    model_dir = _get_model_dir(model_id)
    manifest = _read_manifest(model_dir)

    if not manifest:
        return []

    missing = []
    for entry in manifest.get("protected_files", []):
        fpath = Path(entry["path"])
        if not fpath.exists():
            missing.append(str(fpath))

    return missing


def list_protected() -> list[dict]:
    """List all models with protection manifests."""
    hub = _get_model_dir("").parent  # ~/.cache/huggingface/hub
    if not hub.exists():
        return []

    protected = []
    for model_dir in hub.iterdir():
        if not model_dir.is_dir():
            continue
        manifest = _read_manifest(model_dir)
        if manifest:
            model_id = model_dir.name.replace("models--", "").replace("--", "/")
            missing = [e["path"] for e in manifest.get("protected_files", []) if not Path(e["path"]).exists()]
            protected.append({
                "model_id": model_id,
                "dir": str(model_dir),
                "files": len(manifest.get("protected_files", [])),
                "missing": len(missing),
                "missing_files": missing,
            })

    return protected
