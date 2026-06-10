"""
HuggingFace Hub integration — resolves model files, URLs, and checksums.

Uses ``huggingface_hub.HfApi`` for file listing and HuggingFace's
CDN URL scheme for downloads.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_HF_HOME = Path.home() / ".cache" / "huggingface" / "hub"


@dataclass
class HFFile:
    """A single file in a HuggingFace model repo."""
    path: str
    size: int
    checksum: str
    download_url: str
    is_ignored: bool = False


IGNORED_PATTERNS = ("*.h5", "*.ot", "*.msgpack", "*.onnx", "*.gguf", "*.tflite")


def _matches_ignore(filename: str) -> bool:
    """Check if a filename matches any ignored pattern.

    Also ignores files inside ``onnx/`` or ``tf/`` subdirectories
    (alternative-format variants that aren't needed for core inference).
    """
    import fnmatch
    if any(fnmatch.fnmatch(filename, p) for p in IGNORED_PATTERNS):
        return True
    if filename.startswith("onnx/") or filename.startswith("tf/"):
        return True
    return False


def list_model_files(model_id: str) -> List[HFFile]:
    """List all downloadable files for a HuggingFace model.

    Returns files needed for core inference (config, tokenizer, weights).
    Excludes alternative format variants (ONNX, TFLite, TF, etc.).

    Each file includes its size, SHA-256 checksum (from LFS pointer),
    and HuggingFace CDN download URL.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.model_info(model_id, files_metadata=True)

    files: List[HFFile] = []
    hf_endpoint = os.environ.get(
        "HF_ENDPOINT",
        "https://huggingface.co",
    ).rstrip("/")

    for sibling in (info.siblings or []):
        if sibling.rfilename.startswith("."):
            continue
        ignored = _matches_ignore(sibling.rfilename)

        f = HFFile(
            path=sibling.rfilename,
            size=sibling.size or 0,
            checksum=sibling.lfs.get("sha256", "") if sibling.lfs else "",
            download_url=(
                f"{hf_endpoint}/{model_id}/resolve/main/{sibling.rfilename}"
                if not ignored else ""
            ),
            is_ignored=ignored,
        )
        files.append(f)

    return files


def get_cache_dir(model_id: str, hf_home: Optional[str] = None) -> Path:
    """Get the HF cache directory path for a model.

    Respects ``HF_HOME`` environment variable for custom cache locations.
    """
    base = Path(hf_home) if hf_home else (
        Path(os.environ.get("HF_HOME", str(DEFAULT_HF_HOME.parent))) / "hub"
    )
    return base / f"models--{model_id.replace('/', '--')}"


def is_download_complete(
    model_id: str,
    hf_home: Optional[str] = None,
    deep_check: bool = False,
) -> bool:
    """Check if a model is fully downloaded.

    Quick check (no network): verifies ``refs/main``, snapshot dir,
    no incomplete markers, and at least one weight file > 1 KB.

    Deep check (network, optional): fetches the expected file list from
    Hub API and verifies every weight file exists with correct size.
    Use for verification workflows; skip for batch listing.
    """
    cache_dir = get_cache_dir(model_id, hf_home)
    if not cache_dir.exists():
        return False

    refs_main = cache_dir / "refs" / "main"
    if not refs_main.exists():
        return False
    commit = refs_main.read_text().strip()
    if not commit:
        return False

    snapshot = cache_dir / "snapshots" / commit
    if not snapshot.exists():
        return False

    incomplete = list(cache_dir.rglob("*.incomplete"))
    if incomplete:
        return False
    locks = list(cache_dir.rglob("*.lock"))
    if locks:
        return False

    has_any_weight = False
    for ext in ("*.safetensors", "*.bin"):
        for f in snapshot.rglob(ext):
            try:
                if f.stat().st_size > 1_000:
                    has_any_weight = True
                    break
            except OSError:
                continue
    if not has_any_weight:
        return False

    if deep_check:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            info = api.model_info(model_id, files_metadata=True)
            for sibling in (info.siblings or []):
                if sibling.rfilename.startswith("."):
                    continue
                if not sibling.rfilename.endswith((".safetensors", ".bin", ".gguf")):
                    continue
                expected_size = (sibling.size or 0)
                local = snapshot / sibling.rfilename
                if not local.exists():
                    return False
                if expected_size > 0:
                    try:
                        if local.stat().st_size != expected_size:
                            return False
                    except OSError:
                        return False
        except Exception:
            pass

    return True


def resolve_cached_path(
    model_id: str,
    file_path: str,
    hf_home: Optional[str] = None,
) -> Optional[Path]:
    """Resolve a model file's path in the local HF cache.

    Returns the path to the cached file, or ``None`` if not cached.
    """
    cache_dir = get_cache_dir(model_id, hf_home)
    if not cache_dir.exists():
        return None

    refs_main = cache_dir / "refs" / "main"
    if not refs_main.exists():
        return None
    commit = refs_main.read_text().strip()
    if not commit:
        return None

    cached_file = cache_dir / "snapshots" / commit / file_path
    if cached_file.exists():
        try:
            if cached_file.stat().st_size > 1_000:
                return cached_file
        except OSError:
            pass
    return None
