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

    Returns the standard cache location that downloads target (respects
    ``HF_HOME`` environment variable).  For reading, prefer
    :func:`find_cached_model_dir`, which also locates the project-local
    mirror.
    """
    base = Path(hf_home) if hf_home else (
        Path(os.environ.get("HF_HOME", str(DEFAULT_HF_HOME.parent))) / "hub"
    )
    return base / f"models--{model_id.replace('/', '--')}"


PROJECT_CACHE_RELPATH = Path("models") / "hf-cache" / "hub"


def _model_slug(model_id: str) -> str:
    """Convert a model id to its cache directory name (``models--`` form)."""
    return f"models--{model_id.replace('/', '--')}"


def _project_cache_roots() -> List[Path]:
    """Return existing project-local HF cache roots reachable from CWD.

    This monorepo mirrors the HF hub layout into
    ``<repo>/models/hf-cache/hub``.  Walks upward from the working
    directory (bounded) so the mirror is found regardless of process CWD.
    """
    roots: List[Path] = []
    seen = set()
    current = Path.cwd().resolve()
    for _ in range(5):
        candidate = current / PROJECT_CACHE_RELPATH
        if candidate.is_dir() and str(candidate) not in seen:
            roots.append(candidate)
            seen.add(str(candidate))
        if current.parent == current:
            break
        current = current.parent
    return roots


def _model_cache_candidates(model_id: str, hf_home: Optional[str]) -> List[Path]:
    """Ordered candidate cache dirs: standard HF first, then project-local."""
    candidates = [get_cache_dir(model_id, hf_home)]
    for root in _project_cache_roots():
        candidates.append(root / _model_slug(model_id))
    return candidates


def find_cached_model_dir(model_id: str, hf_home: Optional[str] = None) -> Optional[Path]:
    """Return the first existing cache directory for a model.

    Searches the standard HuggingFace cache, then the project-local
    ``models/hf-cache/hub`` mirror.  Returns ``None`` when the model has
    no directory anywhere (it is not cached at all).
    """
    for candidate in _model_cache_candidates(model_id, hf_home):
        if candidate.is_dir():
            return candidate
    return None


def _snapshot_dir(cache_dir: Path) -> Optional[Path]:
    """Resolve the snapshot dir of the standard HF layout, or ``None``."""
    refs_main = cache_dir / "refs" / "main"
    if not refs_main.is_file():
        return None
    try:
        commit = refs_main.read_text().strip()
    except OSError:
        return None
    if not commit:
        return None
    snapshot = cache_dir / "snapshots" / commit
    return snapshot if snapshot.is_dir() else None


def _has_weight_files(cache_dir: Path) -> bool:
    """Whether a directory holds any model weight file larger than 1 KB."""
    for ext in ("*.safetensors", "*.bin"):
        for f in cache_dir.rglob(ext):
            try:
                if f.stat().st_size > 1_000:
                    return True
            except OSError:
                continue
    return False


def _has_incomplete_markers(cache_dir: Path) -> bool:
    """True when any marker indicates an in-progress or interrupted download.

    Recognizes ``*.incomplete`` (HuggingFace hub) and ``*.sgpart``
    (downcraft resume temp files, renamed to the final name only on
    completion).
    """
    for pattern in ("*.incomplete", "*.sgpart"):
        if list(cache_dir.rglob(pattern)):
            return True
    return False


def _lock_is_stale(cache_dir: Path, lock: Path) -> bool:
    """Whether a ``.lock`` file is leftover because its target file exists.

    HF's local-dir cache mode leaves ``.cache/huggingface/download/x.lock``
    markers whose target ``x`` lives at the model dir root once complete.
    A lock whose target is missing means a download is in progress.
    """
    name = lock.name
    if name.endswith(".lock"):
        name = name[: -len(".lock")]
    return (cache_dir / name).exists()


def _deep_check(model_id: str, target_dir: Path) -> bool:
    """Verify every weight file exists at expected size via Hub API."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(model_id, files_metadata=True)
        for sibling in (info.siblings or []):
            if sibling.rfilename.startswith("."):
                continue
            if not sibling.rfilename.endswith((".safetensors", ".bin", ".gguf")):
                continue
            local = target_dir / sibling.rfilename
            if not local.exists():
                return False
            expected_size = (sibling.size or 0)
            if expected_size > 0:
                try:
                    if local.stat().st_size != expected_size:
                        return False
                except OSError:
                    return False
    except Exception:
        pass
    return True


def _is_snapshot_complete(
    cache_dir: Path,
    snapshot: Path,
    model_id: str,
    deep_check: bool,
) -> bool:
    """Completeness for the standard HF snapshot layout."""
    if _has_incomplete_markers(cache_dir):
        return False
    if list(cache_dir.rglob("*.lock")):
        return False
    if not _has_weight_files(snapshot):
        return False
    if deep_check and not _deep_check(model_id, snapshot):
        return False
    return True


def _is_flat_model_complete(
    cache_dir: Path,
    model_id: str,
    deep_check: bool,
) -> bool:
    """Completeness for the flat project-local / HF local-dir layout.

    Files (``config.json``, ``model.safetensors``, ...) live at the cache
    dir root.  ``refs/`` and ``.cache/huggingface/`` may exist; leftover
    ``.lock`` files are ignored when their target file is already present.
    """
    if not _has_weight_files(cache_dir):
        return False
    if not (cache_dir / "config.json").is_file():
        return False
    if _has_incomplete_markers(cache_dir):
        return False
    for lock in cache_dir.rglob("*.lock"):
        if not _lock_is_stale(cache_dir, lock):
            return False
    if deep_check and not _deep_check(model_id, cache_dir):
        return False
    return True


def is_download_complete(
    model_id: str,
    hf_home: Optional[str] = None,
    deep_check: bool = False,
) -> bool:
    """Check if a model is fully downloaded.

    Recognizes both the standard HF snapshot layout and the flat
    project-local layout (``models/hf-cache/hub``), across the standard
    cache and the project-local mirror.

    Quick check (no network): verifies ``refs/main`` + snapshot dir, or
    a flat model dir with weights and ``config.json``, with no incomplete
    markers and no active download locks.  At least one weight file > 1 KB.

    Deep check (network, optional): fetches the expected file list from
    Hub API and verifies every weight file exists with correct size.
    Use for verification workflows; skip for batch listing.
    """
    for cache_dir in _model_cache_candidates(model_id, hf_home):
        if not cache_dir.is_dir():
            continue
        snapshot = _snapshot_dir(cache_dir)
        if snapshot is not None and _is_snapshot_complete(
            cache_dir, snapshot, model_id, deep_check,
        ):
            return True
        if _is_flat_model_complete(cache_dir, model_id, deep_check):
            return True
    return False


def resolve_cached_path(
    model_id: str,
    file_path: str,
    hf_home: Optional[str] = None,
) -> Optional[Path]:
    """Resolve a model file's path in the local HF cache.

    Handles both the standard snapshot layout and the flat project-local
    layout.  Returns the path to the cached file, or ``None`` if not cached.
    """
    for cache_dir in _model_cache_candidates(model_id, hf_home):
        if not cache_dir.is_dir():
            continue
        snapshot = _snapshot_dir(cache_dir)
        candidate_paths = []
        if snapshot is not None:
            candidate_paths.append(snapshot / file_path)
        candidate_paths.append(cache_dir / file_path)

        for cached_file in candidate_paths:
            if cached_file.exists():
                try:
                    if cached_file.stat().st_size > 1_000:
                        return cached_file
                except OSError:
                    pass
    return None
