"""
HuggingFace Hub loader for the sloughGPT application.

This module owns all HuggingFace-specific knowledge of the project and is
built on top of ``downcraft``'s *generic* HTTP downloader primitives
(``downcraft.downloader``, ``downcraft.state``, ``downcraft.verify``).

``downcraft`` is deliberately HuggingFace-agnostic — it downloads any URL
with cross-session resume via HTTP ``Range`` headers.  Everything here is
the HuggingFace-specific layer:

* the Hub REST API shape and CDN download URL scheme,
* the ``models--<id>`` cache layout and ``refs/main -> snapshots/<commit>``
  resolution (plus the flat project-local mirror),
* LFS ``sha256`` checksums and ignore rules for alternative formats,
* the model-level download / resume / verify workflows composed from the
  generic primitives.

Public API:

* ``list_model_files``, ``fetch_model_info``, ``fetch_dataset_search``
* ``get_cache_dir``, ``find_cached_model_dir``, ``is_download_complete``
* ``download_hf_model``, ``resume_plan``, ``resume_download``, ``resume_model``
* ``verify_model``, ``list_missing_files``
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import requests

from downcraft.download import state as dc_state
from downcraft.download.http import DownloadError, download_file
from downcraft.download.verify import _sha256_of, verify_file

logger = logging.getLogger(__name__)

DEFAULT_HF_HOME = Path.home() / ".cache" / "huggingface" / "hub"


# ---------------------------------------------------------------------------
# Hub REST API
# ---------------------------------------------------------------------------

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


def _hf_endpoint() -> str:
    """Resolve the HuggingFace Hub base URL (respects ``HF_ENDPOINT``)."""
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _hf_api_get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 15) -> Any:
    """GET a HuggingFace Hub REST API endpoint and return parsed JSON.

    Args:
        path: API path after ``/api/`` (e.g. ``models/gpt2``).
        params: Optional query parameters.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON (dict or list), or ``None`` on any request/parse failure.
    """
    try:
        resp = requests.get(
            f"{_hf_endpoint()}/api/{path}",
            params=params,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def fetch_model_info(model_id: str) -> Optional[Dict]:
    """Fetch model repository metadata (file siblings + sizes) from the Hub.

    Args:
        model_id: HuggingFace model ID (e.g. ``gpt2``, ``Qwen/Qwen2.5-0.5B-Instruct``).

    Returns:
        Model metadata dict (with a ``siblings`` list), or ``None`` on failure.
    """
    data = _hf_api_get(f"models/{model_id}", params={"blobs": "true"})
    if not isinstance(data, dict):
        return None
    return data


def fetch_dataset_search(query: str, limit: int = 10) -> List[Dict]:
    """Search HuggingFace datasets via the Hub REST API.

    Args:
        query: Search query string.
        limit: Maximum number of results.

    Returns:
        List of ``{"id", "downloads"}`` dicts (empty on failure).
    """
    data = _hf_api_get("datasets", params={"search": query, "limit": limit})
    if not isinstance(data, list):
        return []
    results: List[Dict] = []
    for ds in data:
        if not isinstance(ds, dict):
            continue
        results.append({
            "id": ds.get("id", ""),
            "downloads": ds.get("downloads") or 0,
        })
    return results


def list_model_files(model_id: str) -> List[HFFile]:
    """List all downloadable files for a HuggingFace model.

    Returns files needed for core inference (config, tokenizer, weights).
    Excludes alternative format variants (ONNX, TFLite, TF, etc.).

    Each file includes its size, SHA-256 checksum (from LFS pointer),
    and HuggingFace CDN download URL.
    """
    info = fetch_model_info(model_id)

    files: List[HFFile] = []
    if not info:
        return files

    for sibling in (info.get("siblings") or []):
        if not isinstance(sibling, dict):
            continue
        rfilename = sibling.get("rfilename", "")
        if rfilename.startswith("."):
            continue
        ignored = _matches_ignore(rfilename)

        lfs = sibling.get("lfs") or {}
        f = HFFile(
            path=rfilename,
            size=sibling.get("size") or 0,
            checksum=lfs.get("sha256", "") if isinstance(lfs, dict) else "",
            download_url=(
                f"{_hf_endpoint()}/{model_id}/resolve/main/{rfilename}"
                if not ignored else ""
            ),
            is_ignored=ignored,
        )
        files.append(f)

    return files


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------

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
        info = fetch_model_info(model_id)
        if not info:
            return True
        for sibling in (info.get("siblings") or []):
            if not isinstance(sibling, dict):
                continue
            rfilename = sibling.get("rfilename", "")
            if rfilename.startswith("."):
                continue
            if not rfilename.endswith((".safetensors", ".bin", ".gguf")):
                continue
            local = target_dir / rfilename
            if not local.exists():
                return False
            expected_size = (sibling.get("size") or 0)
            if expected_size > 0:
                try:
                    if local.stat().st_size != expected_size:
                        return False
                except OSError:
                    return False
    except Exception as exc:
        logger.debug("deep check failed for %s: %s", model_id, exc)
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


# ---------------------------------------------------------------------------
# Model download (composed from downcraft's generic downloader + state)
# ---------------------------------------------------------------------------

def download_hf_model(
    model_id: str,
    hf_home: Optional[str] = None,
    on_progress: Optional[Callable[[str, int, int, float], None]] = None,
    on_file_complete: Optional[Callable[[str, str], None]] = None,
    ignore_cache: bool = False,
) -> Dict:
    """Download a HuggingFace model with cross-session resume.

    Downloads each file of the model repo into the standard
    ``snapshots/default`` cache layout using ``downcraft.download_file``
    (Range-header resume), tracking progress in downcraft's persistent
    state file so multi-file model downloads resume across restarts.

    Args:
        model_id: HF model ID (e.g. ``"gpt2"``, ``"Qwen/Qwen2.5-0.5B-Instruct"``).
        hf_home: Override HF cache directory.
        on_progress: Called per-chunk with ``(model_id, bytes_downloaded, total_bytes, speed_bps)``.
        on_file_complete: Called when each file finishes ``(model_id, file_path)``.
        ignore_cache: If True, redownload even if already fully cached.

    Returns:
        Dict with keys: ``status``, ``cache_dir``, ``elapsed``, ``total_bytes``.

    Side effects:
        - writes files under ``<cache>/snapshots/default``,
        - writes ``<cache>/refs/main``,
        - updates ``~/.downcraft/state.json``.
    """
    # Resolve cache directory.  An explicit ``hf_home`` is the hub root
    # (``<hf_home>/models--<id>``); ``None`` resolves via the HF_HOME env
    # var with ``/hub`` appended — the standard HF layout that
    # is_download_complete and the app's safetensors_loader expect.
    cache_dir = str(get_cache_dir(model_id, hf_home))

    st = dc_state.get_state()

    # Quick check — already complete?  State is a hint only; disk is truth.
    existing = st.get(model_id)
    if existing and existing.status == "complete" and not ignore_cache:
        if is_download_complete(model_id, hf_home=hf_home):
            logger.info("%s already fully downloaded", model_id)
            _write_snapshot_ref(cache_dir)
            return {
                "status": "already_cached",
                "model_id": model_id,
                "cache_dir": existing.cache_dir,
            }
        logger.warning(
            "%s marked complete in state but files missing on disk; redownloading",
            model_id,
        )

    files = list_model_files(model_id)
    weight_files = [f for f in files if not f.is_ignored]

    if not weight_files:
        raise RuntimeError(f"No downloadable files found for {model_id}")

    logger.info(
        "Resolved %d files for %s (%.2f GB total)",
        len(weight_files),
        model_id,
        sum(f.size for f in weight_files) / (1024 ** 3),
    )

    st_state = st.create(model_id, cache_dir)
    start = time.time()
    total_all = sum(f.size for f in weight_files)

    for hf_file in weight_files:
        rel_path = hf_file.path
        dest = Path(cache_dir) / "snapshots" / "default" / rel_path

        # Disk truth beats persistent state: a final file at the expected
        # size is complete even if ~/.downcraft/state.json was lost.
        if hf_file.size > 0 and dest.is_file() and dest.stat().st_size == hf_file.size:
            st.update_file_progress(
                model_id, rel_path, hf_file.download_url,
                hf_file.size, hf_file.size,
                checksum=hf_file.checksum, complete=True,
            )
            continue

        existing_file = st_state.files.get(rel_path)
        if existing_file and existing_file.complete:
            continue

        # Normalize HuggingFace's *.incomplete marker into downcraft's
        # *.sgpart so the generic downloader resumes it via Range.
        incomplete = dest.with_suffix(dest.suffix + ".incomplete")
        sgpart = dest.with_suffix(dest.suffix + ".sgpart")
        if incomplete.is_file() and not sgpart.exists():
            os.replace(str(incomplete), str(sgpart))

        chunk_cb = _make_hf_chunk_cb(
            st, model_id, rel_path,
            hf_file.download_url, hf_file.size,
            hf_file.checksum, start, total_all, on_progress,
        )

        try:
            download_file(
                url=hf_file.download_url,
                dest=dest,
                expected_size=hf_file.size,
                checksum=hf_file.checksum,
                on_chunk=chunk_cb,
                on_complete=lambda p: (
                    on_file_complete(model_id, rel_path)
                    if on_file_complete else None
                ),
            )
            st.update_file_progress(
                model_id, rel_path, hf_file.download_url,
                hf_file.size, hf_file.size,
                checksum=hf_file.checksum, complete=True,
            )
        except DownloadError:
            st.set_status(model_id, "failed", error=f"Failed on {rel_path}")
            st.flush()
            raise

    st.set_status(model_id, "complete")
    st.flush()

    # Record the snapshot ref so is_download_complete recognizes the
    # snapshots/default layout (it resolves refs/main -> snapshot dir).
    _write_snapshot_ref(cache_dir)

    elapsed = time.time() - start
    logger.info(
        "Downloaded %s in %.1fs (%.2f MB/s)",
        model_id, elapsed,
        (total_all / elapsed / 1e6) if elapsed > 0 else 0,
    )

    return {
        "status": "complete",
        "model_id": model_id,
        "cache_dir": cache_dir,
        "elapsed": round(elapsed, 1),
        "total_bytes": total_all,
    }


def _write_snapshot_ref(cache_dir: str) -> None:
    """Write ``refs/main -> default`` so the snapshot layout is recognized.

    ``download_hf_model`` writes files to ``snapshots/default``.  Recording
    the ref lets :func:`is_download_complete` resolve the snapshot dir and
    mark the model complete.
    """
    refs = Path(cache_dir) / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text("default")


def _make_hf_chunk_cb(
    st_obj,
    model_id: str,
    rel_path: str,
    download_url: str,
    file_size: int,
    checksum: str,
    start_time: float,
    total_all: int,
    on_progress: Optional[Callable],
) -> Callable:
    """Factory for HF file chunk callback with persistent state updates."""
    prev_pct = [0]

    def _cb(bytes_done: int, _total: int):
        pct = int(bytes_done / file_size * 100) if file_size else 0
        if pct != prev_pct[0] and pct % 25 == 0:
            logger.info("  %s: %dMB/%dMB (%d%%)", rel_path,
                         bytes_done // (1024*1024),
                         file_size // (1024*1024) if file_size else 0, pct)
            prev_pct[0] = pct

        st_obj.update_file_progress(
            model_id, rel_path, download_url,
            bytes_done, file_size,
            checksum=checksum,
            complete=(bytes_done >= file_size and file_size > 0),
        )

        ms = st_obj.get(model_id)
        all_done = sum(f.bytes_downloaded for f in ms.files.values()) if ms else bytes_done
        elapsed = time.time() - start_time
        speed = all_done / elapsed if elapsed > 0 else 0

        if on_progress:
            on_progress(model_id, all_done, total_all, speed)

    return _cb


download_model_sync = download_hf_model


# ---------------------------------------------------------------------------
# Resume (built on downcraft's generic .sgpart resume mechanism)
# ---------------------------------------------------------------------------

#: Suffixes used for in-progress temp files (downcraft + HuggingFace).
INCOMPLETE_SUFFIXES = (".sgpart", ".incomplete")

MAX_WALK = 8  # ancestor depth limit when deriving the model id from a path


@dataclass
class ResumeInfo:
    """Resume plan for a single incomplete model file.

    Fields:
        model_id: The HuggingFace model id the partial belongs to.
        repo_path: File path within the model repo (e.g. ``model.safetensors``).
        partial_path: The incomplete file on disk.
        final_path: Where the completed file must land.
        resume_offset: Byte offset to resume from (0 = fresh download).
        total_bytes: Expected final size (0 if unknown).
        download_url: URL to fetch the remaining bytes from.
        checksum: Expected SHA-256 of the complete file.
        complete: True when the partial already holds every byte.
    """

    model_id: str
    repo_path: str
    partial_path: Path
    final_path: Path
    resume_offset: int
    total_bytes: int
    download_url: str
    checksum: str
    complete: bool


def _strip_incomplete_suffix(name: str) -> str:
    """Remove a trailing ``.sgpart`` / ``.incomplete`` suffix, if present."""
    for suffix in INCOMPLETE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _derive_model_id(path: Union[str, Path]) -> Optional[str]:
    """Derive a model id from a path nested under a ``models--<id>`` cache dir.

    Walks up from the given path (bounded) looking for a directory whose
    name starts with ``models--`` (the HF cache slug).  Returns ``None``
    when the path is not inside a model cache directory.
    """
    current = Path(path).resolve().parent
    for _ in range(MAX_WALK):
        name = current.name
        if name.startswith("models--"):
            return name[len("models--"):].replace("--", "/")
        if current.parent == current:
            break
        current = current.parent
    return None


def _match_repo_file(final_name: str, repo_files: List[HFFile]):
    """Match a local file name against the model's repo file list.

    Prefers an exact relative-path match, then a basename match (shards
    like ``model-00001-of-00003.safetensors`` resolve by name).  Returns
    the matched :class:`HFFile` or ``None``.
    """
    for f in repo_files:
        if f.path == final_name:
            return f
    wanted = Path(final_name).name
    for f in repo_files:
        if Path(f.path).name == wanted:
            return f
    return None


def _ensure_sgpart(info: ResumeInfo) -> Path:
    """Normalize the partial file into downcraft's ``.sgpart`` temp form.

    Keeps the larger file when both a ``.sgpart`` and another partial
    (``.incomplete`` or a final-name partial) describe the same target.
    """
    sgpart = info.final_path.with_suffix(info.final_path.suffix + ".sgpart")
    if info.partial_path == sgpart:
        return sgpart
    if sgpart.is_file():
        if info.partial_path.stat().st_size > sgpart.stat().st_size:
            sgpart.unlink()
        else:
            info.partial_path.unlink(missing_ok=True)
            return sgpart
    os.replace(str(info.partial_path), str(sgpart))
    return sgpart


def inspect_incomplete(
    partial_path: Union[str, Path],
    model_id: Optional[str] = None,
    hf_home: Optional[str] = None,
    files: Optional[List[HFFile]] = None,
) -> ResumeInfo:
    """Identify the model file a partial download belongs to and where to resume.

    This is the entry point for "I have an incomplete download of a model
    file — which part is it, and where do I resume?".  The partial's name
    (minus its ``.sgpart`` / ``.incomplete`` suffix) is matched against the
    model's file list; the resume offset is the partial's size.

    Args:
        partial_path: Path to the incomplete download on disk.
        model_id: Model id.  Optional when the path sits under a
            ``models--<id>`` cache directory (derived automatically).
        hf_home: HF cache base for resolving files (used only when
            ``files`` is not supplied).
        files: Pre-fetched repo file list.  When omitted,
            :func:`list_model_files` is called.

    Returns:
        A :class:`ResumeInfo` describing the matched repo file, final
        path, resume offset, and expected size/checksum.

    Raises:
        FileNotFoundError: If ``partial_path`` does not exist.
        ValueError: If the model id cannot be derived/None is passed, or
            the partial matches no file in the model's file list.
    """
    partial = Path(partial_path)
    if not partial.is_file():
        raise FileNotFoundError(f"Partial download not found: {partial}")

    if model_id is None:
        model_id = _derive_model_id(partial)
        if model_id is None:
            raise ValueError(
                "model_id is required when the path is not under a "
                "models--<id> cache directory"
            )

    if files is None:
        files = list_model_files(model_id)

    repo_files = [f for f in files if not f.is_ignored]
    final_name = _strip_incomplete_suffix(partial.name)
    match = _match_repo_file(final_name, repo_files)
    if match is None:
        raise ValueError(
            f"{partial.name!r} does not match any file of model {model_id}"
        )

    offset = partial.stat().st_size
    complete = bool(match.size > 0 and offset >= match.size)
    return ResumeInfo(
        model_id=model_id,
        repo_path=match.path,
        partial_path=partial,
        final_path=partial.with_name(final_name),
        resume_offset=offset,
        total_bytes=match.size,
        download_url=match.download_url,
        checksum=match.checksum,
        complete=complete,
    )


def resume_plan(
    model_id: str,
    hf_home: Optional[str] = None,
    files: Optional[List[HFFile]] = None,
) -> List[ResumeInfo]:
    """List every incomplete file of a model with its resume offset.

    Scans the model's cache directory for ``*.sgpart`` and ``*.incomplete``
    markers, plus final-name files that are smaller than their expected
    size (interrupted non-temp writes).

    Args:
        model_id: Model id to scan.
        hf_home: HF cache base (or any project-local cache root).
        files: Pre-fetched repo file list.  When omitted,
            :func:`list_model_files` is called.

    Returns:
        List of :class:`ResumeInfo`, sorted by repo path, one entry per
        incomplete file.  Empty when the model is not cached or complete.
    """
    cache_dir = find_cached_model_dir(model_id, hf_home)
    if cache_dir is None:
        return []
    if not cache_dir.is_dir():
        return []

    if files is None:
        files = list_model_files(model_id)
    repo_files = [f for f in files if not f.is_ignored]

    infos: Dict[str, ResumeInfo] = {}
    for suffix in INCOMPLETE_SUFFIXES:
        for partial in sorted(cache_dir.rglob(f"*{suffix}")):
            if not partial.is_file():
                continue
            info = inspect_incomplete(
                partial, model_id=model_id, hf_home=hf_home, files=files,
            )
            infos[info.repo_path] = info

    # Final-name partials: repo files present but smaller than expected.
    for f in repo_files:
        if f.size <= 0:
            continue
        candidate = cache_dir / f.path
        if not candidate.is_file():
            continue
        if candidate.stat().st_size < f.size:
            infos[f.path] = ResumeInfo(
                model_id=model_id,
                repo_path=f.path,
                partial_path=candidate,
                final_path=candidate,
                resume_offset=candidate.stat().st_size,
                total_bytes=f.size,
                download_url=f.download_url,
                checksum=f.checksum,
                complete=False,
            )

    return [infos[k] for k in sorted(infos)]


def resume_download(
    partial_path: Union[str, Path],
    model_id: Optional[str] = None,
    hf_home: Optional[str] = None,
    files: Optional[List[HFFile]] = None,
    on_chunk: Optional[Callable[[int, int], None]] = None,
    on_complete: Optional[Callable[[Path], None]] = None,
) -> Path:
    """Resume a single incomplete model file to completion.

    Inspects the partial (via :func:`inspect_incomplete`), normalizes it
    into ``.sgpart`` form, and downloads the remaining bytes with a
    ``Range`` header starting at the partial's size.  If the partial
    already holds every byte it is promoted to the final name without a
    network request.

    Args:
        partial_path: The incomplete download (``*.sgpart``,
            ``*.incomplete``, or an under-sized final-name file).
        model_id: Model id; derived from the path when omitted.
        hf_home: HF cache base (used only when ``files`` is omitted).
        files: Pre-fetched repo file list; fetched when omitted.
        on_chunk: Called per chunk with ``(bytes_downloaded, total_bytes)``.
        on_complete: Called with the final path once the file is complete.

    Returns:
        The final path of the completed file.

    Raises:
        ValueError: If the partial matches no file of the model.
        DownloadError: If the resume download fails permanently.
    """
    info = inspect_incomplete(
        partial_path, model_id=model_id, hf_home=hf_home, files=files,
    )

    # Already fully present at the final name — drop the stale partial.
    if info.total_bytes > 0 and info.final_path.is_file():
        try:
            if info.final_path.stat().st_size >= info.total_bytes:
                if info.partial_path != info.final_path:
                    info.partial_path.unlink(missing_ok=True)
                if on_complete:
                    on_complete(info.final_path)
                return info.final_path
        except OSError:
            pass

    if info.complete:
        _ensure_sgpart(info)
        os.replace(
            str(info.final_path.with_suffix(info.final_path.suffix + ".sgpart")),
            str(info.final_path),
        )
        if on_complete:
            on_complete(info.final_path)
        return info.final_path

    _ensure_sgpart(info)
    download_file(
        url=info.download_url,
        dest=info.final_path,
        expected_size=info.total_bytes,
        checksum=info.checksum,
        on_chunk=on_chunk,
        on_complete=on_complete,
    )
    return info.final_path


def resume_model(
    model_id: str,
    hf_home: Optional[str] = None,
    on_progress: Optional[Callable[[str, int, int, float], None]] = None,
    on_file_complete: Optional[Callable[[str, str], None]] = None,
) -> Dict:
    """Resume every incomplete file of a model, then finish the rest.

    Equivalent to :func:`download_hf_model` but driven by the resume plan:
    each incomplete file found on disk is resumed at its exact offset
    first (the "patch out of the parts" step), then any remaining files
    are downloaded fresh.

    Args:
        model_id: Model id to resume.
        hf_home: HF cache base.
        on_progress: Called per chunk with ``(model_id, bytes, total, speed)``.
        on_file_complete: Called as ``(model_id, repo_path)`` per file.

    Returns:
        Dict with keys ``status``, ``model_id``, ``cache_dir``,
        ``elapsed``, ``total_bytes``, ``resumed_files``.
    """
    plan = resume_plan(model_id, hf_home=hf_home)
    resumed = []
    for info in plan:
        resume_download(
            info.partial_path,
            model_id=model_id,
            hf_home=hf_home,
            on_complete=(
                (lambda p, rp=info.repo_path: (
                    on_file_complete(model_id, rp) if on_file_complete else None
                ))
            ),
        )
        resumed.append(info.repo_path)
        logger.info("Resumed %s/%s (%d bytes)", model_id, info.repo_path, info.resume_offset)

    result = download_hf_model(
        model_id,
        hf_home=hf_home,
        on_progress=on_progress,
        on_file_complete=on_file_complete,
    )
    result["resumed_files"] = resumed
    return result


# ---------------------------------------------------------------------------
# Verification (SHA-256 primitives live in downcraft.verify)
# ---------------------------------------------------------------------------

def _find_snapshot_dir(
    model_id: str,
    hf_home: Optional[str] = None,
) -> Optional[Path]:
    """Find the snapshot directory for a cached model."""
    cache_dir = get_cache_dir(model_id, hf_home)
    if not cache_dir.exists():
        return None
    refs_main = cache_dir / "refs" / "main"
    if not refs_main.exists():
        return None
    commit = refs_main.read_text().strip()
    if not commit:
        return None
    snap = cache_dir / "snapshots" / commit
    return snap if snap.exists() else None


def verify_model(
    model_id: str,
    hf_home: Optional[str] = None,
) -> bool:
    """Verify all weight files in a downloaded model against their SHA-256 checksums.

    Returns True if all files pass, False otherwise.
    """
    snap = _find_snapshot_dir(model_id, hf_home)
    if snap is None:
        logger.error("Model %s not found in cache", model_id)
        return False

    files = list_model_files(model_id)
    weight_files = [f for f in files if not f.is_ignored and f.checksum]

    if not weight_files:
        logger.warning("No files with checksums to verify for %s", model_id)
        return False

    all_ok = True
    for hf_file in weight_files:
        local_path = snap / hf_file.path
        if not local_path.exists():
            logger.error("Missing: %s", hf_file.path)
            all_ok = False
            continue

        try:
            actual = _sha256_of(local_path)
        except OSError as e:
            logger.error("Read error %s: %s", hf_file.path, e)
            all_ok = False
            continue

        if actual != hf_file.checksum:
            logger.error(
                "Checksum mismatch: %s (expected %s, got %s)",
                hf_file.path,
                hf_file.checksum[:16],
                actual[:16],
            )
            all_ok = False
        else:
            logger.debug("OK: %s", hf_file.path)

    if all_ok:
        logger.info("All %d files verified ✓", len(weight_files))
    return all_ok


def list_missing_files(
    model_id: str,
    hf_home: Optional[str] = None,
) -> List[str]:
    """List files that are missing or have incorrect checksums."""
    snap = _find_snapshot_dir(model_id, hf_home)
    if snap is None:
        files = list_model_files(model_id)
        return [f.path for f in files if not f.is_ignored]

    files = list_model_files(model_id)
    missing = []
    for hf_file in files:
        if hf_file.is_ignored:
            continue
        local_path = snap / hf_file.path
        if not local_path.exists():
            missing.append(hf_file.path)
        elif hf_file.checksum and not verify_file(local_path, hf_file.checksum):
            missing.append(hf_file.path)
    return missing
