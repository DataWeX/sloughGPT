"""
Resume analysis for incomplete downloads of model files.

Given an incomplete download of a model file (a ``*.sgpart`` or
``*.incomplete`` temp file left behind by a crashed or interrupted
process, or a final-name file smaller than its expected size), these
helpers determine:

* which file of the model repo the partial belongs to (the "part"),
* the exact byte offset to resume from (the partial's size),
* the final path, download URL, expected size, and checksum.

The matching is programmatic: the partial's name (minus its
``.sgpart`` / ``.incomplete`` suffix) is compared against the file
list reported by :func:`downcraft.hf_hub.list_model_files`.  The model
id is taken from the caller or derived from the ``models--<id>``
cache directory found by walking up from the partial's path.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from . import downloader, hf_hub
from .downloader import DownloadError

logger = logging.getLogger(__name__)

#: Suffixes downcraft / HuggingFace use for in-progress temp files.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _match_repo_file(final_name: str, repo_files: List[hf_hub.HFFile]):
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inspect_incomplete(
    partial_path: Union[str, Path],
    model_id: Optional[str] = None,
    hf_home: Optional[str] = None,
    files: Optional[List[hf_hub.HFFile]] = None,
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
            :func:`downcraft.hf_hub.list_model_files` is called.

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
        files = hf_hub.list_model_files(model_id)

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
    files: Optional[List[hf_hub.HFFile]] = None,
) -> List[ResumeInfo]:
    """List every incomplete file of a model with its resume offset.

    Scans the model's cache directory for ``*.sgpart`` and ``*.incomplete``
    markers, plus final-name files that are smaller than their expected
    size (interrupted non-temp writes).

    Args:
        model_id: Model id to scan.
        hf_home: HF cache base (or any project-local cache root).
        files: Pre-fetched repo file list.  When omitted,
            :func:`downcraft.hf_hub.list_model_files` is called.

    Returns:
        List of :class:`ResumeInfo`, sorted by repo path, one entry per
        incomplete file.  Empty when the model is not cached or complete.
    """
    cache_dir = hf_hub.find_cached_model_dir(model_id, hf_home)
    if cache_dir is None:
        return []
    if not cache_dir.is_dir():
        return []

    if files is None:
        files = hf_hub.list_model_files(model_id)
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
    files: Optional[List[hf_hub.HFFile]] = None,
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
    downloader.download_file(
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

    Equivalent to :func:`downcraft.download_hf_model` but driven by the
    resume plan: each incomplete file found on disk is resumed at its
    exact offset first (the "patch out of the parts" step), then any
    remaining files are downloaded fresh.

    Args:
        model_id: Model id to resume.
        hf_home: HF cache base.
        on_progress: Called per chunk with ``(model_id, bytes, total, speed)``.
        on_file_complete: Called as ``(model_id, repo_path)`` per file.

    Returns:
        Dict with keys ``status``, ``model_id``, ``cache_dir``,
        ``elapsed``, ``total_bytes``, ``resumed_files``.
    """
    from . import download_hf_model

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
