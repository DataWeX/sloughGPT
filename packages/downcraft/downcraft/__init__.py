"""
downcraft — Generic HTTP/HTTPS downloader with cross-session resume
via HTTP Range headers and persistent JSON state.

Resumes partial downloads even after power loss, process crash, or
days-long gaps between sessions — as long as the partial file on disk
and server's ``ETag`` still match.

Use cases:
    1. Download any URL::

        download("https://example.com/bigfile.iso", "/tmp/bigfile.iso")

    2. Download a HuggingFace model (convenience plugin)::

        download_hf_model("Qwen/Qwen2.5-0.5B-Instruct")

    3. CLI::

        python -m downcraft url https://...
        python -m downcraft hf Qwen/Qwen2.5-0.5B-Instruct
"""

import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from . import downloader, state
from . import hf_hub
from . import resume
from . import verify

logger = logging.getLogger(__name__)

__all__ = [
    "download",
    "download_hf_model",
    "hf_hub",
    "resume",
    "state",
    "downloader",
    "verify",
    "inspect_incomplete",
    "resume_download",
    "resume_plan",
    "resume_model",
]


# ---------------------------------------------------------------------------
# Generic download — any URL
# ---------------------------------------------------------------------------

def download(
    url: str,
    dest: Union[str, Path],
    expected_size: int = 0,
    checksum: str = "",
    label: str = "",
    on_progress: Optional[Callable[[int, int, float], None]] = None,
) -> Dict:
    """Download a single file from any URL with cross-session resume.

    Uses ``~/.downcraft/state.json`` to track progress so that
    if the process dies mid-download, the next call resumes from
    the last byte received (via HTTP ``Range`` header).

    Args:
        url: HTTP/HTTPS URL to download.
        dest: Local destination path.
        expected_size: Expected total bytes (0 = auto-detect from server).
        checksum: SHA-256 hex string to verify after download.
        label: Human label for logging (defaults to filename).
        on_progress: Called per-chunk with ``(bytes_downloaded, total_bytes, speed_bps)``.

    Returns:
        Dict with keys: ``status``, ``dest``, ``elapsed``, ``total_bytes``.
    """
    dest = Path(dest)
    label = label or dest.name
    st = state.get_state()
    lookup_key = url  # use URL as the state tracking key

    # Check existing state
    existing = st.get(lookup_key)
    if existing and existing.status == "complete":
        logger.info("%s already downloaded", label)
        return {"status": "already_downloaded", "dest": str(dest), "label": label}

    start = time.time()
    total_bytes = [0]

    def _chunk_cb(bytes_done: int, total: int):
        nonlocal total_bytes
        total_bytes[0] = max(total_bytes[0], total)
        st.update_file_progress(
            lookup_key,
            str(dest),
            url,
            bytes_done,
            max(bytes_done, total),
            checksum=checksum,
            complete=(bytes_done >= total and total > 0),
        )
        elapsed = time.time() - start
        speed = bytes_done / elapsed if elapsed > 0 else 0
        if on_progress:
            on_progress(bytes_done, max(bytes_done, total), speed)

    st.create(lookup_key, str(dest.parent))

    try:
        downloader.download_file(
            url=url,
            dest=dest,
            expected_size=expected_size,
            checksum=checksum,
            on_chunk=_chunk_cb,
        )
    except downloader.DownloadError:
        st.set_status(lookup_key, "failed", error=f"Failed to download {label}")
        st.flush()
        raise

    st.set_status(lookup_key, "complete")
    st.flush()
    elapsed = time.time() - start
    return {
        "status": "complete",
        "dest": str(dest),
        "elapsed": round(elapsed, 1),
        "total_bytes": total_bytes[0],
    }


# ---------------------------------------------------------------------------
# HuggingFace model download (one use case)
# ---------------------------------------------------------------------------

def download_hf_model(
    model_id: str,
    hf_home: Optional[str] = None,
    on_progress: Optional[Callable[[str, int, int, float], None]] = None,
    on_file_complete: Optional[Callable[[str, str], None]] = None,
    ignore_cache: bool = False,
) -> Dict:
    """Download a HuggingFace model with cross-session resume.

    Wraps ``download()`` for each file in the model repo, using
    the ``~/.downcraft/state.json`` persistent tracker so that
    multi-file model downloads resume across restarts.

    Args:
        model_id: HF model ID (e.g. ``"gpt2"``, ``"Qwen/Qwen2.5-0.5B-Instruct"``).
        hf_home: Override HF cache directory.
        on_progress: Called per-chunk with ``(model_id, bytes_downloaded, total_bytes, speed_bps)``.
        on_file_complete: Called when each file finishes ``(model_id, file_path)``.
        ignore_cache: If True, redownload even if already fully cached.

    Returns:
        Dict with keys: ``status``, ``cache_dir``, ``elapsed``, ``total_bytes``.
    """
    # Resolve cache directory.  An explicit ``hf_home`` is the hub root
    # (``<hf_home>/models--<id>``); ``None`` resolves via the HF_HOME env
    # var with ``/hub`` appended — the standard HF layout that
    # is_download_complete and the app's safetensors_loader expect.
    cache_dir = str(hf_hub.get_cache_dir(model_id, hf_home))

    st = state.get_state()

    # Quick check — already complete?  State is a hint only; disk is truth.
    existing = st.get(model_id)
    if existing and existing.status == "complete" and not ignore_cache:
        if hf_hub.is_download_complete(model_id, hf_home=hf_home):
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

    files = hf_hub.list_model_files(model_id)
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
            downloader.download_file(
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
        except downloader.DownloadError:
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
    the ref lets :func:`downcraft.hf_hub.is_download_complete` resolve the
    snapshot dir and mark the model complete.
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
