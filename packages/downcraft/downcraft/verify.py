"""
Integrity verification — checks that every weight file in a downloaded model
matches its expected SHA-256 checksum from the HF Hub LFS metadata.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

from . import hf_hub

logger = logging.getLogger(__name__)

CHUNK_SIZE = 64 * 1024 * 1024  # 64 MB


def _sha256_of(path: Path) -> str:
    """Compute SHA-256 of a file, reading in chunks to handle large files."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _find_snapshot_dir(
    model_id: str,
    hf_home: Optional[str] = None,
) -> Optional[Path]:
    """Find the snapshot directory for a cached model."""
    cache_dir = hf_hub.get_cache_dir(model_id, hf_home)
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

    files = hf_hub.list_model_files(model_id)
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


def verify_file(
    file_path: Path,
    expected_checksum: str,
) -> bool:
    """Verify a single file against its expected SHA-256 checksum."""
    try:
        actual = _sha256_of(file_path)
        return actual == expected_checksum
    except OSError:
        return False


def list_missing_files(
    model_id: str,
    hf_home: Optional[str] = None,
) -> List[str]:
    """List files that are missing or have incorrect checksums."""
    snap = _find_snapshot_dir(model_id, hf_home)
    if snap is None:
        files = hf_hub.list_model_files(model_id)
        return [f.path for f in files if not f.is_ignored]

    files = hf_hub.list_model_files(model_id)
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
