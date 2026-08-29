"""
File integrity verification — SHA-256 checksum helpers.

This module is deliberately HuggingFace-agnostic.  It verifies a single
file against an expected SHA-256 checksum.  Model-level verification
(locating snapshots, listing missing weight files) lives in the
application layer (``domains.infrastructure.hf_hub``).
"""

import hashlib
import logging
from pathlib import Path

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
