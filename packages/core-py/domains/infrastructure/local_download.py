"""
Local file download backend — copies files from local paths.

Useful for testing, development, and importing models from local directories.
Unlike other backends, this one copies files instead of downloading them.

Usage::

    from domains.infrastructure.local_download import LocalFileBackend

    backend = LocalFileBackend(source_dir="/path/to/models")
    backend.download("my-model", on_progress=..., on_file_complete=...)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from domains.infrastructure.download_backend import DownloadBackend, FileEstimate

logger = logging.getLogger("slo.local_download")


def _get_cache_root() -> Path:
    """Return the root directory for local model copies."""
    return Path(os.environ.get("SLO_CACHE_DIR", Path.home() / ".cache" / "sloughgpt")) / "local"


class LocalFileBackend(DownloadBackend):
    """Download backend for local file copies.

    Resource IDs map to subdirectories under ``source_dir``.
    Files are copied (not downloaded) to the cache directory.

    Supports SGZ1 compression when source files are .sgz files.
    """

    def __init__(self, source_dir: str, *, compressed: bool = False):
        self._source_dir = Path(source_dir)
        self._compressed = compressed

    def _cache_dir(self, resource_id: str) -> Path:
        safe_name = resource_id.replace("/", "__")
        return _get_cache_root() / safe_name

    def _source_model_dir(self, resource_id: str) -> Path:
        return self._source_dir / resource_id

    def _manifest_path(self, resource_id: str) -> Path:
        return self._cache_dir(resource_id) / ".manifest.json"

    def _load_manifest(self, resource_id: str) -> Dict[str, Any]:
        path = self._manifest_path(resource_id)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return {}

    def _save_manifest(self, resource_id: str, manifest: Dict[str, Any]) -> None:
        path = self._manifest_path(resource_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2))

    def _scan_source(self, resource_id: str) -> Optional[List[Dict]]:
        """Scan source directory for files."""
        source = self._source_model_dir(resource_id)
        if not source.exists():
            return None

        files = []
        for f in sorted(source.rglob("*")):
            if f.is_file():
                rel = str(f.relative_to(source))
                sha256 = hashlib.sha256(f.read_bytes()).hexdigest()
                files.append({
                    "path": rel,
                    "size": f.stat().st_size,
                    "sha256": sha256,
                })
        return files

    def is_cached(self, resource_id: str, deep_check: bool = False) -> bool:
        cache_dir = self._cache_dir(resource_id)
        if not cache_dir.exists():
            return False
        manifest = self._load_manifest(resource_id)
        if not manifest:
            return False
        files = manifest.get("files", [])
        if not files:
            return False
        for f in files:
            fp = cache_dir / f["path"]
            if not fp.exists():
                return False
            if deep_check and fp.stat().st_size != f.get("size", -1):
                return False
        return True

    def get_cache_dir(self, resource_id: str) -> str:
        return str(self._cache_dir(resource_id))

    def estimate_total(self, resource_id: str) -> int:
        manifest = self._load_manifest(resource_id)
        if manifest:
            return sum(f.get("size", 0) for f in manifest.get("files", []))
        # Scan source if no manifest
        files = self._scan_source(resource_id)
        if files:
            return sum(f.get("size", 0) for f in files)
        return 0

    def list_files(self, resource_id: str) -> List[FileEstimate]:
        manifest = self._load_manifest(resource_id)
        files = manifest.get("files", [])
        if not files:
            files = self._scan_source(resource_id) or []
        return [
            FileEstimate(
                path=f["path"],
                size=f.get("size", 0),
                checksum=f.get("sha256", ""),
                download_url=f"file://{self._source_model_dir(resource_id) / f['path']}",
            )
            for f in files
        ]

    def download(
        self,
        resource_id: str,
        on_progress: Callable[[str, int, int, float], None],
        on_file_complete: Callable[[str, str], None],
    ) -> Dict:
        """Copy files from source directory to cache."""
        files = self._scan_source(resource_id)
        if files is None:
            return {
                "status": "error",
                "cache_dir": str(self._cache_dir(resource_id)),
                "error": f"Source directory not found: {self._source_model_dir(resource_id)}",
            }

        if not files:
            return {
                "status": "error",
                "cache_dir": str(self._cache_dir(resource_id)),
                "error": "No files found in source directory",
            }

        cache_dir = self._cache_dir(resource_id)
        cache_dir.mkdir(parents=True, exist_ok=True)
        source_dir = self._source_model_dir(resource_id)

        total_size = sum(f.get("size", 0) for f in files)
        bytes_done = 0

        self._save_manifest(resource_id, {"files": files})

        for f in files:
            src = source_dir / f["path"]
            dest = cache_dir / f["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)

            file_size = f.get("size", 0)

            def _progress(bytes_written: int, expected: int, _file=f):
                current = bytes_done + bytes_written
                on_progress(resource_id, current, total_size, 0)

            # Copy file with progress
            with open(src, "rb") as fin, open(dest, "wb") as fout:
                copied = 0
                while True:
                    chunk = fin.read(65536)
                    if not chunk:
                        break
                    fout.write(chunk)
                    copied += len(chunk)
                    _progress(copied, file_size)

            bytes_done += file_size
            on_file_complete(resource_id, str(dest))

        return {
            "status": "completed",
            "cache_dir": str(cache_dir),
            "total_bytes": total_size,
        }

    def cleanup(self, resource_id: str) -> bool:
        cache_dir = self._cache_dir(resource_id)
        if not cache_dir.exists():
            return False
        logger.warning("Removing local cache for %s: %s", resource_id, cache_dir)
        shutil.rmtree(str(cache_dir), ignore_errors=True)
        return True

    def list_incomplete(self) -> List[str]:
        root = _get_cache_root()
        if not root.exists():
            return []
        result: List[str] = []
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            manifest = self._load_manifest(entry.name)
            if not manifest:
                result.append(entry.name)
                continue
            files = manifest.get("files", [])
            if any(not (entry / f["path"]).exists() for f in files):
                result.append(entry.name)
        return result

    def supports_compression(self, resource_id: str) -> bool:
        return False  # Local copies don't need compression

    def supports_compressed_serve(self) -> bool:
        return False
