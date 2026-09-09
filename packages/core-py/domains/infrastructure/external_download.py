"""
External download backend — downloads models from arbitrary HTTP servers.

Unlike ``HFDownloadBackend`` which talks to HuggingFace Hub, this backend
works with any HTTP server that serves model files.  When the server
supports SGZ1 compression, downloads use ``CompressedDownloader`` for
on-the-fly decompression.

Usage::

    from domains.infrastructure.external_download import ExternalDownloadBackend

    backend = ExternalDownloadBackend(base_url="http://192.168.1.100:8000")
    backend.download("my-model", on_progress=..., on_file_complete=...)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urljoin

from domains.infrastructure.download_backend import DownloadBackend, FileEstimate

logger = logging.getLogger("slo.external_download")


def _get_cache_root() -> Path:
    """Return the root directory for external model downloads."""
    return Path(os.environ.get("SLO_CACHE_DIR", Path.home() / ".cache" / "sloughgpt")) / "external"


class ExternalDownloadBackend(DownloadBackend):
    """Download backend for arbitrary HTTP servers.

    Resource IDs are plain names (e.g. ``my-model``, ``org/model-name``).
    The backend resolves them to ``{base_url}/models/{resource_id}/...``.

    Supports SGZ1 compression when the server advertises it.
    """

    def __init__(self, base_url: str, *, compressed: bool = True):
        self._base_url = base_url.rstrip("/")
        self._compressed = compressed

    def _cache_dir(self, resource_id: str) -> Path:
        safe_name = resource_id.replace("/", "__")
        return _get_cache_root() / safe_name

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

    def _model_url(self, resource_id: str) -> str:
        return f"{self._base_url}/models/{resource_id}"

    def _file_url(self, resource_id: str, file_path: str) -> str:
        return f"{self._model_url(resource_id)}/file/{file_path}"

    def _manifest_url(self, resource_id: str) -> str:
        return f"{self._model_url(resource_id)}/manifest.json"

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
        return sum(f.get("size", 0) for f in manifest.get("files", []))

    def list_files(self, resource_id: str) -> List[FileEstimate]:
        manifest = self._load_manifest(resource_id)
        files = manifest.get("files", [])
        return [
            FileEstimate(
                path=f["path"],
                size=f.get("size", 0),
                checksum=f.get("sha256", ""),
                download_url=self._file_url(resource_id, f["path"]),
            )
            for f in files
        ]

    def _fetch_manifest(self, resource_id: str) -> Optional[Dict]:
        """Download the file manifest from the server."""
        import urllib.request
        import urllib.error

        url = self._manifest_url(resource_id)
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            logger.warning("Failed to fetch manifest for %s: %s", resource_id, e)
            return None
        except Exception as e:
            logger.warning("Failed to fetch manifest for %s: %s", resource_id, e)
            return None

    def download(
        self,
        resource_id: str,
        on_progress: Callable[[str, int, int, float], None],
        on_file_complete: Callable[[str, str], None],
    ) -> Dict:
        """Download all files for a resource from the external server.

        Fetches the manifest first, then downloads each file.
        Uses SGZ1 compression when the server supports it.
        """
        manifest = self._fetch_manifest(resource_id)
        if manifest is None:
            return {
                "status": "error",
                "cache_dir": str(self._cache_dir(resource_id)),
                "error": f"Could not fetch manifest from {self._manifest_url(resource_id)}",
            }

        files = manifest.get("files", [])
        if not files:
            return {
                "status": "error",
                "cache_dir": str(self._cache_dir(resource_id)),
                "error": "Manifest contains no files",
            }

        cache_dir = self._cache_dir(resource_id)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._save_manifest(resource_id, manifest)

        total_size = sum(f.get("size", 0) for f in files)
        bytes_done = 0
        start_time = time.monotonic()

        for f in files:
            file_path = f["path"]
            file_size = f.get("size", 0)
            file_sha256 = f.get("sha256", "")
            dest = cache_dir / file_path
            dest.parent.mkdir(parents=True, exist_ok=True)

            url = self._file_url(resource_id, file_path)

            def _progress(bytes_written: int, expected: int, _file=f):
                current = bytes_done + bytes_written
                elapsed = max(time.monotonic() - start_time, 0.001)
                speed = current / elapsed
                on_progress(resource_id, current, total_size, speed)

            if self._compressed:
                from domains.infrastructure.compressed_transfer import CompressedDownloader

                downloader = CompressedDownloader()
                result = downloader.download_from_url(
                    url,
                    dest,
                    expected_sha256=file_sha256 or None,
                    on_progress=_progress,
                )
                if not result.success:
                    return {
                        "status": "error",
                        "cache_dir": str(cache_dir),
                        "error": result.error,
                    }
            else:
                import urllib.request

                try:
                    req = urllib.request.Request(url)
                    with urllib.request.urlopen(req, timeout=300) as resp:
                        with open(dest, "wb") as out:
                            while True:
                                chunk = resp.read(65536)
                                if not chunk:
                                    break
                                out.write(chunk)
                                _progress(len(chunk), file_size)
                except Exception as e:
                    return {
                        "status": "error",
                        "cache_dir": str(cache_dir),
                        "error": str(e),
                    }

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
        logger.warning("Removing external cache for %s: %s", resource_id, cache_dir)
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
        return self._compressed

    def supports_compressed_serve(self) -> bool:
        return False
