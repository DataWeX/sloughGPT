"""
Git download backend — clones git repositories.

Useful for downloading models distributed via git (e.g., HuggingFace repos, GitHub releases).

Usage::

    from domains.infrastructure.git_download import GitBackend

    backend = GitBackend()
    backend.download("repo/model-name", on_progress=..., on_file_complete=...)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from domains.infrastructure.download_backend import DownloadBackend, FileEstimate

logger = logging.getLogger("slo.git_download")


def _get_cache_root() -> Path:
    """Return the root directory for git clones."""
    return Path(os.environ.get("SLO_CACHE_DIR", Path.home() / ".cache" / "sloughgpt")) / "git"


class GitBackend(DownloadBackend):
    """Download backend for git repositories.

    Resource IDs are parsed as ``[org]/[repo][@branch]``.
    Supports private repos via SSH keys or HTTPS tokens.

    Supports SGZ1 compression when files in the repo are .sgz files.
    """

    def __init__(self, *, ssh_key: Optional[str] = None, token: Optional[str] = None):
        self._ssh_key = ssh_key
        self._token = token

    def _parse_resource(self, resource_id: str) -> tuple[str, str, Optional[str]]:
        """Parse 'org/repo@branch' into (org, repo, branch)."""
        branch = None
        ref = resource_id
        if "@" in ref:
            ref, branch = ref.rsplit("@", 1)

        parts = ref.split("/")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid git resource format: {resource_id!r}. "
                "Expected 'org/repo[@branch]'."
            )
        org = parts[0]
        repo = parts[1]
        return org, repo, branch

    def _clone_url(self, org: str, repo: str) -> str:
        """Build clone URL. Uses HTTPS with token if provided."""
        if self._token:
            return f"https://x-access-token:{self._token}@github.com/{org}/{repo}.git"
        return f"https://github.com/{org}/{repo}.git"

    def _cache_dir(self, resource_id: str) -> Path:
        safe_name = resource_id.replace("/", "__").replace("@", "_at_")
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

    def _scan_repo(self, cache_dir: Path) -> List[Dict]:
        """Scan cloned repo for files."""
        files = []
        for f in sorted(cache_dir.rglob("*")):
            if not f.is_file():
                continue
            # Skip .git directory
            try:
                f.relative_to(cache_dir / ".git")
                continue
            except ValueError:
                pass
            rel = str(f.relative_to(cache_dir))
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
        return 0

    def list_files(self, resource_id: str) -> List[FileEstimate]:
        manifest = self._load_manifest(resource_id)
        files = manifest.get("files", [])
        cache_dir = self._cache_dir(resource_id)
        return [
            FileEstimate(
                path=f["path"],
                size=f.get("size", 0),
                checksum=f.get("sha256", ""),
                download_url=f"file://{cache_dir / f['path']}",
            )
            for f in files
        ]

    def _run_git(self, args: list[str], cwd: str) -> str:
        """Run a git command and return output."""
        env = os.environ.copy()
        if self._ssh_key:
            env["GIT_SSH_COMMAND"] = f"ssh -i {self._ssh_key}"

        cmd = ["git"] + args
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Git command failed: {' '.join(cmd)}\n{result.stderr}")
        return result.stdout

    def download(
        self,
        resource_id: str,
        on_progress: Callable[[str, int, int, float], None],
        on_file_complete: Callable[[str, str], None],
    ) -> Dict:
        """Clone git repository."""
        org, repo, branch = self._parse_resource(resource_id)
        cache_dir = self._cache_dir(resource_id)
        clone_url = self._clone_url(org, repo)

        on_progress(resource_id, 0, 1, 0)

        try:
            if cache_dir.exists():
                logger.info("Updating git repo %s", resource_id)
                on_progress(resource_id, 0, 1, 0)
                # Fetch and reset
                self._run_git(["fetch", "--all"], str(cache_dir))
                if branch:
                    self._run_git(["checkout", branch], str(cache_dir))
                self._run_git(["pull"], str(cache_dir))
            else:
                logger.info("Cloning git repo %s → %s", resource_id, cache_dir)
                cache_dir.parent.mkdir(parents=True, exist_ok=True)
                cmd = ["git", "clone", clone_url]
                if branch:
                    cmd.extend(["--branch", branch])
                cmd.append(str(cache_dir))
                env = os.environ.copy()
                if self._ssh_key:
                    env["GIT_SSH_COMMAND"] = f"ssh -i {self._ssh_key}"
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=600,
                )
                if result.returncode != 0:
                    return {
                        "status": "error",
                        "cache_dir": str(cache_dir),
                        "error": f"Git clone failed: {result.stderr}",
                    }

            on_progress(resource_id, 1, 1, 0)

            # Scan files
            files = self._scan_repo(cache_dir)
            self._save_manifest(resource_id, {"files": files, "branch": branch})

            # Notify file completions
            for f in files:
                on_file_complete(resource_id, str(cache_dir / f["path"]))

            return {
                "status": "completed",
                "cache_dir": str(cache_dir),
                "total_bytes": sum(f.get("size", 0) for f in files),
            }
        except Exception as e:
            return {
                "status": "error",
                "cache_dir": str(cache_dir),
                "error": str(e),
            }

    def cleanup(self, resource_id: str) -> bool:
        import shutil

        cache_dir = self._cache_dir(resource_id)
        if not cache_dir.exists():
            return False
        logger.warning("Removing git cache for %s: %s", resource_id, cache_dir)
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
        return False

    def supports_compressed_serve(self) -> bool:
        return False
