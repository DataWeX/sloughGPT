"""
PermissionsManager — CLI-side download confirmation and size-aware prompts.

Enforces the bandwidth policy: never download large files without user
confirmation. Queries HuggingFace Hub for model size, shows an estimate
with the download details, and prompts the user to confirm.

Usage::

    from core.permissions import PermissionsManager

    pm = PermissionsManager()
    if pm.confirm_download("gpt2"):
        # proceed with download
"""

import os
import sys
import logging
from dataclasses import dataclass
from typing import Optional

import click

from domains.logging import get_global

log = get_global()
from utils.formatting import format_size

logger = logging.getLogger("slo.cli.permissions")

# Threshold: files under 50MB skip confirmation (AGENTS.md policy)
_AUTO_APPROVE_THRESHOLD_MB = 50


@dataclass
class ModelSizeEstimate:
    """Estimated size of a model before download."""

    model_id: str
    total_bytes: int
    file_count: int
    files: list

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 * 1024 * 1024)

    @property
    def human_size(self) -> str:
        if self.total_mb >= 1024:
            return f"{self.total_gb:.2f} GB"
        return f"{self.total_mb:.1f} MB"


class PermissionsManager:
    """Handles download confirmations, size estimation, and permission prompts.

    Enforces bandwidth policy by querying HuggingFace Hub for model metadata,
    displaying size estimates, and requiring user confirmation before downloads.

    Resolution order for auto_download:
    1. Constructor ``auto_yes=True`` (from CLI ``--yes`` flag)
    2. ``SLO_AUTO_DOWNLOAD=1`` env var
    3. ``features.auto_download`` in AppConfig (persistent, settable via shell)
    """

    def __init__(self, auto_yes: bool = False):
        if auto_yes:
            self.auto_yes = True
        elif os.environ.get("SLO_AUTO_DOWNLOAD", "") == "1":
            self.auto_yes = True
        else:
            try:
                from domains.infrastructure.config import get_config
                self.auto_yes = get_config().features.auto_download
            except (ImportError, AttributeError):
                self.auto_yes = False

    def estimate_model_size(self, model_id: str) -> Optional[ModelSizeEstimate]:
        """Query HuggingFace Hub API for model file list and total size.

        Args:
            model_id: HuggingFace model ID (e.g. ``gpt2``, ``Qwen/Qwen2.5-0.5B-Instruct``)

        Returns:
            ModelSizeEstimate with total bytes and file list, or None on failure.
        """
        try:
            from domains.infrastructure.hf_hub import fetch_model_info

            info = fetch_model_info(model_id)
            if info is None:
                return None

            files = []
            total = 0
            siblings = info.get("siblings") or []
            for sf in siblings:
                if not isinstance(sf, dict):
                    continue
                size = sf.get("size") or 0
                if size > 0:
                    files.append({"name": sf.get("rfilename", ""), "size": size})
                    total += size

            if total == 0:
                return None

            return ModelSizeEstimate(
                model_id=model_id,
                total_bytes=total,
                file_count=len(files),
                files=files,
            )
        except Exception as e:
            logger.debug("Could not estimate model size for %s: %s", model_id, e)
            return None

    def confirm_download(self, model_id: str, *, force: bool = False) -> bool:
        """Prompt user to confirm a model download.

        Shows download details with model ID, estimated size, and file count.
        Respects ``--yes`` flag, ``SLO_AUTO_DOWNLOAD`` env var, and the
        50 MB auto-approve threshold.

        Args:
            model_id: HuggingFace model ID to potentially download.
            force: If True, skip confirmation regardless of size.

        Returns:
            True if the user approved the download (or auto-approved).
        """
        if force or self.auto_yes:
            return True

        # Check if already cached
        if self._is_cached(model_id):
            return True

        # Estimate size
        log.step(f"Checking download size for {model_id}...")
        estimate = self.estimate_model_size(model_id)

        if estimate is None:
            # Can't determine size — ask with warning
            log.warning("Could not determine download size (network issue or private model)")
            return click.confirm(
                f"Download '{model_id}'? (size unknown)",
                default=True,
            )

        # Auto-approve small downloads
        if estimate.total_mb < _AUTO_APPROVE_THRESHOLD_MB:
            log.info(f"Small download ({estimate.human_size}) — auto-approving")
            return True

        # Large download — show details and prompt
        self._show_download_panel(estimate)
        return click.confirm("Proceed with download?", default=False)

    def confirm_autoload_download(self, model_id: str) -> bool:
        """Confirm download during server autoload (non-interactive-safe).

        Used by the ``serve`` command to check before starting the server
        if the model needs to be downloaded.

        Args:
            model_id: Model ID that would be downloaded on server start.

        Returns:
            True if approved, False to abort server start.
        """
        if self.auto_yes:
            return True

        if self._is_cached(model_id):
            return True

        estimate = self.estimate_model_size(model_id)

        if estimate and estimate.total_mb >= _AUTO_APPROVE_THRESHOLD_MB:
            self._show_download_panel(estimate, context="server start")
            return click.confirm(
                f"Server will download {estimate.human_size} on startup. Continue?",
                default=True,
            )

        return True

    def check_cached(self, model_id: str) -> bool:
        """Check if a model is already cached locally.

        Args:
            model_id: HuggingFace model ID.

        Returns:
            True if model weights are present in local cache.
        """
        return self._is_cached(model_id)

    def _is_cached(self, model_id: str) -> bool:
        """Check if model has weight files in HF cache."""
        try:
            from domains.infrastructure.download_manager import (
                _cache_dir,
                _has_weight_files,
                _has_complete_snapshot,
            )

            cache = _cache_dir(model_id)
            if cache.exists() and _has_complete_snapshot(cache):
                return True
            if cache.exists() and _has_weight_files(cache):
                return True
        except (ImportError, OSError):
            pass

        # Fallback: check standard HF cache location
        cache_path = (
            os.path.expanduser("~/.cache/huggingface/hub")
            / f"models--{model_id.replace('/', '--')}"
        )
        if os.path.isdir(cache_path):
            for ext in ("*.safetensors", "*.bin", "*.slnc"):
                for f in __import__("pathlib").Path(cache_path).rglob(ext):
                    try:
                        if f.stat().st_size > 1024:
                            return True
                    except OSError:
                        continue
        return False

    def _show_download_panel(self, estimate: ModelSizeEstimate, *, context: str = ""):
        """Display download details with ANSI formatting."""
        import sys

        _tty = sys.stdout.isatty()
        def _c(text, code):
            return f"{code}{text}\033[0m" if _tty else text
        _BOLD = "\033[1m"
        _DIM = "\033[2m"
        _YELLOW = "\033[33m"

        _p = sys.stdout.write
        _flush = sys.stdout.flush

        def _line(text=""):
            _p(text + "\n")
            _flush()

        title = "Download Required"
        if context:
            title = f"Download Required ({context})"

        _line()
        _line(f"  {_c(title, _BOLD + _YELLOW)}")
        _line(f"  {'─' * 40}")
        _line(f"    {_c('Model:', _DIM)} {estimate.model_id}")
        _line(f"    {_c('Size:', _DIM)} {estimate.human_size}")
        _line(f"    {_c('Files:', _DIM)} {estimate.file_count}")

        # Show top 5 largest files
        if estimate.files:
            sorted_files = sorted(estimate.files, key=lambda x: x["size"], reverse=True)[:5]
            top_files = ", ".join(
                f"{f['name'].split('/')[-1]} ({format_size(f['size'])})"
                for f in sorted_files
            )
            _line(f"    {_c('Largest:', _DIM)} {top_files}")
