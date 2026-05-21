"""
PyTorch process bootstrap for SloughGPT.

Centralizes environment tweaks that prevent common hangs and surprises:

* **FastAPI API process** (`apply_api_process_torch_env`): historically forced CPU
  everywhere, which broke CUDA on Linux. We only narrow the device on **macOS**
  by default (MPS + forked workers is a frequent hang source in dev); Linux
  leaves CUDA/MPS untouched unless you set overrides below.

* **DataLoader workers** (`effective_dataloader_num_workers`): on **darwin**,
  multiprocessing ``DataLoader`` workers often deadlock with MPS; use ``0``.

Environment (API / import-time, read before ``import torch``):

* ``SLOUGHGPT_SKIP_TORCH_ENV`` — if set (non-empty), do nothing in
  ``apply_api_process_torch_env``.
* ``SLOUGHGPT_API_ENABLE_MPS`` — on macOS, allow MPS (default: unset → MPS
  disabled for API process only).
* ``PYTORCH_DISABLE_MPS`` / ``CUDA_VISIBLE_DEVICES`` — if already set, we do
  not override (you own the process).
* ``SLOUGHGPT_USE_TORCH_SHIM`` — if set (non-empty), replace ``torch`` with
  the numpy-backed shim in ``domains/training/torch/``. No GPU, no downloads.
  For inference-only deployments or when PyTorch is not installed.

Shell (Intel Mac + AMD / library inject quirks)::

    export DYLD_INSERT_LIBRARIES=""

See also: ``QUICKSTART.md`` (Troubleshooting → macOS PyTorch hangs).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_api_torch_env_applied = False


def apply_api_process_torch_env() -> None:
    """Tune env vars before ``import torch`` in the FastAPI app only.

    Safe to call multiple times; applies at most once per process.
    """
    global _api_torch_env_applied
    if os.environ.get("SLOUGHGPT_SKIP_TORCH_ENV"):
        return
    if _api_torch_env_applied:
        return
    _api_torch_env_applied = True

    if os.environ.get("SLOUGHGPT_USE_TORCH_SHIM"):
        _inject_torch_shim()

    if os.environ.get("PYTORCH_DISABLE_MPS") is not None:
        pass
    elif sys.platform == "darwin" and not os.environ.get("SLOUGHGPT_API_ENABLE_MPS"):
        os.environ.setdefault("PYTORCH_DISABLE_MPS", "1")


def _inject_torch_shim() -> None:
    """Replace torch with the numpy-backed shim.

    Adds ``domains/training/`` to sys.path so that ``import torch`` resolves
    to the shim package at ``domains/training/torch/``. Must be called
    before any ``import torch`` statement.
    """
    shim_root = Path(__file__).parent.parent / "training"
    shim_path = str(shim_root.resolve())
    if shim_path not in sys.path:
        sys.path.insert(0, shim_path)
    for name in list(sys.modules.keys()):
        if name == "torch" or name.startswith("torch."):
            del sys.modules[name]

    # Do not set CUDA_VISIBLE_DEVICES or PYTORCH_NO_CUDA here — that broke
    # Linux GPU deployments when done unconditionally.


def effective_dataloader_num_workers(requested: Any) -> int:
    """Return a safe worker count (``0`` on macOS to avoid fork+MPS hangs)."""
    try:
        n = int(requested)
    except (TypeError, ValueError):
        n = 0
    if n < 0:
        n = 0
    if sys.platform == "darwin":
        return 0
    return n


def prefetch_factor_for_workers(num_workers: int, requested: Any) -> int | None:
    """``prefetch_factor`` only applies when ``num_workers > 0``."""
    if num_workers <= 0:
        return None
    try:
        pf = int(requested)
    except (TypeError, ValueError):
        pf = 2
    return max(1, pf)
