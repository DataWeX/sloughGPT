"""Tests for ``domains.torch_runtime``."""

from __future__ import annotations

import os
import sys

import pytest

from domains.torch_runtime import (
    apply_api_process_torch_env,
    effective_dataloader_num_workers,
    prefetch_factor_for_workers,
)


def test_effective_dataloader_num_workers_non_negative() -> None:
    assert effective_dataloader_num_workers(4) == (0 if sys.platform == "darwin" else 4)
    assert effective_dataloader_num_workers(-1) == 0
    assert effective_dataloader_num_workers("bogus") == 0


def test_prefetch_factor_only_when_workers() -> None:
    assert prefetch_factor_for_workers(0, 2) is None
    assert prefetch_factor_for_workers(2, 3) == 3
    assert prefetch_factor_for_workers(2, "x") == 2


def test_apply_api_process_torch_env_respects_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLOUGHGPT_SKIP_TORCH_ENV", "1")
    monkeypatch.delenv("PYTORCH_DISABLE_MPS", raising=False)
    import domains.torch_runtime as tr

    tr._api_torch_env_applied = False
    apply_api_process_torch_env()
    assert os.environ.get("PYTORCH_DISABLE_MPS") is None
    tr._api_torch_env_applied = False


def test_apply_api_process_darwin_disables_mps_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLOUGHGPT_SKIP_TORCH_ENV", raising=False)
    monkeypatch.delenv("SLOUGHGPT_API_ENABLE_MPS", raising=False)
    monkeypatch.delenv("PYTORCH_DISABLE_MPS", raising=False)
    import domains.torch_runtime as tr

    tr._api_torch_env_applied = False
    monkeypatch.setattr(sys, "platform", "darwin")
    apply_api_process_torch_env()
    assert os.environ.get("PYTORCH_DISABLE_MPS") == "1"
    tr._api_torch_env_applied = False


def test_apply_api_process_linux_leaves_mps_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLOUGHGPT_SKIP_TORCH_ENV", raising=False)
    monkeypatch.delenv("PYTORCH_DISABLE_MPS", raising=False)
    import domains.torch_runtime as tr

    tr._api_torch_env_applied = False
    monkeypatch.setattr(sys, "platform", "linux")
    apply_api_process_torch_env()
    assert os.environ.get("PYTORCH_DISABLE_MPS") is None
    tr._api_torch_env_applied = False
