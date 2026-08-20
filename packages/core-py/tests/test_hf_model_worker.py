"""Tests for domains.infrastructure.hf_model_worker — _resolve_device.

Covers: device string resolution, auto fallback, explicit device passthrough.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.infrastructure.hf_model_worker import _resolve_device


class TestResolveDevice:
    def test_explicit_cpu(self):
        assert _resolve_device("cpu") == "cpu"

    def test_explicit_cuda(self):
        assert _resolve_device("cuda") == "cuda"

    def test_auto_fallback_to_cpu(self):
        # auto should resolve (to cpu or whatever auto_device returns)
        result = _resolve_device("auto")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_custom_device(self):
        assert _resolve_device("mps") == "mps"
