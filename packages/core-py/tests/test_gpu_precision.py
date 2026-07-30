"""
Tests for GPU accelerator precision selection.

Tests the base Accelerator class and module-level set_accelerator_precision()
function. Metal/CUDA backend tests require actual hardware.
"""

import numpy as np
import pytest

from domains.slolib.gpu import _Accelerator, get_accelerator, set_accelerator_precision, reset_accelerator


class TestBaseAcceleratorPrecision:
    """Tests for _Accelerator.set_precision() (base class, CPU path)."""

    def test_set_precision_fp32(self):
        acc = _Accelerator()
        result = acc.set_precision("fp32")
        assert result == "fp32"
        assert not acc._fp16_mode

    def test_set_precision_fp16_no_hardware(self):
        acc = _Accelerator()
        acc._fp16_available = False
        result = acc.set_precision("fp16")
        assert result == "fp32"
        assert not acc._fp16_mode

    def test_set_precision_auto_no_hardware(self):
        acc = _Accelerator()
        result = acc.set_precision("auto")
        assert result == "fp32"
        assert not acc._fp16_mode

    def test_set_precision_fp16_with_hardware(self):
        acc = _Accelerator()
        acc._fp16_available = True
        result = acc.set_precision("fp16")
        assert result == "fp16"
        assert acc._fp16_mode

    def test_set_precision_back_to_fp32(self):
        acc = _Accelerator()
        acc._fp16_available = True
        acc.set_precision("fp16")
        assert acc._fp16_mode
        result = acc.set_precision("fp32")
        assert result == "fp32"
        assert not acc._fp16_mode

    def test_precision_benchmark_no_hardware(self):
        acc = _Accelerator()
        result = acc._prec_benchmark()
        assert result == "fp32"


class TestModuleLevelFunctions:
    """Tests for module-level get_accelerator() / set_accelerator_precision()."""

    def setup_method(self):
        reset_accelerator()

    def test_get_accelerator_returns_instance(self):
        acc = get_accelerator()
        assert isinstance(acc, _Accelerator)

    def test_set_accelerator_precision_fp32(self):
        reset_accelerator()
        result = set_accelerator_precision("fp32")
        assert result == "fp32"

    def test_set_accelerator_precision_auto(self):
        reset_accelerator()
        result = set_accelerator_precision("auto")
        assert result == "fp32"
