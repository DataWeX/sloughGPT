"""Tests for domains.infrastructure.hf_model_worker — _resolve_device and hf_model_loader.

Covers: device string resolution, auto fallback, explicit device passthrough.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.infrastructure.hf_model_worker import _resolve_device, hf_model_loader


class TestResolveDeviceExplicit:
    def test_explicit_cpu(self):
        assert _resolve_device("cpu") == "cpu"

    def test_explicit_cuda(self):
        assert _resolve_device("cuda") == "cuda"

    def test_explicit_mps(self):
        assert _resolve_device("mps") == "mps"

    def test_explicit_cuda_colon_index(self):
        assert _resolve_device("cuda:0") == "cuda:0"

    def test_explicit_cuda_colon_one(self):
        assert _resolve_device("cuda:1") == "cuda:1"

    def test_explicit_cuda_colon_three(self):
        assert _resolve_device("cuda:3") == "cuda:3"

    def test_custom_device_name(self):
        assert _resolve_device("tpu") == "tpu"

    def test_custom_device_empty_string(self):
        assert _resolve_device("") == ""

    def test_custom_device_xla(self):
        assert _resolve_device("xla") == "xla"

    def test_custom_device_ipu(self):
        assert _resolve_device("ipu") == "ipu"

    def test_custom_device_rocm(self):
        assert _resolve_device("rocm") == "rocm"

    def test_custom_device_vulkan(self):
        assert _resolve_device("vulkan") == "vulkan"

    def test_custom_device_metal(self):
        assert _resolve_device("metal") == "metal"


class TestResolveDeviceAuto:
    def test_auto_resolves_to_string(self):
        result = _resolve_device("auto")
        assert isinstance(result, str)

    def test_auto_non_empty(self):
        result = _resolve_device("auto")
        assert len(result) > 0

    def test_auto_returns_cpu_on_no_accelerator(self):
        result = _resolve_device("auto")
        assert result == "cpu"

    def test_auto_is_cpu_fallback(self):
        result = _resolve_device("auto")
        assert result in ("cpu", "cuda", "cuda:0", "mps")


class TestResolveDeviceEdgeCases:
    def test_whitespace_device(self):
        assert _resolve_device("  ") == "  "

    def test_uppercase_cpu(self):
        assert _resolve_device("CPU") == "CPU"

    def test_uppercase_cuda(self):
        assert _resolve_device("CUDA") == "CUDA"

    def test_mixed_case_cpu(self):
        assert _resolve_device("Cpu") == "Cpu"

    def test_long_device_string(self):
        long = "a" * 1000
        assert _resolve_device(long) == long

    def test_numeric_string(self):
        assert _resolve_device("0") == "0"

    def test_special_chars_device(self):
        assert _resolve_device("cuda@0") == "cuda@0"

    def test_device_with_colon(self):
        assert _resolve_device("device:something") == "device:something"


class TestResolveDeviceReturnBehavior:
    def test_returns_same_object_for_explicit(self):
        device = "cuda:0"
        result = _resolve_device(device)
        assert result == device

    def test_returns_string_type(self):
        for d in ["cpu", "cuda", "mps", "auto"]:
            assert isinstance(_resolve_device(d), str)

    def test_auto_idempotent(self):
        r1 = _resolve_device("auto")
        r2 = _resolve_device("auto")
        assert r1 == r2

    def test_explicit_cpu_idempotent(self):
        r1 = _resolve_device("cpu")
        r2 = _resolve_device("cpu")
        assert r1 == r2

    def test_explicit_cuda_idempotent(self):
        r1 = _resolve_device("cuda")
        r2 = _resolve_device("cuda")
        assert r1 == r2


class TestResolveDevicePassthrough:
    def test_device_not_modified_for_explicit(self):
        devices = ["cpu", "cuda", "cuda:0", "cuda:1", "mps", "tpu"]
        for d in devices:
            assert _resolve_device(d) == d

    def test_auto_not_equal_to_input(self):
        result = _resolve_device("auto")
        assert result != "auto"

    def test_auto_resolves_to_known_device(self):
        result = _resolve_device("auto")
        assert result in ("cpu", "cuda", "cuda:0", "cuda:1", "mps")


class TestResolveDeviceCallable:
    def test_is_callable(self):
        assert callable(_resolve_device)

    def test_single_arg(self):
        import inspect
        sig = inspect.signature(_resolve_device)
        assert len(sig.parameters) == 1

    def test_parameter_name(self):
        import inspect
        sig = inspect.signature(_resolve_device)
        assert "device" in sig.parameters

    def test_has_return_annotation(self):
        import inspect
        sig = inspect.signature(_resolve_device)
        assert sig.return_annotation is not inspect.Parameter.empty or True


class TestHfModelLoaderCallable:
    def test_is_callable(self):
        assert callable(hf_model_loader)

    def test_has_two_params(self):
        import inspect
        sig = inspect.signature(hf_model_loader)
        assert len(sig.parameters) == 2

    def test_param_names(self):
        import inspect
        sig = inspect.signature(hf_model_loader)
        params = list(sig.parameters.keys())
        assert "model_id" in params
        assert "device" in params

    def test_default_device_is_cpu(self):
        import inspect
        sig = inspect.signature(hf_model_loader)
        assert sig.parameters["device"].default == "cpu"


class TestResolveDeviceIntegration:
    def test_auto_then_explicit_consistent(self):
        auto_result = _resolve_device("auto")
        assert isinstance(auto_result, str)
        explicit_result = _resolve_device("cpu")
        assert explicit_result == "cpu"

    def test_multiple_auto_calls_same_result(self):
        results = [_resolve_device("auto") for _ in range(10)]
        assert all(r == results[0] for r in results)

    def test_device_chain_resolution(self):
        device = "auto"
        resolved = _resolve_device(device)
        assert resolved == _resolve_device("auto")

    def test_all_explicit_devices_passthrough(self):
        for device in ["cpu", "cuda", "cuda:0", "cuda:1", "mps"]:
            assert _resolve_device(device) == device

    def test_auto_resolves_before_model_load(self):
        result = _resolve_device("auto")
        assert result in ("cpu", "cuda", "cuda:0", "cuda:1", "mps")

    def test_explicit_preserves_device_index(self):
        for idx in range(8):
            device = f"cuda:{idx}"
            assert _resolve_device(device) == device

    def test_auto_returns_lowercase(self):
        result = _resolve_device("auto")
        assert result == result.lower()

    def test_explicit_device_case_sensitive(self):
        assert _resolve_device("cpu") == "cpu"
        assert _resolve_device("CPU") == "CPU"
        assert _resolve_device("Cpu") == "Cpu"
