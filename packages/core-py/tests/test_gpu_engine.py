"""Tests for domains.infrastructure.gpu.gpu_engine — pure logic coverage.

Covers constants, ctypes structures, library search logic, wrapper class
initialisation paths, and convenience functions. No real GPU hardware needed.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

_CORE_PY = Path(__file__).resolve().parents[1]
if str(_CORE_PY) not in sys.path:
    sys.path.insert(0, str(_CORE_PY))

from domains.infrastructure.gpu.gpu_engine import (
    GPU_OK,
    GPU_ERROR_NO_DEVICE,
    GPU_ERROR_NO_MEMORY,
    GPU_ERROR_SHADER_COMPILE,
    GPU_ERROR_PIPELINE,
    GPU_ERROR_BUFFER,
    GPU_ERROR_DISPATCH,
    GPU_ERROR_UNSUPPORTED,
    GPU_BUF_STORAGE,
    GPU_BUF_UNIFORM,
    GPU_BUF_VERTEX,
    GPU_BUF_COPY_SRC,
    GPU_BUF_COPY_DST,
    GpuDeviceT,
    GpuBufferT,
    GpuShaderT,
    GpuPipelineT,
    GpuContextT,
    GpuBufferPoolT,
    GpuBindEntry,
    GpuDevice,
    GpuBuffer,
    GpuShader,
    GpuPipeline,
    GpuContext,
    GpuBufferPool,
    _find_library,
    _setup_functions,
    auto_device,
    is_gpu_available,
)


# ── Constants ───────────────────────────────────────────────────────────


class TestErrorConstants:
    def test_gpu_ok_is_zero(self):
        assert GPU_OK == 0

    def test_errors_are_negative(self):
        errors = [
            GPU_ERROR_NO_DEVICE,
            GPU_ERROR_NO_MEMORY,
            GPU_ERROR_SHADER_COMPILE,
            GPU_ERROR_PIPELINE,
            GPU_ERROR_BUFFER,
            GPU_ERROR_DISPATCH,
            GPU_ERROR_UNSUPPORTED,
        ]
        for e in errors:
            assert e < 0

    def test_errors_are_unique(self):
        errors = [
            GPU_ERROR_NO_DEVICE,
            GPU_ERROR_NO_MEMORY,
            GPU_ERROR_SHADER_COMPILE,
            GPU_ERROR_PIPELINE,
            GPU_ERROR_BUFFER,
            GPU_ERROR_DISPATCH,
            GPU_ERROR_UNSUPPORTED,
        ]
        assert len(errors) == len(set(errors))

    def test_error_values(self):
        assert GPU_ERROR_NO_DEVICE == -1
        assert GPU_ERROR_NO_MEMORY == -2
        assert GPU_ERROR_SHADER_COMPILE == -3
        assert GPU_ERROR_PIPELINE == -4
        assert GPU_ERROR_BUFFER == -5
        assert GPU_ERROR_DISPATCH == -6
        assert GPU_ERROR_UNSUPPORTED == -7


class TestBufferUsageConstants:
    def test_storage_is_bit_0(self):
        assert GPU_BUF_STORAGE == 1 << 0

    def test_uniform_is_bit_1(self):
        assert GPU_BUF_UNIFORM == 1 << 1

    def test_vertex_is_bit_2(self):
        assert GPU_BUF_VERTEX == 1 << 2

    def test_copy_src_is_bit_3(self):
        assert GPU_BUF_COPY_SRC == 1 << 3

    def test_copy_dst_is_bit_4(self):
        assert GPU_BUF_COPY_DST == 1 << 4

    def test_all_unique(self):
        flags = [GPU_BUF_STORAGE, GPU_BUF_UNIFORM, GPU_BUF_VERTEX, GPU_BUF_COPY_SRC, GPU_BUF_COPY_DST]
        assert len(flags) == len(set(flags))

    def test_composable_via_bitwise_or(self):
        combined = GPU_BUF_STORAGE | GPU_BUF_COPY_DST
        assert combined & GPU_BUF_STORAGE
        assert combined & GPU_BUF_COPY_DST
        assert not (combined & GPU_BUF_UNIFORM)


# ── Ctypes structures ──────────────────────────────────────────────────


class TestCtypesTypes:
    def test_device_is_void_p_subclass(self):
        assert issubclass(GpuDeviceT, ctypes.c_void_p)

    def test_buffer_is_void_p_subclass(self):
        assert issubclass(GpuBufferT, ctypes.c_void_p)

    def test_shader_is_void_p_subclass(self):
        assert issubclass(GpuShaderT, ctypes.c_void_p)

    def test_pipeline_is_void_p_subclass(self):
        assert issubclass(GpuPipelineT, ctypes.c_void_p)

    def test_context_is_void_p_subclass(self):
        assert issubclass(GpuContextT, ctypes.c_void_p)

    def test_buffer_pool_is_void_p_subclass(self):
        assert issubclass(GpuBufferPoolT, ctypes.c_void_p)


class TestGpuBindEntry:
    def test_has_expected_fields(self):
        fields = {name for name, _ in GpuBindEntry._fields_}
        assert fields == {"binding", "type", "stages"}

    def test_field_types(self):
        field_map = dict(GpuBindEntry._fields_)
        assert field_map["binding"] is ctypes.c_uint32
        assert field_map["type"] is ctypes.c_uint32
        assert field_map["stages"] is ctypes.c_uint32

    def test_construction(self):
        entry = GpuBindEntry(binding=0, type=1, stages=2)
        assert entry.binding == 0
        assert entry.type == 1
        assert entry.stages == 2

    def test_array_construction(self):
        arr = (GpuBindEntry * 3)(
            GpuBindEntry(binding=0, type=1, stages=2),
            GpuBindEntry(binding=1, type=2, stages=3),
            GpuBindEntry(binding=2, type=0, stages=1),
        )
        assert len(arr) == 3
        assert arr[1].binding == 1


# ── Library search logic ───────────────────────────────────────────────


class TestFindLibrary:
    @patch("domains.infrastructure.gpu.gpu_engine.os.path.exists")
    @patch("domains.infrastructure.gpu.gpu_engine.ctypes.CDLL")
    def test_finds_dev_library_next_to_file(self, mock_cdll, mock_exists, tmp_path):
        lib_path = tmp_path / "libgpu_engine.so"
        mock_exists.side_effect = lambda p: p == str(lib_path)

        with patch("domains.infrastructure.gpu.gpu_engine.os.path.dirname", return_value=str(tmp_path)):
            result = _find_library()

        mock_cdll.assert_called_once_with(str(lib_path))
        assert result is mock_cdll.return_value

    @patch("domains.infrastructure.gpu.gpu_engine.os.path.exists", return_value=False)
    @patch("domains.infrastructure.gpu.gpu_engine.ctypes.CDLL")
    def test_falls_back_to_system_name(self, mock_cdll, mock_exists):
        result = _find_library()
        mock_cdll.assert_called_with("gpu_engine")
        assert result is mock_cdll.return_value

    @patch("domains.infrastructure.gpu.gpu_engine.os.path.exists", return_value=False)
    @patch("domains.infrastructure.gpu.gpu_engine.ctypes.CDLL", side_effect=OSError("no lib"))
    def test_raises_when_not_found(self, mock_cdll, mock_exists):
        with pytest.raises(FileNotFoundError, match="gpu_engine shared library not found"):
            _find_library()

    @patch("domains.infrastructure.gpu.gpu_engine.os.path.exists")
    @patch("domains.infrastructure.gpu.gpu_engine.ctypes.CDLL")
    def test_tries_dll_on_any_platform(self, mock_cdll, mock_exists, tmp_path):
        dll_path = tmp_path / "gpu_engine.dll"
        mock_exists.side_effect = lambda p: p == str(dll_path)

        with patch("domains.infrastructure.gpu.gpu_engine.os.path.dirname", return_value=str(tmp_path)):
            result = _find_library()

        mock_cdll.assert_called_once_with(str(dll_path))


# ── _setup_functions ───────────────────────────────────────────────────


class TestSetupFunctions:
    def test_sets_device_function_signatures(self):
        lib = MagicMock()
        _setup_functions(lib)

        assert lib.gpu_device_create.restype is GpuDeviceT
        assert lib.gpu_device_create.argtypes == []
        assert lib.gpu_device_create_backend.argtypes == [ctypes.c_char_p]
        assert lib.gpu_device_name.argtypes == [GpuDeviceT]
        assert lib.gpu_device_vram.argtypes == [GpuDeviceT]
        assert lib.gpu_device_destroy.argtypes == [GpuDeviceT]

    def test_sets_buffer_function_signatures(self):
        lib = MagicMock()
        _setup_functions(lib)

        assert lib.gpu_buffer_create.argtypes == [GpuDeviceT, ctypes.c_size_t, ctypes.c_uint32]
        assert lib.gpu_buffer_write.argtypes == [GpuBufferT, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        assert lib.gpu_buffer_read.argtypes == [GpuBufferT, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        assert lib.gpu_buffer_map.argtypes == [GpuBufferT]
        assert lib.gpu_buffer_unmap.argtypes == [GpuBufferT]
        assert lib.gpu_buffer_destroy.argtypes == [GpuBufferT]

    def test_sets_shader_function_signatures(self):
        lib = MagicMock()
        _setup_functions(lib)

        assert lib.gpu_shader_create_wgsl.restype is GpuShaderT
        assert lib.gpu_shader_create_spirv.restype is GpuShaderT
        assert lib.gpu_shader_destroy.argtypes == [GpuShaderT]

    def test_sets_pipeline_function_signatures(self):
        lib = MagicMock()
        _setup_functions(lib)

        assert lib.gpu_pipeline_create.restype is GpuPipelineT
        assert lib.gpu_pipeline_destroy.argtypes == [GpuPipelineT]

    def test_sets_compute_function_signatures(self):
        lib = MagicMock()
        _setup_functions(lib)

        assert lib.gpu_compute_begin.restype is GpuContextT
        assert lib.gpu_compute_end.argtypes == [GpuContextT]
        assert lib.gpu_compute_dispatch.argtypes == [GpuContextT, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]

    def test_sets_pool_function_signatures(self):
        lib = MagicMock()
        _setup_functions(lib)

        assert lib.gpu_pool_create.restype is GpuBufferPoolT
        assert lib.gpu_pool_acquire.argtypes == [GpuBufferPoolT, ctypes.c_size_t]
        assert lib.gpu_pool_release.argtypes == [GpuBufferPoolT, GpuBufferT]
        assert lib.gpu_pool_destroy.argtypes == [GpuBufferPoolT]


# ── GpuDevice wrapper ─────────────────────────────────────────────────


def _make_mock_lib():
    """Create a mock ctypes.CDLL with all gpu_engine functions."""
    lib = MagicMock()
    lib.gpu_device_create.return_value = GpuDeviceT(0x1)
    lib.gpu_device_create_backend.return_value = GpuDeviceT(0x1)
    lib.gpu_device_name.return_value = b"MockGPU"
    lib.gpu_device_vram.return_value = 4 * 1024 * 1024 * 1024
    lib.gpu_buffer_create.return_value = GpuBufferT(0x2)
    lib.gpu_shader_create_wgsl.return_value = GpuShaderT(0x3)
    lib.gpu_shader_create_spirv.return_value = GpuShaderT(0x3)
    lib.gpu_pipeline_create.return_value = GpuPipelineT(0x4)
    lib.gpu_compute_begin.return_value = GpuContextT(0x5)
    lib.gpu_compute_end.return_value = GPU_OK
    lib.gpu_pool_create.return_value = GpuBufferPoolT(0x6)
    lib.gpu_pool_acquire.return_value = GpuBufferT(0x7)
    return lib


class TestGpuDeviceWrapper:
    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_init_default_backend(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        lib.gpu_device_create.assert_called_once()
        lib.gpu_device_create_backend.assert_not_called()

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_init_named_backend(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice(backend="vulkan")
        lib.gpu_device_create_backend.assert_called_once_with(b"vulkan")

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_name_property(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        assert device.name == "MockGPU"

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_vram_property(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        assert device.vram == 4 * 1024 * 1024 * 1024

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_raises_on_null_device(self, mock_get_lib):
        lib = _make_mock_lib()
        lib.gpu_device_create.return_value = GpuDeviceT(None)
        mock_get_lib.return_value = lib

        with pytest.raises(RuntimeError, match="Failed to create GPU device"):
            GpuDevice()

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_buffer_create(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        buf = device.buffer_create(1024, GPU_BUF_STORAGE)
        assert isinstance(buf, GpuBuffer)
        lib.gpu_buffer_create.assert_called_once_with(device._ptr, 1024, GPU_BUF_STORAGE)

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_shader_create_wgsl(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        shader = device.shader_create_wgsl("fn main() {}", entry="main")
        assert isinstance(shader, GpuShader)
        lib.gpu_shader_create_wgsl.assert_called_once()

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_shader_create_wgsl_raises_on_null(self, mock_get_lib):
        lib = _make_mock_lib()
        lib.gpu_shader_create_wgsl.return_value = GpuShaderT(None)
        mock_get_lib.return_value = lib

        device = GpuDevice()
        with pytest.raises(RuntimeError, match="Failed to create WGSL shader"):
            device.shader_create_wgsl("fn main() {}")

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_shader_create_spirv(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        code = np.array([0x07230203], dtype=np.uint32)
        shader = device.shader_create_spirv(code, entry="main")
        assert isinstance(shader, GpuShader)
        lib.gpu_shader_create_spirv.assert_called_once()

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_shader_create_spirv_raises_on_null(self, mock_get_lib):
        lib = _make_mock_lib()
        lib.gpu_shader_create_spirv.return_value = GpuShaderT(None)
        mock_get_lib.return_value = lib

        device = GpuDevice()
        code = np.array([0x07230203], dtype=np.uint32)
        with pytest.raises(RuntimeError, match="Failed to create SPIR-V shader"):
            device.shader_create_spirv(code)

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_pipeline_create(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        shader = GpuShader(device, GpuShaderT(0x3))
        entries = [(0, 1, 2), (1, 2, 3)]
        pipeline = device.pipeline_create(shader, entries)
        assert isinstance(pipeline, GpuPipeline)
        lib.gpu_pipeline_create.assert_called_once()

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_pipeline_create_raises_on_null(self, mock_get_lib):
        lib = _make_mock_lib()
        lib.gpu_pipeline_create.return_value = GpuPipelineT(None)
        mock_get_lib.return_value = lib

        device = GpuDevice()
        shader = GpuShader(device, GpuShaderT(0x3))
        with pytest.raises(RuntimeError, match="Failed to create pipeline"):
            device.pipeline_create(shader, [(0, 1, 2)])

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_compute_begin(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        ctx = device.compute_begin()
        assert isinstance(ctx, GpuContext)
        lib.gpu_compute_begin.assert_called_once_with(device._ptr)

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_del_destroys_device(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        device.__del__()
        lib.gpu_device_destroy.assert_called_once_with(device._ptr)

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_del_noop_without_ptr(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        device._ptr = None
        device.__del__()
        lib.gpu_device_destroy.assert_not_called()


# ── GpuBuffer wrapper ─────────────────────────────────────────────────


class TestGpuBufferWrapper:
    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_init_creates_buffer(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        buf = GpuBuffer(device, 2048, GPU_BUF_UNIFORM)
        assert buf.size == 2048
        lib.gpu_buffer_create.assert_called_with(device._ptr, 2048, GPU_BUF_UNIFORM)

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_raises_on_null_buffer(self, mock_get_lib):
        lib = _make_mock_lib()
        lib.gpu_buffer_create.return_value = GpuBufferT(None)
        mock_get_lib.return_value = lib

        device = GpuDevice()
        with pytest.raises(RuntimeError, match="Failed to create buffer"):
            GpuBuffer(device, 1024, GPU_BUF_STORAGE)

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_write_success(self, mock_get_lib):
        lib = _make_mock_lib()
        lib.gpu_buffer_write.return_value = GPU_OK
        mock_get_lib.return_value = lib

        device = GpuDevice()
        buf = GpuBuffer(device, 1024, GPU_BUF_STORAGE)
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        buf.write(data)
        lib.gpu_buffer_write.assert_called_once()

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_write_with_offset(self, mock_get_lib):
        lib = _make_mock_lib()
        lib.gpu_buffer_write.return_value = GPU_OK
        mock_get_lib.return_value = lib

        device = GpuDevice()
        buf = GpuBuffer(device, 1024, GPU_BUF_STORAGE)
        data = np.array([1.0, 2.0], dtype=np.float32)
        buf.write(data, offset=16)
        lib.gpu_buffer_write.assert_called_once()

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_write_raises_on_error(self, mock_get_lib):
        lib = _make_mock_lib()
        lib.gpu_buffer_write.return_value = GPU_ERROR_BUFFER
        mock_get_lib.return_value = lib

        device = GpuDevice()
        buf = GpuBuffer(device, 1024, GPU_BUF_STORAGE)
        with pytest.raises(RuntimeError, match="Buffer write failed"):
            buf.write(np.array([1.0], dtype=np.float32))

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_read_success(self, mock_get_lib):
        lib = _make_mock_lib()
        lib.gpu_buffer_read.return_value = GPU_OK
        mock_get_lib.return_value = lib

        device = GpuDevice()
        buf = GpuBuffer(device, 1024, GPU_BUF_STORAGE)
        result = buf.read(shape=(4,), dtype=np.float32)
        assert result.shape == (4,)
        assert result.dtype == np.float32
        lib.gpu_buffer_read.assert_called_once()

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_read_with_offset(self, mock_get_lib):
        lib = _make_mock_lib()
        lib.gpu_buffer_read.return_value = GPU_OK
        mock_get_lib.return_value = lib

        device = GpuDevice()
        buf = GpuBuffer(device, 1024, GPU_BUF_STORAGE)
        result = buf.read(shape=(2, 3), dtype=np.float64, offset=8)
        assert result.shape == (2, 3)
        assert result.dtype == np.float64

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_read_raises_on_error(self, mock_get_lib):
        lib = _make_mock_lib()
        lib.gpu_buffer_read.return_value = GPU_ERROR_NO_MEMORY
        mock_get_lib.return_value = lib

        device = GpuDevice()
        buf = GpuBuffer(device, 1024, GPU_BUF_STORAGE)
        with pytest.raises(RuntimeError, match="Buffer read failed"):
            buf.read(shape=(4,))

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_del_destroys_buffer(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        buf = GpuBuffer(device, 1024, GPU_BUF_STORAGE)
        buf.__del__()
        lib.gpu_buffer_destroy.assert_called_once_with(buf._ptr)

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_del_noop_without_ptr(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        buf = GpuBuffer(device, 1024, GPU_BUF_STORAGE)
        buf._ptr = None
        buf.__del__()
        lib.gpu_buffer_destroy.assert_not_called()


# ── GpuShader wrapper ─────────────────────────────────────────────────


class TestGpuShaderWrapper:
    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_del_destroys_shader(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        shader = GpuShader(device, GpuShaderT(0x3))
        shader.__del__()
        lib.gpu_shader_destroy.assert_called_once()
        actual_ptr = lib.gpu_shader_destroy.call_args[0][0]
        assert actual_ptr.value == 0x3

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_del_noop_without_ptr(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        shader = GpuShader(device, GpuShaderT(0x3))
        shader._ptr = None
        shader.__del__()
        lib.gpu_shader_destroy.assert_not_called()


# ── GpuPipeline wrapper ───────────────────────────────────────────────


class TestGpuPipelineWrapper:
    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_del_destroys_pipeline(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        pipeline = GpuPipeline(device, GpuPipelineT(0x4))
        pipeline.__del__()
        lib.gpu_pipeline_destroy.assert_called_once()
        actual_ptr = lib.gpu_pipeline_destroy.call_args[0][0]
        assert actual_ptr.value == 0x4

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_del_noop_without_ptr(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        pipeline = GpuPipeline(device, GpuPipelineT(0x4))
        pipeline._ptr = None
        pipeline.__del__()
        lib.gpu_pipeline_destroy.assert_not_called()


# ── GpuContext wrapper ────────────────────────────────────────────────


class TestGpuContextWrapper:
    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_bind_pipeline(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        ctx = GpuContext(device, GpuContextT(0x5))
        pipeline = GpuPipeline(device, GpuPipelineT(0x4))
        ctx.bind_pipeline(pipeline)
        lib.gpu_compute_bind_pipeline.assert_called_once()
        args = lib.gpu_compute_bind_pipeline.call_args[0]
        assert args[0].value == 0x5
        assert args[1].value == 0x4

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_bind_buffer(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        ctx = GpuContext(device, GpuContextT(0x5))
        buf = GpuBuffer(device, 1024, GPU_BUF_STORAGE)
        ctx.bind_buffer(0, buf)
        lib.gpu_compute_bind_buffer.assert_called_once()
        args = lib.gpu_compute_bind_buffer.call_args[0]
        assert args[0].value == 0x5
        assert args[1] == 0
        assert args[2] == buf._ptr

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_set_push(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        ctx = GpuContext(device, GpuContextT(0x5))
        ctx.set_push(b"\x01\x02\x03")
        lib.gpu_compute_set_push.assert_called_once()

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_dispatch_defaults(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        ctx = GpuContext(device, GpuContextT(0x5))
        ctx.dispatch(64)
        lib.gpu_compute_dispatch.assert_called_once()
        args = lib.gpu_compute_dispatch.call_args[0]
        assert args[0].value == 0x5
        assert args[1:] == (64, 1, 1)

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_dispatch_custom_dims(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        ctx = GpuContext(device, GpuContextT(0x5))
        ctx.dispatch(64, 32, 16)
        lib.gpu_compute_dispatch.assert_called_once()
        args = lib.gpu_compute_dispatch.call_args[0]
        assert args[0].value == 0x5
        assert args[1:] == (64, 32, 16)

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_end_returns_status(self, mock_get_lib):
        lib = _make_mock_lib()
        lib.gpu_compute_end.return_value = GPU_OK
        mock_get_lib.return_value = lib

        device = GpuDevice()
        ctx = GpuContext(device, GpuContextT(0x5))
        result = ctx.end()
        assert result == GPU_OK
        lib.gpu_compute_end.assert_called_once()
        actual_ptr = lib.gpu_compute_end.call_args[0][0]
        assert actual_ptr.value == 0x5

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_context_manager(self, mock_get_lib):
        lib = _make_mock_lib()
        lib.gpu_compute_end.return_value = GPU_OK
        mock_get_lib.return_value = lib

        device = GpuDevice()
        ctx = GpuContext(device, GpuContextT(0x5))
        with ctx as c:
            assert c is ctx
        lib.gpu_compute_end.assert_called_once()


# ── GpuBufferPool wrapper ─────────────────────────────────────────────


class TestGpuBufferPoolWrapper:
    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_init_creates_pool(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        pool = GpuBufferPool(device, capacity=32, min_size=512)
        lib.gpu_pool_create.assert_called_once_with(device._ptr, 32, 512)

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_acquire_returns_buffer(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        pool = GpuBufferPool(device)
        buf = pool.acquire(2048)
        assert isinstance(buf, GpuBuffer)
        lib.gpu_pool_acquire.assert_called_once_with(pool._ptr, 2048)

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_release_nullifies_buffer_ptr(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        pool = GpuBufferPool(device)
        buf = GpuBuffer(device, 1024, GPU_BUF_STORAGE)
        buf_ptr = buf._ptr
        pool.release(buf)
        lib.gpu_pool_release.assert_called_once_with(pool._ptr, buf_ptr)
        assert buf._ptr is None

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_del_destroys_pool(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        pool = GpuBufferPool(device)
        pool.__del__()
        lib.gpu_pool_destroy.assert_called_once_with(pool._ptr)

    @patch("domains.infrastructure.gpu.gpu_engine._get_lib")
    def test_del_noop_without_ptr(self, mock_get_lib):
        lib = _make_mock_lib()
        mock_get_lib.return_value = lib

        device = GpuDevice()
        pool = GpuBufferPool(device)
        pool._ptr = None
        pool.__del__()
        lib.gpu_pool_destroy.assert_not_called()


# ── Convenience functions ──────────────────────────────────────────────


class TestConvenienceFunctions:
    @patch("domains.infrastructure.gpu.gpu_engine.GpuDevice")
    def test_auto_device_creates_device(self, mock_cls):
        result = auto_device()
        mock_cls.assert_called_once_with(None)
        assert result is mock_cls.return_value

    @patch("domains.infrastructure.gpu.gpu_engine.GpuDevice")
    def test_auto_device_passes_backend(self, mock_cls):
        result = auto_device(backend="metal")
        mock_cls.assert_called_once_with("metal")

    @patch("domains.infrastructure.gpu.gpu_engine.GpuDevice")
    def test_is_gpu_available_true(self, mock_cls):
        mock_cls.return_value = MagicMock()
        assert is_gpu_available() is True
        mock_cls.assert_called_once()

    @patch("domains.infrastructure.gpu.gpu_engine.GpuDevice", side_effect=RuntimeError("no device"))
    def test_is_gpu_available_false(self, mock_cls):
        assert is_gpu_available() is False
