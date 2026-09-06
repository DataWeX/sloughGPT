"""
gpu_engine.py — Python ctypes bridge to the C GPU engine.

Provides Pythonic access to the platform-agnostic GPU compute engine.
Loads the shared library (libgpu_engine.so/.dylib/.dll) and wraps
all C functions with Python classes.

Usage:
    from domains.infrastructure.gpu.gpu_engine import GpuDevice

    device = GpuDevice()  # auto-selects best backend
    buf = device.buffer_create(1024 * 4, GPU_BUF_STORAGE)
    buf.write(data)
    # ... dispatch shaders ...
    result = buf.read()
"""

from __future__ import annotations

import ctypes
import os
import logging
from ctypes import (
    c_void_p, c_char_p, c_uint32, c_int32, c_uint64, c_size_t,
    Structure, POINTER,
)
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("slo.gpu.engine")

# ── Constants ────────────────────────────────────────────────────────────

GPU_OK = 0
GPU_ERROR_NO_DEVICE = -1
GPU_ERROR_NO_MEMORY = -2
GPU_ERROR_SHADER_COMPILE = -3
GPU_ERROR_PIPELINE = -4
GPU_ERROR_BUFFER = -5
GPU_ERROR_DISPATCH = -6
GPU_ERROR_UNSUPPORTED = -7

GPU_BUF_STORAGE = 1 << 0
GPU_BUF_UNIFORM = 1 << 1
GPU_BUF_VERTEX = 1 << 2
GPU_BUF_COPY_SRC = 1 << 3
GPU_BUF_COPY_DST = 1 << 4


# ── C types ──────────────────────────────────────────────────────────────

class GpuDeviceT(c_void_p):
    pass

class GpuBufferT(c_void_p):
    pass

class GpuShaderT(c_void_p):
    pass

class GpuPipelineT(c_void_p):
    pass

class GpuContextT(c_void_p):
    pass

class GpuBufferPoolT(c_void_p):
    pass

class GpuBindEntry(Structure):
    _fields_ = [
        ("binding", c_uint32),
        ("type", c_uint32),
        ("stages", c_uint32),
    ]


# ── Library loading ──────────────────────────────────────────────────────

def _find_library() -> ctypes.CDLL:
    """Find and load the gpu_engine shared library."""
    search_paths = [
        # Development: next to this file
        os.path.join(os.path.dirname(__file__), "libgpu_engine.so"),
        os.path.join(os.path.dirname(__file__), "libgpu_engine.dylib"),
        os.path.join(os.path.dirname(__file__), "gpu_engine.dll"),
        # Installed: system paths
        "libgpu_engine.so",
        "libgpu_engine.dylib",
        "gpu_engine.dll",
    ]

    for path in search_paths:
        if os.path.exists(path):
            return ctypes.CDLL(path)

    # Try loading by name (system search)
    try:
        return ctypes.CDLL("gpu_engine")
    except OSError:
        pass

    raise FileNotFoundError(
        "gpu_engine shared library not found. "
        "Build with: gcc -shared -o libgpu_engine.so engine.c vulkan.c -lvulkan"
    )


_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _find_library()
        _setup_functions(_lib)
    return _lib


def _setup_functions(lib: ctypes.CDLL) -> None:
    """Declare C function signatures."""

    # Device
    lib.gpu_device_create.restype = GpuDeviceT
    lib.gpu_device_create.argtypes = []

    lib.gpu_device_create_backend.restype = GpuDeviceT
    lib.gpu_device_create_backend.argtypes = [c_char_p]

    lib.gpu_device_name.restype = c_char_p
    lib.gpu_device_name.argtypes = [GpuDeviceT]

    lib.gpu_device_vram.restype = c_uint64
    lib.gpu_device_vram.argtypes = [GpuDeviceT]

    lib.gpu_device_destroy.restype = None
    lib.gpu_device_destroy.argtypes = [GpuDeviceT]

    # Buffers
    lib.gpu_buffer_create.restype = GpuBufferT
    lib.gpu_buffer_create.argtypes = [GpuDeviceT, c_size_t, c_uint32]

    lib.gpu_buffer_write.restype = c_int32
    lib.gpu_buffer_write.argtypes = [GpuBufferT, c_void_p, c_size_t, c_size_t]

    lib.gpu_buffer_read.restype = c_int32
    lib.gpu_buffer_read.argtypes = [GpuBufferT, c_void_p, c_size_t, c_size_t]

    lib.gpu_buffer_map.restype = c_void_p
    lib.gpu_buffer_map.argtypes = [GpuBufferT]

    lib.gpu_buffer_unmap.restype = None
    lib.gpu_buffer_unmap.argtypes = [GpuBufferT]

    lib.gpu_buffer_destroy.restype = None
    lib.gpu_buffer_destroy.argtypes = [GpuBufferT]

    # Shaders
    lib.gpu_shader_create_wgsl.restype = GpuShaderT
    lib.gpu_shader_create_wgsl.argtypes = [GpuDeviceT, c_char_p, c_size_t, c_char_p]

    lib.gpu_shader_create_spirv.restype = GpuShaderT
    lib.gpu_shader_create_spirv.argtypes = [GpuDeviceT, POINTER(c_uint32), c_size_t, c_char_p]

    lib.gpu_shader_destroy.restype = None
    lib.gpu_shader_destroy.argtypes = [GpuShaderT]

    # Pipelines
    lib.gpu_pipeline_create.restype = GpuPipelineT
    lib.gpu_pipeline_create.argtypes = [GpuDeviceT, GpuShaderT, c_char_p,
                                         POINTER(GpuBindEntry), c_uint32]

    lib.gpu_pipeline_destroy.restype = None
    lib.gpu_pipeline_destroy.argtypes = [GpuPipelineT]

    # Compute dispatch
    lib.gpu_compute_begin.restype = GpuContextT
    lib.gpu_compute_begin.argtypes = [GpuDeviceT]

    lib.gpu_compute_bind_pipeline.restype = None
    lib.gpu_compute_bind_pipeline.argtypes = [GpuContextT, GpuPipelineT]

    lib.gpu_compute_bind_buffer.restype = None
    lib.gpu_compute_bind_buffer.argtypes = [GpuContextT, c_uint32, GpuBufferT]

    lib.gpu_compute_set_push.restype = None
    lib.gpu_compute_set_push.argtypes = [GpuContextT, c_void_p, c_size_t]

    lib.gpu_compute_dispatch.restype = None
    lib.gpu_compute_dispatch.argtypes = [GpuContextT, c_uint32, c_uint32, c_uint32]

    lib.gpu_compute_end.restype = c_int32
    lib.gpu_compute_end.argtypes = [GpuContextT]

    # Buffer pool
    lib.gpu_pool_create.restype = GpuBufferPoolT
    lib.gpu_pool_create.argtypes = [GpuDeviceT, c_uint32, c_size_t]

    lib.gpu_pool_acquire.restype = GpuBufferT
    lib.gpu_pool_acquire.argtypes = [GpuBufferPoolT, c_size_t]

    lib.gpu_pool_release.restype = None
    lib.gpu_pool_release.argtypes = [GpuBufferPoolT, GpuBufferT]

    lib.gpu_pool_destroy.restype = None
    lib.gpu_pool_destroy.argtypes = [GpuBufferPoolT]


# ── Python wrappers ──────────────────────────────────────────────────────

class GpuDevice:
    """Python wrapper for GpuDevice."""

    def __init__(self, backend: Optional[str] = None):
        lib = _get_lib()
        if backend:
            self._ptr = lib.gpu_device_create_backend(backend.encode())
        else:
            self._ptr = lib.gpu_device_create()

        if not self._ptr:
            raise RuntimeError("Failed to create GPU device")

        self._lib = lib
        self._name = lib.gpu_device_name(self._ptr).decode()
        self._vram = lib.gpu_device_vram(self._ptr)

    @property
    def name(self) -> str:
        return self._name

    @property
    def vram(self) -> int:
        return self._vram

    def buffer_create(self, size: int, usage: int) -> "GpuBuffer":
        return GpuBuffer(self, size, usage)

    def shader_create_wgsl(self, source: str, entry: str = "main") -> "GpuShader":
        src_bytes = source.encode()
        ptr = self._lib.gpu_shader_create_wgsl(
            self._ptr, src_bytes, len(src_bytes), entry.encode()
        )
        if not ptr:
            raise RuntimeError("Failed to create WGSL shader")
        return GpuShader(self, ptr)

    def shader_create_spirv(self, code: np.ndarray, entry: str = "main") -> "GpuShader":
        code_ptr = code.ctypes.data_as(POINTER(c_uint32))
        ptr = self._lib.gpu_shader_create_spirv(
            self._ptr, code_ptr, len(code), entry.encode()
        )
        if not ptr:
            raise RuntimeError("Failed to create SPIR-V shader")
        return GpuShader(self, ptr)

    def pipeline_create(self, shader: "GpuShader", entries: List[Tuple[int, int, int]]) -> "GpuPipeline":
        """Create compute pipeline.

        Args:
            shader: Compiled shader.
            entries: List of (binding, type, stages) tuples.
        """
        bind_entries = (GpuBindEntry * len(entries))(*[
            GpuBindEntry(binding=e[0], type=e[1], stages=e[2]) for e in entries
        ])
        ptr = self._lib.gpu_pipeline_create(
            self._ptr, shader._ptr, b"main",
            bind_entries, len(entries)
        )
        if not ptr:
            raise RuntimeError("Failed to create pipeline")
        return GpuPipeline(self, ptr)

    def compute_begin(self) -> "GpuContext":
        return GpuContext(self, self._lib.gpu_compute_begin(self._ptr))

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            self._lib.gpu_device_destroy(self._ptr)


class GpuBuffer:
    """Python wrapper for GpuBuffer."""

    def __init__(self, device: GpuDevice, size: int, usage: int):
        self._device = device
        self._lib = device._lib
        self._ptr = self._lib.gpu_buffer_create(device._ptr, size, usage)
        if not self._ptr:
            raise RuntimeError(f"Failed to create buffer ({size} bytes)")
        self._size = size

    @property
    def size(self) -> int:
        return self._size

    def write(self, data: np.ndarray, offset: int = 0) -> None:
        """Write numpy array to GPU buffer."""
        arr = np.ascontiguousarray(data)
        ptr = arr.ctypes.data_as(c_void_p)
        err = self._lib.gpu_buffer_write(self._ptr, ptr, arr.nbytes, offset)
        if err != GPU_OK:
            raise RuntimeError(f"Buffer write failed: {err}")

    def read(self, shape: Tuple[int, ...], dtype=np.float32, offset: int = 0) -> np.ndarray:
        """Read GPU buffer into numpy array."""
        arr = np.empty(shape, dtype=dtype)
        ptr = arr.ctypes.data_as(c_void_p)
        err = self._lib.gpu_buffer_read(self._ptr, ptr, arr.nbytes, offset)
        if err != GPU_OK:
            raise RuntimeError(f"Buffer read failed: {err}")
        return arr

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            self._lib.gpu_buffer_destroy(self._ptr)


class GpuShader:
    """Python wrapper for GpuShader."""

    def __init__(self, device: GpuDevice, ptr: GpuShaderT):
        self._device = device
        self._lib = device._lib
        self._ptr = ptr

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            self._lib.gpu_shader_destroy(self._ptr)


class GpuPipeline:
    """Python wrapper for GpuPipeline."""

    def __init__(self, device: GpuDevice, ptr: GpuPipelineT):
        self._device = device
        self._lib = device._lib
        self._ptr = ptr

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            self._lib.gpu_pipeline_destroy(self._ptr)


class GpuContext:
    """Python wrapper for GpuContext (compute dispatch)."""

    def __init__(self, device: GpuDevice, ptr: GpuContextT):
        self._device = device
        self._lib = device._lib
        self._ptr = ptr

    def bind_pipeline(self, pipeline: GpuPipeline) -> None:
        self._lib.gpu_compute_bind_pipeline(self._ptr, pipeline._ptr)

    def bind_buffer(self, binding: int, buffer: GpuBuffer) -> None:
        self._lib.gpu_compute_bind_buffer(self._ptr, binding, buffer._ptr)

    def set_push(self, data: bytes) -> None:
        buf = ctypes.create_string_buffer(data)
        self._lib.gpu_compute_set_push(self._ptr, buf, len(data))

    def dispatch(self, x: int, y: int = 1, z: int = 1) -> None:
        self._lib.gpu_compute_dispatch(self._ptr, x, y, z)

    def end(self) -> int:
        return self._lib.gpu_compute_end(self._ptr)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end()


class GpuBufferPool:
    """Python wrapper for GpuBufferPool."""

    def __init__(self, device: GpuDevice, capacity: int = 64, min_size: int = 1024):
        self._device = device
        self._lib = device._lib
        self._ptr = self._lib.gpu_pool_create(device._ptr, capacity, min_size)

    def acquire(self, min_size: int) -> GpuBuffer:
        self._lib.gpu_pool_acquire(self._ptr, min_size)
        return GpuBuffer(self._device, min_size, GPU_BUF_STORAGE)

    def release(self, buffer: GpuBuffer) -> None:
        self._lib.gpu_pool_release(self._ptr, buffer._ptr)
        buffer._ptr = None

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            self._lib.gpu_pool_destroy(self._ptr)


# ── Convenience ──────────────────────────────────────────────────────────

def auto_device(backend: Optional[str] = None) -> GpuDevice:
    """Create the best available GPU device."""
    return GpuDevice(backend)


def is_gpu_available() -> bool:
    """Check if any GPU backend is available."""
    try:
        dev = GpuDevice()
        del dev
        return True
    except Exception:
        return False
