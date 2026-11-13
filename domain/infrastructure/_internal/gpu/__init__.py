"""
GPU compute engine for pugeeq.

Own engine from scratch — no third-party ML/GPU frameworks.
Platform-agnostic: Vulkan, Metal, DX12, CPU fallback.

Files:
  engine.h/c    — Core API, CPU fallback, buffer pool
  vulkan.c       — Vulkan compute backend
  metal.c        — Metal compute backend (macOS)
  dx12.c         — DX12 compute backend (Windows)
  gpu_engine.py  — Python ctypes bridge
  wgpu_be.py     — ComputeBackend impl
  shaders/       — WGSL compute shaders
"""

from __future__ import annotations
