"""
Serving Profiles — predefined configurations for different hardware tiers.

Each profile bundles model loading, inference, process guard, and resource
allocation settings into a single named preset. Users can switch profiles
via ``POST /profiles/apply`` without manually tuning dozens of env vars.

Profiles are **not** persisted — they apply to the running process until
restart or until a different profile is applied.

Usage::

    from domains.infrastructure.serving_profiles import (
        get_profile, list_profiles, apply_profile,
    )

    profiles = list_profiles()              # all available profiles
    current = get_profile("balanced")      # fetch one
    apply_profile("balanced")              # apply to runtime config
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("slo.infrastructure.serving_profiles")


# ---------------------------------------------------------------------------
# Profile schema
# ---------------------------------------------------------------------------


class ProfileTier(str, Enum):
    """Hardware tier classification."""

    CPU_ONLY = "cpu_only"
    CPU_OPTIMIZED = "cpu_optimized"
    BALANCED = "balanced"
    GPU_PERFORMANCE = "gpu_performance"
    TRAINING = "training"


@dataclass(frozen=True)
class ServingProfile:
    """A named configuration preset for model serving."""

    id: str
    name: str
    description: str
    tier: ProfileTier

    # -- Model loading --
    device: str = "auto"
    quantize: bool = True
    quant_bits: int = 8
    quant_mode: str = "symmetric"
    quant_clip: float = 0.999

    # -- Inference --
    inference_pool_size: int = 2
    generate_timeout: float = 120.0
    request_timeout: float = 120.0
    idle_timeout: float = 300.0

    # -- Process guard --
    enable_guard: bool = True
    lazy_autoload: bool = True
    memory_limit_mb: float = 0.0  # 0 = auto
    max_restarts: int = 3

    # -- Resource allocation --
    workload_mode: str = "inference"  # inference | training | balanced
    compute_threads: int = 2
    io_threads: int = 2

    # -- Memory pressure --
    memory_pressure_warning: float = 80.0
    memory_pressure_critical: float = 90.0
    memory_pressure_emergency: float = 95.0

    # -- Generation defaults --
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    max_new_tokens: int = 200

    # -- Metadata --
    tags: list[str] = field(default_factory=list)
    min_ram_gb: float = 0.0
    recommended_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        d: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if k == "tier":
                d[k] = v.value
            else:
                d[k] = v
        return d


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

_PROFILES: dict[str, ServingProfile] = {}


def _register(p: ServingProfile) -> None:
    _PROFILES[p.id] = p


_register(
    ServingProfile(
        id="cpu_only",
        name="CPU Only",
        description="Minimal footprint for machines without GPU. Quantized model, minimal threads, "
        "single inference slot. Best for quick testing on laptops.",
        tier=ProfileTier.CPU_ONLY,
        device="cpu",
        quantize=True,
        quant_bits=8,
        inference_pool_size=1,
        generate_timeout=60.0,
        request_timeout=60.0,
        idle_timeout=120.0,
        enable_guard=False,
        lazy_autoload=False,
        compute_threads=1,
        io_threads=1,
        workload_mode="inference",
        memory_pressure_warning=75.0,
        memory_pressure_critical=85.0,
        memory_pressure_emergency=92.0,
        temperature=0.7,
        max_new_tokens=150,
        tags=["laptop", "testing", "minimal"],
        min_ram_gb=4.0,
        recommended_model="Qwen/Qwen2.5-0.5B-Instruct",
    )
)

_register(
    ServingProfile(
        id="cpu_optimized",
        name="CPU Optimized",
        description="Tuned for multi-core CPU servers. More inference slots and threads, "
        "with process guard for crash isolation.",
        tier=ProfileTier.CPU_OPTIMIZED,
        device="cpu",
        quantize=True,
        quant_bits=8,
        inference_pool_size=4,
        generate_timeout=90.0,
        request_timeout=90.0,
        idle_timeout=300.0,
        enable_guard=True,
        lazy_autoload=True,
        compute_threads=4,
        io_threads=2,
        workload_mode="inference",
        tags=["server", "cpu", "production"],
        min_ram_gb=8.0,
        recommended_model="Qwen/Qwen2.5-0.5B-Instruct",
    )
)

_register(
    ServingProfile(
        id="balanced",
        name="Balanced",
        description="Default profile. Auto-detects device, moderate resource usage. "
        "Good starting point for most setups.",
        tier=ProfileTier.BALANCED,
        device="auto",
        quantize=True,
        quant_bits=8,
        inference_pool_size=4,
        generate_timeout=120.0,
        request_timeout=120.0,
        idle_timeout=300.0,
        enable_guard=True,
        lazy_autoload=True,
        compute_threads=4,
        io_threads=2,
        workload_mode="balanced",
        tags=["default", "auto-detect"],
        min_ram_gb=8.0,
        recommended_model="Qwen/Qwen2.5-0.5B-Instruct",
    )
)

_register(
    ServingProfile(
        id="gpu_performance",
        name="GPU Performance",
        description="Optimized for GPU servers. More inference slots, higher timeouts, "
        "aggressive memory thresholds. Best with CUDA.",
        tier=ProfileTier.GPU_PERFORMANCE,
        device="cuda",
        quantize=False,
        quant_bits=16,
        inference_pool_size=8,
        generate_timeout=180.0,
        request_timeout=180.0,
        idle_timeout=600.0,
        enable_guard=True,
        lazy_autoload=True,
        memory_limit_mb=0,  # auto
        compute_threads=8,
        io_threads=4,
        workload_mode="inference",
        memory_pressure_warning=85.0,
        memory_pressure_critical=92.0,
        memory_pressure_emergency=97.0,
        temperature=0.8,
        max_new_tokens=512,
        tags=["gpu", "cuda", "high-throughput"],
        min_ram_gb=16.0,
        recommended_model="Qwen/Qwen2.5-1.5B-Instruct",
    )
)

_register(
    ServingProfile(
        id="training",
        name="Training",
        description="Optimized for training workloads. Switches to training mode, "
        "maximizes thread pools, enables checkpointing.",
        tier=ProfileTier.TRAINING,
        device="auto",
        quantize=True,
        quant_bits=8,
        inference_pool_size=2,
        generate_timeout=120.0,
        request_timeout=120.0,
        idle_timeout=0,  # don't unload during training
        enable_guard=True,
        lazy_autoload=True,
        compute_threads=8,
        io_threads=4,
        workload_mode="training",
        tags=["training", "fine-tuning", "distill"],
        min_ram_gb=16.0,
        recommended_model="Qwen/Qwen2.5-0.5B-Instruct",
    )
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_profiles() -> list[dict[str, Any]]:
    """Return all available profiles as dicts."""
    return [p.to_dict() for p in _PROFILES.values()]


def get_profile(profile_id: str) -> Optional[ServingProfile]:
    """Fetch a profile by ID, or None if not found."""
    return _PROFILES.get(profile_id)


def get_active_profile_id() -> str:
    """Return the ID of the currently active profile, or 'custom'."""
    return _active_profile_id


_active_profile_id: str = "balanced"


def apply_profile(profile_id: str) -> dict[str, Any]:
    """Validate and activate a profile. Returns what was set and what needs restart.

    This function only tracks the active profile. Use ``_apply_live_settings``
    in the router layer to patch live config (ServerConfig lives in the API server,
    not in this core package).
    """
    global _active_profile_id

    profile = _PROFILES.get(profile_id)
    if profile is None:
        raise ValueError(f"Unknown profile: {profile_id!r}. Available: {list(_PROFILES.keys())}")

    _active_profile_id = profile_id

    # Settings that can be applied at runtime by the caller
    live_settings = {
        "enable_guard": profile.enable_guard,
        "temperature": profile.temperature,
        "top_p": profile.top_p,
        "top_k": profile.top_k,
        "repetition_penalty": profile.repetition_penalty,
        "max_new_tokens": profile.max_new_tokens,
        "memory_pressure_warning": profile.memory_pressure_warning,
        "memory_pressure_critical": profile.memory_pressure_critical,
        "memory_pressure_emergency": profile.memory_pressure_emergency,
    }

    # Settings that require a server restart
    restart_settings = [
        "device", "quantize", "quant_bits", "quant_mode", "quant_clip",
        "inference_pool_size", "idle_timeout",
        "lazy_autoload", "memory_limit_mb", "max_restarts",
        "compute_threads", "io_threads", "workload_mode",
    ]

    logger.info("Applied serving profile: %s (%s)", profile.name, profile_id)

    return {
        "profile": profile.to_dict(),
        "live_settings": live_settings,
        "requires_restart": restart_settings,
        "active_profile_id": _active_profile_id,
    }


def detect_recommended_profile(ram_gb: float, has_gpu: bool = False) -> str:
    """Suggest the best profile based on available hardware."""
    if has_gpu and ram_gb >= 16:
        return "gpu_performance"
    if ram_gb >= 16:
        return "balanced"
    if ram_gb >= 8:
        return "cpu_optimized"
    return "cpu_only"
