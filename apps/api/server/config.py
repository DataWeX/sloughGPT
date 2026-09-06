"""
Server configuration — single source of truth for all environment-derived settings.

All config classes are frozen dataclasses with ``from_env()`` factory methods.
Import pattern::

    from config import gen_config, ServerConfig

    port = ServerConfig.from_env().port
    temp = gen_config.temperature
"""

from __future__ import annotations

import multiprocessing
import os
import secrets
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GenerationConfig:
    """Production defaults for text generation. All overridable via env vars."""

    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    max_new_tokens: int = 200
    max_context_length: int = 1024

    @classmethod
    def from_env(cls) -> GenerationConfig:
        def _env(canonical: str, legacy: str, default: str) -> str:
            return os.getenv(canonical, os.getenv(legacy, default))

        return cls(
            temperature=float(_env("SLO_TEMPERATURE", "SLOUGHGT_TEMPERATURE", "0.8")),
            top_p=float(_env("SLO_TOP_P", "SLOUGHGT_TOP_P", "0.9")),
            top_k=int(_env("SLO_TOP_K", "SLOUGHGT_TOP_K", "50")),
            repetition_penalty=float(
                _env("SLO_REPETITION_PENALTY", "SLOUGHGT_REPETITION_PENALTY", "1.2")
            ),
            max_new_tokens=int(_env("SLO_MAX_NEW_TOKENS", "SLOUGHGT_MAX_NEW_TOKENS", "200")),
            max_context_length=int(
                _env("SLO_MAX_CONTEXT_LENGTH", "SLOUGHGT_MAX_CONTEXT_LENGTH", "1024")
            ),
        )


@dataclass(frozen=True)
class ServerConfig:
    """Server-level configuration from environment variables."""

    port: int = 8000
    host: str = "0.0.0.0"
    log_level: str = "INFO"
    reload: bool = False

    autoload_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    autoload_device: str = "auto"
    use_slonet: bool = False
    quantize_slonet: bool = True  # int8 quantization enabled by default (saves ~50% RAM)
    quant_bits: int = 8
    quant_mode: str = "symmetric"
    quant_clip: float = 0.999

    inference_pool_size: int = field(
        default_factory=lambda: max(1, multiprocessing.cpu_count() // 2)
    )
    request_timeout_seconds: float = 120.0
    # timeout for model.generate() calls (env: SLO_GENERATE_TIMEOUT)
    generate_timeout: float = 120.0
    # timeout for model loading during startup (env: SLO_STARTUP_MODEL_LOAD_TIMEOUT)
    startup_model_load_timeout: float = 120.0
    # timeout for entire startup sequence (env: SLO_STARTUP_TOTAL_TIMEOUT)
    startup_total_timeout: float = 120.0
    startup_register_generate_timeout: float = (
        120.0  # timeout for register_generate() calls (env: SLO_REGISTER_GENERATE_TIMEOUT)
    )
    streaming_skip_paths: tuple[str, ...] = (
        "/chat/stream",
        "/training/stream",
        "/session/",
        "/generate/stream",
        "/models/load",
        "/inference/generate",
        "/chat",
    )

    enable_watchdog: bool = True
    watchdog_poll_seconds: int = 15
    watchdog_max_failures: int = 3
    watchdog_grace_seconds: int = 120

    enable_workflow: bool = True
    enable_health_monitor: bool = True
    health_monitor_interval: int = 300

    native_soul_path: str = ""  # path to .soul file for native-trained model
    enable_process_guard: bool = True  # production default: guard _load_hf_model() in a subprocess
    lazy_guard_autoload: bool = (
        True  # defer parent weight load when a ProcessGuard + .slnc are available
    )
    enable_inference_engine: bool = False  # run model in separate subprocess (isolated memory)
    inference_engine_host: str = "127.0.0.1"  # inference engine bind address
    inference_engine_port: int = 0  # 0 = auto-assign
    inference_engine_timeout: float = 300.0  # seconds to wait for engine startup
    process_guard_memory_limit_mb: float = 0.0  # 0 = auto-size from model file
    enable_web: bool = False

    idle_timeout_seconds: float = 300.0  # unload model after 5 min idle (0 = disabled)

    memory_pressure_warning: float = 80.0  # percent — log warning, clear caches
    memory_pressure_critical: float = 90.0  # percent — force GC, drop KV caches, release weights
    memory_pressure_emergency: float = 95.0  # percent — block new model loads + inference

    jwt_secret: str = ""  # auto-generated if empty
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    @classmethod
    def from_env(cls) -> ServerConfig:
        raw_port = os.getenv("SLO_API_PORT", "").strip()
        port = int(raw_port) if raw_port else 8000
        return cls(
            port=port,
            host=os.getenv("SLO_HOST", "0.0.0.0"),
            log_level=os.getenv("SLO_LOG_LEVEL", "INFO").upper(),
            reload=os.getenv("SLO_RELOAD", "").lower() in ("1", "true", "yes"),
            autoload_model=os.getenv("SLO_AUTOLOAD_MODEL", "Qwen/Qwen2.5-0.5B-Instruct").strip(),
            autoload_device=(os.getenv("SLO_AUTOLOAD_DEVICE") or "auto").strip(),
            use_slonet=os.getenv("SLO_USE_SLONET", "0").strip().lower() in ("1", "true", "yes"),
            quantize_slonet=os.getenv("SLO_QUANTIZE", "true").lower() in ("1", "true", "yes"),
            quant_bits=int(os.getenv("SLO_QUANT_BITS", "8")),
            quant_mode=os.getenv("SLO_QUANT_MODE", "symmetric").strip(),
            quant_clip=float(os.getenv("SLO_QUANT_CLIP", "0.999")),
            inference_pool_size=int(
                os.getenv("SLO_INFERENCE_POOL_SIZE", str(max(1, multiprocessing.cpu_count() // 2)))
            ),
            request_timeout_seconds=float(os.getenv("SLO_REQUEST_TIMEOUT", "120.0")),
            generate_timeout=float(os.getenv("SLO_GENERATE_TIMEOUT", "120.0")),
            startup_model_load_timeout=float(os.getenv("SLO_STARTUP_MODEL_LOAD_TIMEOUT", "120.0")),
            startup_total_timeout=float(os.getenv("SLO_STARTUP_TOTAL_TIMEOUT", "120.0")),
            startup_register_generate_timeout=float(
                os.getenv("SLO_REGISTER_GENERATE_TIMEOUT", "120.0")
            ),
            enable_watchdog=os.getenv("SLO_WATCHDOG", "true").lower() == "true",
            native_soul_path=os.getenv("SLO_NATIVE_SOUL_PATH", "").strip(),
            enable_process_guard=os.getenv("SLO_ENABLE_PROCESS_GUARD", "true").lower()
            in ("1", "true", "yes"),
            lazy_guard_autoload=os.getenv("SLO_LAZY_GUARD_AUTOLOAD", "true").lower()
            in ("1", "true", "yes"),
            enable_inference_engine=os.getenv("SLO_INFERENCE_ENGINE", "false").lower()
            in ("1", "true", "yes"),
            inference_engine_host=os.getenv("SLO_INFERENCE_ENGINE_HOST", "127.0.0.1").strip(),
            inference_engine_port=int(os.getenv("SLO_INFERENCE_ENGINE_PORT", "0")),
            inference_engine_timeout=float(os.getenv("SLO_INFERENCE_ENGINE_TIMEOUT", "300")),
            process_guard_memory_limit_mb=float(
                os.getenv("SLO_PROCESS_GUARD_MEMORY_LIMIT_MB", "0")
            ),
            enable_workflow=os.getenv("SLO_AUTO_WORKFLOW", "true").lower() == "true",
            enable_health_monitor=os.getenv("SLO_HEALTH_MONITOR", "true").lower() == "true",
            health_monitor_interval=int(os.getenv("SLO_HEALTH_INTERVAL", "300")),
            enable_web=os.getenv("SLO_WEB", "").lower() in ("1", "true", "yes"),
            idle_timeout_seconds=float(os.getenv("SLO_IDLE_TIMEOUT", "300")),
            memory_pressure_warning=float(os.getenv("SLO_MEMORY_PRESSURE_WARNING", "80")),
            memory_pressure_critical=float(os.getenv("SLO_MEMORY_PRESSURE_CRITICAL", "90")),
            memory_pressure_emergency=float(os.getenv("SLO_MEMORY_PRESSURE_EMERGENCY", "95")),
            jwt_secret=os.getenv("SLO_JWT_SECRET")
            or os.getenv("JWT_SECRET")
            or secrets.token_urlsafe(64),
            jwt_algorithm=os.getenv("SLO_JWT_ALGORITHM", os.getenv("JWT_ALGORITHM", "HS256")),
            jwt_expiration_hours=int(
                os.getenv("SLO_JWT_EXPIRATION_HOURS", os.getenv("JWT_EXPIRATION_HOURS", "24"))
            ),
        )


gen_config = GenerationConfig.from_env()

# ── Runtime ProcessGuard toggle ──────────────────────────────────────
# Initialised from ServerConfig (SLO_ENABLE_PROCESS_GUARD, default true —
# ProcessGuard is the production default). Can be changed at runtime via
# POST /models/process-guard; persists until restart.
_runtime_process_guard_enabled: bool = ServerConfig.from_env().enable_process_guard


def get_process_guard_enabled() -> bool:
    """Return the current runtime ProcessGuard enabled state."""
    return _runtime_process_guard_enabled


def set_process_guard_enabled(enabled: bool) -> None:
    """Set the runtime ProcessGuard enabled state (persists until restart)."""
    global _runtime_process_guard_enabled
    _runtime_process_guard_enabled = enabled
