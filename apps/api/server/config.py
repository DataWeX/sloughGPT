"""
Server configuration — single source of truth for all environment-derived settings.

All config classes are frozen dataclasses with ``from_env()`` factory methods.
Import pattern::

    from config import gen_config, ServerConfig

    port = ServerConfig.from_env().port
    temp = gen_config.temperature
"""

from __future__ import annotations

import os
import secrets
import multiprocessing
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
            repetition_penalty=float(_env("SLO_REPETITION_PENALTY", "SLOUGHGT_REPETITION_PENALTY", "1.2")),
            max_new_tokens=int(_env("SLO_MAX_NEW_TOKENS", "SLOUGHGT_MAX_NEW_TOKENS", "200")),
            max_context_length=int(_env("SLO_MAX_CONTEXT_LENGTH", "SLOUGHGT_MAX_CONTEXT_LENGTH", "1024")),
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

    inference_pool_size: int = field(default_factory=lambda: max(1, multiprocessing.cpu_count() // 2))
    request_timeout_seconds: float = 120.0
    streaming_skip_paths: tuple[str, ...] = (
        "/chat/stream", "/auto-train/stream", "/session/",
        "/generate/stream", "/models/load", "/inference/generate", "/chat",
    )

    enable_watchdog: bool = True
    watchdog_poll_seconds: int = 15
    watchdog_max_failures: int = 3
    watchdog_grace_seconds: int = 120

    enable_workflow: bool = True
    enable_health_monitor: bool = True
    health_monitor_interval: int = 300

    enable_process_guard: bool = True
    enable_web: bool = False

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
            inference_pool_size=int(os.getenv("SLO_INFERENCE_POOL_SIZE", str(max(1, multiprocessing.cpu_count() // 2)))),
            request_timeout_seconds=float(os.getenv("SLO_REQUEST_TIMEOUT", "120.0")),
            enable_watchdog=os.getenv("SLO_WATCHDOG", "true").lower() == "true",
            enable_process_guard=os.getenv("SLO_ENABLE_PROCESS_GUARD", "true").lower() in ("1", "true", "yes"),
            enable_workflow=os.getenv("SLO_AUTO_WORKFLOW", "true").lower() == "true",
            enable_health_monitor=os.getenv("SLO_HEALTH_MONITOR", "true").lower() == "true",
            health_monitor_interval=int(os.getenv("SLO_HEALTH_INTERVAL", "300")),
            enable_web=os.getenv("SLO_WEB", "").lower() in ("1", "true", "yes"),
            jwt_secret=os.getenv("SLO_JWT_SECRET") or os.getenv("JWT_SECRET") or secrets.token_urlsafe(64),
            jwt_algorithm=os.getenv("SLO_JWT_ALGORITHM", os.getenv("JWT_ALGORITHM", "HS256")),
            jwt_expiration_hours=int(os.getenv("SLO_JWT_EXPIRATION_HOURS", os.getenv("JWT_EXPIRATION_HOURS", "24"))),
        )


gen_config = GenerationConfig.from_env()
