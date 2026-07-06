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
        return cls(
            temperature=float(os.getenv("SLOUGHGT_TEMPERATURE", "0.8")),
            top_p=float(os.getenv("SLOUGHGT_TOP_P", "0.9")),
            top_k=int(os.getenv("SLOUGHGT_TOP_K", "50")),
            repetition_penalty=float(os.getenv("SLOUGHGT_REPETITION_PENALTY", "1.2")),
            max_new_tokens=int(os.getenv("SLOUGHGT_MAX_NEW_TOKENS", "200")),
            max_context_length=int(os.getenv("SLOUGHGT_MAX_CONTEXT_LENGTH", "1024")),
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

    inference_pool_size: int = field(default_factory=lambda: max(1, multiprocessing.cpu_count() // 2))
    request_timeout_seconds: float = 60.0
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

    enable_process_guard: bool = False
    enable_web: bool = False

    jwt_secret: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    @classmethod
    def from_env(cls) -> ServerConfig:
        raw_port = os.getenv("MAN_API_PORT", "").strip()
        port = int(raw_port) if raw_port else 8000
        return cls(
            port=port,
            host=os.getenv("MAN_HOST", "0.0.0.0"),
            log_level=os.getenv("MAN_LOG_LEVEL", "INFO").upper(),
            reload=os.getenv("MAN_RELOAD", "").lower() in ("1", "true", "yes"),
            autoload_model=os.getenv("MAN_AUTOLOAD_MODEL", "Qwen/Qwen2.5-0.5B-Instruct").strip(),
            autoload_device=(os.getenv("MAN_AUTOLOAD_DEVICE") or "auto").strip(),
            use_slonet=os.getenv("MAN_USE_SLONET", "0").strip().lower() in ("1", "true", "yes"),
            inference_pool_size=int(os.getenv("MAN_INFERENCE_POOL_SIZE", str(max(1, multiprocessing.cpu_count() // 2)))),
            request_timeout_seconds=float(os.getenv("MAN_REQUEST_TIMEOUT", "60.0")),
            enable_watchdog=os.getenv("MAN_WATCHDOG", "true").lower() == "true",
            enable_process_guard=os.getenv("MAN_ENABLE_PROCESS_GUARD", "").lower() in ("1", "true", "yes"),
            enable_workflow=os.getenv("MAN_AUTO_WORKFLOW", "true").lower() == "true",
            enable_health_monitor=os.getenv("MAN_HEALTH_MONITOR", "true").lower() == "true",
            health_monitor_interval=int(os.getenv("MAN_HEALTH_INTERVAL", "300")),
            enable_web=os.getenv("MAN_WEB", "").lower() in ("1", "true", "yes"),
            jwt_secret=os.getenv("JWT_SECRET", "dev-secret-change-in-production"),
            jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            jwt_expiration_hours=int(os.getenv("JWT_EXPIRATION_HOURS", "24")),
        )


gen_config = GenerationConfig.from_env()
