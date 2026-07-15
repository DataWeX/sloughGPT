"""
Config System — typed hierarchical config with env override, file defaults,
per-environment profiles, and runtime reload via EventBus.

Usage:
    from domains.infrastructure.config import get_config
    cfg = get_config()
    cfg.model.name          # "Qwen/Qwen2.5-0.5B-Instruct"
    cfg.model.device        # "cpu"
    cfg.server.port         # 8000
    cfg.log_level           # "INFO"

Env overrides use double-underscore for nesting:
    SLO_MODEL__NAME=gpt2    overrides config.model.name
    SLO_SERVER__PORT=9000   overrides config.server.port
"""

from __future__ import annotations

import os
import logging
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("slo.config")

# ── Nested Config Models ──


class ModelConfig(BaseModel):
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "auto"
    max_length: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_new_tokens: int = 200
    use_slonet: bool = False
    autoload: bool = True
    use_flash_attention: bool = False


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    reload: bool = False
    request_timeout: float = 60.0
    inference_pool_size: int = 2


class FeaturesConfig(BaseModel):
    auto_workflow: bool = True
    health_monitor: bool = True
    health_interval: int = 300
    watchdog: bool = True
    web: bool = False


class AuthConfig(BaseModel):
    jwt_secret: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    api_keys_enabled: bool = False


class TrackingConfig(BaseModel):
    """Experiment tracking (W&B, MLflow) configuration."""

    provider: str = "none"
    wandb_training_enabled: bool = False
    wandb_server_enabled: bool = False
    wandb_entity: str = ""
    wandb_project: str = "sloughgpt"
    wandb_api_key: str = ""
    wandb_mode: str = "online"
    wandb_dir: str = ""
    wandb_server_interval: int = 60
    wandb_server_run_name: str = ""
    mlflow_tracking_uri: str = ""


class EmbeddingConfig(BaseModel):
    """Embedding provider configuration."""

    provider: str = "n_gram"
    openai_api_key: str = ""


class StorageConfig(BaseModel):
    data_dir: str = "data"
    datasets_dir: str = "datasets"
    models_dir: str = "models"
    checkpoint_dir: str = "models/auto-training"


class AppConfig(BaseModel):
    """Top-level config — all sub-configs are nested here."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    model_config = ConfigDict(extra="forbid")


_ENV_PREFIX = "SLO_"
_SEPARATOR = "__"


def _env_to_key(env_name: str) -> str:
    """SLO_MODEL__NAME → model.name"""
    rest = env_name[len(_ENV_PREFIX):]
    parts = rest.lower().split(_SEPARATOR)
    return ".".join(parts)


# Env vars consumed by external systems (ServerConfig, etc.) — silently ignored.
_SKIP_ENV_KEYS: frozenset[str] = frozenset({
    # ServerConfig (apps/api/server/config.py) — not in infrastructure AppConfig schema
    "SLO_ENABLE_PROCESS_GUARD",
    "SLO_AUTOLOAD_MODEL",
    "SLO_AUTOLOAD_DEVICE",
    "SLO_USE_SLONET",
    "SLO_STARTUP_PROFILE",
    "SLO_RELOAD",
    "SLO_GUARD_MEMORY_LIMIT_MB",
    "SLO_GUARD_MAX_RESTARTS",
    "SLO_GUARD_RESTART_DELAY",
    # Server feature flags consumed directly in main.py lifecycle
    "SLO_HEALTH_MONITOR",
    "SLO_WATCHDOG",
    "SLO_WEB",
    "SLO_AUTO_WORKFLOW",
})


def _apply_env_overrides(config: AppConfig) -> AppConfig:
    """Walk environment variables, parse SLO_* keys, override config values."""
    overrides: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        if key in _SKIP_ENV_KEYS:
            continue
        if key.startswith("SLO_RATE_LIMIT__"):
            continue
        if key == _ENV_PREFIX.rstrip("_"):
            continue
        dot_path = _env_to_key(key)
        overrides[dot_path] = value

    if not overrides:
        return config

    current = config.model_dump()
    for dot_path, value in overrides.items():
        parts = dot_path.split(".")
        target = current
        for part in parts[:-1]:
            if part not in target:
                logger.warning("Unknown config key: %s (from env %s)", dot_path, _ENV_PREFIX + dot_path.upper().replace(".", _SEPARATOR),
                    extra={"tag": "INFRA"})
                break
            target = target[part]
        else:
            leaf = parts[-1]
            if leaf in target:
                expected_type = type(target[leaf])
                try:
                    if expected_type is bool:
                        target[leaf] = value.lower() in ("1", "true", "yes")
                    elif expected_type is int:
                        target[leaf] = int(value)
                    elif expected_type is float:
                        target[leaf] = float(value)
                    else:
                        target[leaf] = value
                except (ValueError, TypeError):
                    logger.warning("Cannot coerce %s=%s to %s", dot_path, value, expected_type.__name__,
                        extra={"tag": "INFRA"})
            else:
                logger.warning("Unknown config key: %s (from env %s)", dot_path, _ENV_PREFIX + dot_path.upper().replace(".", _SEPARATOR),
                    extra={"tag": "INFRA"})

    return AppConfig.model_validate(current)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file. Returns empty dict if file doesn't exist or parse fails."""
    try:
        import yaml
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        logger.warning("Failed to load config file: %s", path,
            extra={"tag": "INFRA"})
    return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge — override values win."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ── Config Manager ──


class ConfigManager:
    """Manages config lifecycle: load, merge, reload, subscribe."""

    def __init__(self, config_dir: str | Path | None = None):
        self._config_dir = Path(config_dir) if config_dir else Path.cwd() / "config"
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._config: AppConfig = self._build()
        self._lock = threading.Lock()
        self._reload_callbacks: list[callable] = []

    def _build(self) -> AppConfig:
        """Load defaults + file overrides + env overrides."""
        defaults = AppConfig().model_dump()

        # Load YAML defaults file
        yaml_data = _load_yaml(self._config_dir / "defaults.yaml")
        merged = _deep_merge(defaults, yaml_data)

        # Load profile override (SLO_ENV or SLO_PROFILE)
        profile = os.environ.get("SLO_ENV") or os.environ.get("SLO_PROFILE") or ""
        if profile:
            profile_data = _load_yaml(self._config_dir / f"{profile}.yaml")
            merged = _deep_merge(merged, profile_data)

        config = AppConfig.model_validate(merged)
        config = _apply_env_overrides(config)
        return config

    @property
    def config(self) -> AppConfig:
        return self._config

    def reload(self) -> AppConfig:
        """Reload config from disk + env. Fires callbacks and EventBus event."""
        with self._lock:
            new_config = self._build()
            old_config = self._config
            self._config = new_config

        for cb in self._reload_callbacks:
            try:
                cb(new_config, old_config)
            except Exception:
                logger.exception("Config reload callback failed", extra={"tag": "INFRA"})

        try:
            from domains.infrastructure.event_bus import get_event_bus
            bus = get_event_bus()
            import asyncio
            payload = {
                "old": old_config.model_dump() if hasattr(old_config, "model_dump") else {},
                "new": new_config.model_dump(),
            }
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(bus.emit("config.changed", payload, source="config_manager"))
            except RuntimeError:
                # No running event loop — use asyncio.run() so async handlers are awaited
                asyncio.run(bus.emit("config.changed", payload, source="config_manager"))
        except Exception:
            logger.warning("Could not emit config.changed event",
                extra={"tag": "INFRA"})

        return new_config

    def on_reload(self, callback: callable):
        """Register a callback for config reloads. Receives (new, old)."""
        self._reload_callbacks.append(callback)

    def dump(self) -> dict[str, Any]:
        return self._config.model_dump()


# ── Singleton ──

_default_manager: ConfigManager | None = None
_manager_lock = threading.Lock()


def get_config_manager() -> ConfigManager:
    global _default_manager
    if _default_manager is None:
        with _manager_lock:
            if _default_manager is None:
                _default_manager = ConfigManager()
    return _default_manager


def set_config_manager(manager: ConfigManager):
    global _default_manager
    _default_manager = manager


def get_config() -> AppConfig:
    return get_config_manager().config


def reload_config() -> AppConfig:
    return get_config_manager().reload()
