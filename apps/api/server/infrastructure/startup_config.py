"""Startup configuration — allows customizing startup stages and hooks.

Usage:
    from infrastructure.startup_config import get_startup_config

    config = get_startup_config()
    config.disable_hook("wandb")
    config.set_hook_timeout("model_load", 120.0)
    config.set_stage_timeout("CRITICAL", 60.0)

Environment variables:
    SLO_STARTUP_DISABLE_HOOKS: Comma-separated list of hooks to disable
    SLO_STARTUP_HOOK_TIMEOUTS: Comma-separated hook=timeout pairs
    SLO_STARTUP_STAGE_TIMEOUTS: Comma-separated stage=timeout pairs
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field


@dataclass
class StartupConfig:
    """Configuration for startup stages and hooks."""

    # Hooks to disable
    disabled_hooks: set[str] = field(default_factory=set)

    # Custom hook timeouts (hook_name -> timeout_seconds)
    hook_timeouts: dict[str, float] = field(default_factory=dict)

    # Custom stage timeouts (stage_name -> timeout_seconds)
    stage_timeouts: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._load_from_env()

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Disabled hooks
        disabled = os.environ.get("SLO_STARTUP_DISABLE_HOOKS", "")
        if disabled:
            self.disabled_hooks = {h.strip() for h in disabled.split(",") if h.strip()}

        # Hook timeouts
        hook_timeouts = os.environ.get("SLO_STARTUP_HOOK_TIMEOUTS", "")
        if hook_timeouts:
            for pair in hook_timeouts.split(","):
                if "=" in pair:
                    name, timeout = pair.split("=", 1)
                    try:
                        self.hook_timeouts[name.strip()] = float(timeout.strip())
                    except ValueError:
                        pass

        # Stage timeouts
        stage_timeouts = os.environ.get("SLO_STARTUP_STAGE_TIMEOUTS", "")
        if stage_timeouts:
            for pair in stage_timeouts.split(","):
                if "=" in pair:
                    name, timeout = pair.split("=", 1)
                    try:
                        self.stage_timeouts[name.strip().upper()] = float(timeout.strip())
                    except ValueError:
                        pass

    def is_hook_disabled(self, hook_name: str) -> bool:
        """Check if a hook is disabled."""
        return hook_name in self.disabled_hooks

    def get_hook_timeout(self, hook_name: str, default: float) -> float:
        """Get timeout for a hook (custom or default)."""
        return self.hook_timeouts.get(hook_name, default)

    def get_stage_timeout(self, stage_name: str, default: float) -> float:
        """Get timeout for a stage (custom or default)."""
        return self.stage_timeouts.get(stage_name.upper(), default)

    def disable_hook(self, hook_name: str) -> None:
        """Disable a startup hook."""
        self.disabled_hooks.add(hook_name)

    def enable_hook(self, hook_name: str) -> None:
        """Enable a startup hook."""
        self.disabled_hooks.discard(hook_name)

    def set_hook_timeout(self, hook_name: str, timeout: float) -> None:
        """Set custom timeout for a hook."""
        self.hook_timeouts[hook_name] = timeout

    def set_stage_timeout(self, stage_name: str, timeout: float) -> None:
        """Set custom timeout for a stage."""
        self.stage_timeouts[stage_name.upper()] = timeout

    def to_dict(self) -> dict:
        """Export configuration as dict."""
        return {
            "disabled_hooks": sorted(self.disabled_hooks),
            "hook_timeouts": dict(self.hook_timeouts),
            "stage_timeouts": dict(self.stage_timeouts),
        }


# ── Singleton ───────────────────────────────────────────────────────
_config: StartupConfig | None = None
_config_lock = threading.Lock()


def get_startup_config() -> StartupConfig:
    """Return the global startup config instance."""
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = StartupConfig()
    return _config
