"""
Feature flags for sloughGPT.

FEATURE: feature-flags — Runtime feature toggles with env var overrides.
DO NOT DELETE. Controls which features are active.

Usage:
    from domains.shared.feature_flags import is_enabled, FeatureFlags

    if is_enabled("native_c_inference"):
        from domains.inference.native.engine import NativeEngine
        ...

    # Register new features
    FeatureFlags.register("my_feature", enabled=False, description="My new feature")

Env overrides:
    SLO_FF_NATIVE_C_INFERENCE=1   → enable
    SLO_FF_NATIVE_C_INFERENCE=0   → disable
"""

import os
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
from enum import Enum

logger = logging.getLogger("slo.feature_flags")


class FlagStatus(str, Enum):
    """Feature flag status."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    EXPERIMENTAL = "experimental"  # enabled but not guaranteed


@dataclass
class FeatureFlag:
    """A single feature flag."""
    name: str
    description: str
    status: FlagStatus = FlagStatus.DISABLED
    env_var: str = ""  # auto-generated from name if empty

    def __post_init__(self):
        if not self.env_var:
            self.env_var = f"SLO_FF_{self.name.upper()}"

    @property
    def is_enabled(self) -> bool:
        """Check if feature is enabled. Env var overrides config."""
        env_val = os.environ.get(self.env_var)
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return self.status in (FlagStatus.ENABLED, FlagStatus.EXPERIMENTAL)


class FeatureFlags:
    """Feature flag registry."""

    _flags: Dict[str, FeatureFlag] = {}
    _config_path: Optional[Path] = None

    @classmethod
    def register(
        cls,
        name: str,
        description: str = "",
        status: FlagStatus = FlagStatus.DISABLED,
    ) -> FeatureFlag:
        """Register a new feature flag."""
        if name in cls._flags:
            return cls._flags[name]
        flag = FeatureFlag(name=name, description=description, status=status)
        cls._flags[name] = flag
        return flag

    @classmethod
    def is_enabled(cls, name: str) -> bool:
        """Check if a feature flag is enabled."""
        flag = cls._flags.get(name)
        if flag is None:
            logger.warning("Unknown feature flag: %s (defaulting to disabled)", name)
            return False
        return flag.is_enabled

    @classmethod
    def set_status(cls, name: str, status: FlagStatus) -> None:
        """Update a feature flag's status."""
        if name not in cls._flags:
            raise KeyError(f"Unknown feature flag: {name}")
        cls._flags[name].status = status

    @classmethod
    def list_all(cls) -> Dict[str, dict]:
        """List all registered feature flags."""
        return {
            name: {
                "description": flag.description,
                "status": flag.status.value,
                "enabled": flag.is_enabled,
                "env_var": flag.env_var,
            }
            for name, flag in cls._flags.items()
        }

    @classmethod
    def load_config(cls, path: Optional[Path] = None) -> None:
        """Load feature flag statuses from JSON config file."""
        if path is None:
            path = Path(__file__).parent.parent.parent.parent / "config" / "feature_flags.json"
        cls._config_path = path
        if not path.exists():
            logger.info("No feature flags config at %s, using defaults", path)
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for name, status_str in data.items():
                if name in cls._flags:
                    try:
                        cls._flags[name].status = FlagStatus(status_str)
                    except ValueError:
                        logger.warning("Invalid status '%s' for flag %s", status_str, name)
            logger.info("Loaded feature flags from %s (%d flags)", path, len(data))
        except Exception as e:
            logger.warning("Failed to load feature flags: %s", e)

    @classmethod
    def save_config(cls, path: Optional[Path] = None) -> None:
        """Save current flag statuses to JSON config file."""
        if path is None:
            path = cls._config_path or Path(__file__).parent.parent.parent.parent / "config" / "feature_flags.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: flag.status.value for name, flag in cls._flags.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved feature flags to %s", path)


def is_enabled(name: str) -> bool:
    """Check if a feature flag is enabled. Convenience shortcut."""
    return FeatureFlags.is_enabled(name)


# =============================================================================
# REGISTER ALL KNOWN FEATURES
# =============================================================================

def _register_defaults():
    """Register all known features from FEATURE tags in the codebase."""
    FeatureFlags.register(
        "slonet_provider",
        description="Sole torch-free inference engine (SloNetChatProvider)",
        status=FlagStatus.ENABLED,
    )
    FeatureFlags.register(
        "native_c_inference",
        description="C transformer forward pass using Apple Accelerate BLAS",
        status=FlagStatus.DISABLED,  # opt-in: use setup_providers(native_slnc_path=...) instead
    )
    FeatureFlags.register(
        "cloud_vector_store",
        description="Pinecone vector store setup utility",
        status=FlagStatus.DISABLED,  # not yet wired into main app
    )
    FeatureFlags.register(
        "soul_format",
        description="Self-contained .soul model format (weights + personality)",
        status=FlagStatus.ENABLED,
    )
    FeatureFlags.register(
        "soul_manager",
        description="Runtime personality/soul switching without restart",
        status=FlagStatus.ENABLED,
    )
    FeatureFlags.register(
        "slonet_kernels",
        description="Numba JIT compiled kernels for inference hot path",
        status=FlagStatus.EXPERIMENTAL,
    )
    FeatureFlags.register(
        "multimodal",
        description="Vision, speech, image captioning, diffusion, TTS",
        status=FlagStatus.ENABLED,
    )
    FeatureFlags.register(
        "cross_attention",
        description="Cross-attention for multimodal patch injection",
        status=FlagStatus.ENABLED,
    )
    FeatureFlags.register(
        "kv_cache",
        description="KV cache for greedy generation (per-layer)",
        status=FlagStatus.ENABLED,
    )
    FeatureFlags.register(
        "session_kv_cache",
        description="Cross-turn KV cache for multi-turn conversations",
        status=FlagStatus.ENABLED,
    )
    FeatureFlags.register(
        "model_server",
        description="Crash-resilient model serving (semaphore, circuit breaker)",
        status=FlagStatus.ENABLED,
    )
    FeatureFlags.register(
        "model_registry",
        description="Composable model registry wrapping ModelServer instances",
        status=FlagStatus.ENABLED,
    )
    FeatureFlags.register(
        "process_isolation",
        description="Subprocess-based model isolation (ProcessGuard)",
        status=FlagStatus.ENABLED,
    )
    FeatureFlags.register(
        "on_device_training",
        description="Activity classifier training on-device",
        status=FlagStatus.EXPERIMENTAL,
    )
    FeatureFlags.register(
        "quantization",
        description="Per-tensor int8/int4 quantization (pure NumPy)",
        status=FlagStatus.ENABLED,
    )
    FeatureFlags.register(
        "hf_finetune",
        description="HuggingFace model fine-tuning with optional LoRA",
        status=FlagStatus.ENABLED,
    )
    FeatureFlags.register(
        "vlm",
        description="Vision-Language Model (training + inference)",
        status=FlagStatus.EXPERIMENTAL,
    )
    FeatureFlags.register(
        "dpo",
        description="Direct Preference Optimization training",
        status=FlagStatus.EXPERIMENTAL,
    )
    FeatureFlags.register(
        "context_managers",
        description="Personality, Memory, Style, Task steering managers",
        status=FlagStatus.ENABLED,
    )
    FeatureFlags.register(
        "knowledge_memory",
        description="Fact storage, search, and context injection",
        status=FlagStatus.ENABLED,
    )
    FeatureFlags.register(
        "semantic_cache",
        description="Response caching via embedding similarity",
        status=FlagStatus.ENABLED,
    )
    FeatureFlags.register(
        "llm_nlp",
        description="LLM-powered natural language processing for shell",
        status=FlagStatus.EXPERIMENTAL,
    )
    FeatureFlags.register(
        "feature_flags",
        description="Runtime feature toggles with env var overrides",
        status=FlagStatus.ENABLED,
    )
    FeatureFlags.register(
        "slonet_provider_tests",
        description="Tests for SloNetChatProvider features",
        status=FlagStatus.ENABLED,
    )
    FeatureFlags.register(
        "slonet_provider_wave_i",
        description="Wave I: slonet_provider 100% coverage, chat_stream error propagation",
        status=FlagStatus.ENABLED,
    )
    FeatureFlags.register(
        "slonet_wave_f",
        description="Wave F: from_slnc (plain + quantized), chat/chat_stream coverage",
        status=FlagStatus.ENABLED,
    )


_register_defaults()
