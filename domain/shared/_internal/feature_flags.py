from __future__ import annotations

"""
Feature flags for sloughGPT.

FEATURE: feature-flags — Runtime feature toggles with env var overrides.
DO NOT DELETE. Controls which features are active.

Usage:
    from domain.shared._internal.feature_flags import is_enabled, FeatureFlags

    if is_enabled("native_c_inference"):
        from domain.inference._internal.native.engine import NativeEngine
        ...

    # Register new features
    FeatureFlags.register("my_feature", enabled=False, description="My new feature")

Env overrides:
    SLO_FF_NATIVE_C_INFERENCE=1   → enable
    SLO_FF_NATIVE_C_INFERENCE=0   → disable

Versioning & change ledger:
    Every flag carries a semver ``version`` (+ ``since`` date). Status
    changes and version bumps are appended to ``config/flag_history.jsonl``
    (append-only — nothing is ever overwritten), so flag drift such as an
    out-of-band config edit can be attributed after the fact::

        FeatureFlags.bump_version("native_c_inference", "1.1.0")
        FeatureFlags.read_history()   # -> [{"kind": "version", ...}, ...]
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger("slo.feature_flags")

#: History file (append-only change ledger) lives next to the config file.
HISTORY_FILENAME = "flag_history.jsonl"

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


class FlagStatus(StrEnum):
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
    #: Semantic version of the feature behind this flag (bumped via
    #: ``FeatureFlags.bump_version``; every bump is ledgered).
    version: str = "1.0.0"
    #: ISO date the feature entered its current version (defaults to today).
    since: str = ""

    def __post_init__(self):
        if not self.env_var:
            self.env_var = f"SLO_FF_{self.name.upper()}"
        if not self.since:
            self.since = datetime.now(UTC).date().isoformat()

    @property
    def is_enabled(self) -> bool:
        """Check if feature is enabled. Env var overrides config."""
        env_val = os.environ.get(self.env_var)
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return self.status in (FlagStatus.ENABLED, FlagStatus.EXPERIMENTAL)


class FeatureFlags:
    """Feature flag registry."""

    _flags: dict[str, FeatureFlag] = {}
    _config_path: Path | None = None
    _history_path: Path | None = None

    @classmethod
    def register(
        cls,
        name: str,
        description: str = "",
        status: FlagStatus = FlagStatus.DISABLED,
        version: str = "1.0.0",
        since: str = "",
    ) -> FeatureFlag:
        """Register a new feature flag."""
        if name in cls._flags:
            return cls._flags[name]
        flag = FeatureFlag(
            name=name, description=description, status=status, version=version, since=since
        )
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
    def set_status(cls, name: str, status: FlagStatus, source: str = "runtime") -> None:
        """Update a feature flag's status, ledgering the change.

        *source* names where the change came from ("runtime", "config",
        "cli") and is recorded in the append-only history file so flag
        drift can be attributed later.
        """
        if name not in cls._flags:
            raise KeyError(f"Unknown feature flag: {name}")
        flag = cls._flags[name]
        previous = flag.status
        flag.status = status
        if previous != status:
            cls._append_history(
                {
                    "kind": "status",
                    "flag": name,
                    "from": previous.value,
                    "to": status.value,
                    "version": flag.version,
                    "source": source,
                }
            )

    @classmethod
    def bump_version(cls, name: str, new_version: str, source: str = "manual") -> str:
        """Bump a flag's feature version (semver) and ledger the bump."""
        if name not in cls._flags:
            raise KeyError(f"Unknown feature flag: {name}")
        if not _SEMVER_RE.match(new_version):
            raise ValueError(f"Version must be MAJOR.MINOR.PATCH: {new_version!r}")
        flag = cls._flags[name]
        previous = flag.version
        if previous == new_version:
            return previous
        flag.version = new_version
        flag.since = datetime.now(UTC).date().isoformat()
        cls._append_history(
            {
                "kind": "version",
                "flag": name,
                "from": previous,
                "to": new_version,
                "version": new_version,
                "source": source,
            }
        )
        return new_version

    @classmethod
    def list_all(cls) -> dict[str, dict]:
        """List all registered feature flags."""
        return {
            name: {
                "description": flag.description,
                "status": flag.status.value,
                "enabled": flag.is_enabled,
                "env_var": flag.env_var,
                "version": flag.version,
                "since": flag.since,
            }
            for name, flag in cls._flags.items()
        }

    # ── Change ledger (append-only) ──────────────────────────────────────

    @classmethod
    def history_path(cls) -> Path:
        """Where the append-only flag change ledger lives."""
        if cls._history_path is not None:
            return cls._history_path
        if cls._config_path is not None:
            return cls._config_path.parent / HISTORY_FILENAME
        return Path(__file__).parent.parent.parent.parent / "config" / HISTORY_FILENAME

    @classmethod
    def _append_history(cls, entry: dict) -> None:
        """Append one ledger line. Never raises — history must not break reads."""
        entry = {**entry, "at": datetime.now(UTC).isoformat()}
        path = cls.history_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Failed to append flag history to %s: %s", path, exc)

    @classmethod
    def read_history(cls) -> list[dict]:
        """Read the ledger back (malformed lines are skipped)."""
        path = cls.history_path()
        if not path.exists():
            return []
        entries: list[dict] = []
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(obj, dict) and "flag" in obj:
                        entries.append(obj)
        except OSError as exc:
            logger.warning("Failed to read flag history from %s: %s", path, exc)
        return entries

    @classmethod
    def _last_recorded_statuses(cls) -> dict[str, str]:
        """Last status recorded per flag in the ledger."""
        out: dict[str, str] = {}
        for entry in cls.read_history():
            if entry.get("kind", "status") == "status":
                out[entry["flag"]] = entry.get("to", "")
        return out

    @classmethod
    def load_config(cls, path: Path | None = None) -> None:
        """Load feature flag statuses from JSON config file.

        A config status that differs from the in-memory one is treated as an
        out-of-band change and appended to the ledger — but only once: the
        last recorded status is consulted first, so restarts never spam.
        """
        if path is None:
            path = Path(__file__).parent.parent.parent.parent / "config" / "feature_flags.json"
        cls._config_path = path
        cls._history_path = path.parent / HISTORY_FILENAME
        if not path.exists():
            logger.info("No feature flags config at %s, using defaults", path)
            return
        try:
            with open(path) as f:
                data = json.load(f)
            before = {name: flag.status for name, flag in cls._flags.items()}
            for name, status_str in data.items():
                if name in cls._flags:
                    try:
                        cls._flags[name].status = FlagStatus(status_str)
                    except ValueError:
                        logger.warning("Invalid status '%s' for flag %s", status_str, name)
            recorded = cls._last_recorded_statuses()
            for name, flag in cls._flags.items():
                if flag.status != before.get(name) and recorded.get(name) != flag.status.value:
                    cls._append_history(
                        {
                            "kind": "status",
                            "flag": name,
                            "from": before[name].value,
                            "to": flag.status.value,
                            "version": flag.version,
                            "source": "config",
                        }
                    )
            logger.info("Loaded feature flags from %s (%d flags)", path, len(data))
        except Exception as e:
            logger.warning("Failed to load feature flags: %s", e)

    @classmethod
    def save_config(cls, path: Path | None = None) -> None:
        """Save current flag statuses to JSON config file."""
        if path is None:
            path = (
                cls._config_path
                or Path(__file__).parent.parent.parent.parent / "config" / "feature_flags.json"
            )
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
        description="Sole inference engine (SloNetChatProvider)",
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
        description="HuggingFace model fine-tuning with optional LoRA (requires torch)",
        status=FlagStatus.DISABLED,
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
