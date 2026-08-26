"""Slo Manager - Hot-swappable personality system.

FEATURE: soul-manager — Runtime soul/personality switching without restart.
DO NOT DELETE. Core infrastructure for soul loading, trait weights, snapshots.
"""

import os
import glob
import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import logging
from domains.shared import find_repo_root

logger = logging.getLogger("slo.soul_manager")


@dataclass
class SloInfo:
    """Information about a registered soul."""

    name: str
    path: str
    description: str = ""
    personality: Dict[str, float] = field(default_factory=dict)
    traits: List[str] = field(default_factory=list)
    loaded_at: Optional[float] = None
    born_at: str = ""
    training_dataset: str = ""
    epochs_trained: int = 0
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    lineage: str = ""
    base_model: str = ""
    version: str = ""
    size_mb: float = 0.0
    behavior: Dict[str, Any] = field(default_factory=dict)
    cognition: Dict[str, float] = field(default_factory=dict)
    emotion: Dict[str, float] = field(default_factory=dict)
    generation_params: Dict[str, Any] = field(default_factory=dict)


class SloManager:
    """
    Hot-swappable soul/personality manager.

    Features:
    - List available souls
    - Switch personalities at runtime
    - Persist soul preference
    - Profile-based switching

    Usage:
        manager = SloManager()
        souls = manager.list_souls()
        manager.switch_soul("helpful_assistant")
        current = manager.get_current_soul()
    """

    def __init__(self, souls_dir: str = "models"):
        self.slos_dir = Path(souls_dir)
        self._current_soul: Optional[str] = None
        self._souls_cache: Dict[str, SloInfo] = {}
        self._preference_file = find_repo_root(Path(__file__).resolve()) / "data" / ".soul_preference"

        # Load cached souls
        self._scan_souls()

        # Load saved preference
        self._load_preference()

    def _scan_souls(self) -> None:
        """Scan for available .slo (personality profiles) and .soul (checkpoint) files.

        Skips the scan if the cache is already populated (idempotent).
        Call ``rescan_souls()`` to force a refresh.
        """
        if self._souls_cache:
            return

        if not self.slos_dir.exists():
            logger.warning("Slos directory not found: %s", self.slos_dir, extra={"tag": "SOUL"})
            return

        # Find personality profiles (.slo) and checkpoint files (.soul) — recursive
        for ext in ("**/*.slo", "**/*.soul"):
            for sou_path in glob.glob(str(self.slos_dir / ext), recursive=True):
                try:
                    soul_info = self._parse_soul_info(sou_path)
                    if soul_info:
                        self._souls_cache[soul_info.name] = soul_info
                except Exception as e:  # pragma: no cover (unreachable — _parse_soul_info swallows all errors)
                    logger.debug("Failed to parse soul %s: %s", sou_path, e)  # pragma: no cover

        # Find text profile files in souls/ subdirectory (both .slo and .soul)
        soul_candidates = [self.slos_dir / "souls"]
        # Try to resolve from this file's location to repo root
        for p in Path(__file__).resolve().parents:
            candidate = p / "models" / "souls"
            if candidate.exists() and candidate not in soul_candidates:
                soul_candidates.append(candidate)
                break
        for candidate in soul_candidates:
            if candidate.exists():
                for ext in ("*.slo", "*.soul"):
                    for soul_path in glob.glob(str(candidate / ext)):
                        try:
                            soul_info = self._parse_soul_info(soul_path)
                            if soul_info and soul_info.name not in self._souls_cache:
                                self._souls_cache[soul_info.name] = soul_info
                        except Exception as e:  # pragma: no cover (unreachable — _parse_soul_info swallows all errors)
                            logger.debug("Failed to parse soul profile %s: %s", soul_path, e)  # pragma: no cover

        if not hasattr(self, '_scanned'):
            logger.info("Found %s souls", len(self._souls_cache), extra={"tag": "SOUL"})
            self._scanned = True

    def _parse_soul_info(self, sou_path: str) -> Optional[SloInfo]:
        """Parse soul file for metadata.

        Handles both binary .soul files (SOUL magic + JSON config header + weights)
        and plain-text .slo personality profiles (via SouParser).
        """
        # ── Binary .soul format: SOUL + version + config_len + JSON_config ──
        try:
            with open(sou_path, "rb") as f:
                header = f.read(4)
                if header == b"SOUL":
                    ver = struct.unpack("<I", f.read(4))[0]
                    config_len = struct.unpack("<I", f.read(4))[0]
                    file_size = os.path.getsize(sou_path)
                    if config_len > file_size - 12:
                        raise ValueError(
                            f"config_len ({config_len}) exceeds remaining file size "
                            f"({file_size - 12}) in {sou_path}"
                        )
                    config_bytes = f.read(config_len)
                    try:
                        config = json.loads(config_bytes.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise ValueError(
                            f"Corrupted config header in {sou_path}: "
                            f"config_len={config_len}, read={len(config_bytes)} bytes: {exc}"
                        ) from exc
                    name = config.get("name", Path(sou_path).stem)
                    description = config.get("description", "") or config.get("tagline", "") or name
                    traits = config.get("traits", config.get("personality_traits", []))
                    if not traits:
                        behavior = config.get("behavior", {})
                        approach = behavior.get("reasoning_approach", "balanced")
                        traits = [approach]
                        personality_traits = config.get("personality", {})
                        if personality_traits:
                            traits.extend(
                                k for k, v in personality_traits.items()
                                if isinstance(v, (int, float)) and v > 0.6
                            )
                    personality = config.get("personality", config.get("personality_traits", {}))
                    try:
                        size_mb = os.path.getsize(sou_path) / (1024 * 1024)
                    except OSError:
                        size_mb = 0.0
                    return SloInfo(
                        name=name,
                        path=sou_path,
                        description=description,
                        personality=personality,
                        traits=list(traits) if isinstance(traits, (list, tuple)) else [],
                        born_at=config.get("born_at", ""),
                        training_dataset=config.get("training_dataset", ""),
                        epochs_trained=config.get("epochs_trained", 0) or 0,
                        final_train_loss=config.get("final_train_loss"),
                        final_val_loss=config.get("final_val_loss"),
                        lineage=config.get("lineage", ""),
                        base_model=config.get("base_model", ""),
                        version=config.get("version", ""),
                        size_mb=size_mb,
                        behavior=config.get("behavior", {}),
                        cognition=config.get("cognition", {}),
                        emotion=config.get("emotion", {}),
                        generation_params=config.get("generation", {}),
                    )
        except Exception:
            pass  # Not a binary .soul file, try text format below

        # ── Plain-text .slo personality profile — parse via SouParser ──
        try:
            from .slo_format import SouParser
            with open(sou_path, "r", encoding="utf-8") as f:
                content = f.read()
            soul = SouParser.parse(content)
            personality = {k: v for k, v in soul.personality.to_dict().items()}
            traits = []
            behavior = getattr(soul, "behavior", None)
            if behavior:
                traits.append(getattr(behavior, "reasoning_approach", "balanced"))
                for k, v in personality.items():
                    if isinstance(v, (int, float)) and v > 0.6 and k not in traits:
                        traits.append(k)
            return SloInfo(
                name=soul.name,
                path=sou_path,
                description=getattr(soul, "description", "") or soul.name,
                personality=personality,
                traits=traits,
            )
        except Exception as e:
            logger.debug("Parse error for %s: %s", sou_path, e)
            return None

    def _load_preference(self) -> None:
        """Load saved soul preference."""
        if self._preference_file.exists():
            try:
                name = self._preference_file.read_text().strip()
                if name in self._souls_cache:
                    self._current_soul = name
                    logger.info("Restored soul preference: %s", name, extra={"tag": "SOUL"})
            except Exception as e:
                logger.warning("Failed to load soul preference: %s", e)

    def _save_preference(self) -> None:
        """Save current soul preference."""
        if self._current_soul:
            try:
                self._preference_file.parent.mkdir(parents=True, exist_ok=True)
                self._preference_file.write_text(self._current_soul)
            except Exception as e:
                logger.warning("Failed to save soul preference: %s", e)

    def list_souls(self) -> List[SloInfo]:
        """List all available souls."""
        self._scan_souls()
        return list(self._souls_cache.values())

    def rescan_souls(self) -> List[SloInfo]:
        """Force a full re-scan of the souls directory.

        Clears the cached results and re-globs the filesystem.  Call after
        soul files are created, deleted, or renamed.
        """
        self._souls_cache.clear()
        self._scan_souls()
        return list(self._souls_cache.values())

    def get_soul(self, name: str) -> Optional[SloInfo]:
        """Get soul by name."""
        return self._souls_cache.get(name)

    def get_current_soul(self) -> Optional[SloInfo]:
        """Get currently active soul."""
        if self._current_soul:
            return self._souls_cache.get(self._current_soul)
        return None

    def switch_soul(self, name: str) -> Dict[str, Any]:
        """
        Switch to a different soul/personality.

        Returns info about the new soul.
        """
        if name not in self._souls_cache:
            return {
                "success": False,
                "error": f"Slo '{name}' not found",
                "available": list(self._souls_cache.keys()),
            }

        self._current_soul = name
        self._save_preference()

        soul = self._souls_cache[name]
        soul.loaded_at = os.times().elapsed

        logger.info("Switched to soul: %s", name, extra={"tag": "SOUL"})

        return {
            "success": True,
            "name": soul.name,
            "path": soul.path,
            "description": soul.description,
            "personality": soul.personality,
            "traits": soul.traits,
        }

    def register_soul(self, path: str, name: Optional[str] = None) -> SloInfo:
        """
        Register a new soul file.

        Args:
            path: Path to .soul file
            name: Optional custom name

        Returns:
            SloInfo for the registered soul
        """
        soul_info = self._parse_soul_info(path)

        if not soul_info:
            raise ValueError(f"Failed to parse soul file: {path}")

        if name:
            soul_info.name = name

        self._souls_cache[soul_info.name] = soul_info

        return soul_info

    def create_default_souls(self) -> None:
        """Create default souls if none exist."""
        default_souls = [
            {
                "name": "assistant",
                "description": "Helpful and informative assistant",
                "personality": {
                    "warmth": 0.7,
                    "creativity": 0.5,
                    "curiosity": 0.8,
                    "confidence": 0.6,
                },
            },
            {
                "name": "creative",
                "description": "Creative and imaginative AI",
                "personality": {
                    "warmth": 0.6,
                    "creativity": 0.9,
                    "curiosity": 0.9,
                    "confidence": 0.5,
                },
            },
            {
                "name": "analyst",
                "description": "Analytical and precise AI",
                "personality": {
                    "warmth": 0.4,
                    "creativity": 0.3,
                    "curiosity": 0.7,
                    "confidence": 0.8,
                },
            },
        ]

        for soul_def in default_souls:
            if soul_def["name"] not in self._souls_cache:
                soul = SloInfo(**soul_def, path="")
                self._souls_cache[soul.name] = soul

    def get_stats(self) -> Dict[str, Any]:
        """Get soul manager statistics."""
        return {
            "total_souls": len(self._souls_cache),
            "current_soul": self._current_soul,
            "souls_dir": str(self.slos_dir),
            "available_souls": [s.name for s in self._souls_cache.values()],
        }

    def get_trait_weights(self) -> Dict[str, Any]:
        """
        Read trait weight attributes from the currently active soul file,
        overlaid with live values from the context manager TraitWeightsConfig.

        Returns a dict with personality, cognition, and emotion trait groups
        showing each trait's current weight (0.0–1.0). These configure the
        context managers — like a player card in a sports game.

        Always returns the full trait structure. Without a soul, returns
        TraitWeightsConfig defaults (0.5) overlaid with any feedback-driven
        changes.
        """
        soul = self.get_current_soul()

        # Always start with canonical defaults from TRAIT_SCHEMA
        from domains.context.managers import TRAIT_SCHEMA

        result = {
            group: {t: 0.5 for t in traits}
            for group, traits in TRAIT_SCHEMA.items()
        }

        # Override with soul file's personality if loaded
        if soul:
            if soul.personality:
                for k, v in soul.personality.items():
                    if k in result.get("personality", {}):
                        result["personality"][k] = v

            # Read metadata from the .slo/.soul file
            try:
                with open(soul.path, "rb") as f:
                    magic = f.read(4)
                    if magic in (b"SOUL",):
                        meta_len = struct.unpack("<I", f.read(4))[0]
                        meta = json.loads(f.read(meta_len))
                        for group in ("personality", "cognition", "emotion"):
                            if group in meta:
                                for k, v in meta[group].items():
                                    if k in result.get(group, {}):
                                        result[group][k] = v
            except Exception as e:
                logger.warning("Failed to merge trait weights from soul: %s", e)

        # Overlay live values from TraitWeightsConfig (feedback-driven)
        try:
            from domains.context.managers import get_trait_config
            config = get_trait_config()
            live = config.all()
            for group in ("personality", "cognition", "emotion"):
                if group in live:
                    for k, v in live[group].items():
                        if k in result.get(group, {}):
                            result[group][k] = v
        except Exception as e:
            logger.warning("Failed to merge live trait weights: %s", e)

        return result


# Global manager instance
_slo_manager: Optional[SloManager] = None


def get_slo_manager() -> SloManager:
    """Get the global soul manager instance."""
    global _slo_manager
    if _slo_manager is None:
        _slo_manager = SloManager()
    return _slo_manager


def switch_soul(name: str) -> Dict[str, Any]:
    """Quick function to switch soul."""
    return get_slo_manager().switch_soul(name)


def list_souls() -> List[SloInfo]:
    """Quick function to list souls."""
    return get_slo_manager().list_souls()
