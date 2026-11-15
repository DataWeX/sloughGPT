"""Slo Manager - Hot-swappable personality system.

FEATURE: soul-manager — Runtime soul/personality switching without restart.
DO NOT DELETE. Core infrastructure for soul loading, trait weights, snapshots.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import struct
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from domain.infrastructure._internal.pugqeep.singleton import ThreadHandle, spawn_loop
from domain.shared import find_repo_root, repair_iso

logger = logging.getLogger("slo.soul_manager")


@dataclass
class SloInfo:
    """Information about a registered soul."""

    name: str
    path: str
    description: str = ""
    personality: dict[str, float] = field(default_factory=dict)
    traits: list[str] = field(default_factory=list)
    loaded_at: float | None = None
    born_at: str = ""
    training_dataset: str = ""
    epochs_trained: int = 0
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    lineage: str = ""
    base_model: str = ""
    version: str = ""
    size_mb: float = 0.0
    behavior: dict[str, Any] = field(default_factory=dict)
    cognition: dict[str, float] = field(default_factory=dict)
    emotion: dict[str, float] = field(default_factory=dict)
    generation_params: dict[str, Any] = field(default_factory=dict)


#: SloInfo attributes persisted in the on-disk soul index. ``loaded_at`` is
#: runtime-only (it changes on every switch) and would dirty the index.
_INDEXED_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(SloInfo) if f.name != "loaded_at")
_INDEX_VERSION = 1


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
        self._current_soul: str | None = None
        self._souls_cache: dict[str, SloInfo] = {}
        self._preference_file = (
            find_repo_root(Path(__file__).resolve()) / "data" / ".soul_preference"
        )
        # Persisted scan index. Lives in ``data/`` for the repo's own models dir
        # (gitignored, canonical cache location); any other souls dir — tests,
        # alternate checkpoints — keeps it beside its own data so a test run can
        # never rewrite the production index.
        repo_root = find_repo_root(Path(__file__).resolve())
        index_override = os.environ.get("SLO_SOUL_INDEX_PATH")
        if index_override:
            self._index_file = Path(index_override)
        elif self.slos_dir.resolve() == (repo_root / "models").resolve():
            self._index_file = repo_root / "data" / ".soul_index.json"
        else:
            self._index_file = self.slos_dir / ".soul_index.json"
        self._scan_lock = threading.Lock()
        self._scanned = False

        # The scan is deliberately NOT run here. Each candidate soul embeds a
        # JSON config header that can be tens of MB (a trained ``.soul`` carries
        # ``metadata.training_state.optimizer`` — the Adam moments — as JSON
        # floats, ~41 MB per file, and ``json.loads`` on that costs ~2.2 s).
        # On the 2026-11-03 cold start this cost 11 s of the 40 s boot, before
        # the server accepted a single request. Inventory is not readiness:
        # ``_ensure_scanned()`` populates the cache on first real use, and the
        # ``(path, mtime, size) → SloInfo`` index below makes every boot after
        # the first pay only a ``stat`` per candidate.
        #
        # The preference is restored at the end of ``_scan_souls()`` because it
        # can only be validated against a populated cache.

    def _ensure_scanned(self) -> None:
        """Populate the soul cache on first use (lazy ``_scan_souls``)."""
        if not self._scanned:
            self._scan_souls()

    # ── persisted index ──────────────────────────────────────────────────
    # Keyed by absolute path so parallel worktrees (each with its own models/
    # and data/) never share entries. Entries are invalidated by (mtime, size).

    def _index_read(self) -> dict[str, Any]:
        """Read the persisted soul index, or ``{}`` when absent/corrupt."""
        try:
            raw = json.loads(self._index_file.read_text())
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.debug("Soul index unreadable, rebuilding: %s", e, extra={"tag": "SOUL"})
            return {}
        if not isinstance(raw, dict) or raw.get("version") != _INDEX_VERSION:
            return {}
        entries = raw.get("entries")
        return entries if isinstance(entries, dict) else {}

    def _index_write(self, entries: dict[str, Any]) -> None:
        """Persist the soul index atomically (last writer wins)."""
        try:
            self._index_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._index_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"version": _INDEX_VERSION, "entries": entries}))
            os.replace(tmp, self._index_file)
        except Exception as e:
            logger.debug("Failed to write soul index: %s", e, extra={"tag": "SOUL"})

    def _info_to_entry(self, info: SloInfo) -> dict[str, Any]:
        """Serialise a ``SloInfo`` into its index representation."""
        raw = asdict(info)
        return {k: raw[k] for k in _INDEXED_FIELDS if k in raw}

    def _info_from_entry(self, entry: Any) -> SloInfo | None:
        """Rebuild a ``SloInfo`` from an index entry (``None`` if unusable)."""
        if not isinstance(entry, dict):
            return None
        kwargs = {k: v for k, v in entry.items() if k in _INDEXED_FIELDS}
        try:
            return SloInfo(**kwargs)
        except TypeError:
            return None

    def _soul_candidates(self) -> list[str]:
        """Absolute paths of every candidate soul file, deduped, deterministic.

        Order is the historical precedence: the recursive ``.slo``/``.soul``
        glob over the souls dir first, then the ``souls/`` profile subdirectory.
        ``glob`` order is filesystem-dependent, so it is sorted here — a
        duplicate name must resolve the same way on every boot.
        """
        paths: list[str] = []
        seen: set[str] = set()

        def _add(path: str) -> None:
            abspath = os.path.abspath(path)
            if abspath not in seen:
                seen.add(abspath)
                paths.append(abspath)

        if self.slos_dir.exists():
            for ext in ("**/*.slo", "**/*.soul"):
                for soul_path in sorted(glob.glob(str(self.slos_dir / ext), recursive=True)):
                    _add(soul_path)

        soul_candidates = [self.slos_dir / "souls"]
        for p in Path(__file__).resolve().parents:
            candidate = p / "models" / "souls"
            if candidate.exists() and candidate not in soul_candidates:
                soul_candidates.append(candidate)
                break
        for candidate in soul_candidates:
            if not candidate.exists():
                continue
            for ext in ("*.slo", "*.soul"):
                for soul_path in sorted(glob.glob(str(candidate / ext))):
                    _add(soul_path)
        return paths

    def _scan_souls(self) -> None:
        """Scan for available .slo (personality profiles) and .soul checkpoints.

        Idempotent and thread-safe: the result is cached until
        ``rescan_souls()`` clears it. Parsed metadata is kept in
        ``data/.soul_index.json`` keyed by ``(abspath, mtime_ns, size)``, so a
        restart with unchanged soul files stats each candidate and parses none.
        """
        if self._scanned:
            return

        with self._scan_lock:
            if self._scanned:
                return
            self._scan_locked()

    def _scan_locked(self) -> None:
        """Body of ``_scan_souls``; caller holds ``_scan_lock``."""
        if not self.slos_dir.exists():
            logger.warning("Slos directory not found: %s", self.slos_dir, extra={"tag": "SOUL"})
            self._scanned = True
            self._load_preference()
            return

        started = time.monotonic()
        candidates = self._soul_candidates()
        index = self._index_read()
        new_index: dict[str, Any] = {}
        parsed = 0

        for path in candidates:
            try:
                st = os.stat(path)
            except OSError:
                continue
            entry = index.get(path)
            info: SloInfo | None = None
            if (
                isinstance(entry, dict)
                and entry.get("mtime_ns") == st.st_mtime_ns
                and entry.get("size") == st.st_size
            ):
                info = self._info_from_entry(entry.get("info"))
            if info is None:
                try:
                    info = self._parse_soul_info(path)
                except (
                    Exception
                ) as e:  # pragma: no cover (unreachable — _parse_soul_info swallows all errors)
                    logger.debug("Failed to parse soul %s: %s", path, e)  # pragma: no cover
                    info = None
                parsed += 1
            if info is None:
                continue

            new_index[path] = {
                "mtime_ns": st.st_mtime_ns,
                "size": st.st_size,
                "info": self._info_to_entry(info),
            }
            # First candidate wins on duplicate names (historical behaviour:
            # ``<name>.soul`` beats ``<name>.soul.soul`` copies of the same soul).
            if info.name not in self._souls_cache:
                self._souls_cache[info.name] = info

        if new_index != index:
            self._index_write(new_index)

        self._scanned = True
        logger.info(
            "Found %s souls (%s parsed, %s indexed, %.2fs)",
            len(self._souls_cache),
            parsed,
            len(candidates) - parsed,
            time.monotonic() - started,
            extra={"tag": "SOUL"},
        )
        self._load_preference()

    def _parse_soul_info(self, sou_path: str) -> SloInfo | None:
        """Parse soul file for metadata.

        Handles both binary .soul files (SOUL magic + JSON config header + weights)
        and plain-text .slo personality profiles (via SouParser).
        """
        # ── Binary .soul format: SOUL + version + config_len + JSON_config ──
        try:
            with open(sou_path, "rb") as f:
                header = f.read(4)
                if header == b"SOUL":
                    struct.unpack("<I", f.read(4))[0]
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
                                k
                                for k, v in personality_traits.items()
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
                        born_at=repair_iso(config.get("born_at", "")) or "",
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

            with open(sou_path, encoding="utf-8") as f:
                content = f.read()
            soul = SouParser.parse(content)
            personality = dict(soul.personality.to_dict().items())
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

    def list_souls(self) -> list[SloInfo]:
        """List all available souls."""
        self._scan_souls()
        return list(self._souls_cache.values())

    def rescan_souls(self) -> list[SloInfo]:
        """Force a full re-scan of the souls directory.

        Clears the cached results and re-globs the filesystem.  Call after
        soul files are created, deleted, or renamed.
        """
        with self._scan_lock:
            self._souls_cache.clear()
            self._scanned = False
        self._scan_souls()
        return list(self._souls_cache.values())

    def warm(self) -> int:
        """Populate the soul cache eagerly (used by the startup warm thread).

        Returns the number of souls discovered. Safe to call concurrently with
        request handlers — ``_scan_souls`` serialises on ``_scan_lock``.
        """
        self._ensure_scanned()
        return len(self._souls_cache)

    def get_soul(self, name: str) -> SloInfo | None:
        """Get soul by name."""
        self._ensure_scanned()
        return self._souls_cache.get(name)

    def get_current_soul(self) -> SloInfo | None:
        """Get currently active soul."""
        self._ensure_scanned()
        if self._current_soul:
            return self._souls_cache.get(self._current_soul)
        return None

    def switch_soul(self, name: str) -> dict[str, Any]:
        """
        Switch to a different soul/personality.

        Returns info about the new soul.
        """
        self._ensure_scanned()
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

    def register_soul(self, path: str, name: str | None = None) -> SloInfo:
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
        self._ensure_scanned()
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

    def get_stats(self) -> dict[str, Any]:
        """Get soul manager statistics."""
        self._ensure_scanned()
        return {
            "total_souls": len(self._souls_cache),
            "current_soul": self._current_soul,
            "souls_dir": str(self.slos_dir),
            "available_souls": [s.name for s in self._souls_cache.values()],
        }

    def get_trait_weights(self) -> dict[str, Any]:
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
        from domain.context._internal.managers import TRAIT_SCHEMA

        result = {group: dict.fromkeys(traits, 0.5) for group, traits in TRAIT_SCHEMA.items()}

        # Override with soul file's personality if loaded
        if soul:
            # Use the values already parsed (and indexed) by ``_scan_souls``.
            # Re-opening the file here meant a second ``json.loads`` of the
            # ~41 MB config header on every request that renders trait weights.
            for group in ("personality", "cognition", "emotion"):
                cached = getattr(soul, group) or {}
                if not isinstance(cached, dict):
                    continue
                for k, v in cached.items():
                    if k in result.get(group, {}) and isinstance(v, (int, float)):
                        result[group][k] = v

        # Overlay live values from TraitWeightsConfig (feedback-driven)
        try:
            from domain.context._internal.managers import get_trait_config

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
_slo_manager: SloManager | None = None
_slo_manager_lock = threading.Lock()


def get_slo_manager() -> SloManager:
    """Get the global soul manager instance."""
    global _slo_manager
    if _slo_manager is None:
        with _slo_manager_lock:
            if _slo_manager is None:
                _slo_manager = SloManager()
    return _slo_manager


def warm_soul_index_async() -> ThreadHandle | None:
    """Warm the soul index in the background (startup hook; never blocks boot).

    Readiness must not be gated on soul inventory: a trained ``.soul`` embeds
    its Adam optimizer state as JSON (~41 MB), and parsing one costs ~2.2 s.
    Running this from ``main.py`` startup keeps that off the critical path while
    still making the first ``/souls`` request free.

    Caveat: ``json.loads`` is C code and holds the GIL, so the cold (first-ever)
    build still causes short latency spikes while it runs. Every boot after the
    first hits ``data/.soul_index.json`` and parses nothing. Set
    ``SLO_SOUL_INDEX_WARM=0`` to skip warming entirely.

    Returns the thread, or ``None`` when the cache was already populated.
    """
    if os.environ.get("SLO_SOUL_INDEX_WARM", "1").lower() in ("0", "false", "no"):
        return None
    manager = get_slo_manager()
    if manager._scanned:
        return None

    def _warm() -> None:
        try:
            logger.info("Warming soul index", extra={"tag": "SOUL"})
            manager.warm()
        except Exception as e:  # pragma: no cover — never break boot
            logger.warning("Soul index warm failed: %s", e, extra={"tag": "SOUL"})

    handle = spawn_loop(_warm, name="soul-index-warm")
    return handle


def switch_soul(name: str) -> dict[str, Any]:
    """Quick function to switch soul."""
    return get_slo_manager().switch_soul(name)


def list_souls() -> list[SloInfo]:
    """Quick function to list souls."""
    return get_slo_manager().list_souls()
