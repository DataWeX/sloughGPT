"""Consciousness engine — unified self-awareness, qualia, meta-cognition, narrative."""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any

from domain.cognition._internal.consciousness.config import ConsciousnessConfig
from domain.cognition._internal.consciousness.meta_cognition import MetaCognition
from domain.cognition._internal.consciousness.mind_state import MindLog, MindRecord
from domain.cognition._internal.consciousness.narrative import NarrativeGenerator
from domain.cognition._internal.consciousness.qualia import QualiaEngine
from domain.cognition._internal.consciousness.self_model import SelfEpisode, SelfModel


class ConsciousnessEngine:
    """Unified consciousness engine combining all 4 modules."""

    def __init__(self, config: ConsciousnessConfig | None = None) -> None:
        self.config = config or ConsciousnessConfig()
        self.self_model = SelfModel(store_path=self.config.store_path)
        self.qualia = QualiaEngine()
        self.meta_cognition = MetaCognition(self.self_model, self.qualia)
        self.narrative = NarrativeGenerator(self.self_model, self.qualia, self.meta_cognition)

        # Reflection cache for propose-only reads. reflect() draws from
        # random.choice pools, so without a cache the SAME episodes produced
        # a DIFFERENT injected system message on every request — a per-request
        # dice roll pretending to be current state. The cache is keyed on a
        # fingerprint of exactly what reflect() reads (episode count, recent
        # growth/qualia, beliefs) — not a bump counter: several writers mutate
        # self_model directly (routers/consciousness.py appends episodes and
        # rewrites beliefs without going through the engine), and push
        # invalidation is only as complete as its least disciplined writer.
        self._reflection_cache: tuple[tuple[Any, ...], Any] | None = None

        # Mind-state substrate: the append-only event log of the inner life
        # (episodes, dreams, reflections, belief deltas as typed records).
        # self_model.json stays the windowed state document; this log is the
        # complete timeline every renderer queries instead of hand-shaping.
        self.mind_log = MindLog(self.config.store_path)
        # Last record id per type — true edges (dream consolidates episode,
        # belief_delta applies reflection); omitted when unknown (restart).
        self._mind_last: dict[str, str] = {}

        if self.config.is_enabled():
            self.self_model.load()

        # Idle-dream baseline: episode count as of boot/last tick — the
        # world being quiet (count unchanged) is what creates sleep pressure.
        self._dream_baseline = len(self.self_model.episodes)

    def _reflection_fingerprint(self) -> tuple[Any, ...]:
        """State of exactly the inputs SelfModel.reflect() reads."""
        sm = self.self_model
        recent = sm.episodes[-5:]
        return (
            len(sm.episodes),
            tuple((e.growth_delta, tuple(sorted(e.qualia.items()))) for e in recent),
            tuple(sorted(sm.self_beliefs.items())),
        )

    def _propose_reflection(self) -> Any:
        """Propose-only reflection, memoized while the fingerprint is unchanged."""
        fp = self._reflection_fingerprint()
        cached = self._reflection_cache
        if cached is not None and cached[0] == fp:
            return cached[1]
        r = self.self_model.reflect()
        self._reflection_cache = (fp, r)
        return r

    def reflection_state_key(self) -> str:
        """Stable key of the current reflection state (fingerprint hash).

        Renderers diff this instead of polling narrative text; it moves
        exactly when the propose-only reflection would regenerate.
        """
        digest = hashlib.sha1(str(self._reflection_fingerprint()).encode()).hexdigest()
        return digest[:16]

    def _mind_append(
        self,
        record_type: str,
        payload: dict[str, Any],
        provenance: str,
        edge_type: str | tuple[str, ...] | None = None,
        edge_rel: str = "reflects_on",
    ) -> MindRecord:
        """Append one fact to the mind-state log, wiring a true edge if known."""
        edges: list[dict[str, str]] = []
        if edge_type is not None:
            candidates = (edge_type,) if isinstance(edge_type, str) else edge_type
            for candidate in candidates:
                ref = self._mind_last.get(candidate)
                if ref is not None:
                    edges.append({"rel": edge_rel, "id": ref})
                    break
        record = MindRecord(type=record_type, payload=payload, provenance=provenance, edges=edges)
        self.mind_log.append(record)
        self._mind_last[record_type] = record.id
        return record

    def maybe_dream(self) -> dict[str, Any]:
        """Idle-dream policy: nights happen when the world is quiet.

        Dream only when no episode landed since the last tick and there is
        something to consolidate — sleep pressure is the absence of new
        experience. Called by the host's ambient loop (interval from
        config.reflection_interval, re-read per tick); synchronous and
        wall-clock-free so tests call it directly.
        """
        if not self.config.is_enabled():
            return {"dreamed": False, "reason": "disabled"}
        count = len(self.self_model.episodes)
        if count != self._dream_baseline:
            self._dream_baseline = count
            return {"dreamed": False, "reason": "new experience since last tick"}
        if count == 0:
            return {"dreamed": False, "reason": "no experiences to consolidate"}
        result = self.dream()
        self._dream_baseline = len(self.self_model.episodes)
        return result

    def process(self, input_text: str, response: str, level: int | None = None) -> str:
        lvl = level if level is not None else self.config.level
        if lvl == 0:
            return ""

        experience: dict[str, Any] = {
            "input_text": input_text,
            "response": response,
        }

        qualia_state = self.qualia.experience(input_text)
        experience["qualia"] = qualia_state.to_dict()

        episode = self.self_model.observe(experience)
        self._mind_append(
            "episode",
            {
                "input_text": episode.input_text[:2000],
                "response": episode.response[:2000],
                "growth_delta": episode.growth_delta,
                "qualia": dict(episode.qualia),
            },
            provenance="experience",
        )

        return self.narrative.generate(input_text, response, level=lvl)

    def get_status(self) -> dict[str, Any]:
        return {
            "enabled": self.config.is_enabled(),
            "level": self.config.level,
            "episodes": len(self.self_model.episodes),
            "dreams": sum(1 for e in self.self_model.episodes if e.is_dream),
            "state_key": self.reflection_state_key(),
            "beliefs": dict(self.self_model.self_beliefs),
            "current_qualia": self.qualia.current.to_dict(),
            "curiosity_topics": dict(self.meta_cognition.curiosity_topics),
        }

    def save(self) -> None:
        self.self_model.save()

    def reflect(self, apply: bool = True) -> Any:
        """Return structured Reflection (caller should use .narrative for text).

        When ``apply`` is True (default), belief deltas are committed and
        persisted via ``apply_beliefs`` so the reflection loop closes at the
        engine boundary — that path reflects fresh every call. Use
        ``apply=False`` for a propose-only read (e.g. prompt injection) that
        must not mutate beliefs; that path is cached per state fingerprint,
        so repeated reads without new experience return the identical
        narrative instead of re-rolling the random pools.
        """
        if not apply:
            return self._propose_reflection()
        r = self.self_model.reflect()
        record = self._mind_append(
            "reflection",
            {
                "narrative": r.narrative,
                "trajectory": r.trajectory,
                "avg_growth": r.avg_growth,
                "episode_count": r.episode_count,
                "belief_deltas": dict(r.belief_deltas or {}),
            },
            provenance="reflection",
            edge_type=("dream", "episode"),
            edge_rel="reflects_on",
        )
        if getattr(r, "belief_deltas", None):
            self.apply_beliefs(r.belief_deltas, source=record.id)
        return r

    def reflect_text(self) -> str:
        r = self._propose_reflection()
        return r.narrative if hasattr(r, "narrative") else str(r)

    def dream(self) -> dict[str, Any]:
        """Consolidate inner state with zero input — the inner loop, unplugged.

        The propose-only reflection over current state is recorded as a dream
        episode, so the next reflection sees its own past reflections: the
        loop closes on itself. That self-reference is what makes the inner
        life intrinsic rather than input-driven — no chat, no tools, no feeds
        required. Dreams never commit belief deltas: consolidation reads
        state, experience writes it.
        """
        if not self.self_model.episodes:
            return {"dreamed": False, "reason": "no experiences to consolidate", "episodes": 0}
        r = self._propose_reflection()
        narrative = r.narrative if hasattr(r, "narrative") else str(r)
        dream = SelfEpisode(
            timestamp=time.time(),
            input_text="(dream)",
            response="",
            qualia=dict(self.qualia.current.to_dict()),
            self_insight=narrative,
            growth_delta=0.0,
            is_dream=True,
        )
        self.self_model.episodes.append(dream)
        self.save()
        self._mind_append(
            "dream",
            {"narrative": narrative, "qualia": dict(dream.qualia)},
            provenance="dream",
            edge_type="episode",
            edge_rel="consolidates",
        )
        return {
            "dreamed": True,
            "narrative": narrative,
            "episodes": len(self.self_model.episodes),
            "dreams": sum(1 for e in self.self_model.episodes if e.is_dream),
        }

    def apply_beliefs(
        self, deltas: dict[str, float], source: str | None = None
    ) -> dict[str, float]:
        """Commit belief deltas; log the fact, edged to its source record."""
        result = self.self_model.apply_beliefs(deltas)
        edges = [{"rel": "applies", "id": source}] if source else []
        self.mind_log.append(
            MindRecord(
                type="belief_delta",
                payload={"deltas": dict(deltas), "after": dict(result)},
                provenance="reflection",
                edges=edges,
            )
        )
        return result

    def clear_episodes(self) -> int:
        count = self.self_model.clear_episodes()
        self.save()
        return count

    def reset_beliefs(self) -> dict[str, float]:
        beliefs = self.self_model.reset_beliefs()
        self.save()
        return beliefs


_module_lock = threading.Lock()
_instance: ConsciousnessEngine | None = None


def get_consciousness(config: ConsciousnessConfig | None = None) -> ConsciousnessEngine:
    global _instance
    with _module_lock:
        if _instance is None:
            if config is None:
                config = ConsciousnessConfig.load()
            _instance = ConsciousnessEngine(config)
        elif config is not None:
            _instance.config = config
        return _instance


def reset_consciousness() -> None:
    global _instance
    with _module_lock:
        _instance = None
