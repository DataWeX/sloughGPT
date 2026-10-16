"""Consciousness engine — unified self-awareness, qualia, meta-cognition, narrative."""

from __future__ import annotations

import threading
import time
from typing import Any

from domain.cognition._internal.consciousness.config import ConsciousnessConfig
from domain.cognition._internal.consciousness.meta_cognition import MetaCognition
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

        if self.config.is_enabled():
            self.self_model.load()

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

        self.self_model.observe(experience)

        return self.narrative.generate(input_text, response, level=lvl)

    def get_status(self) -> dict[str, Any]:
        return {
            "enabled": self.config.is_enabled(),
            "level": self.config.level,
            "episodes": len(self.self_model.episodes),
            "dreams": sum(1 for e in self.self_model.episodes if e.is_dream),
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
        if getattr(r, "belief_deltas", None):
            self.apply_beliefs(r.belief_deltas)
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
        return {
            "dreamed": True,
            "narrative": narrative,
            "episodes": len(self.self_model.episodes),
            "dreams": sum(1 for e in self.self_model.episodes if e.is_dream),
        }

    def apply_beliefs(self, deltas: dict[str, float]) -> dict[str, float]:
        return self.self_model.apply_beliefs(deltas)

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
