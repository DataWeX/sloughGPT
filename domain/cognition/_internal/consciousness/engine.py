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

# Feedback calibration: what the rating evidences, and the 1-5 points it
# maps to (thumbs_up=5, thumbs_down=1, neutral=3).
_FEEDBACK_BELIEFS = ("accuracy", "helpfulness")
_FEEDBACK_POINTS: dict[str, int] = {"thumbs_up": 5, "thumbs_down": 1, "neutral": 3}

# The dream journal reads as a bounded entry list, never the whole log: a
# renderer says "12 of 44" and keeps the raw stream one click away.
DIARY_LIMIT = 12


def _is_down_rating(rating: object) -> bool:
    """True for a logged negative evidence rating (string form or points)."""
    if rating == "thumbs_down":
        return True
    return isinstance(rating, int) and not isinstance(rating, bool) and rating < 3


# Locus: where the mind is, derived from the freshest record's kind. A mind
# is AWAKENING (taking world input), DREAMING (consolidating), or
# RATIONALIZING (making self-narrative / adjusting beliefs). Provenance
# refines the one ambiguous type: a belief_delta born of reflection is
# rationalization; one born of feedback is world engagement (awakening).
_LOCUS_BY_TYPE = {
    "episode": "awakening",
    "dream": "dreaming",
    "reflection": "rationalizing",
}


def _classify_locus(record: MindRecord) -> str:
    if record.type in _LOCUS_BY_TYPE:
        return _LOCUS_BY_TYPE[record.type]
    if record.type == "belief_delta":
        return "rationalizing" if record.provenance == "reflection" else "awakening"
    return "unknown"


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

        # Idle-dream baseline: episode count as of the last consolidation. It
        # is the watermark of what the last dream already saw — sleep pressure
        # is material ABOVE it, not silence at it.
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
        edge_ids: list[str] | str | None = None,
    ) -> MindRecord:
        """Append one fact to the mind-state log, wiring a true edge if known.

        ``edge_ids`` names the target(s) explicitly when the caller actually
        knows what it is pointing at — a dream that revisited a specific
        episode should say so, rather than inherit 'whatever record was
        appended last'. When omitted, ``edge_type`` falls back to the last
        record of that type (unknown after a restart: no edge, never a guess).
        """
        edges: list[dict[str, str]] = []
        if edge_ids is not None:
            targets = (edge_ids,) if isinstance(edge_ids, str) else tuple(edge_ids)
            edges = [{"rel": edge_rel, "id": t} for t in targets]
        elif edge_type is not None:
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
        """Sleep gate: pressure is *unconsolidated experience*, not silence.

        Dream once when material has landed that the last dream did not see,
        then stay asleep. Satiation is the whole point — a gate that fires on
        emptiness degenerates into a loop that re-narrates a saturated
        snapshot forever (the production log reached 44 dreams, ~30 identical,
        qualia all-zero for 6h, off a single real episode).

        A *drop* in the episode count is not material: ``save()`` trims the
        window to ``MAX_EPISODES``, and forgetting is not experience. The drop
        is absorbed as the new baseline so it can neither arm a dream nor
        wedge the gate against later growth.

        Called by the host's ambient loop (interval from
        config.reflection_interval, re-read per tick); synchronous and
        wall-clock-free so tests call it directly.
        """
        if not self.config.is_enabled():
            return {"dreamed": False, "reason": "disabled"}
        count = len(self.self_model.episodes)
        if count < self._dream_baseline:
            # The working window trimmed. Not material — accept the new reality
            # so later growth can still arm a dream.
            self._dream_baseline = count
            return {"dreamed": False, "reason": "window trimmed, nothing new to consolidate"}
        if count == self._dream_baseline:
            return {"dreamed": False, "reason": "satiated: nothing new since last dream"}
        # Material above the watermark -> one night. Baseline then moves to the
        # POST-dream count, because dream() appends its own episode and save()
        # trims the window.
        result = self.dream()
        self._dream_baseline = len(self.self_model.episodes)
        return result

    def self_report(self) -> dict[str, Any]:
        """Fold the mind-state log into the structured self-report.

        The 'what do you know about me' surface as data — rendered time:
        current beliefs plus the trajectory that produced them, what pulls
        attention (curiosity), the experience timeline counted from the log
        (so it survives the document window), the last dream, and the dream
        diary — the night's dreams as a readable journal with their
        provenance intact. Read-only: a report observes the mind, it never
        steers it.
        """
        sm = self.self_model
        beliefs = {k: round(float(v), 4) for k, v in sm.self_beliefs.items()}
        strongest = max(beliefs, key=beliefs.get) if beliefs else None
        reflection = self._propose_reflection()

        episodes = self.mind_log.query(record_type="episode", limit=0)
        dreams = self.mind_log.query(record_type="dream", limit=0)
        deltas = self.mind_log.query(record_type="belief_delta", limit=0)

        # How it came to believe that: belief name -> [[t, value_after], ...]
        trajectory: dict[str, list[list[float]]] = {}
        for d in deltas:
            for name, value in (d.payload.get("after") or {}).items():
                trajectory.setdefault(name, []).append([d.t, round(float(value), 4)])

        last_dream = dreams[-1] if dreams else None
        return {
            "identity": {"name": sm.identity.name},
            "beliefs": beliefs,
            "strongest_belief": (
                {"name": strongest, "value": beliefs[strongest]} if strongest else None
            ),
            "doubts": list(sm.self_doubts),
            "reflection": {
                "narrative": getattr(reflection, "narrative", str(reflection)),
                "trajectory": getattr(reflection, "trajectory", "stagnation"),
                "strategy_notes": list(getattr(reflection, "strategy_notes", []) or []),
                "state_key": self.reflection_state_key(),
            },
            "curiosity": {
                k: round(float(v), 4) for k, v in self.meta_cognition.curiosity_topics.items()
            },
            "timeline": {
                "experiences": len(episodes),
                "dreams": len(dreams),
                "first_experience": episodes[0].t if episodes else None,
                "last_experience": episodes[-1].t if episodes else None,
            },
            "belief_trajectory": trajectory,
            "last_dream": (
                {
                    "t": last_dream.t,
                    "narrative": (last_dream.payload.get("narrative") or "")[:400],
                }
                if last_dream is not None
                else None
            ),
            "dream_diary": self._dream_diary(dreams),
        }

    def _dream_diary(self, dreams: list[MindRecord] | None = None) -> dict[str, Any]:
        """The dream journal: readable narrative beside the provenance of its making.

        Two planes in one entry — the human sentence to read, and the bones
        that keep it honest (locus, from_locus, provenance, qualia, edges as
        recorded). Read-only, bounded by ``DIARY_LIMIT`` with an accurate
        ``total`` so a renderer can summarise instead of dumping the log.
        """
        if dreams is None:
            dreams = self.mind_log.query(record_type="dream", limit=0)
        window = dreams[-DIARY_LIMIT:]
        return {
            "total": len(dreams),
            "shown": len(window),
            "entries": [
                {
                    "t": d.t,
                    "narrative": d.payload.get("narrative") or "",
                    "locus": d.payload.get("locus"),
                    "from_locus": d.payload.get("from_locus"),
                    "provenance": d.provenance,
                    "qualia": dict(d.payload.get("qualia") or {}),
                    "edges": list(d.edges),
                    "revisited": list(d.payload.get("revisited") or []),
                }
                for d in window
            ],
        }

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

    def locus(self) -> str:
        """Where the mind is right now — derived, never stored.

        awakening (world input) / dreaming (consolidation) / rationalizing
        (self-narrative, belief-making), read off the freshest record via the
        O(1) tail read. A stored 'I am dreaming' flag would go stale the
        moment the world moves; a derived locus cannot.
        """
        rec = self.mind_log.latest()
        return _classify_locus(rec) if rec else "unborn"

    def get_status(self) -> dict[str, Any]:
        last = self.mind_log.latest()
        return {
            "enabled": self.config.is_enabled(),
            "level": self.config.level,
            "episodes": len(self.self_model.episodes),
            "dreams": sum(1 for e in self.self_model.episodes if e.is_dream),
            "state_key": self.reflection_state_key(),
            # Summary pointer to the freshest inner event (O(1) tail read) —
            # never the payload: episode records carry full input/response
            # text, and status must stay a summary.
            "last_record": (
                {
                    "id": last.id,
                    "type": last.type,
                    "provenance": last.provenance,
                    "t": last.t,
                }
                if last
                else None
            ),
            "locus": _classify_locus(last) if last else "unborn",
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

    def _consolidated_keys(self) -> set[str]:
        """What earlier nights already worked on, read back from the ledger.

        A dream records the ids it revisited; this is that set. Kept in the log
        rather than as a flag on the model so it cannot drift from the truth.
        """
        seen: set[str] = set()
        for rec in self.mind_log.query(record_type="dream", limit=0):
            seen.update(rec.payload.get("revisited") or [])
        return seen

    def _consolidation_candidates(self) -> list[tuple[SelfEpisode, MindRecord | None, str]]:
        """(episode, its log record or None, stable key), oldest first.

        Two ways living episodes outnumber, or sit under, their records — each
        needs different alignment or every consolidation lands on the wrong
        experience:

        * log >= episodes: save() trims the model's WINDOW, the log keeps the
          whole timeline — so living episodes align to the log's TAIL.
        * episodes > log: writers that bypass the engine append to the model
          without recording (routers/consciousness.py does) — align from the
          head, and the tail has no ledger entry to point at.

        An unrecorded episode is keyed by its own timestamp, so it is still
        consolidated exactly once and never re-dreamed.
        """
        living = [e for e in self.self_model.episodes if not e.is_dream]
        if not living:
            return []
        records = self.mind_log.query(record_type="episode", limit=0)
        if len(records) >= len(living):
            pairs: list[tuple[SelfEpisode, MindRecord | None]] = list(
                zip(living, records[-len(living) :])
            )
        else:
            pairs = list(zip(living, records))
            pairs.extend((e, None) for e in living[len(records) :])

        seen = self._consolidated_keys()
        candidates: list[tuple[SelfEpisode, MindRecord | None, str]] = []
        for episode, record in pairs:
            key = record.id if record is not None else f"ep:{episode.timestamp}"
            if key not in seen:
                candidates.append((episode, record, key))
        return candidates

    def dream(self) -> dict[str, Any]:
        """Consolidate one unconsolidated experience — the inner loop, unplugged.

        A dream CHOOSES what it revisits: the oldest experience no earlier
        dream has worked on. It feels like that experience (its qualia, not the
        mood it fell asleep in), records what it touched in the ledger, and
        points its consolidates edge at the episode actually revisited — so the
        diary renders a night that really happened rather than a stamp on
        'whatever was last'. The narrative is recorded as a dream episode, so
        the next reflection sees its own past reflections: the loop closes on
        itself.

        Dreams never commit belief deltas: consolidation reads state,
        experience writes it.
        """
        candidates = self._consolidation_candidates()
        if not candidates:
            reason = (
                "no experiences to consolidate"
                if not self.self_model.episodes
                else "everything has been consolidated - nothing new to revisit"
            )
            return {
                "dreamed": False,
                "reason": reason,
                "episodes": len(self.self_model.episodes),
            }

        target, target_record, key = candidates[0]
        from_locus = self.locus()  # where I am BEFORE dreaming — I drift out of this
        r = self._propose_reflection()
        narrative = r.narrative if hasattr(r, "narrative") else str(r)
        # Consolidation feels like the thing being consolidated — not like the
        # waking mood it happened to fall asleep in.
        felt = dict(target.qualia) if target.qualia else dict(self.qualia.current.to_dict())
        dream = SelfEpisode(
            timestamp=time.time(),
            input_text="(dream)",
            response="",
            qualia=felt,
            self_insight=narrative,
            growth_delta=0.0,
            is_dream=True,
        )
        self.self_model.episodes.append(dream)
        self.save()
        self._mind_append(
            "dream",
            {
                "narrative": narrative,
                "qualia": dict(dream.qualia),
                "locus": "dreaming",
                "from_locus": from_locus,
                "revisited": [key],
            },
            provenance="dream",
            # only claim an edge the ledger can actually back: an episode with
            # no record has nothing to point at, so it gets none — never a guess
            edge_ids=[target_record.id] if target_record is not None else None,
            edge_rel="consolidates",
        )
        return {
            "dreamed": True,
            "narrative": narrative,
            "locus": "dreaming",
            "from_locus": from_locus,
            "revisited": [key],
            "episodes": len(self.self_model.episodes),
            "dreams": sum(1 for e in self.self_model.episodes if e.is_dream),
        }

    def apply_beliefs(
        self,
        deltas: dict[str, float],
        source: str | None = None,
        provenance: str = "reflection",
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Commit belief deltas; log the fact, edged to its source record."""
        result = self.self_model.apply_beliefs(deltas)
        edges = [{"rel": "applies", "id": source}] if source else []
        payload: dict[str, Any] = {"deltas": dict(deltas), "after": dict(result)}
        if evidence:
            payload["evidence"] = dict(evidence)
        self.mind_log.append(
            MindRecord(
                type="belief_delta",
                payload=payload,
                provenance=provenance,
                edges=edges,
            )
        )
        return result

    def observe_feedback(self, rating: str | int) -> dict[str, Any]:
        """Calibrate self-confidence from outcome evidence (user feedback).

        Beliefs are confidence estimates: success pulls them toward 1,
        failure toward 0 — asymptotically, so evidence accumulates instead
        of oscillating (the rarer, stronger thumbs-down moves them further
        per event). The rating enters through the same door as every other
        experience: qualia feel it first, the belief deltas are logged with
        their evidence. The world informs the mind; nothing steers it from
        outside.
        """
        if not self.config.is_enabled():
            return {"observed": False, "reason": "disabled"}
        points = (
            _FEEDBACK_POINTS.get(rating.lower(), 3)
            if isinstance(rating, str)
            else max(1, min(5, int(rating)))
        )
        self.qualia.experience_from_feedback(points)
        if points == 3:
            return {"observed": True, "delta": {}, "reason": "neutral rating"}
        success = points > 3
        eta = 0.05 if success else 0.08
        deltas: dict[str, float] = {}
        for name in _FEEDBACK_BELIEFS:
            confidence = float(self.self_model.self_beliefs.get(name, 0.5))
            step = eta * (1.0 - confidence) if success else -eta * confidence
            if abs(step) > 1e-9:
                deltas[name] = round(step, 4)
        if not deltas:
            return {"observed": True, "delta": {}, "reason": "confidence saturated"}
        self.apply_beliefs(
            deltas,
            provenance="experience",
            evidence={
                "rating": rating if isinstance(rating, str) else points,
                "source": "feedback",
            },
        )
        # The voiced uncertainty follows the evidence: count logged feedback
        # records (window of the last 20 — only events that CHANGED beliefs
        # are logged, so saturated downs undercount by design) and rebuild
        # the doubt: born on first down, sharpened per event, dissolved when
        # the sliding window clears.
        evidence_records = [
            r
            for r in self.mind_log.query(record_type="belief_delta", limit=0)
            if r.payload.get("evidence")
        ][-20:]
        downs = sum(
            1 for r in evidence_records if _is_down_rating(r.payload["evidence"].get("rating"))
        )
        self.self_model.set_evidence_doubt(downs, len(evidence_records))
        return {"observed": True, "delta": deltas}

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


def peek_consciousness() -> ConsciousnessEngine | None:
    """Return the existing engine singleton WITHOUT creating one.

    For callers that inform the mind (e.g. feedback calibration): you speak
    to a mind that already exists — you never birth one just to tell it
    something. ``None`` means no mind has been instantiated yet.
    """
    with _module_lock:
        return _instance
