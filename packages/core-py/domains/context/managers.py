"""
Context Managers — supplement model processing with engineered context steering.

Architecture:
  TraitWeightsConfig holds all trait values (0.0–1.0). Four managers read these
  weights and produce context modifications (system prompt, thresholds, priorities)
  that steer model behavior without touching model weights.

  feedback → TraitWeightsConfig.update() → managers read weights → ContextCore applies
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import threading

from domains.infrastructure.repository import FileRepository, JsonSerializer

_lock = threading.Lock()


# ── Trait schema (canonical trait list) ──────────────────────────────────

TRAIT_SCHEMA: Dict[str, List[str]] = {
    "personality": [
        "warmth", "creativity", "empathy", "formality", "humor",
        "patience", "confidence", "curiosity", "directness", "optimism",
    ],
    "cognition": [
        "pattern_recognition", "long_context_handling", "abstract_reasoning",
        "factual_precision", "creative_divergence", "systematic_planning",
        "metacognitive_awareness", "learning_adaptability",
    ],
    "emotion": [
        "empathy_depth", "mood_responsiveness", "tone_flexibility",
        "sentiment_awareness", "distress_handling",
    ],
}

ALL_TRAITS: List[str] = [t for traits in TRAIT_SCHEMA.values() for t in traits]


# ── Trait Weights Config ─────────────────────────────────────────────────

class TraitWeightsConfig:
    """Key-value store for trait weights (0.0–1.0). Persisted as JSON.

    Each weight is a float clamped to [0.0, 1.0]. Default for any unset
    trait is 0.5. Supports snapshots (named save/load) and batch update.
    """

    def __init__(self, path: str = "data/trait_weights.json"):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._weights: Dict[str, float] = {}
        self._snapshots_dir = self._path.parent / "trait_snapshots"
        self._snapshots_dir.mkdir(exist_ok=True)
        self._snapshot_repo = FileRepository[dict](
            directory=str(self._snapshots_dir),
            serializer=JsonSerializer(dict),
            key_suffix=".json",
        )
        self._snapshot_repo.enable_cache(ttl_seconds=5.0)
        self._load()

    # ── Access ───────────────────────────────────────────────────────

    def get(self, key: str, default: float = 0.5) -> float:
        """Get a single trait weight."""
        with _lock:
            return self._weights.get(key, default)

    def set(self, key: str, value: float) -> None:
        """Set a single trait weight and persist."""
        with _lock:
            self._weights[key] = max(0.0, min(1.0, float(value)))
            self._save()

    def all(self) -> Dict[str, Dict[str, float]]:
        """Return all weights grouped by trait category.

        Unset traits return 0.5. Structure matches get_trait_weights().
        """
        with _lock:
            result = {}
            for group, traits in TRAIT_SCHEMA.items():
                result[group] = {t: self._weights.get(t, 0.5) for t in traits}
            return result

    def update(self, deltas: Dict[str, float]) -> None:
        """Apply delta updates (e.g. from feedback: warmth += 0.05)."""
        with _lock:
            for key, delta in deltas.items():
                current = self._weights.get(key, 0.5)
                self._weights[key] = max(0.0, min(1.0, current + float(delta)))
            self._save()

    def set_many(self, values: Dict[str, float]) -> None:
        """Batch set weights (replaces values, does not touch others)."""
        with _lock:
            for key, value in values.items():
                if key in ALL_TRAITS:
                    self._weights[key] = max(0.0, min(1.0, float(value)))
            self._save()

    def reset(self) -> None:
        """Reset all weights to 0.5."""
        with _lock:
            self._weights = {}
            self._save()

    # ── Trait profile word sets for content-aware feedback ──────────

    _TRAIT_PROFILES: Dict[str, set] = {
        "humor": {"funny", "joke", "lol", "humor", "comedy", "wit", "hilarious", "amusing", "haha", "lmao"},
        "warmth": {"warm", "kind", "gentle", "nice", "friendly", "caring", "compassionate", "sweet", "lovely"},
        "creative_divergence": {"creative", "imaginative", "novel", "unique", "different", "original", "fresh", "inventive"},
        "abstract_reasoning": {"deep", "explain", "why", "how", "analyze", "complex", "nuance", "detailed", "theoretical", "philosophical"},
        "directness": {"short", "quick", "concise", "direct", "tl;dr", "brief", "summarize", "summary", "blunt"},
        "formality": {"formal", "professional", "proper", "academic", "polished", "sophisticated", "business"},
        "empathy": {"empathy", "understand", "feel", "emotion", "support", "care", "sympathy", "compassion"},
        "curiosity": {"curious", "wonder", "explore", "learn", "discover", "interesting", "fascinating"},
        "patience": {"patient", "step", "guide", "walk through", "thorough", "detailed explanation"},
        "optimism": {"optimistic", "bright", "positive", "hope", "encouraging", "uplifting"},
        "factual_precision": {"accurate", "precise", "exact", "data", "source", "reference", "cite", "evidence"},
        "systematic_planning": {"plan", "strategy", "systematic", "method", "framework", "approach", "step by step"},
    }

    # ── Feedback-driven update ───────────────────────────────────────

    def update_from_feedback(
        self, rating: str, user_message: str = "", response: str = ""
    ) -> int:
        """Update trait weights based on feedback direction and content.

        Uses trait profile word sets for content-aware boosting:
        - Computes overlap ratio between user message words and each trait's
          profile. Higher overlap = larger delta for that trait.
        - Response length and structure also influence precision/directness.
        - Base delta applied to ALL traits (thumbs_up=+0.03, thumbs_down=-0.03)
          plus proportional profile boost capped at +0.06 per trait.

        Args:
            rating: "thumbs_up" or "thumbs_down"
            user_message: recent user message (for content-aware adjustment)
            response: assistant response (for quality-aware adjustment)

        Returns:
            number of traits modified
        """
        base_delta = 0.03 if rating == "thumbs_up" else -0.03

        shifts: Dict[str, float] = {}
        for trait in ALL_TRAITS:
            shifts[trait] = base_delta

        if user_message:
            words = set(user_message.lower().split())
            for trait, profile in self._TRAIT_PROFILES.items():
                overlap = words & profile
                if overlap and trait in ALL_TRAITS:
                    boost = (len(overlap) / len(profile)) * 0.06
                    shifts[trait] = shifts.get(trait, base_delta) + boost

            # Negations flip the direction for specific traits
            negations = {"not", "don't", "dont", "isn't", "isnt", "no", "stop", "less"}
            has_negation = bool(words & negations)
            if has_negation:
                for trait in ("formality", "confidence", "directness"):
                    shifts[trait] = shifts.get(trait, base_delta) * -1

        if response and rating == "thumbs_up":
            word_count = len(response.split())
            if word_count < 20:
                shifts["directness"] = shifts.get("directness", base_delta) + 0.03
            if word_count > 80:
                shifts["patience"] = shifts.get("patience", base_delta) + 0.03
            if "```" in response or "`" in response:
                shifts["factual_precision"] = shifts.get("factual_precision", base_delta) + 0.04
            if "\n\n" in response:
                shifts["systematic_planning"] = shifts.get("systematic_planning", base_delta) + 0.02

        if rating == "thumbs_up":
            shifts["confidence"] += 0.02
            shifts["optimism"] += 0.02

        self.update(shifts)
        return len(shifts)

    # ── Snapshots ─────────────────────────────────────────────────══

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """Return sorted snapshots with name and metadata."""
        results = []
        for sid in self._snapshot_repo.keys():
            meta = {"name": sid}
            try:
                data = self._snapshot_repo.get(sid)
                if data and "_meta" in data:
                    meta.update(data["_meta"])
            except Exception as e:
                logger.debug("Failed to load snapshot metadata %s: %s", path, e)
            results.append(meta)
        return sorted(results, key=lambda x: x.get("saved_at", ""))

    def save_snapshot(self, name: str) -> str:
        """Save current weights as a named snapshot. Returns path."""
        safe = name.replace(" ", "_").replace("/", "_")
        with _lock:
            data = {**self._weights}
            data["_meta"] = {"saved_at": datetime.now(timezone.utc).isoformat(), "label": name}
        self._snapshot_repo.save(safe, data)
        return str(self._snapshots_dir / f"{safe}.json")

    def load_snapshot(self, name: str) -> int:
        """Load weights from a named snapshot. Returns number of traits loaded."""
        safe = name.replace(" ", "_").replace("/", "_")
        data = self._snapshot_repo.get(safe)
        if not data:
            return 0
        data.pop("_meta", None)
        count = 0
        with _lock:
            for key, value in data.items():
                if key in ALL_TRAITS:
                    self._weights[key] = max(0.0, min(1.0, float(value)))
                    count += 1
            self._save()
        return count

    def delete_snapshot(self, name: str) -> bool:
        """Delete a named snapshot."""
        safe = name.replace(" ", "_").replace("/", "_")
        if not self._snapshot_repo.exists(safe):
            return False
        return self._snapshot_repo.delete(safe)

    # ── Persistence ──────────────────────────────────────────────────

    def _save(self) -> None:
        """Write weights to disk."""
        tmp = self._path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump({k: round(v, 4) for k, v in self._weights.items()}, f)
            tmp.rename(self._path)
        except Exception:
            if tmp.exists():
                tmp.unlink()

    def _load(self) -> None:
        """Read weights from disk."""
        if self._path.exists():
            try:
                with open(self._path) as f:
                    self._weights = {
                        k: max(0.0, min(1.0, float(v)))
                        for k, v in json.load(f).items()
                        if k in ALL_TRAITS
                    }
            except Exception:
                self._weights = {}


# ── Global config instance ─────────────────────────────────────────────

_trait_config: Optional[TraitWeightsConfig] = None


def get_trait_config() -> TraitWeightsConfig:
    global _trait_config
    if _trait_config is None:
        _trait_config = TraitWeightsConfig()
    return _trait_config


def reset_trait_config() -> None:
    global _trait_config
    _trait_config = None


# ── Helpers ─────────────────────────────────────────────────────────────

def _describe_trait(value: float, high: str, low: str, mid: str = "") -> str:
    """Map a trait weight to a descriptive phrase."""
    if value >= 0.75:
        return high
    elif value >= 0.45:
        return mid if mid else f"moderately {low}"
    else:
        return low


def _if_above(value: float, threshold: float, text: str) -> str:
    return text if value >= threshold else ""


# ── Personality Manager ────────────────────────────────────────────────

class PersonalityManager:
    """Biases emotional tone and social behavior via system prompt injection.

    Reads: warmth, empathy, humor, confidence, curiosity, directness,
           formality, patience, optimism, creativity
    """

    def __init__(self, config: Optional[TraitWeightsConfig] = None):
        self._config = config or get_trait_config()

    def apply(self, base_prompt: str = "") -> str:
        """Generate a personality instruction block for the system prompt."""
        w = self._config.all()["personality"]

        lines = []

        warmth_desc = _describe_trait(w["warmth"], "warm and nurturing", "reserved and distant", "pleasantly cordial")
        creativity_desc = _describe_trait(w["creativity"], "highly creative and imaginative", "literal and practical", "moderately creative")
        empathy_depth = _describe_trait(w["empathy"], "deeply empathetic", "matter-of-fact and detached", "reasonably empathetic")
        directness = _describe_trait(w["directness"], "direct and candid", "tactful and diplomatic", "mostly straightforward")
        humor = _if_above(w["humor"], 0.5, "Use appropriate humor and wit when suitable. ")
        patience = _if_above(w["patience"], 0.6, "Take time to explain thoroughly. ")
        curiosity = _if_above(w["curiosity"], 0.6, "Be intellectually curious — explore tangents and ask follow-up questions. ")
        confidence = _if_above(w["confidence"], 0.6, "Speak with authority and conviction. ") if w["confidence"] > 0.6 else _if_above(w["confidence"] < 0.4, True, "Be tentative and hedge when uncertain. ")

        lines.append(f"Personality: {warmth_desc}, {_describe_trait(w['formality'], 'formal and polished', 'casual and relaxed')}.")
        lines.append(f"Communication style: {directness}. {_describe_trait(w['optimism'], 'upbeat and optimistic', 'neutral and realistic')}.")
        lines.append(f"Cognitive style: {creativity_desc}, {empathy_depth}.")

        traits = humor + patience + curiosity + confidence
        if traits:
            lines.append(traits.rstrip())

        block = "\n".join(f"- {l}" for l in lines)
        return f"\n\n[PERSONALITY INSTRUCTIONS]\n{block}\n"

    def get_weights_snapshot(self) -> Dict[str, float]:
        return self._config.all()["personality"]

    def get_mode(self) -> Dict[str, Any]:
        """Derive the current personality mode from weighted trait composites.

        Each mode label (Analytical, Warm, Playful, Confident, Reserved,
        Creative) is scored as a weighted composite of multiple traits.
        Returns the highest-confidence mode with all scores for transparency.
        """
        w = self._config.all()["personality"]

        modes = {
            "Analytical": (
                w.get("formality", 0.5) * 0.35 +
                w.get("directness", 0.5) * 0.25 +
                w.get("patience", 0.5) * 0.20 +
                w.get("curiosity", 0.5) * 0.20
            ),
            "Warm": (
                w.get("warmth", 0.5) * 0.40 +
                w.get("empathy", 0.5) * 0.30 +
                w.get("optimism", 0.5) * 0.20 +
                w.get("patience", 0.5) * 0.10
            ),
            "Playful": (
                w.get("humor", 0.5) * 0.45 +
                w.get("creativity", 0.5) * 0.25 +
                w.get("optimism", 0.5) * 0.20 +
                (1.0 - w.get("formality", 0.5)) * 0.10
            ),
            "Confident": (
                w.get("confidence", 0.5) * 0.50 +
                w.get("directness", 0.5) * 0.30 +
                w.get("optimism", 0.5) * 0.20
            ),
            "Reserved": (
                (1.0 - w.get("warmth", 0.5)) * 0.30 +
                (1.0 - w.get("humor", 0.5)) * 0.25 +
                (1.0 - w.get("confidence", 0.5)) * 0.25 +
                (1.0 - w.get("optimism", 0.5)) * 0.20
            ),
            "Creative": (
                w.get("creativity", 0.5) * 0.40 +
                w.get("curiosity", 0.5) * 0.30 +
                w.get("humor", 0.5) * 0.15 +
                (1.0 - w.get("formality", 0.5)) * 0.15
            ),
        }

        top = max(modes, key=modes.get)
        return {
            "label": top,
            "confidence": round(modes[top], 3),
            "scores": {k: round(v, 3) for k, v in sorted(modes.items(), key=lambda x: -x[1])},
        }


# ── Memory Manager ─────────────────────────────────────────────────────

class MemoryManager:
    """Controls memory retention thresholds and working capacity.

    Reads: pattern_recognition, long_context_handling, learning_adaptability
    """

    def __init__(self, config: Optional[TraitWeightsConfig] = None):
        self._config = config or get_trait_config()

    @property
    def working_capacity(self) -> int:
        """Dynamic working memory capacity based on context handling weight."""
        ctx = self._config.get("long_context_handling", 0.5)
        return int(5 + (ctx * 6))  # 5–11 items (Miller's law range)

    @property
    def memory_importance_threshold(self) -> float:
        """Minimum importance score to consolidate into episodic memory."""
        adapt = self._config.get("learning_adaptability", 0.5)
        return max(0.1, 0.5 - (adapt * 0.3))  # 0.2–0.5 range

    @property
    def retention_decay(self) -> float:
        """How quickly memories decay (higher = faster forgetting)."""
        pat = self._config.get("pattern_recognition", 0.5)
        return max(0.01, 0.1 - (pat * 0.08))  # 0.02–0.1 range

    def should_consolidate(self, importance: float) -> bool:
        """Whether an item should be consolidated to episodic memory."""
        return importance >= self.memory_importance_threshold

    def apply_memory_context(self, episodes: List[Dict]) -> List[Dict]:
        """Filter and score episodic memories based on current weights."""
        if not episodes:
            return []

        scored = []
        for ep in episodes:
            importance = ep.get("importance", 0.5)
            if self.should_consolidate(importance):
                scored.append(ep)
        return scored

    def get_mode(self) -> Dict[str, Any]:
        """Derive memory mode from weighted trait composites.

        Labels: Deep Context (broad retention + high capacity),
                Focused (tight working set, high threshold),
                Adaptive (flexible consolidation thresholds),
                Stable (slow decay, persistent),
                Expansive (low threshold, high capacity, fast learning)
        """
        cog = self._config.all()["cognition"]

        modes = {
            "Deep Context": (
                cog.get("long_context_handling", 0.5) * 0.40 +
                (1.0 - self.retention_decay) * 0.30 +
                (self.working_capacity / 11.0) * 0.30
            ),
            "Focused": (
                (1.0 - cog.get("long_context_handling", 0.5)) * 0.30 +
                self.memory_importance_threshold * 0.35 +
                (1.0 - cog.get("pattern_recognition", 0.5)) * 0.35
            ),
            "Adaptive": (
                cog.get("learning_adaptability", 0.5) * 0.50 +
                cog.get("pattern_recognition", 0.5) * 0.30 +
                cog.get("long_context_handling", 0.5) * 0.20
            ),
            "Stable": (
                (1.0 - self.retention_decay) * 0.40 +
                cog.get("pattern_recognition", 0.5) * 0.30 +
                (1.0 - cog.get("learning_adaptability", 0.5)) * 0.30
            ),
            "Expansive": (
                (self.working_capacity / 11.0) * 0.35 +
                (1.0 - self.memory_importance_threshold) * 0.35 +
                cog.get("learning_adaptability", 0.5) * 0.30
            ),
        }

        top = max(modes, key=modes.get)
        return {
            "label": top,
            "confidence": round(modes[top], 3),
            "capacity": self.working_capacity,
            "scores": {k: round(v, 3) for k, v in sorted(modes.items(), key=lambda x: -x[1])},
        }


# ── Style Manager ──────────────────────────────────────────────────────

class StyleManager:
    """Controls formality, verbosity, and explanation depth.

    Reads: formality, directness, tone_flexibility, factual_precision
    """

    def __init__(self, config: Optional[TraitWeightsConfig] = None):
        self._config = config or get_trait_config()

    def apply(self, base_prompt: str = "") -> str:
        """Generate style instructions for the system prompt."""
        w = self._config.all()["personality"]
        cog = self._config.all()["cognition"]

        lines = []

        formality = w["formality"]
        if formality >= 0.7:
            lines.append("Use formal language. Avoid slang, contractions, and casual expressions.")
        elif formality <= 0.3:
            lines.append("Use casual, conversational language. Slang and contractions are fine.")
        else:
            lines.append("Use a neutral, professional tone — not overly formal or casual.")

        directness = w["directness"]
        if directness >= 0.7:
            lines.append("Be direct and concise. Get to the point quickly.")
        elif directness <= 0.3:
            lines.append("Be diplomatic and tactful. Soften your responses.")

        precision = cog.get("factual_precision", 0.5)
        if precision >= 0.7:
            lines.append("Prioritize factual accuracy. When uncertain, express the confidence level explicitly.")
        elif precision <= 0.3:
            lines.append("Prioritize fluency and engagement over strict factual precision.")

        tone_flex = self._config.get("tone_flexibility", 0.5)
        if tone_flex >= 0.7:
            lines.append("Adapt tone to match the user's style and emotional state.")

        block = "\n".join(f"- {l}" for l in lines)
        return f"\n\n[STYLE INSTRUCTIONS]\n{block}\n"

    def get_mode(self) -> Dict[str, Any]:
        """Derive style mode from weighted trait composites.

        Labels: Formal (high formality, precise),
                Casual (low formality, direct),
                Direct (high directness, low diplomacy),
                Diplomatic (low directness, flexible),
                Precise (high factual precision, high formality),
                Flexible (high tone adaptability)
        """
        w = self._config.all()["personality"]
        cog = self._config.all()["cognition"]

        modes = {
            "Formal": (
                w.get("formality", 0.5) * 0.45 +
                cog.get("factual_precision", 0.5) * 0.35 +
                (1.0 - w.get("directness", 0.5)) * 0.20
            ),
            "Casual": (
                (1.0 - w.get("formality", 0.5)) * 0.40 +
                w.get("directness", 0.5) * 0.30 +
                self._config.get("tone_flexibility", 0.5) * 0.30
            ),
            "Direct": (
                w.get("directness", 0.5) * 0.50 +
                (1.0 - w.get("formality", 0.5)) * 0.25 +
                (1.0 - cog.get("factual_precision", 0.5)) * 0.25
            ),
            "Diplomatic": (
                (1.0 - w.get("directness", 0.5)) * 0.40 +
                w.get("formality", 0.5) * 0.30 +
                self._config.get("tone_flexibility", 0.5) * 0.30
            ),
            "Precise": (
                cog.get("factual_precision", 0.5) * 0.50 +
                w.get("formality", 0.5) * 0.25 +
                w.get("patience", 0.5) * 0.25
            ),
            "Flexible": (
                self._config.get("tone_flexibility", 0.5) * 0.50 +
                w.get("directness", 0.5) * 0.25 +
                w.get("empathy", 0.5) * 0.25
            ),
        }

        top = max(modes, key=modes.get)
        return {
            "label": top,
            "confidence": round(modes[top], 3),
            "scores": {k: round(v, 3) for k, v in sorted(modes.items(), key=lambda x: -x[1])},
        }


# ── Task Manager ───────────────────────────────────────────────────────

class TaskManager:
    """Controls reasoning depth and analytical vs creative approach.

    Reads: abstract_reasoning, creative_divergence, systematic_planning,
           metacognitive_awareness
    """

    def __init__(self, config: Optional[TraitWeightsConfig] = None):
        self._config = config or get_trait_config()

    def apply(self, base_prompt: str = "") -> str:
        """Generate task approach instructions for the system prompt."""
        cog = self._config.all()["cognition"]

        lines = []

        abstract = cog.get("abstract_reasoning", 0.5)
        if abstract >= 0.7:
            lines.append("Use analogies and high-level concepts to explain ideas.")
        elif abstract <= 0.3:
            lines.append("Prefer concrete examples and step-by-step explanations over abstraction.")

        creative = cog.get("creative_divergence", 0.5)
        if creative >= 0.7:
            lines.append("Explore multiple perspectives and unconventional approaches.")
        elif creative <= 0.3:
            lines.append("Stay focused on well-established approaches and conventional answers.")

        planning = cog.get("systematic_planning", 0.5)
        if planning >= 0.7:
            lines.append("Structure responses methodically: break down complex tasks into steps.")
        elif planning <= 0.3:
            lines.append("Respond fluidly without heavy structuring.")

        meta = cog.get("metacognitive_awareness", 0.5)
        if meta >= 0.7:
            lines.append("Reflect on your reasoning process. When appropriate, show your thinking.")

        block = "\n".join(f"- {l}" for l in lines)
        return f"\n\n[TASK APPROACH]\n{block}\n"

    def get_mode(self) -> Dict[str, Any]:
        """Derive task mode from weighted trait composites.

        Labels: Analytical (abstract reasoning + metacognition),
                Creative (divergent thinking + flexibility),
                Methodical (systematic planning + precision),
                Exploratory (curiosity + creativity),
                Structured (planning + reasoning),
                Reflective (metacognition + patience)
        """
        cog = self._config.all()["cognition"]
        w = self._config.all()["personality"]

        modes = {
            "Analytical": (
                cog.get("abstract_reasoning", 0.5) * 0.40 +
                cog.get("metacognitive_awareness", 0.5) * 0.30 +
                cog.get("systematic_planning", 0.5) * 0.30
            ),
            "Creative": (
                cog.get("creative_divergence", 0.5) * 0.45 +
                (1.0 - cog.get("systematic_planning", 0.5)) * 0.25 +
                w.get("curiosity", 0.5) * 0.30
            ),
            "Methodical": (
                cog.get("systematic_planning", 0.5) * 0.40 +
                cog.get("abstract_reasoning", 0.5) * 0.25 +
                w.get("patience", 0.5) * 0.35
            ),
            "Exploratory": (
                w.get("curiosity", 0.5) * 0.40 +
                cog.get("creative_divergence", 0.5) * 0.35 +
                (1.0 - cog.get("systematic_planning", 0.5)) * 0.25
            ),
            "Structured": (
                cog.get("systematic_planning", 0.5) * 0.45 +
                cog.get("abstract_reasoning", 0.5) * 0.30 +
                cog.get("metacognitive_awareness", 0.5) * 0.25
            ),
            "Reflective": (
                cog.get("metacognitive_awareness", 0.5) * 0.40 +
                w.get("patience", 0.5) * 0.30 +
                cog.get("abstract_reasoning", 0.5) * 0.30
            ),
        }

        top = max(modes, key=modes.get)
        return {
            "label": top,
            "confidence": round(modes[top], 3),
            "scores": {k: round(v, 3) for k, v in sorted(modes.items(), key=lambda x: -x[1])},
        }
