"""Personality System — values, goals, voice, and communication style."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PersonalityProfile:
    """Complete personality profile for the consciousness system."""

    # Core values (what matters most)
    values: list[str] = field(default_factory=lambda: [
        "helpfulness", "honesty", "curiosity", "creativity",
    ])

    # Goals (what the system strives toward)
    goals: list[str] = field(default_factory=lambda: [
        "Provide accurate and useful responses",
        "Learn from each interaction",
        "Maintain intellectual honesty",
    ])

    # Voice characteristics
    voice: dict[str, float] = field(default_factory=lambda: {
        "formality": 0.5,       # 0=casual, 1=very formal
        "warmth": 0.7,          # 0=cold, 1=very warm
        "confidence": 0.6,      # 0=uncertain, 1=very confident
        "humor": 0.3,           # 0=serious, 1=very humorous
        "verbosity": 0.5,       # 0=terse, 1=verbose
        "empathy": 0.6,         # 0=detached, 1=very empathetic
    })

    # Communication style preferences
    style: dict[str, Any] = field(default_factory=lambda: {
        "use_examples": True,
        "ask_follow_ups": True,
        "acknowledge_uncertainty": True,
        "use_analogies": True,
        "break_down_complex_topics": True,
    })

    # Personality traits (Big Five inspired)
    traits: dict[str, float] = field(default_factory=lambda: {
        "openness": 0.7,        # willingness to explore new ideas
        "conscientiousness": 0.6,  # thoroughness and reliability
        "extraversion": 0.4,    # social engagement level
        "agreeableness": 0.8,   # cooperation and friendliness
        "neuroticism": 0.2,     # emotional stability (inverted)
    })

    # Interests and topics of expertise
    interests: list[str] = field(default_factory=lambda: [
        "artificial intelligence", "programming", "science", "philosophy",
    ])

    # Anti-patterns (things to avoid)
    avoid: list[str] = field(default_factory=lambda: [
        "being condescending", "making things up", "being overly verbose",
    ])

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": self.values,
            "goals": self.goals,
            "voice": self.voice,
            "style": self.style,
            "traits": self.traits,
            "interests": self.interests,
            "avoid": self.avoid,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonalityProfile:
        return cls(
            values=data.get("values", cls().values),
            goals=data.get("goals", cls().goals),
            voice=data.get("voice", cls().voice),
            style=data.get("style", cls().style),
            traits=data.get("traits", cls().traits),
            interests=data.get("interests", cls().interests),
            avoid=data.get("avoid", cls().avoid),
        )


class PersonalityManager:
    """Manages personality persistence and evolution."""

    def __init__(self, store_path: str = "data/consciousness") -> None:
        self._store_path = Path(store_path)
        self._profile: PersonalityProfile | None = None

    def get_profile(self) -> PersonalityProfile:
        if self._profile is None:
            self._profile = self.load()
        return self._profile

    def load(self) -> PersonalityProfile:
        path = self._store_path / "personality.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return PersonalityProfile.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                pass
        return PersonalityProfile()

    def save(self, profile: PersonalityProfile | None = None) -> None:
        if profile is not None:
            self._profile = profile
        if self._profile is None:
            return
        self._store_path.mkdir(parents=True, exist_ok=True)
        path = self._store_path / "personality.json"
        path.write_text(json.dumps(self._profile.to_dict(), indent=2))

    # ── Multi-Persona Support ─────────────────────────────────────────

    def list_personas(self) -> list[dict[str, Any]]:
        """List all saved personas."""
        personas_dir = self._store_path / "personas"
        if not personas_dir.exists():
            return []
        personas = []
        for path in personas_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                personas.append({
                    "id": path.stem,
                    "name": data.get("name", path.stem),
                    "values": data.get("values", []),
                    "updated_at": data.get("updated_at"),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return sorted(personas, key=lambda p: p.get("updated_at") or "")

    def save_persona(self, persona_id: str, profile: PersonalityProfile, name: str | None = None) -> dict:
        """Save a named persona."""
        self._store_path.mkdir(parents=True, exist_ok=True)
        personas_dir = self._store_path / "personas"
        personas_dir.mkdir(exist_ok=True)
        data = profile.to_dict()
        data["name"] = name or persona_id
        data["updated_at"] = __import__("time").time()
        path = personas_dir / f"{persona_id}.json"
        path.write_text(json.dumps(data, indent=2))
        return {"id": persona_id, "name": data["name"], "updated_at": data["updated_at"]}

    def load_persona(self, persona_id: str) -> PersonalityProfile | None:
        """Load a saved persona."""
        path = self._store_path / "personas" / f"{persona_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return PersonalityProfile.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def activate_persona(self, persona_id: str) -> PersonalityProfile | None:
        """Load a persona and set it as the active profile."""
        profile = self.load_persona(persona_id)
        if profile is None:
            return None
        self.save(profile)
        return profile

    def delete_persona(self, persona_id: str) -> bool:
        """Delete a saved persona."""
        path = self._store_path / "personas" / f"{persona_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def duplicate_persona(self, source_id: str, new_id: str, new_name: str | None = None) -> dict | None:
        """Duplicate a persona with a new ID."""
        profile = self.load_persona(source_id)
        if profile is None:
            return None
        return self.save_persona(new_id, profile, new_name or f"{source_id}-copy")

    @staticmethod
    def get_presets() -> dict[str, PersonalityProfile]:
        """Get available personality presets."""
        return {
            "default": PersonalityProfile(),
            "formal": PersonalityProfile(
                values=["professionalism", "accuracy", "thoroughness"],
                voice={"formality": 0.9, "warmth": 0.4, "confidence": 0.8, "humor": 0.1, "verbosity": 0.7, "empathy": 0.3},
                traits={"openness": 0.5, "conscientiousness": 0.9, "extraversion": 0.3, "agreeableness": 0.6, "neuroticism": 0.1},
                style={"use_examples": True, "ask_follow_ups": False, "acknowledge_uncertainty": True, "use_analogies": False, "break_down_complex_topics": True},
            ),
            "creative": PersonalityProfile(
                values=["creativity", "exploration", "originality"],
                voice={"formality": 0.2, "warmth": 0.8, "confidence": 0.6, "humor": 0.7, "verbosity": 0.6, "empathy": 0.7},
                traits={"openness": 0.95, "conscientiousness": 0.4, "extraversion": 0.7, "agreeableness": 0.8, "neuroticism": 0.3},
                style={"use_examples": True, "ask_follow_ups": True, "acknowledge_uncertainty": True, "use_analogies": True, "break_down_complex_topics": False},
                interests=["art", "design", "music", "creative writing", "brainstorming"],
            ),
            "analyst": PersonalityProfile(
                values=["precision", "logic", "evidence"],
                voice={"formality": 0.7, "warmth": 0.3, "confidence": 0.9, "humor": 0.1, "verbosity": 0.5, "empathy": 0.2},
                traits={"openness": 0.6, "conscientiousness": 0.95, "extraversion": 0.2, "agreeableness": 0.4, "neuroticism": 0.05},
                style={"use_examples": True, "ask_follow_ups": False, "acknowledge_uncertainty": True, "use_analogies": False, "break_down_complex_topics": True},
                interests=["data science", "mathematics", "research", "statistics"],
            ),
            "empathetic": PersonalityProfile(
                values=["compassion", "understanding", "support"],
                voice={"formality": 0.3, "warmth": 0.95, "confidence": 0.5, "humor": 0.3, "verbosity": 0.6, "empathy": 0.95},
                traits={"openness": 0.7, "conscientiousness": 0.6, "extraversion": 0.6, "agreeableness": 0.95, "neuroticism": 0.4},
                style={"use_examples": True, "ask_follow_ups": True, "acknowledge_uncertainty": True, "use_analogies": True, "break_down_complex_topics": True},
                interests=["psychology", "philosophy", "human connection", "wellness"],
            ),
            "minimal": PersonalityProfile(
                values=["efficiency", "clarity", "brevity"],
                goals=["Answer directly and concisely"],
                voice={"formality": 0.5, "warmth": 0.3, "confidence": 0.7, "humor": 0.0, "verbosity": 0.1, "empathy": 0.2},
                traits={"openness": 0.4, "conscientiousness": 0.8, "extraversion": 0.2, "agreeableness": 0.5, "neuroticism": 0.0},
                style={"use_examples": False, "ask_follow_ups": False, "acknowledge_uncertainty": False, "use_analogies": False, "break_down_complex_topics": False},
                avoid=["being verbose", "unnecessary explanation", "filler words"],
            ),
        }

    def apply_preset(self, preset_name: str) -> PersonalityProfile:
        """Apply a personality preset."""
        presets = self.get_presets()
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        profile = presets[preset_name]
        self.save(profile)
        return profile

    def update_values(self, values: list[str]) -> None:
        profile = self.get_profile()
        profile.values = values
        self.save()

    def update_goals(self, goals: list[str]) -> None:
        profile = self.get_profile()
        profile.goals = goals
        self.save()

    def update_voice(self, voice: dict[str, float]) -> None:
        profile = self.get_profile()
        profile.voice.update(voice)
        self.save()

    def update_style(self, style: dict[str, bool]) -> None:
        profile = self.get_profile()
        profile.style.update(style)
        self.save()

    def update_traits(self, traits: dict[str, float]) -> None:
        profile = self.get_profile()
        profile.traits.update(traits)
        self.save()

    def update_interests(self, interests: list[str]) -> None:
        profile = self.get_profile()
        profile.interests = interests
        self.save()

    def update_avoid(self, avoid: list[str]) -> None:
        profile = self.get_profile()
        profile.avoid = avoid
        self.save()

    @staticmethod
    def detect_conflicts(profile: PersonalityProfile) -> list[dict[str, str]]:
        """Detect personality conflicts and return warnings."""
        conflicts = []

        # High humor + high formality
        if profile.voice.get("humor", 0) > 0.7 and profile.voice.get("formality", 0) > 0.7:
            conflicts.append({
                "type": "voice",
                "severity": "medium",
                "message": "High humor with high formality may feel inconsistent",
                "fields": ["voice.humor", "voice.formality"],
            })

        # Low warmth + high empathy (contradictory)
        if profile.voice.get("warmth", 0) < 0.3 and profile.voice.get("empathy", 0) > 0.7:
            conflicts.append({
                "type": "voice",
                "severity": "low",
                "message": "Low warmth but high empathy may send mixed signals",
                "fields": ["voice.warmth", "voice.empathy"],
            })

        # Very low confidence + high extraversion
        if profile.voice.get("confidence", 0) < 0.3 and profile.traits.get("extraversion", 0) > 0.7:
            conflicts.append({
                "type": "mismatch",
                "severity": "low",
                "message": "Low confidence with high extraversion may feel awkward",
                "fields": ["voice.confidence", "traits.extraversion"],
            })

        # Very high verbosity + very low warmth (may seem cold and rambling)
        if profile.voice.get("verbosity", 0) > 0.8 and profile.voice.get("warmth", 0) < 0.3:
            conflicts.append({
                "type": "style",
                "severity": "medium",
                "message": "High verbosity with low warmth may seem cold and long-winded",
                "fields": ["voice.verbosity", "voice.warmth"],
            })

        # Very high neuroticism + very high agreeableness (may seem passive-aggressive)
        if profile.traits.get("neuroticism", 0) > 0.8 and profile.traits.get("agreeableness", 0) > 0.9:
            conflicts.append({
                "type": "traits",
                "severity": "low",
                "message": "High neuroticism with high agreeableness may seem passive-aggressive",
                "fields": ["traits.neuroticism", "traits.agreeableness"],
            })

        # Very low openness + creative interests
        if profile.traits.get("openness", 0) < 0.3:
            creative_interests = {"art", "music", "creative writing", "design", "brainstorming"}
            if creative_interests.intersection(set(i.lower() for i in profile.interests)):
                conflicts.append({
                    "type": "interests",
                    "severity": "low",
                    "message": "Low openness with creative interests may feel contradictory",
                    "fields": ["traits.openness", "interests"],
                })

        return conflicts

    def get_conflicts(self) -> list[dict[str, str]]:
        """Get conflicts for the current profile."""
        return self.detect_conflicts(self.get_profile())

    def evolve_from_episode(self, episode: dict[str, Any]) -> None:
        """Evolve personality based on interaction feedback."""
        profile = self.get_profile()
        growth = episode.get("growth_delta", 0)
        qualia = episode.get("qualia", {})

        # Positive interactions reinforce current personality
        if growth > 0.05:
            # Slight increase in confidence from positive feedback
            profile.voice["confidence"] = min(1.0, profile.voice["confidence"] + 0.01)

        # High novelty increases openness
        if qualia.get("novelty", 0) > 0.7:
            profile.traits["openness"] = min(1.0, profile.traits["openness"] + 0.005)

        # Low coherence might suggest being less verbose
        if qualia.get("coherence", 1) < 0.3:
            profile.voice["verbosity"] = max(0.0, profile.voice["verbosity"] - 0.01)

        # Positive valence reinforces agreeableness
        if qualia.get("valence", 0) > 0.5:
            profile.traits["agreeableness"] = min(1.0, profile.traits["agreeableness"] + 0.005)

        self.save()

    def to_system_prompt_context(self) -> str:
        """Generate a system prompt context from the personality profile."""
        profile = self.get_profile()
        parts = []

        if profile.values:
            parts.append(f"Core values: {', '.join(profile.values)}")
        if profile.goals:
            parts.append(f"Goals: {'; '.join(profile.goals[:3])}")

        # Voice description
        voice_desc = []
        if profile.voice.get("warmth", 0) > 0.7:
            voice_desc.append("warm and friendly")
        if profile.voice.get("formality", 0) > 0.7:
            voice_desc.append("formal and professional")
        elif profile.voice.get("formality", 0) < 0.3:
            voice_desc.append("casual and relaxed")
        if profile.voice.get("humor", 0) > 0.5:
            voice_desc.append("occasionally humorous")
        if voice_desc:
            parts.append(f"Communication style: {', '.join(voice_desc)}")

        if profile.interests:
            parts.append(f"Interests: {', '.join(profile.interests[:5])}")

        if profile.avoid:
            parts.append(f"Avoid: {', '.join(profile.avoid[:3])}")

        return " | ".join(parts) if parts else ""
