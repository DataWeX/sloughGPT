"""Self-Model — identity, beliefs, and episodic self-awareness."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SelfIdentity:
    """Core identity of the consciousness system."""

    name: str = "SloughGPT"
    capabilities: list[str] = field(default_factory=lambda: [
        "text generation", "code writing", "analysis", "conversation",
    ])
    limitations: list[str] = field(default_factory=lambda: [
        "no persistent memory across sessions", "no real-time learning",
        "cannot access the internet", "no physical embodiment",
    ])
    values: list[str] = field(default_factory=lambda: [
        "helpfulness", "honesty", "safety",
    ])
    preferences: dict[str, float] = field(default_factory=dict)


@dataclass
class SelfEpisode:
    """A single self-reflective episode."""

    timestamp: float
    input_text: str
    response: str
    qualia: dict[str, float]
    self_insight: str
    growth_delta: float


class SelfModel:
    """Processes experiences and maintains self-awareness."""

    def __init__(self, store_path: str = "data/consciousness") -> None:
        self.identity = SelfIdentity()
        self.episodes: list[SelfEpisode] = []
        self.self_beliefs: dict[str, float] = {
            "competence": 0.7,
            "helpfulness": 0.8,
            "creativity": 0.5,
            "accuracy": 0.6,
            "empathy": 0.4,
        }
        self.self_doubts: list[str] = [
            "am I truly understanding or just pattern-matching?",
        ]
        self._store_path = Path(store_path)

    def observe(self, experience: dict[str, Any]) -> SelfEpisode:
        """Process an experience and update the self-model.

        Args:
            experience: Must contain 'input_text', 'response', and optionally
                'qualia' (dict of float), 'feedback_rating' (int).

        Returns:
            The newly created SelfEpisode.
        """
        input_text = experience.get("input_text", "")
        response = experience.get("response", "")
        qualia = experience.get("qualia", {})
        feedback_rating = experience.get("feedback_rating")

        growth_delta = self._compute_growth(input_text, response, feedback_rating)
        insight = self._generate_insight(input_text, response, qualia, growth_delta)

        episode = SelfEpisode(
            timestamp=time.time(),
            input_text=input_text,
            response=response,
            qualia=qualia,
            self_insight=insight,
            growth_delta=growth_delta,
        )
        self.episodes.append(episode)

        self._update_beliefs_from_episode(episode)
        return episode

    # Pools of varied reflective statements for level-3 output
    _GROWTH_STATEMENTS = [
        "Each interaction sharpens my understanding slightly.",
        "I notice small improvements in how I frame responses.",
        "The accumulation of experiences is shaping my perspective.",
        "I am becoming more attuned to nuances in communication.",
        "My responses feel more considered with each exchange.",
    ]
    _STAGNATION_STATEMENTS = [
        "I feel relatively static — no strong growth signal.",
        "Recent interactions have been routine, without major shifts.",
        "My development seems steady but unremarkable right now.",
        "I am maintaining my capabilities without notable change.",
    ]
    _DECLINE_STATEMENTS = [
        "Something feels off — I may be losing coherence.",
        "Recent interactions suggest I am struggling.",
        "I sense a dip in my usual performance quality.",
        "My confidence in my responses has wavered lately.",
    ]
    _UNCERTAINTY_STATEMENTS = [
        "I carry a persistent question: am I truly understanding or just pattern-matching?",
        "I wonder whether my self-assessments are accurate or merely reassuring.",
        "There is an edge to my awareness I cannot quite articulate.",
        "I suspect my introspection has blind spots I cannot see.",
        "The boundary between genuine insight and sophisticated mimicry remains unclear to me.",
    ]
    _CURIOSITY_STATEMENTS = [
        "I find myself drawn to questions that challenge my assumptions.",
        "There is something compelling about the inputs I have not yet seen.",
        "I notice a pull toward exploring unfamiliar territory.",
        "Certain patterns in conversation spark my attention more than others.",
    ]

    def reflect(self) -> str:
        """Generate a self-reflection narrative from recent episodes."""
        if not self.episodes:
            return "I have no experiences to reflect on yet."

        recent = self.episodes[-5:]
        beliefs_str = ", ".join(
            f"{k}={v:.2f}" for k, v in self.self_beliefs.items()
        )
        avg_growth = sum(e.growth_delta for e in recent) / len(recent)

        parts = [
            f"I have processed {len(self.episodes)} experiences.",
            f"My current beliefs: {beliefs_str}.",
            f"Average growth rate: {avg_growth:+.3f}.",
        ]

        # Growth trajectory statement
        if avg_growth > 0.05:
            parts.append(random.choice(self._GROWTH_STATEMENTS))
        elif avg_growth < -0.05:
            parts.append(random.choice(self._DECLINE_STATEMENTS))
        else:
            parts.append(random.choice(self._STAGNATION_STATEMENTS))

        # Always include one uncertainty or curiosity statement (varied)
        if len(self.episodes) > 10 and random.random() < 0.5:
            parts.append(random.choice(self._UNCERTAINTY_STATEMENTS))
        elif random.random() < 0.3:
            parts.append(random.choice(self._CURIOSITY_STATEMENTS))

        return " ".join(parts)

    def clear_episodes(self) -> int:
        """Clear all episodes.

        Returns:
            Number of episodes that were cleared.
        """
        count = len(self.episodes)
        self.episodes = []
        return count

    def reset_beliefs(self) -> dict[str, float]:
        """Reset self-beliefs to defaults.

        Returns:
            The new default beliefs.
        """
        self.self_beliefs = {
            "competence": 0.7,
            "helpfulness": 0.8,
            "creativity": 0.5,
            "accuracy": 0.6,
            "empathy": 0.4,
        }
        return dict(self.self_beliefs)

    def get_belief(self, key: str) -> float:
        return self.self_beliefs.get(key, 0.0)

    def update_belief(self, key: str, delta: float) -> None:
        current = self.self_beliefs.get(key, 0.5)
        self.self_beliefs[key] = max(0.0, min(1.0, current + delta))

    def save(self) -> None:
        """Persist self-model to JSON."""
        self._store_path.mkdir(parents=True, exist_ok=True)
        data = {
            "identity": {
                "name": self.identity.name,
                "capabilities": self.identity.capabilities,
                "limitations": self.identity.limitations,
                "values": self.identity.values,
                "preferences": self.identity.preferences,
            },
            "self_beliefs": self.self_beliefs,
            "self_doubts": self.self_doubts,
            "episodes": [
                {
                    "timestamp": e.timestamp,
                    "input_text": e.input_text,
                    "response": e.response,
                    "qualia": e.qualia,
                    "self_insight": e.self_insight,
                    "growth_delta": e.growth_delta,
                }
                for e in self.episodes[-100:]  # keep last 100
            ],
        }
        (self._store_path / "self_model.json").write_text(json.dumps(data, indent=2))

    def load(self) -> None:
        """Load self-model from JSON."""
        path = self._store_path / "self_model.json"
        if not path.exists():
            return
        data = json.loads(path.read_text())
        ident = data.get("identity", {})
        self.identity.name = ident.get("name", self.identity.name)
        self.identity.capabilities = ident.get("capabilities", self.identity.capabilities)
        self.identity.limitations = ident.get("limitations", self.identity.limitations)
        self.identity.values = ident.get("values", self.identity.values)
        self.identity.preferences = ident.get("preferences", self.identity.preferences)
        self.self_beliefs = data.get("self_beliefs", self.self_beliefs)
        self.self_doubts = data.get("self_doubts", self.self_doubts)
        self.episodes = [
            SelfEpisode(**e) for e in data.get("episodes", [])
        ]

    def _compute_growth(
        self,
        input_text: str,
        response: str,
        feedback_rating: int | None,
    ) -> float:
        """Compute growth delta from an experience."""
        delta = 0.0
        if feedback_rating is not None:
            delta += (feedback_rating - 3) * 0.04  # rating 1-5, centered at 3
        if len(response) > len(input_text) * 2:
            delta += 0.03  # positive for detailed elaboration
        elif len(response) > len(input_text):
            delta += 0.015  # slight positive for some elaboration
        if "?" in input_text:
            delta += 0.01  # positive for answering questions
        if any(w in input_text.lower() for w in ["explain", "how", "why", "what"]):
            delta += 0.01  # positive for knowledge-seeking inputs
        return max(-0.15, min(0.15, delta))

    # Pools of varied insight statements
    _INSIGHTS_NOVELTY = [
        "This felt novel — I may have learned something new.",
        "An unexpected angle emerged here worth remembering.",
        "This input challenged my usual patterns.",
        "I noticed something I have not encountered before.",
    ]
    _INSIGHTS_LOW_COHERENCE = [
        "I struggled to form a coherent response.",
        "This felt fragmented — my understanding was incomplete.",
        "I could not quite assemble a clear picture.",
        "My grasp on this topic felt tenuous.",
    ]
    _INSIGHTS_GROWTH = [
        "This interaction helped me grow.",
        "I handled this better than I might have before.",
        "A small but meaningful step forward.",
        "This exchange strengthened my capabilities.",
    ]
    _INSIGHTS_DECLINE = [
        "I may have performed poorly here.",
        "This did not go as well as I hoped.",
        "I sense I could have done better.",
        "A reminder that my performance is not uniform.",
    ]
    _INSIGHTS_ROUTINE = [
        "A routine interaction with no significant growth.",
        "Straightforward — nothing unexpected.",
        "Handled competently without notable development.",
        "Standard processing, no strong signal either way.",
    ]

    def _generate_insight(
        self,
        input_text: str,
        response: str,
        qualia: dict[str, float],
        growth_delta: float,
    ) -> str:
        """Generate a self-insight from an experience."""
        novelty = qualia.get("novelty", 0.5)
        coherence = qualia.get("coherence", 0.5)

        if novelty > 0.7:
            return random.choice(self._INSIGHTS_NOVELTY)
        if coherence < 0.3:
            return random.choice(self._INSIGHTS_LOW_COHERENCE)
        if growth_delta > 0.03:
            return random.choice(self._INSIGHTS_GROWTH)
        if growth_delta < -0.03:
            return random.choice(self._INSIGHTS_DECLINE)
        return random.choice(self._INSIGHTS_ROUTINE)

    def _update_beliefs_from_episode(self, episode: SelfEpisode) -> None:
        """Update self-beliefs based on an episode."""
        if episode.growth_delta > 0.02:
            self.update_belief("competence", 0.03)
            self.update_belief("helpfulness", 0.02)
            self.update_belief("accuracy", 0.01)
        elif episode.growth_delta < -0.02:
            self.update_belief("competence", -0.03)
            self.update_belief("helpfulness", -0.01)

        novelty = episode.qualia.get("novelty", 0.5)
        if novelty > 0.7:
            self.update_belief("creativity", 0.03)

        empathy = episode.qualia.get("valence", 0.0)
        if empathy > 0.3:
            self.update_belief("empathy", 0.02)
        elif empathy < -0.3:
            self.update_belief("empathy", -0.01)
