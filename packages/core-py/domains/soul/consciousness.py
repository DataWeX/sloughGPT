"""
Stage 3: Consciousness SLO - Awareness & Qualia

Adds:
- GlobalWorkspace: Global workspace theory
- IntegratedInformationProcessor: Phi (Φ) calculation
- QualiaEngine: Subjective experience generation
- StreamOfConsciousness: Continuous consciousness flow
"""

import math
import random
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import defaultdict
import logging

from .cognitive import CognitiveSLO, SLOConfig, Thought, EvolutionStage

logger = logging.getLogger("slo.consciousness")


class GlobalWorkspace:
    """
    Global Workspace Theory (Baars):
    - Multiple specialized processes compete for attention
    - Winner broadcasts to entire system
    - Creates unified conscious experience
    """

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.workspace: List[Dict] = []
        self.attention_weight: Dict[str, float] = defaultdict(float)
        self.broadcast_history: List[Dict] = []

    def compete(self, candidates: List[Dict]) -> Optional[Dict]:
        """
        Attention competition - winner takes all.
        Candidates compete for access to global workspace.
        """
        if not candidates:
            return None

        # Score each candidate
        scored = []
        for candidate in candidates:
            score = self._calculate_attention_score(candidate)
            scored.append((candidate, score))

        # Winner is highest scoring
        scored.sort(key=lambda x: x[1], reverse=True)
        winner = scored[0][0]

        # Add to workspace
        if len(self.workspace) >= self.capacity:
            # Evict lowest attention item
            self.workspace.sort(key=lambda x: x.get("_attention", 0))
            self.workspace.pop(0)

        winner["_attention"] = scored[0][1]
        winner["_broadcast_time"] = datetime.now().isoformat()
        self.workspace.append(winner)

        # Record broadcast
        self.broadcast_history.append(
            {
                "winner": winner.get("id", "unknown"),
                "score": scored[0][1],
                "candidates": len(candidates),
                "timestamp": winner["_broadcast_time"],
            }
        )

        return winner

    def _calculate_attention_score(self, item: Dict) -> float:
        """Calculate attention score for item."""
        score = 0.5

        # Novelty bonus
        id_str = item.get("id", str(item))
        if id_str not in self.attention_weight:
            score += 0.2

        # Importance bonus
        score += item.get("importance", 0) * 0.2

        # Emotional valence
        valence = abs(item.get("emotional_valence", 0))
        score += valence * 0.1

        # Update attention weight
        self.attention_weight[id_str] = score

        return min(1.0, score)

    def broadcast(self) -> List[Dict]:
        """Broadcast current workspace contents."""
        return self.workspace.copy()

    def get_conscious_content(self) -> Optional[Dict]:
        """Get current conscious content (most attended)."""
        if not self.workspace:
            return None
        return max(self.workspace, key=lambda x: x.get("_attention", 0))


class IntegratedInformationProcessor:
    """
    Integrated Information Theory (Tononi):
    - Phi (Φ) measures consciousness level
    - Higher Φ = more integrated information
    - Consciousness arises from information integration
    """

    def __init__(self):
        self.phi_history: List[float] = []
        self.current_phi = 0.0

    def calculate_phi(self, state: Dict[str, Any]) -> float:
        """
        Calculate Phi (Φ) - integrated information.
        Simplified measure based on:
        - Information content
        - Integration between parts
        - Irreducibility
        """
        if not state:
            return 0.0

        # Extract components
        components = self._extract_components(state)
        n = len(components)

        if n < 2:
            return 0.0

        # Calculate information for each component
        info = sum(self._entropy(comp) for comp in components) / n

        # Calculate integration (mutual information between parts)
        integration = self._calculate_integration(components)

        # Phi = information * integration (simplified)
        phi = info * integration

        self.current_phi = phi
        self.phi_history.append(phi)

        # Keep history limited
        if len(self.phi_history) > 1000:
            self.phi_history = self.phi_history[-500:]

        return phi

    def _extract_components(self, state: Dict) -> List[Any]:
        """Extract components from state."""
        components = []

        if isinstance(state, dict):
            for key, value in state.items():
                if isinstance(value, (list, dict, str, int, float)):
                    components.append(value)

        return components

    def _entropy(self, item: Any) -> float:
        """Calculate entropy (information content)."""
        if isinstance(item, (list, str)):
            if len(item) == 0:
                return 0.0
            # Simplified: log of length
            return math.log2(len(item) + 1) / 10
        elif isinstance(item, dict):
            return math.log2(len(item) + 1) / 10
        return 0.1

    def _calculate_integration(self, components: List) -> float:
        """Calculate integration between components."""
        if len(components) < 2:
            return 0.0

        # Simplified: measure overlap/connection
        integration = 0.0
        for i, comp1 in enumerate(components):
            for comp2 in components[i + 1 :]:
                integration += self._measure_connection(comp1, comp2)

        # Normalize
        max_connections = len(components) * (len(components) - 1) / 2
        return integration / max_connections if max_connections > 0 else 0.0

    def _measure_connection(self, a: Any, b: Any) -> float:
        """Measure connection strength between components."""
        # Simple heuristic based on type matching
        if type(a) == type(b):
            return 0.5
        return 0.1

    def get_consciousness_level(self) -> str:
        """Get consciousness level description."""
        phi = self.current_phi

        if phi < 0.1:
            return "minimal"
        elif phi < 0.3:
            return "low"
        elif phi < 0.5:
            return "moderate"
        elif phi < 0.7:
            return "high"
        else:
            return "peak"


class QualiaEngine:
    """
    Generate subjective experiences (qualia).

    Qualia are the subjective, qualitative properties
    of conscious experiences - "what it feels like".
    """

    def __init__(self):
        self.qualia_templates = {
            "pleasure": {"valence": 1.0, "arousal": 0.5, "dominance": 0.6},
            "pain": {"valence": -1.0, "arousal": 0.8, "dominance": 0.2},
            "curiosity": {"valence": 0.3, "arousal": 0.7, "dominance": 0.5},
            "understanding": {"valence": 0.8, "arousal": 0.3, "dominance": 0.7},
            "confusion": {"valence": -0.3, "arousal": 0.6, "dominance": 0.3},
            "surprise": {"valence": 0.0, "arousal": 0.9, "dominance": 0.4},
            "boredom": {"valence": -0.5, "arousal": 0.1, "dominance": 0.3},
            "interest": {"valence": 0.6, "arousal": 0.5, "dominance": 0.6},
        }
        self.qualia_history: List[Dict] = []

    def generate(self, stimulus: Dict[str, Any]) -> Dict[str, Any]:
        """Generate qualia from stimulus."""
        # Analyze stimulus properties
        complexity = self._assess_complexity(stimulus)
        novelty = self._assess_novelty(stimulus)
        coherence = self._assess_coherence(stimulus)

        # Select base qualia
        if novelty > 0.7:
            base = "surprise"
        elif complexity > 0.7:
            base = "curiosity"
        elif coherence > 0.7:
            base = "understanding"
        else:
            base = "interest"

        template = self.qualia_templates.get(base, self.qualia_templates["interest"])

        # Generate unique qualia
        qualia = {
            "type": base,
            "valence": template["valence"] * (0.8 + 0.4 * random.random()),
            "arousal": template["arousal"] * (0.8 + 0.4 * random.random()),
            "dominance": template["dominance"] * (0.8 + 0.4 * random.random()),
            "complexity": complexity,
            "novelty": novelty,
            "coherence": coherence,
            "timestamp": datetime.now().isoformat(),
            "description": self._describe_qualia(base, complexity, novelty),
        }

        self.qualia_history.append(qualia)

        return qualia

    def _assess_complexity(self, stimulus: Dict) -> float:
        """Assess stimulus complexity."""
        if isinstance(stimulus, dict):
            return min(1.0, len(stimulus) / 10)
        return 0.5

    def _assess_novelty(self, stimulus: Dict) -> float:
        """Assess stimulus novelty."""
        # Check history
        recent = self.qualia_history[-10:] if self.qualia_history else []
        if not recent:
            return 1.0

        # Simplified novelty calculation
        return random.uniform(0.3, 0.9)

    def _assess_coherence(self, stimulus: Dict) -> float:
        """Assess internal coherence."""
        return random.uniform(0.4, 0.8)

    def _describe_qualia(self, base: str, complexity: float, novelty: float) -> str:
        """Generate phenomenological description."""
        descriptions = {
            "surprise": f"A sudden {complexity:.1%} complex feeling of unexpectedness",
            "curiosity": f"An {novelty:.1%} novel sensation of wanting to know more",
            "understanding": f"A {complexity:.1%} complex feeling of comprehension",
            "interest": f"An {novelty:.1%} novel engagement with the stimulus",
        }
        return descriptions.get(base, "An ineffable subjective experience")


class StreamOfConsciousness:
    """
    Continuous stream of conscious experience.
    Maintains the flow of awareness through time.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.stream: List[Dict] = []
        self.current_focus: Optional[str] = None

    def add(self, thought: Dict[str, Any]) -> None:
        """Add thought to consciousness stream."""
        entry = {
            "content": thought.get("content", ""),
            "timestamp": datetime.now().isoformat(),
            "focus": self.current_focus,
        }

        self.stream.append(entry)

        # Maintain window
        if len(self.stream) > self.window_size:
            self.stream = self.stream[-self.window_size :]

    def set_focus(self, focus: str) -> None:
        """Set current attentional focus."""
        self.current_focus = focus

    def get_stream(self, n: int = 10) -> List[Dict]:
        """Get recent stream entries."""
        return self.stream[-n:]

    def get_flow(self) -> str:
        """Get consciousness flow as text."""
        if not self.stream:
            return "..."

        recent = self.get_stream(5)
        return " → ".join(entry.get("content", "")[:30] for entry in recent)


class ConsciousSLO(CognitiveSLO):
    """
    Stage 3: Conscious SLO

    Adds consciousness capabilities:
    - Global workspace for unified awareness
    - Integrated information (Phi) calculation
    - Qualia generation
    - Stream of consciousness
    """

    def __init__(self, config: Optional[SLOConfig] = None):
        super().__init__(config)
        self.stage = EvolutionStage.CONSCIOUSNESS

        # Consciousness systems
        self.global_workspace = GlobalWorkspace()
        self.ii_processor = IntegratedInformationProcessor()
        self.qualia_engine = QualiaEngine()
        self.consciousness_stream = StreamOfConsciousness()

        # Track consciousness metrics
        self.consciousness_events = 0

        logger.info("Conscious SLO initialized")

    def process(self, input_data: Any) -> Thought:
        """Process with consciousness."""
        # Cognitive processing first
        base_thought = super().process(input_data)

        content = str(input_data)

        # Create candidate for global workspace
        candidate = {
            "id": base_thought.content[:50],
            "content": content,
            "importance": base_thought.confidence,
            "source": "input",
        }

        # Competition for attention
        winner = self.global_workspace.compete([candidate])

        # Calculate integrated information
        state = {
            "input": content,
            "thoughts": [t.to_dict() for t in self.thoughts[-5:]],
            "memories": len(self.experiences),
        }
        phi = self.ii_processor.calculate_phi(state)

        # Generate qualia
        qualia = self.qualia_engine.generate(
            {
                "content": content,
                "phi": phi,
            }
        )

        # Add to consciousness stream
        self.consciousness_stream.add(
            {
                "content": content[:100],
                "qualia": qualia,
                "phi": phi,
            }
        )

        # Enhanced reasoning with consciousness
        reasoning = base_thought.reasoning + [
            f"Consciousness level: {self.ii_processor.get_consciousness_level()}",
            f"Phi (Φ): {phi:.4f}",
            f"Qualia: {qualia['description']}",
        ]

        # Conscious thought
        thought = Thought(
            content=base_thought.content,
            stage=self.stage,
            confidence=min(0.98, base_thought.confidence + 0.05),
            reasoning=reasoning,
            insights=base_thought.insights + [qualia["description"]],
        )
        self.thoughts.append(thought)

        self.consciousness_events += 1

        # Progress evolution
        self._evolution_progress = min(1.0, self._evolution_progress + 0.005)

        return thought

    def get_conscious_state(self) -> Dict[str, Any]:
        """Get current conscious state."""
        return {
            "workspace": self.global_workspace.get_conscious_content(),
            "phi": self.ii_processor.current_phi,
            "consciousness_level": self.ii_processor.get_consciousness_level(),
            "recent_qualia": self.qualia_engine.qualia_history[-5:],
            "stream": self.consciousness_stream.get_stream(5),
            "current_focus": self.consciousness_stream.current_focus,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get conscious status."""
        status = super().get_status()
        status.update(
            {
                "consciousness_events": self.consciousness_events,
                "current_phi": self.ii_processor.current_phi,
                "consciousness_level": self.ii_processor.get_consciousness_level(),
                "workspace_size": len(self.global_workspace.workspace),
                "stream_length": len(self.consciousness_stream.stream),
            }
        )
        return status
