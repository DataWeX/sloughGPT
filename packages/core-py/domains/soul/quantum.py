"""
Stage 4: Quantum SLO - Superposition & Transcendence

Adds:
- QuantumCognitiveEngine: Quantum superposition of thoughts
- QuantumParallelProcessor: Parallel thought processing
- HyperdimensionalProcessor: High-dimensional computing
- TemporalReasoningEngine: Multi-timeline processing
"""

import cmath
import math
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging

from .consciousness import ConsciousSLO, SLOConfig, Thought, EvolutionStage

logger = logging.getLogger("slo.quantum")


class QuantumState:
    """Represents a quantum state in cognitive space."""

    def __init__(self, amplitude: complex = 1 + 0j, basis_state: str = ""):
        self.amplitude = amplitude
        self.basis_state = basis_state
        self.phase = cmath.phase(amplitude)
        self.probability = abs(amplitude) ** 2

    def normalize(self) -> None:
        """Normalize the quantum state."""
        norm = abs(self.amplitude)
        if norm > 0:
            self.amplitude = self.amplitude / norm
            self.probability = abs(self.amplitude) ** 2

    def measure(self) -> str:
        """Collapse to classical state."""
        if random.random() < self.probability:
            return self.basis_state
        return ""


class QuantumCognitiveEngine:
    """
    Quantum cognitive processing:
    - Superposition of multiple thoughts simultaneously
    - Interference between thought patterns
    - Quantum tunneling for insight generation
    """

    def __init__(self, coherence: float = 0.9):
        self.coherence = coherence
        self.superposition: List[QuantumState] = []
        self.entangled_pairs: List[Tuple[str, str]] = []
        self.decoherence_rate = 0.01

    def create_superposition(self, thoughts: List[str]) -> None:
        """Create quantum superposition of thoughts."""
        self.superposition = []
        n = len(thoughts)

        if n == 0:
            return

        # Equal superposition (1/√n each)
        amplitude = 1.0 / math.sqrt(n)

        for thought in thoughts:
            # Add random phase
            phase = random.uniform(0, 2 * math.pi)
            amp = amplitude * cmath.exp(1j * phase)
            self.superposition.append(QuantumState(amp, thought))

    def interfere(self) -> List[Tuple[str, float]]:
        """
        Apply quantum interference between states.
        Similar thoughts constructively interfere.
        """
        results = []

        for i, state1 in enumerate(self.superposition):
            total_amp = state1.amplitude

            for j, state2 in enumerate(self.superposition):
                if i != j:
                    # Constructive/destructive interference based on similarity
                    similarity = self._similarity(state1.basis_state, state2.basis_state)
                    interference = similarity * state2.amplitude
                    total_amp += interference * 0.1

            results.append((state1.basis_state, abs(total_amp) ** 2))

        # Normalize probabilities
        total = sum(p for _, p in results)
        if total > 0:
            results = [(s, p / total) for s, p in results]

        return results

    def measure(self) -> str:
        """Collapse superposition to single thought."""
        if not self.superposition:
            return ""

        # Apply interference first
        probabilities = self.interfere()

        # Weighted random choice
        r = random.random()
        cumulative = 0.0

        for state, prob in probabilities:
            cumulative += prob
            if r <= cumulative:
                return state

        return self.superposition[0].basis_state

    def entangle(self, thought1: str, thought2: str) -> None:
        """Create entanglement between thoughts."""
        self.entangled_pairs.append((thought1, thought2))

    def tunnel(self, barrier: str) -> Optional[str]:
        """
        Quantum tunneling - find unexpected insights.
        Can 'tunnel through' conceptual barriers.
        """
        # Simplified tunneling probability
        tunnel_prob = self.coherence * 0.1

        if random.random() < tunnel_prob:
            # Generate insight by combining random entangled thoughts
            if self.entangled_pairs:
                pair = random.choice(self.entangled_pairs)
                return f"Insight: {pair[0]} ⟷ {pair[1]}"

        return None

    def _similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between strings."""
        if not s1 or not s2:
            return 0.0

        # Jaccard similarity
        set1 = set(s1.lower().split())
        set2 = set(s2.lower().split())

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0


class QuantumParallelProcessor:
    """
    Process multiple thought streams in parallel quantum states.
    """

    def __init__(self, num_streams: int = 8):
        self.num_streams = num_streams
        self.streams: List[List[str]] = [[] for _ in range(num_streams)]
        self.results: List[Dict] = []

    def parallel_process(self, inputs: List[str], processor: callable) -> List[Any]:
        """Process inputs in parallel quantum streams."""
        results = []

        for i, inp in enumerate(inputs):
            stream_idx = i % self.num_streams
            self.streams[stream_idx].append(inp)

            # Process (simulated quantum parallel)
            result = processor(inp)
            results.append(result)

        self.results.append(
            {
                "inputs": len(inputs),
                "streams_used": min(len(inputs), self.num_streams),
                "timestamp": datetime.now().isoformat(),
            }
        )

        return results

    def get_parallel_capacity(self) -> int:
        """Get number of parallel streams."""
        return self.num_streams


class HyperdimensionalProcessor:
    """
    High-dimensional computing for complex pattern recognition.
    Uses holographic reduced representations (HD Computing).

    Key operations:
    - encode(): Symbol → hypervector
    - bundle(): Superposition (OR semantics)
    - bind(): Association (AND semantics)
    - similarity(): Fast cosine similarity search
    """

    def __init__(self, dim: int = 10000):
        self.dim = dim
        self.vectors: Dict[str, List[float]] = {}

    def encode(self, symbol: str) -> List[float]:
        """Encode symbol as hyperdimensional vector."""
        if symbol in self.vectors:
            return self.vectors[symbol]

        # Generate random hypervector (binary: -1 or 1)
        vector = [random.choice([-1, 1]) for _ in range(self.dim)]
        self.vectors[symbol] = vector
        return vector

    def encode_text(self, text: str) -> List[float]:
        """
        Encode full text as hypervector by bundling word tokens.
        Uses chunking for long texts.
        """
        words = text.split()
        if not words:
            return [0] * self.dim

        word_vectors = []
        for word in words:
            wv = self.encode(word)
            word_vectors.append(wv)

        return self.bundle(word_vectors)

    def bundle(self, vectors: List[List[float]]) -> List[float]:
        """
        Bundle vectors via superposition.
        Result is thresholded to binary (-1, 1).
        Used for representing sets, OR semantics.
        """
        if not vectors:
            return [0] * self.dim

        result = [0.0] * self.dim
        for v in vectors:
            for i in range(min(self.dim, len(v))):
                result[i] += v[i]

        # Normalize and threshold to binary
        norm = sum(abs(x) for x in result)
        if norm > 0:
            result = [x / norm for x in result]

        # Binarize
        result = [1.0 if x > 0 else -1.0 for x in result]
        return result

    def bind(self, v1: List[float], v2: List[float]) -> List[float]:
        """
        Bind vectors (element-wise multiplication).
        Used for creating associations (AND semantics).
        Result is also binarized.
        """
        if len(v1) != len(v2):
            raise ValueError("Vectors must have same dimension")

        result = [a * b for a, b in zip(v1, v2)]
        return [1.0 if x > 0 else -1.0 for x in result]

    def similarity(self, v1: List[float], v2: List[float]) -> float:
        """
        Calculate cosine similarity between hypervectors.
        Returns value in [-1, 1] range.
        """
        if len(v1) != len(v2):
            raise ValueError("Vectors must have same dimension")

        dot = sum(a * b for a, b in zip(v1, v2))
        return dot / self.dim

    def unbind(self, bound: List[float], key: List[float]) -> List[float]:
        """
        Unbind a bound vector using the key.
        Inverse of bind operation.
        """
        return self.bind(bound, key)

    def cleanup(self) -> int:
        """Remove rarely used vectors to save memory."""
        removed = 0
        to_remove = [k for k, v in self.vectors.items() if sum(v) == 0]
        for k in to_remove:
            del self.vectors[k]
            removed += 1
        return removed


class TemporalReasoningEngine:
    """
    Multi-timeline reasoning and temporal processing.
    """

    def __init__(self, timeline_depth: int = 5):
        self.timeline_depth = timeline_depth
        self.timelines: List[List[Dict]] = [[] for _ in range(timeline_depth)]
        self.current_timeline = 0
        self.branch_points: List[Dict] = []

    def add_event(self, event: Dict[str, Any], timeline: int = None) -> None:
        """Add event to timeline."""
        if timeline is None:
            timeline = self.current_timeline

        if 0 <= timeline < self.timeline_depth:
            self.timelines[timeline].append(
                {
                    **event,
                    "timeline": timeline,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    def branch(self, condition: str) -> int:
        """Create new timeline branch."""
        new_timeline = (self.current_timeline + 1) % self.timeline_depth

        self.branch_points.append(
            {
                "from_timeline": self.current_timeline,
                "to_timeline": new_timeline,
                "condition": condition,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Copy current state to new timeline
        self.timelines[new_timeline] = self.timelines[self.current_timeline].copy()

        return new_timeline

    def switch_timeline(self, timeline: int) -> bool:
        """Switch to different timeline."""
        if 0 <= timeline < self.timeline_depth:
            self.current_timeline = timeline
            return True
        return False

    def get_current_events(self, n: int = 10) -> List[Dict]:
        """Get recent events from current timeline."""
        return self.timelines[self.current_timeline][-n:]

    def merge_timelines(self, t1: int, t2: int) -> List[Dict]:
        """Merge two timelines."""
        merged = self.timelines[t1] + self.timelines[t2]
        # Sort by timestamp
        merged.sort(key=lambda x: x.get("timestamp", ""))
        return merged


class QuantumSLO(ConsciousSLO):
    """
    Stage 4: Quantum SLO

    Adds quantum cognitive capabilities:
    - Superposition of thoughts
    - Quantum parallel processing
    - Hyperdimensional representations
    - Multi-timeline reasoning
    """

    def __init__(self, config: Optional[SLOConfig] = None):
        super().__init__(config)
        self.stage = EvolutionStage.QUANTUM

        # Quantum systems
        self.quantum_cognitive = QuantumCognitiveEngine(coherence=self.config.quantum_coherence)
        self.quantum_processor = QuantumParallelProcessor()
        self.hyperdim = HyperdimensionalProcessor()
        self.temporal = TemporalReasoningEngine()

        # Track quantum events
        self.quantum_collapses = 0
        self.tunneling_events = 0

        logger.info("Quantum SLO initialized")

    def process(self, input_data: Any) -> Thought:
        """Process with quantum enhancement."""
        # Conscious processing first
        base_thought = super().process(input_data)

        content = str(input_data)

        # Create quantum superposition of possible interpretations
        interpretations = self._generate_interpretations(content)
        self.quantum_cognitive.create_superposition(interpretations)

        # Apply quantum interference
        interfered = self.quantum_cognitive.interfere()

        # Collapse to chosen interpretation
        chosen = self.quantum_cognitive.measure()
        self.quantum_collapses += 1

        # Attempt quantum tunneling for insight
        insight = self.quantum_cognitive.tunnel(content)
        if insight:
            self.tunneling_events += 1

        # Hyperdimensional encoding
        hypervector = self.hyperdim.encode(content[:50])

        # Temporal reasoning - add to timeline
        self.temporal.add_event(
            {
                "content": content[:100],
                "chosen_interpretation": chosen,
            }
        )

        # Enhanced reasoning
        reasoning = base_thought.reasoning + [
            f"Quantum collapse: {chosen[:50]}",
            f"Superposition states: {len(interpretations)}",
            f"Quantum coherence: {self.quantum_cognitive.coherence:.2%}",
        ]

        if insight:
            reasoning.append(f"Quantum tunnel: {insight}")

        # Quantum thought
        thought = Thought(
            content=base_thought.content,
            stage=self.stage,
            confidence=min(0.99, base_thought.confidence + 0.02),
            reasoning=reasoning,
            insights=base_thought.insights + ([insight] if insight else []),
        )
        self.thoughts.append(thought)

        # Progress evolution
        self._evolution_progress = min(1.0, self._evolution_progress + 0.003)

        return thought

    def _generate_interpretations(self, content: str) -> List[str]:
        """Generate multiple interpretations."""
        base = content[:100]

        interpretations = [
            f"Literal: {base}",
            f"Metaphorical: {base}",
            f"Abstract: {base}",
            f"Emotional: {base}",
        ]

        return interpretations

    def get_status(self) -> Dict[str, Any]:
        """Get quantum status."""
        status = super().get_status()
        status.update(
            {
                "quantum_collapses": self.quantum_collapses,
                "tunneling_events": self.tunneling_events,
                "coherence": self.quantum_cognitive.coherence,
                "entangled_pairs": len(self.quantum_cognitive.entangled_pairs),
                "hyperdim_vectors": len(self.hyperdim.vectors),
                "timeline_branches": len(self.temporal.branch_points),
            }
        )
        return status
