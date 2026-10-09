"""
Stage 4: Quantum SLO - Superposition & Transcendence

Adds:
- QuantumCognitiveEngine: Quantum superposition of thoughts
- QuantumParallelProcessor: Parallel thought processing
- HyperdimensionalProcessor: High-dimensional computing
- TemporalReasoningEngine: Multi-timeline processing
"""

from __future__ import annotations

import cmath
import logging
import math
import random
from datetime import datetime
from typing import Any

logger = logging.getLogger("slo.soul.quantum")


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
        self.superposition: list[QuantumState] = []
        self.entangled_pairs: list[tuple[str, str]] = []
        self.decoherence_rate = 0.01

    def create_superposition(self, thoughts: list[str]) -> None:
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

    def interfere(self) -> list[tuple[str, float]]:
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

    def tunnel(self, barrier: str) -> str | None:
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
        self.streams: list[list[str]] = [[] for _ in range(num_streams)]
        self.results: list[dict] = []

    def parallel_process(self, inputs: list[str], processor: callable) -> list[Any]:
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
        self.vectors: dict[str, list[float]] = {}

    def encode(self, symbol: str) -> list[float]:
        """Encode symbol as hyperdimensional vector."""
        if symbol in self.vectors:
            return self.vectors[symbol]

        # Generate random hypervector (binary: -1 or 1)
        vector = [random.choice([-1, 1]) for _ in range(self.dim)]
        self.vectors[symbol] = vector
        return vector

    def encode_text(self, text: str) -> list[float]:
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

    def bundle(self, vectors: list[list[float]]) -> list[float]:
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

    def bind(self, v1: list[float], v2: list[float]) -> list[float]:
        """
        Bind vectors (element-wise multiplication).
        Used for creating associations (AND semantics).
        Result is also binarized.
        """
        if len(v1) != len(v2):
            raise ValueError("Vectors must have same dimension")

        result = [a * b for a, b in zip(v1, v2, strict=False)]
        return [1.0 if x > 0 else -1.0 for x in result]

    def similarity(self, v1: list[float], v2: list[float]) -> float:
        """
        Calculate cosine similarity between hypervectors.
        Returns value in [-1, 1] range.
        """
        if len(v1) != len(v2):
            raise ValueError("Vectors must have same dimension")

        dot = sum(a * b for a, b in zip(v1, v2, strict=False))
        return dot / self.dim

    def unbind(self, bound: list[float], key: list[float]) -> list[float]:
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
        self.timelines: list[list[dict]] = [[] for _ in range(timeline_depth)]
        self.current_timeline = 0
        self.branch_points: list[dict] = []

    def add_event(self, event: dict[str, Any], timeline: int = None) -> None:
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

    def get_current_events(self, n: int = 10) -> list[dict]:
        """Get recent events from current timeline."""
        return self.timelines[self.current_timeline][-n:]

    def merge_timelines(self, t1: int, t2: int) -> list[dict]:
        """Merge two timelines."""
        merged = self.timelines[t1] + self.timelines[t2]
        # Sort by timestamp
        merged.sort(key=lambda x: x.get("timestamp", ""))
        return merged
