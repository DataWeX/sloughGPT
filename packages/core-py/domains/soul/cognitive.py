"""
Stage 2: Cognitive SLO - Memory & Learning

Memory Hierarchy:
1. Working Memory (Session) - Current conversation context
2. Long-term Memory (HaulsStore) - Persistent across all sessions
3. Episodic Memory (Conversations) - Individual chat sessions stored for reference

Adds:
- CognitiveArchitecture: Multi-layered memory
- NeuralPlasticityEngine: Hebbian learning
- MetaLearningEngine: Learn how to learn
- DreamProcessingEngine: Sleep consolidation
"""

import random
import hashlib
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import defaultdict
import logging

try:
    from .foundation import FoundationSLO, SLOConfig, Experience, Thought, EvolutionStage
except ImportError:
    FoundationSLO = SLOConfig = Experience = Thought = EvolutionStage = None

try:
    from ..infrastructure import RAGEngine, SpacedRepetitionScheduler, SLOKnowledgeGraph
except ImportError:
    RAGEngine = SpacedRepetitionScheduler = SLOKnowledgeGraph = None

logger = logging.getLogger("slo.soul.cognitive")


class SentimentAnalyzer:
    """
    Sentiment and Emotion Detection.

    Analyzes user input to detect emotional state and sentiment.
    """

    def __init__(self):
        self.emotion_keywords = {
            "happy": [
                "happy",
                "joy",
                "excited",
                "great",
                "wonderful",
                "love",
                "awesome",
                "amazing",
                "fantastic",
            ],
            "sad": [
                "sad",
                "unhappy",
                "depressed",
                "down",
                "upset",
                "disappointed",
                "feel bad",
                "terrible",
            ],
            "angry": ["angry", "mad", "frustrated", "annoyed", "irritated", "furious", "hate"],
            "fear": ["afraid", "scared", "worried", "anxious", "nervous", "fear", "panic"],
            "surprise": ["surprised", "shocked", "amazing", "unexpected", "wow", "unbelievable"],
            "neutral": ["okay", "fine", "alright", "normal", "regular"],
        }

        self.sentiment_words = {
            "positive": [
                "good",
                "great",
                "excellent",
                "wonderful",
                "amazing",
                "love",
                "best",
                "fantastic",
                "happy",
                "joy",
            ],
            "negative": [
                "bad",
                "terrible",
                "awful",
                "horrible",
                "worst",
                "hate",
                "sad",
                "angry",
                "disappointed",
                "frustrated",
            ],
        }

    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment: -1 (negative) to 1 (positive)
        """
        text_lower = text.lower()
        words = text_lower.split()

        positive_count = sum(1 for w in words if w in self.sentiment_words["positive"])
        negative_count = sum(1 for w in words if w in self.sentiment_words["negative"])

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        return (positive_count - negative_count) / total

    def detect_emotion(self, text: str) -> str:
        """
        Detect primary emotion in text.
        """
        text_lower = text.lower()

        emotion_scores = {}
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            emotion_scores[emotion] = score

        if not emotion_scores or max(emotion_scores.values()) == 0:
            return "neutral"

        return max(emotion_scores.keys(), key=lambda e: emotion_scores[e])

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Complete emotional analysis.
        """
        sentiment = self.analyze_sentiment(text)
        emotion = self.detect_emotion(text)

        intensity = abs(sentiment)

        return {
            "sentiment": sentiment,
            "emotion": emotion,
            "intensity": intensity,
            "is_positive": sentiment > 0.1,
            "is_negative": sentiment < -0.1,
            "is_neutral": -0.1 <= sentiment <= 0.1,
        }


class EmotionalResponseGenerator:
    """
    Generates emotionally appropriate responses.
    """

    def __init__(self):
        self.empathy_responses = {
            "happy": [
                "I'm so glad to hear that!",
                "That's wonderful!",
                "I'm happy for you!",
                "Great to hear!",
            ],
            "sad": [
                "I'm sorry you're feeling this way.",
                "That sounds difficult. I'm here to help.",
                "I understand this is tough.",
                "Take care of yourself.",
            ],
            "angry": [
                "I understand your frustration.",
                "That's definitely upsetting.",
                "I hear you.",
                "Let's work through this together.",
            ],
            "fear": [
                "It's okay to feel worried.",
                "I'm here to help you through this.",
                "Take it one step at a time.",
                "You're not alone in this.",
            ],
            "surprise": [
                "That's quite surprising!",
                "I can see why that would shock you.",
                "What an unexpected turn!",
            ],
            "neutral": [
                "I understand.",
                "Got it.",
                "I see.",
                "Alright.",
            ],
        }

        self.qualifiers = {
            "high": ["definitely", "certainly", "absolutely"],
            "medium": ["probably", "likely", "possibly"],
            "low": ["might", "may", "could"],
        }

    def generate_empathetic_response(self, emotion: str, sentiment: float) -> str:
        """
        Generate an empathetic response based on emotion.
        """
        if emotion not in self.empathy_responses:
            emotion = "neutral"

        responses = self.empathy_responses[emotion]
        return random.choice(responses)

    def adapt_response(self, response: str, emotion: str, sentiment: float) -> str:
        """
        Adapt response based on emotional context.
        """
        if sentiment > 0.5:
            return f"{response}! 😊"
        elif sentiment < -0.5:
            return f"{response} 😔"

        return response

    def format_emotional_response(
        self, base_response: str, emotion: str, sentiment: float, include_empathy: bool = True
    ) -> str:
        """
        Format a complete emotional response.
        """
        parts = []

        if include_empathy and emotion != "neutral":
            empathy = self.generate_empathetic_response(emotion, sentiment)
            parts.append(empathy)

        parts.append(base_response)

        result = " ".join(parts)
        return self.adapt_response(result, emotion, sentiment)


class RelationshipMemory:
    """
    Tracks user relationships over time.
    """

    def __init__(self):
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.interaction_history: Dict[str, List[Dict]] = defaultdict(list)

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "user_id": user_id,
                "created_at": time.time(),
                "last_interaction": None,
                "total_interactions": 0,
                "emotional_tendencies": defaultdict(int),
                "topics_of_interest": defaultdict(int),
                "satisfaction_score": 0.5,
                "mood_history": [],
            }

        return self.user_profiles[user_id]

    def update_from_interaction(
        self,
        user_id: str,
        user_input: str,
        response: str,
        sentiment: float,
        emotion: str,
        feedback: Optional[str] = None,
    ) -> None:
        """
        Update user profile from interaction.
        """
        profile = self.get_user_profile(user_id)

        profile["last_interaction"] = time.time()
        profile["total_interactions"] += 1

        profile["emotional_tendencies"][emotion] += 1

        words = user_input.lower().split()
        topics = [w for w in words if len(w) > 5]
        for topic in topics[:3]:
            profile["topics_of_interest"][topic] += 1

        profile["mood_history"].append(
            {
                "timestamp": time.time(),
                "emotion": emotion,
                "sentiment": sentiment,
            }
        )

        if len(profile["mood_history"]) > 50:
            profile["mood_history"] = profile["mood_history"][-50:]

        if feedback == "good":
            profile["satisfaction_score"] = min(1.0, profile["satisfaction_score"] + 0.1)
        elif feedback == "bad":
            profile["satisfaction_score"] = max(0.0, profile["satisfaction_score"] - 0.1)

        self.interaction_history[user_id].append(
            {
                "timestamp": time.time(),
                "user_input": user_input,
                "response": response,
                "emotion": emotion,
                "sentiment": sentiment,
                "feedback": feedback,
            }
        )

        if len(self.interaction_history[user_id]) > 100:
            self.interaction_history[user_id] = self.interaction_history[user_id][-100:]

    def get_user_summary(self, user_id: str) -> Dict[str, Any]:
        """Get user summary."""
        profile = self.get_user_profile(user_id)

        emotional_tendencies = profile["emotional_tendencies"]
        dominant_emotion = (
            max(emotional_tendencies, key=emotional_tendencies.get)
            if emotional_tendencies
            else "neutral"
        )

        topics = profile["topics_of_interest"]
        top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "user_id": user_id,
            "total_interactions": profile["total_interactions"],
            "dominant_emotion": dominant_emotion,
            "satisfaction_score": profile["satisfaction_score"],
            "top_topics": [t[0] for t in top_topics],
            "last_interaction": profile["last_interaction"],
        }

    def get_relationship_context(self, user_id: str, current_emotion: str) -> str:
        """
        Get context for relationship-aware responses.
        """
        profile = self.get_user_profile(user_id)

        context_parts = []

        if profile["total_interactions"] > 5:
            context_parts.append(
                f"You've been feeling {profile['emotional_tendencies'].most_common(1)[0][0]} lately."
            )

        if current_emotion != "neutral":
            context_parts.append(f"Currently feeling {current_emotion}.")

        if profile["satisfaction_score"] < 0.4:
            context_parts.append("User seems dissatisfied - be extra helpful.")
        elif profile["satisfaction_score"] > 0.7:
            context_parts.append("User is happy - maintain positive tone.")

        return " ".join(context_parts)


class SessionMemory:
    """
    Working Memory (Session) - Current conversation context.
    Stores the active conversation with role-based messages.
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.conversation: List[Dict] = []
        self.session_id = self._generate_session_id()
        self.session_start = datetime.now().isoformat()

    def _generate_session_id(self) -> str:
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"

    def add(self, role: str, content: str) -> Dict:
        """Add a message to the session."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "turn": len(self.conversation),
        }
        self.conversation.append(message)

        # Maintain max turns
        if len(self.conversation) > self.max_turns:
            self.conversation = self.conversation[-self.max_turns :]

        return message

    def get_context(self, n: int = 5) -> List[Dict]:
        """Get recent context."""
        return self.conversation[-n:]

    def get_full_session(self) -> List[Dict]:
        """Get entire session."""
        return self.conversation.copy()

    def clear(self) -> None:
        """Clear session for new conversation."""
        self.conversation = []
        self.session_id = self._generate_session_id()
        self.session_start = datetime.now().isoformat()

    def get_summary(self) -> Dict:
        """Get session summary."""
        return {
            "session_id": self.session_id,
            "start": self.session_start,
            "turns": len(self.conversation),
            "roles": defaultdict(int, {m["role"]: 1 for m in self.conversation}),
        }


class EpisodicMemoryStore:
    """
    Episodic Memory (Conversations) - Individual chat sessions stored for future reference.
    Persists complete conversations as episodes.
    """

    def __init__(self, max_episodes: int = 100):
        self.max_episodes = max_episodes
        self.episodes: Dict[str, List[Dict]] = {}
        self.episode_metadata: Dict[str, Dict] = {}

    def save_episode(self, session_id: str, conversation: List[Dict]) -> str:
        """Save a complete conversation episode."""
        episode_id = f"conv_{hashlib.md5(session_id.encode()).hexdigest()[:12]}"

        self.episodes[episode_id] = conversation.copy()
        self.episode_metadata[episode_id] = {
            "session_id": session_id,
            "turns": len(conversation),
            "saved": datetime.now().isoformat(),
            "importance": self._calculate_importance(conversation),
        }

        # Maintain max episodes
        if len(self.episodes) > self.max_episodes:
            self._evict_least_important()

        return episode_id

    def _calculate_importance(self, conversation: List[Dict]) -> float:
        """Calculate importance score for conversation."""
        if not conversation:
            return 0.0

        score = 0.5

        # Longer conversations may be more important
        if len(conversation) > 10:
            score += 0.2

        # Check for important keywords
        content = " ".join(m.get("content", "") for m in conversation).lower()
        important_words = ["important", "remember", "critical", "key", "learn"]
        score += sum(0.1 for w in important_words if w in content)

        return min(1.0, score)

    def _evict_least_important(self) -> None:
        """Remove least important episode."""
        if not self.episode_metadata:
            return

        least_important = min(self.episode_metadata.items(), key=lambda x: x[1]["importance"])
        episode_id = least_important[0]

        del self.episodes[episode_id]
        del self.episode_metadata[episode_id]

    def get_episode(self, episode_id: str) -> Optional[List[Dict]]:
        """Retrieve a specific episode."""
        return self.episodes.get(episode_id)

    def search_episodes(self, query: str, limit: int = 5) -> List[Dict]:
        """Search episodes for relevant conversations."""
        results = []
        query_lower = query.lower()

        for episode_id, conversation in self.episodes.items():
            # Simple text matching
            content = " ".join(m.get("content", "") for m in conversation)
            if query_lower in content.lower():
                results.append(
                    {
                        "episode_id": episode_id,
                        "relevance": 0.5,  # Simplified relevance
                        "turns": len(conversation),
                        "metadata": self.episode_metadata.get(episode_id, {}),
                    }
                )

        return results[:limit]

    def get_recent_episodes(self, n: int = 10) -> List[str]:
        """Get most recent episode IDs."""
        sorted_episodes = sorted(
            self.episode_metadata.items(), key=lambda x: x[1]["saved"], reverse=True
        )
        return [ep_id for ep_id, _ in sorted_episodes[:n]]


class CognitiveArchitecture:
    """
    Multi-layered memory system:
    - Sensory: Immediate input buffer
    - Working (Session): Current conversation context
    - Episodic: Stored conversation sessions
    - Semantic: Facts and concepts
    - Long-term: Persistent via HaulsStore
    """

    def __init__(self, working_capacity: int = 7):
        # Memory layers
        self.sensory_buffer: List[Any] = []
        self.working_memory: List[Any] = []
        self.working_capacity = working_capacity  # Miller's law (7±2)

        # Session memory (current conversation)
        self.session_memory = SessionMemory()

        # Episodic memory (stored conversations)
        self.episodic_store = EpisodicMemoryStore()

        # Semantic memory (facts/concepts)
        self.semantic_memory: Dict[str, Any] = {}

    def process_sensory(self, input_data: Any) -> bool:
        """Process sensory input."""
        self.sensory_buffer.append(
            {
                "data": input_data,
                "timestamp": datetime.now().isoformat(),
            }
        )
        # Keep buffer small
        if len(self.sensory_buffer) > 100:
            self.sensory_buffer = self.sensory_buffer[-50:]
        return True

    def to_working(self, item: Any) -> bool:
        """Move item to working memory."""
        if len(self.working_memory) >= self.working_capacity:
            # FIFO eviction
            evicted = self.working_memory.pop(0)
            self._consolidate_to_episodic(evicted)

        self.working_memory.append(item)
        return True

    def _consolidate_to_episodic(self, item: Any) -> bool:
        """Consolidate working memory to episodic."""
        episode = {
            "content": item,
            "timestamp": datetime.now().isoformat(),
            "importance": random.random(),  # Simplified
        }
        return True

    def add_to_session(self, role: str, content: str) -> Dict:
        """Add message to current session memory."""
        return self.session_memory.add(role, content)

    def get_session_context(self, n: int = 5) -> List[Dict]:
        """Get recent session context."""
        return self.session_memory.get_context(n)

    def save_session_as_episode(self) -> str:
        """Save current session as episodic memory."""
        episode_id = self.episodic_store.save_episode(
            self.session_memory.session_id, self.session_memory.get_full_session()
        )
        return episode_id

    def recall_episodes(self, query: str, limit: int = 5) -> List[Dict]:
        """Recall relevant past episodes."""
        return self.episodic_store.search_episodes(query, limit)

    def to_semantic(self, key: str, value: Any) -> bool:
        """Store in semantic memory."""
        if key in self.semantic_memory:
            # Strengthen existing
            self.semantic_memory[key]["strength"] += 0.1
        else:
            self.semantic_memory[key] = {
                "value": value,
                "strength": 1.0,
                "created": datetime.now().isoformat(),
            }
        return True

    def retrieve_semantic(self, key: str) -> Optional[Any]:
        """Retrieve from semantic memory."""
        if key in self.semantic_memory:
            self.semantic_memory[key]["last_accessed"] = datetime.now().isoformat()
            return self.semantic_memory[key]["value"]
        return None


class NeuralPlasticityEngine:
    """
    Hebbian learning: "Neurons that fire together, wire together"

    Implements synaptic plasticity for learning patterns.
    """

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.connections: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.activation_history: Dict[str, List[float]] = defaultdict(list)

    def activate(self, neuron_id: str, strength: float = 1.0) -> None:
        """Record neuron activation."""
        self.activation_history[neuron_id].append(strength)
        # Keep history limited
        if len(self.activation_history[neuron_id]) > 100:
            self.activation_history[neuron_id] = self.activation_history[neuron_id][-50:]

    def hebbian_learn(self, pre: str, post: str, reward: float = 1.0) -> float:
        """
        Hebbian learning rule: Δw = η * pre * post
        Strengthens connection between co-activated neurons.
        """
        pre_strength = self.activation_history[pre][-1] if self.activation_history[pre] else 1.0
        post_strength = self.activation_history[post][-1] if self.activation_history[post] else 1.0

        delta = self.learning_rate * pre_strength * post_strength * reward
        self.connections[pre][post] += delta

        return self.connections[pre][post]

    def get_connection_strength(self, pre: str, post: str) -> float:
        """Get connection strength between neurons."""
        return self.connections[pre][post]

    def prune_weak_connections(self, threshold: float = 0.01) -> int:
        """Remove weak connections (synaptic pruning)."""
        pruned = 0
        for pre in list(self.connections.keys()):
            for post in list(self.connections[pre].keys()):
                if abs(self.connections[pre][post]) < threshold:
                    del self.connections[pre][post]
                    pruned += 1
        return pruned


class MetaLearningEngine:
    """
    Learn how to learn better.
    Optimizes learning strategies based on performance.
    """

    def __init__(self):
        self.strategies: Dict[str, Dict] = {
            "rote": {"success": 0, "attempts": 0, "weight": 1.0},
            "spaced": {"success": 0, "attempts": 0, "weight": 1.0},
            "interleaved": {"success": 0, "attempts": 0, "weight": 1.0},
            "elaborative": {"success": 0, "attempts": 0, "weight": 1.0},
        }
        self.best_strategy = "spaced"

    def record_outcome(self, strategy: str, success: bool) -> None:
        """Record learning outcome for strategy."""
        if strategy in self.strategies:
            self.strategies[strategy]["attempts"] += 1
            if success:
                self.strategies[strategy]["success"] += 1

    def update_weights(self) -> None:
        """Update strategy weights based on performance."""
        for name, data in self.strategies.items():
            if data["attempts"] > 0:
                success_rate = data["success"] / data["attempts"]
                data["weight"] = 0.7 * data["weight"] + 0.3 * success_rate

        # Find best strategy
        self.best_strategy = max(self.strategies.keys(), key=lambda k: self.strategies[k]["weight"])

    def get_strategy(self) -> str:
        """Get recommended learning strategy."""
        return self.best_strategy


class DreamProcessingEngine:
    """
    Sleep consolidation: Process and integrate memories.
    Runs during idle periods to strengthen important memories.
    """

    def __init__(self):
        self.dream_cycles = 0
        self.consolidated = 0

    def dream(self, memories: List[Experience], plasticity: NeuralPlasticityEngine) -> List[str]:
        """
        Process memories during 'sleep'.
        Returns insights generated during dreaming.
        """
        self.dream_cycles += 1
        insights = []

        # Replay important memories
        important = sorted(memories, key=lambda m: m.importance, reverse=True)[:10]

        for i, memory in enumerate(important):
            # Connect related memories (Hebbian)
            for j, other in enumerate(important):
                if i != j:
                    plasticity.hebbian_learn(memory.id, other.id)

        # Generate insights from patterns
        if len(important) >= 3:
            insight = f"Pattern detected across {len(important)} memories"
            insights.append(insight)

        self.consolidated += len(important)

        return insights
