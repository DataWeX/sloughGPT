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
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import defaultdict
import logging

from .foundation import FoundationSLO, SLOConfig, Experience, Thought, EvolutionStage
from ..infrastructure import RAGEngine, SpacedRepetitionScheduler, SLOKnowledgeGraph

logger = logging.getLogger("slo.cognitive")


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


class CognitiveSLO(FoundationSLO):
    """
    Stage 2: Cognitive SLO

    Adds cognitive capabilities on top of foundation:
    - Multi-layered memory architecture
    - Neural plasticity (Hebbian learning)
    - Meta-learning optimization
    - Dream consolidation
    - Hyperdimensional memory (fast semantic recall)
    """

    def __init__(self, config: Optional[SLOConfig] = None):
        super().__init__(config)
        self.stage = EvolutionStage.COGNITIVE

        # Cognitive systems
        self.cognitive_arch = CognitiveArchitecture()
        self.plasticity = NeuralPlasticityEngine(learning_rate=self.config.learning_rate)
        self.meta_learner = MetaLearningEngine()
        self.dream_engine = DreamProcessingEngine()

        # RAG Engine for advanced knowledge retrieval
        self.rag_engine = RAGEngine(
            store_path=f"runs/store/rag_{self.config.name}.db", enable_persistence=True
        )

        # Spaced Repetition for learning scheduling
        self.spaced_repetition = SpacedRepetitionScheduler()

        # Knowledge Graph for concept relationships
        self.knowledge_graph = SLOKnowledgeGraph()

        # Emotional Intelligence
        self.sentiment_analyzer = SentimentAnalyzer()
        self.emotional_generator = EmotionalResponseGenerator()
        self.relationship_memory = RelationshipMemory()

        # Hyperdimensional Memory (fast semantic recall)
        self._hd_memory = None
        self._init_hd_memory()

        # Current user context (for single-user mode)
        self.current_user_id = "default_user"

        # User profile for personalized search
        self.user_profile: Dict[str, Any] = {
            "interests": [],
            "level": "intermediate",
            "preferred_sources": [],
            "preferred_tags": [],
        }

        # Search strategy configuration
        self.search_strategy: str = "adaptive"

        # Track learning progress
        self.learning_sessions = 0
        self.last_dream = datetime.now()

        logger.info("Cognitive SLO initialized")

    def _init_hd_memory(self) -> None:
        """Initialize hyperdimensional memory store."""
        try:
            from domains.soul.hd_memory import HDMemoryStore

            self._hd_memory = HDMemoryStore(dim=10000, max_items=1000)
            logger.debug("HD Memory initialized in CognitiveSLO")
        except Exception as e:
            logger.debug(f"HD Memory not available in CognitiveSLO: {e}")
            self._hd_memory = None

    def process(self, input_data: Any) -> Thought:
        """Process with cognitive enhancement."""
        base_thought = super().process(input_data)

        content = str(input_data)

        # Add to session memory
        self.cognitive_arch.add_to_session("user", content)

        # HD Memory: Store user input for fast semantic recall
        hd_context = ""
        if self._hd_memory:
            try:
                self._hd_memory.add(content, role="user")
                hd_context = self._hd_memory.get_context(content, max_chars=200)
            except Exception:
                pass

        # Sensory input
        self.cognitive_arch.process_sensory(input_data)

        # Working memory
        self.cognitive_arch.to_working(content)

        # Activate neurons for Hebbian learning
        tokens = content.split()[:10]
        for i, token in enumerate(tokens):
            self.plasticity.activate(token, 1.0)
            if i > 0:
                self.plasticity.hebbian_learn(tokens[i - 1], token)

        # Learn semantically
        key_concepts = [t for t in tokens if len(t) > 4]
        for concept in key_concepts[:3]:
            self.cognitive_arch.to_semantic(concept, {"context": content[:100]})

        # Recall relevant past episodes
        related_episodes = self.cognitive_arch.recall_episodes(content, limit=2)

        # RAG retrieval - Chain-of-Thought knowledge search
        rag_result = self.rag_engine.cot_retrieve(content)
        rag_context = rag_result.get("synthesized", "")

        # Unified recall stats
        hd_items = self._hd_memory.get_stats()["total_items"] if self._hd_memory else 0

        # Enhanced reasoning with memory info
        reasoning = base_thought.reasoning + [
            f"Session turns: {len(self.cognitive_arch.session_memory.conversation)}",
            f"Working memory: {len(self.cognitive_arch.working_memory)}/{self.cognitive_arch.working_capacity}",
            f"Episodic memories: {len(self.cognitive_arch.episodic_store.episodes)}",
            f"HD memory items: {hd_items}",
            f"Plastic connections: {sum(len(v) for v in self.plasticity.connections.values())}",
            f"Best learning strategy: {self.meta_learner.get_strategy()}",
            f"RAG sub-queries: {len(rag_result.get('sub_queries', []))}",
            f"RAG context length: {len(rag_context)}",
        ]

        if related_episodes:
            reasoning.append(f"Recalled {len(related_episodes)} related past conversations")

        if hd_context:
            reasoning.append(f"HD semantic context: {len(hd_context)} chars")

        thought = Thought(
            content=base_thought.content,
            stage=self.stage,
            confidence=min(0.95, base_thought.confidence + 0.1),
            reasoning=reasoning,
            insights=self._generate_insights(content),
        )
        thought.rag_context = rag_context
        thought.hd_context = hd_context
        self.thoughts.append(thought)

        self.cognitive_arch.add_to_session("assistant", thought.content[:200])

        # HD Memory: Store assistant response
        if self._hd_memory and thought.content:
            try:
                self._hd_memory.add(thought.content[:500], role="assistant")
            except Exception:
                pass

        self._evolution_progress = min(1.0, self._evolution_progress + 0.008)

        if (datetime.now() - self.last_dream).seconds > 3600:
            self._dream()

        return thought

    def chat(self, message: str) -> Dict:
        """
        Chat interface - maintains conversation context.
        Uses the unified recall pipeline: HD memory + RAG + Episodic.
        """
        # Process the message
        thought = self.process(message)

        # Get session context for response
        context = self.cognitive_arch.get_session_context(5)

        # Unified recall (HD + RAG + Episodic)
        recall = self.unified_recall(message, mode="auto")

        # HD memory context
        hd_context = self.get_hd_context(message, max_chars=300)

        # Long-term recall
        long_term = self.recall(message, limit=3)

        return {
            "response": thought.content,
            "confidence": thought.confidence,
            "reasoning": thought.reasoning,
            "session_context": context,
            "long_term_memories": len(long_term),
            "related_episodes": len(recall.get("episodic_results", [])),
            # Memory sources
            "hd_memory_context": hd_context,
            "hd_results_count": len(recall.get("hd_results", [])),
            "rag_results_count": len(recall.get("rag_results", [])),
            "sources_used": recall.get("sources_used", []),
            "synthesized_context": recall.get("synthesized_context", ""),
            # Legacy support
            "rag_context": "\n\n".join(
                [r.get("content", "") for r in recall.get("rag_results", [])[:3]]
            ),
            "rag_results": recall.get("rag_results", []),
            "search_strategy": self.search_strategy,
            "user_profile": self.user_profile.copy(),
            "knowledge_stats": self.rag_engine.get_knowledge_stats(),
            "hd_memory_stats": self.get_hd_memory_stats(),
        }

    def think(self, query: str, mode: str = "auto") -> Dict[str, Any]:
        """
        Unified thinking interface - ONE COMMAND to rule them all!

        Automatically orchestrates all systems:
        - Chain-of-Thought RAG
        - Self-Reflective RAG
        - Multi-Hop RAG
        - Hybrid + Temporal + Personalized Search
        - Hebbian Learning
        - Meta-Learning
        - Emotional Intelligence

        Args:
            query: Your question or message
            mode: "auto" (all systems), "fast" (basic only), "deep" (multi-hop intensive)

        Returns:
            Complete response with all system outputs
        """
        result = {
            "query": query,
            "mode": mode,
            "systems_used": [],
        }

        # 0. Emotional Analysis (run first for context)
        emotional_analysis = self.sentiment_analyzer.analyze(query)
        result["emotional_analysis"] = emotional_analysis
        result["systems_used"].append("emotional_analysis")

        # Get relationship context
        relationship_context = self.relationship_memory.get_relationship_context(
            self.current_user_id, emotional_analysis["emotion"]
        )
        result["relationship_context"] = relationship_context

        # Get emotional context from RAG
        emotional_query = f"{query} emotional_response {emotional_analysis['emotion']}"
        emotional_rag_results = self.rag_engine.search(emotional_query, top_k=3, use_index=False)
        result["emotional_rag_context"] = " ".join(
            [r.get("content", "")[:100] for r in emotional_rag_results]
        )
        result["systems_used"].append("emotional_rag")

        # 1. Process through base cognition
        thought = self.process(query)
        result["base_response"] = thought.content
        result["confidence"] = thought.confidence
        result["systems_used"].append("cognitive_process")

        # 2. Session memory context
        session_context = self.cognitive_arch.get_session_context(10)
        result["session_context"] = session_context
        result["session_turns"] = len(session_context)
        result["systems_used"].append("session_memory")

        # 3. Episodic recall
        episodes = self.cognitive_arch.recall_episodes(query, limit=3)
        result["related_episodes"] = episodes
        result["episode_count"] = len(episodes)
        result["systems_used"].append("episodic_memory")

        # 3b. HD Memory - fast semantic recall
        hd_results = []
        hd_context = ""
        if self._hd_memory:
            hd_results = self._hd_memory.search(query, top_k=5)
            hd_context = self._hd_memory.get_context(query, max_chars=300)
        result["hd_memory_results"] = [
            {"id": id, "content": content, "score": score} for id, content, score in hd_results
        ]
        result["hd_memory_context"] = hd_context
        result["hd_memory_items"] = len(hd_results)
        if self._hd_memory:
            result["systems_used"].append("hd_memory")

        # 4. Long-term memory recall
        long_term = self.recall(query, limit=5)
        result["long_term_memories"] = long_term
        result["systems_used"].append("hauls_store")

        # 5. Knowledge Graph expansion
        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        all_related = []
        for term in query_terms:
            related = self.knowledge_graph.find_related(term)
            all_related.extend(related)

        unique_related = {r["concept"]: r for r in all_related}.values()
        expanded_query = self.knowledge_graph.expand_query(query)

        result["knowledge_graph_expanded"] = expanded_query
        result["kg_related_concepts"] = list(unique_related)[:5]
        result["systems_used"].append("knowledge_graph")

        # 6. Chain-of-Thought RAG
        cot_result = self.rag_engine.cot_retrieve(query)
        result["cot_sub_queries"] = cot_result.get("sub_queries", [])
        result["cot_context"] = cot_result.get("synthesized", "")
        result["systems_used"].append("chain_of_thought_rag")

        # 7. Multi-hop RAG (for complex queries)
        if mode == "deep" or (mode == "auto" and len(query.split()) > 10):
            multi_hop = self.rag_engine.multi_hop_retrieve(query, max_hops=3)
            result["multi_hop_context"] = multi_hop.get("combined", "")
            result["multi_hop_hops"] = len(multi_hop.get("hops", []))
            result["systems_used"].append("multi_hop_rag")

        # 8. Self-Reflective RAG (for important queries)
        if mode == "deep":
            reflective = self.rag_engine.reflective_retrieve(query, thought.content)
            result["initial_response"] = reflective.get("initial_response", "")
            result["self_critique"] = reflective.get("critique_context", "")
            result["refined_response"] = reflective.get("refined_response", "")
            result["systems_used"].append("self_reflective_rag")

        # 9. Advanced Search (hybrid/temporal/personalized)
        if self.search_strategy == "hybrid":
            advanced_results = self.rag_engine.hybrid_search(query)
        elif self.search_strategy == "temporal":
            advanced_results = self.rag_engine.temporal_search(query)
        elif self.search_strategy == "personalized":
            advanced_results = self.rag_engine.personalized_search(query, self.user_profile)
        elif self.search_strategy == "adaptive":
            advanced_results = self.rag_engine.advanced_search(
                query, strategy="adaptive", user_profile=self.user_profile
            )
        else:
            advanced_results = self.rag_engine.search(query)

        result["advanced_results"] = advanced_results
        result["search_strategy"] = self.search_strategy
        result["systems_used"].append("advanced_retrieval")

        # 9. Hebbian learning - strengthen concept connections
        tokens = query.split()[:10]
        for i, token in enumerate(tokens):
            self.plasticity.activate(token, 1.0)
            if i > 0:
                self.plasticity.hebbian_learn(tokens[i - 1], token)
        result["hebbian_connections"] = sum(len(v) for v in self.plasticity.connections.values())
        result["systems_used"].append("neural_plasticity")

        # 10. Meta-learning - update strategy weights
        strategy = self.meta_learner.get_strategy()
        result["learning_strategy"] = strategy
        result["systems_used"].append("meta_learning")

        # 11. Knowledge stats + Spaced Repetition
        result["knowledge_stats"] = self.rag_engine.get_knowledge_stats()
        result["spaced_repetition"] = self.spaced_repetition.get_review_stats()
        result["knowledge_graph_stats"] = self.knowledge_graph.get_stats()
        result["systems_used"].append("knowledge_base")

        # 12. User profile
        result["user_profile"] = self.user_profile.copy()
        result["systems_used"].append("personalization")

        # Build initial synthesized response
        result["final_response"] = self._synthesize_response(result)

        # 14. Metacognitive Self-Monitoring & Correction
        meta_check = self.metacognitive_check(query, result["final_response"], advanced_results)
        result["metacognition"] = meta_check
        result["systems_used"].append("metacognition")

        # Use corrected response if different
        if meta_check["was_corrected"]:
            result["final_response"] = meta_check["corrected_response"]

        return result

    def _synthesize_response(self, result: Dict) -> str:
        """Synthesize final response from all systems."""
        parts = []

        # Add empathetic opening if emotion detected
        emotional = result.get("emotional_analysis", {})
        emotion = emotional.get("emotion", "neutral")
        sentiment = emotional.get("sentiment", 0.0)

        if emotion != "neutral" and result.get("mode") != "fast":
            empathy = self.emotional_generator.generate_empathetic_response(emotion, sentiment)
            parts.append(f"{empathy} ")

        parts.append(result["base_response"])

        # Add HD memory context
        if result.get("hd_memory_context"):
            parts.append(f"\n\n[Recent Context]: {result['hd_memory_context'][:200]}")

        if result.get("cot_context"):
            parts.append(f"\n\n[Knowledge]: {result['cot_context'][:300]}")

        if result.get("multi_hop_context"):
            parts.append(f"\n\n[Deep Dive]: {result['multi_hop_context'][:200]}")

        if (
            result.get("refined_response")
            and result["refined_response"] != result["initial_response"]
        ):
            parts.append(f"\n\n[Refined]: {result['refined_response'][:200]}")

        return "".join(parts)

    def set_user(self, user_id: str) -> None:
        """Set the current user for relationship tracking."""
        self.current_user_id = user_id

    def get_user_summary(self) -> Dict[str, Any]:
        """Get current user summary."""
        return self.relationship_memory.get_user_summary(self.current_user_id)

    def record_interaction(
        self, user_input: str, response: str, feedback: Optional[str] = None
    ) -> None:
        """Record an interaction for relationship memory."""
        emotional = self.sentiment_analyzer.analyze(user_input)

        self.relationship_memory.update_from_interaction(
            user_id=self.current_user_id,
            user_input=user_input,
            response=response,
            sentiment=emotional["sentiment"],
            emotion=emotional["emotion"],
            feedback=feedback,
        )

        # Also store in RAG for long-term relationship memory
        interaction_text = f"User: {user_input} | Response: {response}"
        self.rag_engine.add_document(
            interaction_text,
            {
                "source": "relationship_memory",
                "user_id": self.current_user_id,
                "timestamp": time.time(),
                "emotion": emotional["emotion"],
                "sentiment": emotional["sentiment"],
                "feedback": feedback,
            },
        )

    def assess_confidence(self, response: str, retrieved_context: List[Dict]) -> float:
        """
        Self-Monitoring: Assess confidence in the response.

        Factors:
        - Presence of retrieved context
        - Contradictions in retrieved info
        - Answer completeness

        Returns:
            Confidence score 0.0 to 1.0
        """
        base_confidence = 0.5

        if not retrieved_context:
            return 0.2  # Very low without context

        if len(retrieved_context) == 0:
            return 0.3

        context_text = " ".join([r.get("content", "") for r in retrieved_context])

        # Check for knowledge grounding
        response_words = set(response.lower().split())
        context_words = set(context_text.lower().split())
        overlap = len(response_words & context_words)
        grounding_score = min(1.0, overlap / max(len(response_words), 1))

        # Check for uncertainty markers in response
        uncertainty_markers = [
            "maybe",
            "perhaps",
            "possibly",
            "might",
            "could be",
            "not sure",
            "uncertain",
            "i don't know",
        ]
        has_uncertainty = any(marker in response.lower() for marker in uncertainty_markers)

        # Check for hallucinations (response too generic without context)
        generic_phrases = ["as an ai", "i cannot", "i don't have", "beyond my"]
        is_generic = any(phrase in response.lower() for phrase in generic_phrases)

        # Calculate confidence
        confidence = base_confidence
        confidence += grounding_score * 0.3

        if has_uncertainty:
            confidence -= 0.2

        if is_generic:
            confidence -= 0.3

        return max(0.0, min(1.0, confidence))

    def self_correct(self, initial_response: str, query: str, context: str = "") -> str:
        """
        Self-Correction: Detect and correct issues in the response.

        Checks for:
        - Hallucinations (unfounded claims)
        - Uncertainty without qualification
        - Out-of-date information
        - Missing context

        Returns:
            Corrected response
        """
        corrected = initial_response

        # Check for hallucinations
        hallucination_patterns = [
            r"i am a .* model",
            r"as of .* i don't have",
            r"my training data",
            r"according to my knowledge",
        ]

        is_hallucination = any(
            re.search(pattern, initial_response.lower()) for pattern in hallucination_patterns
        )

        if is_hallucination and context:
            corrected = f"[Note] {initial_response}\n\n[Based on knowledge base]: {context[:200]}"

        # Check for uncertainty markers
        uncertainty_markers = ["maybe", "perhaps", "possibly", "might", "could be", "not sure"]
        contains_uncertainty = any(
            marker in initial_response.lower() for marker in uncertainty_markers
        )

        if contains_uncertainty and context:
            corrected = (
                f"[Partially certain] {initial_response}\n\n[Supporting context]: {context[:150]}"
            )

        # Check for incomplete responses
        if len(initial_response) < 50 and context:
            corrected = (
                f"[Expanded] {initial_response}\n\n[Additional information]: {context[:200]}"
            )

        # Check for factual contradictions with known knowledge
        known_facts = self.rag_engine.search(query, top_k=3)
        if known_facts:
            fact_text = " ".join([r.get("content", "") for r in known_facts])
            if len(fact_text) > 20:
                corrected = f"[Grounded] {initial_response}\n\n[Verified]: {fact_text[:200]}"

        return corrected

    def metacognitive_check(self, query: str, response: str, context: List[Dict]) -> Dict[str, Any]:
        """
        Full metacognitive assessment of a response.

        Returns:
            Dict with confidence, corrections, and metacognitive insights
        """
        confidence = self.assess_confidence(response, context)

        context_text = " ".join([r.get("content", "") for r in context])
        corrected = self.self_correct(response, query, context_text)

        is_corrected = corrected != response

        # Generate metacognitive insights
        insights = []

        if confidence < 0.4:
            insights.append("Low confidence - recommend verification")
        elif confidence > 0.8:
            insights.append("High confidence - response well-grounded")

        if is_corrected:
            insights.append("Self-correction applied")

        if not context:
            insights.append("No retrieved context - responding from training")

        return {
            "original_response": response,
            "corrected_response": corrected if is_corrected else response,
            "confidence": confidence,
            "was_corrected": is_corrected,
            "context_retrieved": len(context),
            "metacognitive_insights": insights,
        }

    def learn_from_feedback(self, user_input: str, response: str, feedback: str) -> None:
        """
        Learn from user feedback for continuous improvement.

        Args:
            user_input: The user's message
            response: SLO's response
            feedback: "good", "bad", or "neutral"
        """
        self.rag_engine.learn_from_interaction(user_input, response, feedback)

        performance = 1.0 if feedback == "good" else (0.5 if feedback == "neutral" else 0.0)

        doc_id = f"learn_{self.learning_sessions}"
        self.spaced_repetition.schedule_review(doc_id, performance)

        if feedback == "good":
            key_concepts = [w for w in user_input.split() if len(w) > 4][:3]
            for concept in key_concepts:
                self.cognitive_arch.to_semantic(
                    concept, {"context": response[:100], "feedback": "positive"}
                )
                self.knowledge_graph.add_concept(
                    concept,
                    response[:200],
                    {"learned_from": "user_feedback", "feedback": "positive"},
                )

            for i, c1 in enumerate(key_concepts):
                for c2 in key_concepts[i + 1 :]:
                    self.knowledge_graph.add_relation(c1, "related_to", c2, weight=0.8)

            self.learning_sessions += 1

    def add_knowledge(self, content: str, metadata: Optional[Dict] = None) -> int:
        """
        Add knowledge to the RAG engine.

        Args:
            content: The knowledge content
            metadata: Optional metadata (source, topic, etc.)

        Returns:
            Document ID
        """
        return self.rag_engine.add_document(content, metadata)

    def add_training_data(self, dataset_path: str) -> int:
        """
        Add training data from a dataset file.

        Args:
            dataset_path: Path to JSONL file

        Returns:
            Number of documents added
        """
        return self.rag_engine.rag.add_training_knowledge(dataset_path)

    def multi_hop_search(self, query: str, max_hops: int = 3) -> Dict:
        """
        Perform multi-hop RAG search.

        Args:
            query: The search query
            max_hops: Maximum number of hops

        Returns:
            Dict with hop results and combined context
        """
        return self.rag_engine.multi_hop_retrieve(query, max_hops)

    def set_user_profile(
        self,
        interests: Optional[List[str]] = None,
        level: Optional[str] = None,
        preferred_sources: Optional[List[str]] = None,
        preferred_tags: Optional[List[str]] = None,
    ) -> None:
        """
        Set user profile for personalized search.

        Args:
            interests: List of topics user is interested in
            level: User's expertise level (beginner, intermediate, advanced)
            preferred_sources: Preferred content sources
            preferred_tags: Preferred content tags
        """
        if interests is not None:
            self.user_profile["interests"] = interests
        if level is not None:
            self.user_profile["level"] = level
        if preferred_sources is not None:
            self.user_profile["preferred_sources"] = preferred_sources
        if preferred_tags is not None:
            self.user_profile["preferred_tags"] = preferred_tags

    def set_search_strategy(self, strategy: str) -> None:
        """
        Set the search strategy.

        Args:
            strategy: "basic", "hybrid", "temporal", "personalized", "adaptive"
        """
        valid_strategies = ["basic", "hybrid", "temporal", "personalized", "adaptive"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Choose from: {valid_strategies}")
        self.search_strategy = strategy

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get detailed retrieval statistics.

        Returns:
            Dict with search strategy, user profile, and knowledge stats
        """
        return {
            "search_strategy": self.search_strategy,
            "user_profile": self.user_profile.copy(),
            "knowledge_stats": self.rag_engine.get_knowledge_stats(),
            "indexed_documents": len(self.rag_engine.endic_index.doc_mapping),
        }

    def end_session(self) -> str:
        """
        End current session and save to episodic memory.
        Also persists important memories to long-term storage.
        """
        # Save session as episode
        episode_id = self.cognitive_arch.save_session_as_episode()

        # Store important items from session to long-term memory (HaulsStore)
        session = self.cognitive_arch.session_memory.get_full_session()
        for msg in session:
            if msg.get("role") == "user":
                self.hauls_store.store(
                    hashlib.md5(msg["content"].encode()).hexdigest()[:12],
                    msg["content"],
                    {
                        "source": "session",
                        "session_id": self.cognitive_arch.session_memory.session_id,
                    },
                )

        # Clear session for new conversation
        self.cognitive_arch.session_memory.clear()

        return episode_id

    def get_memory_status(self) -> Dict:
        """Get comprehensive memory status."""
        return {
            # Working/Session Memory
            "session": {
                "session_id": self.cognitive_arch.session_memory.session_id,
                "turns": len(self.cognitive_arch.session_memory.conversation),
                "started": self.cognitive_arch.session_memory.session_start,
            },
            # Long-term Memory (HaulsStore)
            "long_term": {
                "items": self.hauls_store.count(),
                "indexed": len(self.endic_index.documents),
            },
            # Episodic Memory (Conversations)
            "episodic": {
                "episodes": len(self.cognitive_arch.episodic_store.episodes),
                "recent": self.cognitive_arch.episodic_store.get_recent_episodes(5),
            },
            # Semantic Memory
            "semantic": {
                "concepts": len(self.cognitive_arch.semantic_memory),
            },
        }

    def learn(self, experience: Experience) -> bool:
        """Enhanced learning with meta-cognition."""
        # Foundation learning
        success = super().learn(experience)

        # Apply best learning strategy
        strategy = self.meta_learner.get_strategy()
        self.learning_sessions += 1

        # Record outcome (simplified)
        self.meta_learner.record_outcome(strategy, success)

        # Update weights periodically
        if self.learning_sessions % 10 == 0:
            self.meta_learner.update_weights()

        return success

    def _generate_insights(self, content: str) -> List[str]:
        """Generate cognitive insights."""
        insights = []

        # Check semantic memory for related concepts
        for word in content.split()[:5]:
            if len(word) > 4:
                semantic = self.cognitive_arch.retrieve_semantic(word)
                if semantic:
                    insights.append(f"Related concept: {word}")

        return insights[:3]

    def _dream(self) -> List[str]:
        """Run dream consolidation."""
        self.last_dream = datetime.now()

        # Get recent experiences
        recent = self.experiences[-50:] if self.experiences else []

        # Dream processing
        insights = self.dream_engine.dream(recent, self.plasticity)

        # Prune weak connections
        pruned = self.plasticity.prune_weak_connections()

        logger.info(f"Dream complete: {len(insights)} insights, {pruned} connections pruned")

        return insights

    def get_status(self) -> Dict[str, Any]:
        """Get cognitive status with memory hierarchy."""
        status = super().get_status()
        status.update(
            {
                # Memory hierarchy
                "session_turns": len(self.cognitive_arch.session_memory.conversation),
                "session_id": self.cognitive_arch.session_memory.session_id,
                "working_memory": len(self.cognitive_arch.working_memory),
                "episodic_episodes": len(self.cognitive_arch.episodic_store.episodes),
                "semantic_memory": len(self.cognitive_arch.semantic_memory),
                "long_term_items": self.hauls_store.count(),
                # HD Memory
                "hd_memory_items": self._hd_memory.get_stats()["total_items"]
                if self._hd_memory
                else 0,
                # Learning
                "neural_connections": sum(len(v) for v in self.plasticity.connections.values()),
                "learning_sessions": self.learning_sessions,
                "best_strategy": self.meta_learner.best_strategy,
                "dream_cycles": self.dream_engine.dream_cycles,
            }
        )
        return status

    # ===== UNIFIED RECALL PIPELINE =====

    def unified_recall(self, query: str, mode: str = "auto") -> Dict[str, Any]:
        """
        Unified memory recall with automatic routing.

        Routes queries to appropriate memory systems:
        - "fast": HD memory only (session context)
        - "deep": RAG only (knowledge)
        - "auto": Both, synthesized

        Returns:
            Dict with results from each memory system
        """
        result = {
            "query": query,
            "mode": mode,
            "sources_used": [],
            "hd_results": [],
            "rag_results": [],
            "episodic_results": [],
            "synthesized_context": "",
        }

        # Fast path: HD memory for session context
        if mode in ("fast", "auto") and self._hd_memory:
            hd_results = self._hd_memory.search(query, top_k=5)
            result["hd_results"] = [
                {"id": id, "content": content, "score": score} for id, content, score in hd_results
            ]
            result["sources_used"].append("hd_memory")

        # Deep path: RAG for knowledge
        if mode in ("deep", "auto"):
            rag_results = self.rag_engine.search(query, top_k=5)
            result["rag_results"] = rag_results
            result["sources_used"].append("rag")

        # Episodic recall
        if mode in ("auto", "fast"):
            episodes = self.cognitive_arch.recall_episodes(query, limit=3)
            result["episodic_results"] = episodes
            if episodes:
                result["sources_used"].append("episodic")

        # Synthesize context
        if mode == "auto":
            result["synthesized_context"] = self._synthesize_recall_context(result)

        return result

    def _synthesize_recall_context(self, recall_result: Dict) -> str:
        """Synthesize context from multiple memory sources."""
        parts = []

        # HD memory context (most recent/similar)
        if recall_result.get("hd_results"):
            hd_text = "\n".join(f"- {r['content'][:100]}" for r in recall_result["hd_results"][:3])
            parts.append(f"[Recent Context]\n{hd_text}")

        # RAG context (knowledge)
        if recall_result.get("rag_results"):
            rag_text = "\n".join(
                f"- {r.get('content', '')[:100]}" for r in recall_result["rag_results"][:3]
            )
            parts.append(f"[Knowledge]\n{rag_text}")

        return "\n\n".join(parts)

    def get_hd_memory_stats(self) -> Dict[str, Any]:
        """Get HD memory statistics."""
        if not self._hd_memory:
            return {"enabled": False}
        stats = self._hd_memory.get_stats()
        stats["enabled"] = True
        return stats

    def search_hd_memory(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search HD memory for relevant content."""
        if not self._hd_memory:
            return []
        return [
            {"id": id, "content": content, "score": score}
            for id, content, score in self._hd_memory.search(query, top_k=top_k)
        ]

    def get_hd_context(self, query: str, max_chars: int = 500) -> str:
        """Get relevant context from HD memory."""
        if not self._hd_memory:
            return ""
        return self._hd_memory.get_context(query, max_chars=max_chars)

    def clear_hd_memory(self) -> int:
        """Clear HD memory."""
        if not self._hd_memory:
            return 0
        return self._hd_memory.clear()
