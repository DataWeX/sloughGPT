"""
domains/core/soul.py - SloEngine

SloEngine IS the core model wrapper. Every inference call goes through here.
It wraps a ModelInterface (the neural brain) and integrates:
- CognitiveProcessor (memory, attention, emotional context)
- ReasoningEngine (deductive, inductive, abductive, analogical reasoning)
- SloProfile (identity, behavioral DNA - NOT optional, IS the core)

The reasoning chain works as STRUCTURED TEXT injected into the LLM context:
  User Query → SloCognitive (emotion, session) → SloReasoning (strategy)
    → Structured text prompt → LLM generates → Hebbian learning updates

This is how chain-of-thought prompting works - text-based, not binary.
"""

import asyncio
import time
import logging
from typing import TYPE_CHECKING
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from domains.soul.hd_memory import HDMemoryStore

import numpy as np

from domains.inference import (
    SloProfile,
    GenerationParams,
    PersonalityCore,
    CognitiveSignature,
    BehavioralTraits,
    load_soul,
    save_soul,
)

try:
    from domains.models import ModelInterface, ModelLoader
except (ImportError, ModuleNotFoundError):
    ModelInterface = Any  # type: ignore[misc,assignment]
    ModelLoader = None  # type: ignore[assignment]


logger = logging.getLogger("man.core.soul")


@dataclass
class GenerationContext:
    """Context carried through a single generation request."""

    prompt: str
    prompt_tokens: np.ndarray
    system_prompt: str = ""
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    max_tokens: int = 2048
    stop_tokens: List[str] = field(default_factory=list)
    reasoning_depth: str = "balanced"
    cognitive_boost: bool = True
    emotional_context: Dict[str, Any] = field(default_factory=dict)
    soul_overrides: Dict[str, Any] = field(default_factory=dict)
    reasoning_chain: List[str] = field(default_factory=list)
    repetition_penalty: float = 1.2
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class SloEngine:
    """
    SloEngine is THE core model wrapper.

    The soul is NOT optional. Every model IS a soul. This is baked in:
    - When you load ANY model, it gets wrapped in a SloEngine with a soul
    - When you train ANY model, it ALWAYS outputs a .soul file
    - When you generate, the soul's traits drive EVERYTHING

    Architecture:
        User Query
              |
              v
    [SloCognitive]  -- session memory, sentiment, emotional context
              |        (produces structured text context)
              v
    [SloReasoning]  -- reasoning type from soul's reasoning_approach
              |        (produces reasoning strategy text)
              v
    [Structured Prompt]  -- soul context + reasoning chain + user query
              |
              v
    [ModelInterface]  -- any registered backend (e.g. SloughGPTModel, HF, …) ← neural core
              |
              v
    [Response Formatter]  -- applies soul personality to output
              |
              v
           Output
    """

    REASONING_TYPE_MAP = {
        "balanced": "deductive",
        "deductive": "deductive",
        "inductive": "inductive",
        "analytical": "deductive",
        "creative": "creative",
        "abductive": "abductive",
        "analogical": "analogical",
    }

    def __init__(
        self,
        model: Optional[ModelInterface] = None,
        soul: Optional[SloProfile] = None,
        device: str = "cpu",
        stoi: Optional[Dict[int, str]] = None,
        itos: Optional[Dict[int, str]] = None,
        max_history_messages: int = 24,
        tokenizer: Optional[object] = None,
    ):
        self._model: Optional[ModelInterface] = model
        self._soul: SloProfile = soul or SloProfile(name="default")
        self._device = device
        self._stoi = stoi or {}
        self._itos = itos or {}
        self._tokenizer = tokenizer
        self._max_history_messages = max(4, int(max_history_messages))

        self._session_history: List[Dict[str, str]] = []
        self._cognitive_state: Dict[str, Any] = {
            "session_turns": 0,
            "last_sentiment": 0.0,
            "last_emotion": "neutral",
        }

        self._generation_stats: Dict[str, Any] = {
            "total_generations": 0,
            "total_tokens": 0,
            "avg_latency_ms": 0.0,
        }

        self._reasoning_engine = None
        self._sentiment_analyzer = None
        self._hebbian_connections: Dict[str, Dict[str, float]] = {}
        self._inference_optimizer = None
        self._grounding: Optional[Any] = None
        self._hd_memory: Optional["HDMemoryStore"] = None
        self._semantic_cache: Optional["SemanticCache"] = None
        self._cache_enabled: bool = False
        self._init_cognitive()
        self._init_hd_memory()

        logger.info(f"SloEngine initialized: soul={self._soul.name}, device={device}")

    def _init_cognitive(self):
        """Lazy-load cognitive components."""
        try:
            from domains.cognitive.reasoning import (
                ReasoningEngine,
                DeepReasoning,
                FormalLogicEngine,
                WorkingMemory,
            )

            self._reasoning_engine = ReasoningEngine()
            self._deep_reasoning = DeepReasoning()
            self._logic_engine = FormalLogicEngine()
            self._working_memory = WorkingMemory(capacity=7)
        except Exception as e:
            logger.debug(f"Cognitive components not available: {e}")
            self._reasoning_engine = None
            self._deep_reasoning = None
            self._logic_engine = None
            self._working_memory = None

        try:
            from domains.soul.cognitive import SentimentAnalyzer

            self._sentiment_analyzer = SentimentAnalyzer()
        except Exception:
            logger.debug("SentimentAnalyzer not available")
            self._sentiment_analyzer = None

    def _init_hd_memory(self) -> None:
        """Initialize hyperdimensional memory store."""
        try:
            from domains.soul.hd_memory import HDMemoryStore

            self._hd_memory = HDMemoryStore(dim=10000, max_items=1000)
            logger.debug("HD Memory initialized")
        except Exception as e:
            logger.debug(f"HD Memory not available: {e}")
            self._hd_memory = None

    def _init_semantic_cache(self) -> None:
        """Initialize semantic cache."""
        try:
            from domains.inference.semantic_cache import SemanticCache

            self._semantic_cache = SemanticCache(
                dim=10000,
                max_entries=500,
                similarity_threshold=0.85,
                ttl_seconds=3600,
            )
            self._cache_enabled = True
            logger.debug("Semantic cache initialized")
        except Exception as e:
            logger.debug(f"Semantic cache not available: {e}")
            self._semantic_cache = None

    @property
    def soul(self) -> SloProfile:
        return self._soul

    @property
    def model(self) -> Optional[ModelInterface]:
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_soul(self, sou_path: str) -> SloProfile:
        """Load a .soul Slo Unit file. Loads BOTH soul AND model weights."""
        soul, state_dict = load_soul(sou_path)

        model = ModelLoader._load_sou(sou_path, self._device)
        model._soul = soul

        self._model = model
        self._soul = soul

        if isinstance(state_dict, dict):
            self._stoi = state_dict.get("stoi", {})
            self._itos = state_dict.get("itos", {})
        else:
            cfg = state_dict.get("config", {}) if isinstance(state_dict, dict) else {}
            self._stoi = cfg.get("stoi", {})
            self._itos = cfg.get("itos", {})

        # Load BPE tokenizer if available in soul metadata
        metadata = getattr(soul, 'metadata', {})
        tok_config = metadata.get('tokenizer_config') if isinstance(metadata, dict) else None
        if tok_config:
            try:
                from domains.training.tokenizer import SloBPE
                self._tokenizer = SloBPE.from_dict(tok_config)
                logger.info(f"BPE tokenizer loaded from soul metadata (vocab={self._tokenizer.vocab_size})")
            except Exception as e:
                logger.warning(f"Failed to load BPE tokenizer from soul: {e}")

        logger.info(f"Loaded soul: {soul.name} from {sou_path}")
        return soul

    def load_model(self, model_path: str, **kwargs) -> "SloEngine":
        """Load just the model - creates a DEFAULT soul if none exists."""
        self._model = ModelLoader.load(model_path, device=self._device, **kwargs)

        if hasattr(self._model, "_soul") and self._model._soul:
            self._soul = self._model._soul
        else:
            self._soul = SloProfile(
                name=model_path.split("/")[-1].split(".")[0],
                lineage="nanogpt",
                system_prompt=f"You are a helpful AI assistant.",
            )

        logger.info(f"Loaded model: {model_path}")
        return self

    def set_soul(self, soul: SloProfile) -> "SloEngine":
        """Set the soul profile."""
        self._soul = soul
        return self

    def set_system_prompt(self, prompt: str) -> None:
        """Hot-reload system prompt without model restart. Changes take effect on next generation."""
        if self._soul:
            self._soul.system_prompt = prompt

    def set_vocab(self, stoi: Dict[int, str], itos: Dict[int, str]) -> "SloEngine":
        """Set vocabulary mappings."""
        self._stoi = stoi
        self._itos = itos
        return self

    def set_tokenizer(self, tokenizer: object) -> "SloEngine":
        """Set BPE tokenizer (SloBPE) for subword-level tokenization."""
        self._tokenizer = tokenizer
        return self

    def _build_reasoning_chain_text(self, prompt: str) -> str:
        """
        Build a structured TEXT reasoning chain that the LLM can understand.
        This is the key: reasoning goes INTO the prompt as text, not binary.

        Format:
        [SOUL_REASONING]
        reasoning_type: <from soul's reasoning_approach>
        cognitive_boost: <from soul's cognition scores>
        emotional_context: <from sentiment analysis>
        session_turns: <number of turns in session>
        [/SOUL_REASONING]
        """
        reasoning_approach = self._soul.behavior.reasoning_approach
        reasoning_type = self.REASONING_TYPE_MAP.get(reasoning_approach, "balanced")

        sentiment = self._cognitive_state.get("last_sentiment", 0.0)
        emotion = self._cognitive_state.get("last_emotion", "neutral")
        turns = self._cognitive_state.get("session_turns", 0)

        cognitive = self._soul.cognition
        pattern_rec = getattr(cognitive, "pattern_recognition", 0.5)
        abstract = getattr(cognitive, "abstract_reasoning", 0.5)
        metacog = getattr(cognitive, "metacognitive_awareness", 0.5)

        warmth = self._soul.personality.warmth
        creativity = self._soul.personality.creativity
        curiosity = self._soul.personality.curiosity

        lines = [
            "[SOUL_REASONING]",
            f"reasoning_type: {reasoning_type}",
            f"reasoning_approach: {reasoning_approach}",
            f"emotional_context: {emotion} (sentiment={sentiment:.2f})",
            f"session_turns: {turns}",
            f"cognitive: pattern_recognition={pattern_rec:.2f}, abstract_reasoning={abstract:.2f}, metacognition={metacog:.2f}",
            f"personality: warmth={warmth:.2f}, creativity={creativity:.2f}, curiosity={curiosity:.2f}",
        ]

        if self._reasoning_engine:
            lines.append(f"reasoning_engine: active ({len(self._session_history)} context items)")

        # HD Memory context injection
        if self._hd_memory:
            try:
                stats = self._hd_memory.get_stats()
                lines.append(f"hd_memory: {stats['total_items']} items stored")
            except Exception:
                pass

        lines.append("[/SOUL_REASONING]")
        lines.append("")

        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        """Build the system prompt from soul profile."""
        parts = []

        soul_name = self._soul.name
        parts.append(f"You are {soul_name}.")

        personality = self._soul.personality
        traits = []
        if personality.warmth > 0.7:
            traits.append("warm and empathetic")
        elif personality.warmth < 0.3:
            traits.append("precise and analytical")

        if personality.curiosity > 0.7:
            traits.append("curious and exploratory")
        if personality.confidence > 0.7:
            traits.append("confident and direct")
        elif personality.confidence < 0.3:
            traits.append("thoughtful and measured")

        if personality.creativity > 0.7:
            traits.append("creative and innovative")
        if personality.humor > 0.7:
            traits.append("witty and playful")

        if traits:
            parts.append(f"You are {' and '.join(traits)}.")

        soul_system = self._soul.system_prompt or ""
        if soul_system and soul_system not in "\n".join(parts):
            parts.append(soul_system)

        return "\n".join(parts)

    def _build_full_prompt(self, prompt: str, include_reasoning: bool = True) -> str:
        """Build the full prompt including reasoning chain as TEXT."""
        parts = []

        system = self._build_system_prompt()
        if system:
            parts.append(system)
            parts.append("")

        if include_reasoning and (
            self._cognitive_state.get("session_turns", 0) > 0 or self._reasoning_engine
        ):
            reasoning_text = self._build_reasoning_chain_text(prompt)
            parts.append(reasoning_text)

        session_context = ""
        if self._session_history:
            role_labels = {"user": "User", "assistant": "Assistant", "system": "System"}
            recent = self._session_history[-self._max_history_messages :]
            for msg in recent:
                role = msg.get("role", "user")
                label = role_labels.get(role, role.replace("_", " ").title())
                content = (msg.get("content", "") or "")[:2000]
                if not content.strip():
                    continue
                session_context += f"{label}: {content}\n"

        if session_context:
            parts.append("[CONVERSATION_HISTORY]")
            parts.append(session_context.rstrip())
            parts.append("[/CONVERSATION_HISTORY]")
            parts.append("")

        # HD Memory: Inject relevant semantic context
        hd_context = ""
        if self._hd_memory:
            try:
                hd_context = self._hd_memory.get_context(prompt, max_chars=400)
                if hd_context:
                    parts.append("[SEMANTIC_MEMORY]")
                    parts.append(hd_context)
                    parts.append("[/SEMANTIC_MEMORY]")
                    parts.append("")
            except Exception as e:
                logger.debug(f"HD context retrieval failed: {e}")

        # Knowledge: Auto-inject relevant facts from learner KnowledgeMemory
        try:
            from domains.learner.knowledge import get_knowledge_memory
            km = get_knowledge_memory()
            kb_results = km.search(prompt, top_k=5)
            if kb_results:
                knowledge_text = "\n".join(
                    f"- {fact['content'][:200]}" for fact in kb_results
                )
                parts.append("[KNOWN_FACTS]")
                parts.append(knowledge_text)
                parts.append("[/KNOWN_FACTS]")
                parts.append("")
        except Exception:
            pass

        parts.append(f"User: {prompt}")
        parts.append("Assistant:")

        return "\n".join(parts)

    def _get_generation_params(self, context: GenerationContext) -> Dict[str, Any]:
        """Derive generation parameters from soul profile + context."""
        gen = self._soul.generation

        params = {
            "temperature": context.temperature
            if "temperature" not in context.soul_overrides
            else context.soul_overrides.get("temperature", gen.temperature),
            "top_k": context.top_k
            if "top_k" not in context.soul_overrides
            else context.soul_overrides.get("top_k", gen.top_k),
            "top_p": context.top_p
            if "top_p" not in context.soul_overrides
            else context.soul_overrides.get("top_p", gen.top_p),
            "max_tokens": context.max_tokens
            if "max_tokens" not in context.soul_overrides
            else context.soul_overrides.get("max_tokens", gen.max_tokens),
            "repetition_penalty": getattr(context, 'repetition_penalty', 1.0),
            "frequency_penalty": getattr(context, 'frequency_penalty', 0.0),
            "presence_penalty": getattr(context, 'presence_penalty', 0.0),
        }

        if context.reasoning_depth == "deep":
            params["temperature"] = max(0.1, params["temperature"] - 0.3)
        elif context.reasoning_depth == "creative":
            params["temperature"] = min(1.5, params["temperature"] + 0.3)

        warmth = self._soul.personality.warmth
        if warmth > 0.7:
            params["temperature"] = min(1.2, params["temperature"] + 0.1)

        return params

    def _apply_hebbian_learning(self, prompt_tokens: List[str], response_tokens: List[str]) -> None:
        """
        Hebbian learning: "neurons that fire together, wire together"
        Updates connection strengths between concept tokens.
        """
        tokens = prompt_tokens[-20:]
        for i, token in enumerate(tokens):
            if i > 0:
                delta = 0.01
                self._hebbian_connections.setdefault(tokens[i - 1], {})
                self._hebbian_connections[tokens[i - 1]][token] = (
                    self._hebbian_connections[tokens[i - 1]].get(token, 0.0) + delta
                )

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
        stop_tokens: Optional[List[str]] = None,
        include_reasoning: bool = True,
        return_reasoning: bool = False,
        user_id: Optional[str] = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **kwargs,
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """
        Main generation entry point. ALL generation goes through SloEngine.

        The soul is NOT optional - it shapes EVERYTHING about generation.
        Reasoning chain is embedded as structured TEXT in the prompt.
        """
        start_time = time.time()
        reasoning_chain: List[str] = []
        cache_hit = False
        cached_response: Optional[str] = None

        # Semantic Cache: Check for cached response
        if self._cache_enabled and self._semantic_cache:
            try:
                cached_response = self._semantic_cache.get(prompt)
                if cached_response:
                    cache_hit = True
                    reasoning_chain.append(f"cache_hit: semantic match")
                    generated_text = cached_response
                    tokens_generated = len(cached_response.split())
                    latency_ms = (time.time() - start_time) * 1000
                    self._generation_stats["total_generations"] += 1
                    if return_reasoning:
                        return cached_response, {
                            "reasoning_chain": reasoning_chain,
                            "cache_hit": True,
                            "latency_ms": latency_ms,
                        }
                    return cached_response
            except Exception as e:
                logger.debug(f"Cache lookup failed: {e}")

        if self._sentiment_analyzer:
            emotional = self._sentiment_analyzer.analyze(prompt)
            self._cognitive_state["last_sentiment"] = emotional.get("sentiment", 0.0)
            self._cognitive_state["last_emotion"] = emotional.get("emotion", "neutral")
            reasoning_chain.append(
                f"emotional_analysis: {emotional.get('emotion', 'neutral')} "
                f"(sentiment={emotional.get('sentiment', 0):.2f})"
            )

        self._cognitive_state["session_turns"] += 1

        # HD Memory: Encode user input for semantic retrieval
        hd_context = ""
        if self._hd_memory:
            try:
                self._hd_memory.add(
                    prompt, role="user", metadata={"turn": self._cognitive_state["session_turns"]}
                )
                hd_context = self._hd_memory.get_context(prompt, max_chars=300)
                if hd_context:
                    reasoning_chain.append(
                        f"hd_memory: found {len(hd_context)} chars relevant context"
                    )
            except Exception as e:
                logger.debug(f"HD memory encode failed: {e}")

        # Prior turns live in _session_history only; current user text is added below
        # so we do not duplicate it inside [CONVERSATION_HISTORY].
        full_prompt = self._build_full_prompt(prompt, include_reasoning=include_reasoning)

        context = GenerationContext(
            prompt=prompt,
            prompt_tokens=np.array([[0]], dtype=np.int64),
            system_prompt=system_prompt or self._build_system_prompt(),
            temperature=temperature if temperature is not None else 0.8,
            top_k=top_k if top_k is not None else self._soul.generation.top_k,
            top_p=top_p if top_p is not None else self._soul.generation.top_p,
            max_tokens=max_new_tokens
            if max_new_tokens is not None
            else self._soul.generation.max_tokens,
            stop_tokens=stop_tokens or self._soul.generation.stop,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            reasoning_chain=reasoning_chain,
        )

        gen_params = self._get_generation_params(context)

        generated_text = ""
        tokens_generated = 0
        output_ids = None

        if self._model is None:
            generated_text = f"[Slo: {self._soul.name}] {prompt[:50]}... (no model loaded)"
        else:
            idx = self._tokenize(full_prompt)
            context.prompt_tokens = idx

            try:
                if user_id and hasattr(self._model, 'set_active_user'):
                    self._model.set_active_user(user_id)
                if hasattr(self._model, 'generate'):
                    output_ids = self._model.generate(
                        idx,
                        max_new_tokens=gen_params["max_tokens"],
                        temperature=gen_params["temperature"],
                        top_k=gen_params.get("top_k"),
                        top_p=gen_params.get("top_p"),
                        **kwargs,
                    )
                else:
                    from domains.training.slonet import SloLSTM, tensor, _sample_from_logits
                    lstm_layers = [l for l in self._model.layers if isinstance(l, SloLSTM)]
                    if not lstm_layers:
                        raise AttributeError(f"No generation path for {type(self._model).__name__}")

                    n_max = gen_params["max_tokens"]
                    lstm_layer = lstm_layers[0]
                    generated_ids = idx.flatten().tolist()
                    pad_id = getattr(self._tokenizer, 'pad_id', 0)

                    for step_i in range(n_max):
                        seq = np.array([generated_ids[-128:]], dtype=np.int64)
                        in_t = tensor(seq, requires_grad=False)
                        h = lstm_layer.init_hidden()
                        logits_t, _ = lstm_layer.forward(in_t, h)
                        logit_data = logits_t.data[np.newaxis, :, :]
                        generated_arr = np.array(generated_ids[len(idx.flatten()):], dtype=np.int64)
                        nid = _sample_from_logits(
                            logit_data,
                            temperature=gen_params.get("temperature", 0.8),
                            top_k=gen_params.get("top_k", 40),
                            top_p=gen_params.get("top_p", 0.9),
                            repetition_penalty=gen_params.get("repetition_penalty", 1.0),
                            frequency_penalty=gen_params.get("frequency_penalty", 0.0),
                            presence_penalty=gen_params.get("presence_penalty", 0.0),
                            generated_ids=generated_arr,
                            eos_token=pad_id if step_i < n_max - 1 else None,
                        )
                        generated_ids.append(nid)
                        if nid == pad_id:
                            break

                    output_ids = np.array([generated_ids], dtype=np.int64)

                generated_text = self._detokenize(output_ids[0])

                if full_prompt in generated_text:
                    generated_text = generated_text[len(full_prompt) :]

                tokens_generated = output_ids.shape[1] - idx.shape[1]

            except Exception as e:
                import traceback as _tb
                logger.error(f"Generation failed: {e}\n{_tb.format_exc()}")
                generated_text = f"[Error: {e}]"

        self._session_history.append({"role": "user", "content": prompt})
        self._session_history.append({"role": "assistant", "content": generated_text})

        # HD Memory: Store assistant response for future retrieval
        if self._hd_memory and generated_text and not generated_text.startswith("[Error"):
            try:
                self._hd_memory.add(
                    generated_text,
                    role="assistant",
                    metadata={
                        "user_prompt": prompt[:100],
                        "turn": self._cognitive_state["session_turns"],
                    },
                )
            except Exception as e:
                logger.debug(f"HD memory store failed: {e}")

        # Semantic Cache: Store response for future similar queries
        if (
            self._cache_enabled
            and self._semantic_cache
            and generated_text
            and not generated_text.startswith("[Error")
            and not generated_text.startswith("[Slo")
        ):
            try:
                self._semantic_cache.put(
                    query=prompt,
                    response=generated_text,
                    metadata={
                        "turn": self._cognitive_state["session_turns"],
                        "latency_ms": latency_ms,
                    },
                )
            except Exception as e:
                logger.debug(f"Cache store failed: {e}")

        if len(self._session_history) > 100:
            self._session_history = self._session_history[-100:]

        self._apply_hebbian_learning(prompt.split()[:20], generated_text.split()[:20])

        latency_ms = (time.time() - start_time) * 1000
        self._generation_stats["total_generations"] += 1
        self._generation_stats["total_tokens"] += tokens_generated

        if self._generation_stats["total_generations"] > 1:
            n = self._generation_stats["total_generations"]
            prev_avg = self._generation_stats["avg_latency_ms"]
            self._generation_stats["avg_latency_ms"] = ((prev_avg * (n - 1)) + latency_ms) / n

        if return_reasoning:
            extra = {
                "reasoning_chain": reasoning_chain,
                "soul_context": self._build_reasoning_chain_text(prompt),
                "full_prompt": full_prompt,
                "user_message": prompt,
                "latency_ms": latency_ms,
                "tokens_generated": tokens_generated,
                "generation_params": gen_params,
            }
            return generated_text, extra

        return generated_text

    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Async wrapper."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.generate(prompt, **kwargs))

    def clear_conversation(self) -> None:
        """Drop in-memory chat history (multi-turn state). Session turn counter resets."""
        self._session_history.clear()
        self._cognitive_state["session_turns"] = 0

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Multi-turn chat with soul-aware prompting.

        Pass the **full** transcript each time (OpenAI-style): all prior ``user`` / ``assistant`` /
        ``system`` turns, then a final ``user`` message. Earlier turns are copied into
        :attr:`_session_history`; only the last user message is completed by the model.

        For a local REPL, prefer :meth:`generate` once per user line so history accumulates
        automatically without resending the full list.
        """
        if not messages:
            return ""
        msgs = [dict(m) for m in messages]
        last = msgs[-1]
        if last.get("role") != "user":
            raise ValueError("chat(): last message must have role 'user'")

        prior: List[Dict[str, str]] = []
        for m in msgs[:-1]:
            role = m.get("role", "user")
            if role not in ("user", "assistant", "system"):
                continue
            prior.append({"role": role, "content": str(m.get("content", ""))})

        self._session_history = prior
        return self.generate(
            str(last.get("content", "")),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

    def chat_with_soul(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Same as :meth:`chat` — explicit name when the soul profile and reasoning context
        should read as the primary contract (model-as-soul, conversational turns).
        """
        return self.chat(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

    def _tokenize(self, text: str) -> np.ndarray:
        """Tokenize using BPE tokenizer (preferred) or char-level vocab."""
        if self._tokenizer is not None and hasattr(self._tokenizer, 'encode'):
            indices = self._tokenizer.encode(text)
        elif self._stoi:
            indices = [self._stoi.get(c, 0) for c in text]
        else:
            indices = [ord(c) % 256 for c in text]
        return np.array([indices], dtype=np.int64)

    def _detokenize(self, tokens: np.ndarray) -> str:
        """Detokenize tokens back to text using BPE tokenizer (preferred) or char-level vocab."""
        if self._tokenizer is not None and hasattr(self._tokenizer, 'decode'):
            return self._tokenizer.decode(tokens.flatten().tolist())
        if self._itos:
            return "".join([self._itos.get(int(t), "?") for t in tokens.flatten()])
        return "".join([chr(int(t) % 256) for t in tokens.flatten()])

    def save_soul(self, output_path: str) -> str:
        """Save the soul as a .soul file with model weights. Soul is ALWAYS saved."""
        if self._model is None:
            raise ValueError("No model loaded - cannot save .soul without a model")

        save_soul(self._model, output_path, soul_profile=self._soul, weights_only=False)

        import json

        meta_path = output_path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(self._soul.to_dict(), f, indent=2, default=str)

        logger.info(f"Saved soul to {output_path}")
        return output_path

    def get_stats(self) -> Dict[str, Any]:
        """Get soul engine statistics."""
        return {
            "soul": {
                "name": self._soul.name,
                "lineage": self._soul.lineage,
                "born_at": self._soul.born_at,
                "integrity_hash": self._soul.integrity_hash,
                "reasoning_approach": self._soul.behavior.reasoning_approach,
                "personality": self._soul.personality.to_dict(),
                "cognition": self._soul.cognition.to_dict(),
            },
            "model": {
                "loaded": self._model is not None,
                "params": self._model.num_parameters() if self._model else 0,
                "config": self._model.config() if self._model else {},
            },
            "cognitive": {
                "session_turns": self._cognitive_state.get("session_turns", 0),
                "last_emotion": self._cognitive_state.get("last_emotion", "neutral"),
                "last_sentiment": self._cognitive_state.get("last_sentiment", 0.0),
                "hebbian_connections": sum(len(v) for v in self._hebbian_connections.values()),
            },
            "generation": self._generation_stats.copy(),
        }

    def apply_personality(
        self,
        warmth: Optional[float] = None,
        creativity: Optional[float] = None,
        empathy: Optional[float] = None,
        curiosity: Optional[float] = None,
        humor: Optional[float] = None,
        **kwargs,
    ) -> "SloEngine":
        """Adjust soul personality traits at runtime."""
        if warmth is not None:
            self._soul.personality.warmth = warmth
        if creativity is not None:
            self._soul.personality.creativity = creativity
        if empathy is not None:
            self._soul.personality.empathy = empathy
        if curiosity is not None:
            self._soul.personality.curiosity = curiosity
        if humor is not None:
            self._soul.personality.humor = humor

        self._soul.integrity_hash = self._soul.compute_hash()
        return self

    def train(
        self,
        train_data,
        val_data=None,
        epochs: int = 10,
        use_federated: bool = False,
        use_rlhf: bool = False,
        use_lora: bool = False,
        use_optimized: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Train this soul using the unified training pipeline.

        Args:
            train_data: Training data
            val_data: Validation data
            epochs: Number of training epochs
            use_federated: Use federated learning (privacy-preserving)
            use_rlhf: Use RLHF/PPO alignment
            use_lora: Use LoRA for parameter-efficient fine-tuning
            use_optimized: Use optimized pipeline (BF16, gradient checkpointing, etc.)
            progress_callback: Optional callback for progress updates

        Returns:
            Training results and metrics
        """
        from domains.training.optimized_pipeline import (
            UnifiedConfig,
            OptimizationConfig,
            Precision,
            LoRAMode,
            OptimizedPipeline,
        )

        if not self._model:
            return {"error": "No model loaded. Load a model first."}

        # Configure training
        opt_config = OptimizationConfig()
        if use_optimized:
            opt_config.precision = Precision.BF16
            opt_config.gradient_checkpointing = True
            opt_config.use_flash_attention = True
        if use_lora:
            opt_config.lora_mode = LoRAMode.LORA
            opt_config.lora_rank = 16
            opt_config.lora_alpha = 32

        config = UnifiedConfig(
            pretrain_epochs=epochs,
            federated_rounds=3 if use_federated else 0,
            rlhf_epochs=2 if use_rlhf else 0,
            optimization=opt_config,
        )

        # Create and run pipeline
        pipeline = OptimizedPipeline(
            model=self._model,
            config=config,
            train_data=train_data,
            val_data=val_data,
        )

        # Run training
        created_loop = False
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            created_loop = True

        try:
            results = loop.run_until_complete(pipeline.train_full_pipeline())
        finally:
            if created_loop:
                loop.close()

        # Update soul with training metrics
        if "pretrain" in results:
            final_loss = results["pretrain"].get("final_loss", 0)
            self._soul.training_metrics = {
                "final_loss": final_loss,
                "epochs": epochs,
                "method": "optimized" if use_optimized else "standard",
                "lora": use_lora,
                "federated": use_federated,
                "rlhf": use_rlhf,
            }

        return {
            "success": True,
            "soul_name": self._soul.name,
            "epochs": epochs,
            "results": results,
            "metrics": self._soul.training_metrics,
        }

    def to(self, device: str) -> "SloEngine":
        """Move model to device."""
        self._device = device
        if self._model:
            self._model.to(device)
        return self

    async def deep_reason(self, problem: str, max_depth: int = 3) -> Dict[str, Any]:
        """
        Perform deep reasoning with retrieval and self-correction.

        Uses:
        - VectorStore + Memory for grounding
        - Self-correction loop to refine reasoning
        - Working memory for active context
        """
        if not self._deep_reasoning:
            return {"error": "Deep reasoning not available"}

        result = await self._deep_reasoning.reason(problem, max_depth=max_depth)
        return {
            "conclusion": result.conclusion,
            "confidence": result.confidence,
            "steps": [
                {"id": s.step_id, "thought": s.thought, "type": s.reasoning_type}
                for s in result.steps
            ],
            "metadata": result.metadata,
        }

    def prove_syllogism(
        self,
        premise1: Tuple[str, str, str],
        premise2: Tuple[str, str, str],
        conclusion: Tuple[str, str, str],
    ) -> Dict[str, Any]:
        """
        Prove a categorical syllogism using formal logic.

        Args:
            premise1: (quantifier, copula, predicate) e.g., ("All", "are", "mortal")
            premise2: (quantifier, copula, predicate)
            conclusion: (quantifier, copula, predicate)

        Example:
            soul.prove_syllogism(
                premise1=("All", "are", "mortal"),
                premise2=("All", "are", "human"),
                conclusion=("All", "are", "mortal"),
            )
        """
        if not self._logic_engine:
            return {"error": "Logic engine not available"}

        return self._logic_engine.prove_syllogism(premise1, premise2, conclusion)

    def assert_knowledge(self, predicate_name: str, *terms: str) -> None:
        """Assert a fact to the soul's knowledge base."""
        if self._logic_engine:
            self._logic_engine.assert_predicate(predicate_name, *terms)

    def query_knowledge(self, predicate_name: str, *terms: str) -> bool:
        """Query the soul's knowledge base."""
        if not self._logic_engine:
            return False
        from domains.cognitive.reasoning import Predicate, Term

        return self._logic_engine.query(
            Predicate(name=predicate_name, terms=[Term(name=t) for t in terms])
        )

    def add_to_working_memory(self, item: str) -> None:
        """Add item to working memory (active reasoning context)."""
        if self._working_memory:
            self._working_memory.add(item)

    def get_working_memory(self, n: int = 5) -> List[str]:
        """Get n most relevant items from working memory."""
        if self._working_memory:
            return self._working_memory.get_recent(n)
        return []

    def clear_working_memory(self) -> None:
        """Clear working memory."""
        if self._working_memory:
            self._working_memory.clear()

    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get reasoning system statistics."""
        return {
            "deep_reasoning": self._deep_reasoning is not None,
            "logic_engine": self._logic_engine is not None,
            "working_memory_items": len(self._working_memory.items) if self._working_memory else 0,
            "session_turns": self._cognitive_state.get("session_turns", 0),
            "hebbian_connections": len(self._hebbian_connections),
        }

    def optimize_inference(
        self,
        use_kv_cache: bool = True,
        use_flash_attention: bool = True,
        use_quantization: bool = False,
        quantization_bits: int = 8,
    ) -> Dict[str, Any]:
        """
        Enable inference optimizations.

        Args:
            use_kv_cache: Enable KV cache for faster generation
            use_flash_attention: Use Flash Attention if available
            use_quantization: Quantize model weights
            quantization_bits: Bits for quantization (4, 8)

        Returns:
            Optimization status
        """
        try:
            from domains.inference.optimizer import InferenceOptimizer, InferenceConfig

            config = InferenceConfig(
                use_kv_cache=use_kv_cache,
                use_flash_attention=use_flash_attention,
                use_quantization=use_quantization,
                quantization_bits=quantization_bits,
            )

            self._inference_optimizer = InferenceOptimizer(
                model=self._model,
                config=config,
                device=self._device,
            )

            return {
                "success": True,
                "kv_cache": use_kv_cache,
                "flash_attention": use_flash_attention,
                "quantization": use_quantization if use_quantization else f"{quantization_bits}bit",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def benchmark_inference(
        self,
        prompt_tokens: int = 50,
        generated_tokens: int = 50,
        num_runs: int = 10,
    ) -> Dict[str, Any]:
        """
        Benchmark inference performance.

        Args:
            prompt_tokens: Length of input prompt
            generated_tokens: Tokens to generate
            num_runs: Number of benchmark runs

        Returns:
            Benchmark results
        """
        if not hasattr(self, "_inference_optimizer") or not self._inference_optimizer:
            return {"error": "Run optimize_inference() first"}

        try:
            from domains.inference.optimizer import InferenceBenchmark

            benchmark = InferenceBenchmark(self._inference_optimizer)

            # Create dummy prompt as numpy array
            prompt = np.random.randint(0, 1000, (1, prompt_tokens)).astype(np.int64)

            results = benchmark.run_benchmark(
                prompt="benchmark",
                num_tokens=generated_tokens,
                num_runs=num_runs,
            )

            return results
        except Exception as e:
            return {"error": str(e)}

    def enable_grounding(self) -> Dict[str, bool]:
        """
        Enable grounding system to prevent hallucinations and ensure accuracy.

        Features:
        - RAG: Retrieve relevant documents for grounding
        - Knowledge Graph: Verify statements against structured knowledge
        - Curriculum Learning: Efficient training
        """
        try:
            from domains.cognitive.grounding import GroundingOrchestrator

            self._grounding = GroundingOrchestrator()

            return {
                "enabled": True,
                "rag": True,
                "knowledge_graph": True,
                "curriculum_learning": True,
            }
        except Exception as e:
            return {"enabled": False, "error": str(e)}

    def add_knowledge(self, text: str, source: str = "user") -> Dict[str, Any]:
        """
        Add knowledge for grounding.

        This data will be used to:
        - Verify LLM outputs
        - Provide context for generation
        - Prevent hallucinations
        """
        if not self._grounding:
            self.enable_grounding()

        if self._grounding:
            self._grounding.add_data(text, source)
            return {"success": True, "chunks": len(text.split()) // 512 + 1}

        return {"success": False, "error": "Grounding not available"}

    def ground_output(self, response: str, query: str) -> Dict[str, Any]:
        """
        Ground an LLM output in real data.

        Returns:
        - Verification status
        - Confidence score
        - Supporting documents
        """
        if not self._grounding:
            return {"error": "Enable grounding first with enable_grounding()"}

        return self._grounding.ground_output(response, query)

    def get_knowledge_context(self, query: str) -> str:
        """
        Get relevant knowledge context for a query.
        Use this to provide grounding context to the LLM.
        """
        if not self._grounding:
            return ""

        return self._grounding.get_knowledge_context(query)

    # ===== HYPERDIMENSIONAL MEMORY =====

    def get_hd_memory_stats(self) -> Dict[str, Any]:
        """
        Get hyperdimensional memory statistics.

        Returns:
            HD memory stats including item count, dimension, etc.
        """
        if not self._hd_memory:
            return {"enabled": False, "error": "HD memory not initialized"}

        try:
            return {"enabled": True, **self._hd_memory.get_stats()}
        except Exception as e:
            return {"enabled": True, "error": str(e)}

    def search_hd_memory(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Search hyperdimensional memory for relevant content.

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            List of (id, content, similarity) tuples
        """
        if not self._hd_memory:
            return []

        try:
            return self._hd_memory.search(query, top_k=top_k)
        except Exception as e:
            logger.debug(f"HD memory search failed: {e}")
            return []

    def add_to_hd_memory(self, content: str, role: str = "user") -> str:
        """
        Add content to hyperdimensional memory.

        Args:
            content: Text to encode and store
            role: Role (user/assistant/system)

        Returns:
            Memory item ID
        """
        if not self._hd_memory:
            return ""

        try:
            return self._hd_memory.add(content, role=role)
        except Exception as e:
            logger.debug(f"HD memory add failed: {e}")
            return ""

    def clear_hd_memory(self) -> int:
        """
        Clear hyperdimensional memory.

        Returns:
            Number of items cleared
        """
        if not self._hd_memory:
            return 0

        try:
            return self._hd_memory.clear()
        except Exception as e:
            logger.debug(f"HD memory clear failed: {e}")
            return 0

    def get_hd_context(self, query: str, max_chars: int = 500) -> str:
        """
        Get relevant context from HD memory.

        Args:
            query: Query text
            max_chars: Maximum characters to return

        Returns:
            Relevant context string
        """
        if not self._hd_memory:
            return ""

        try:
            return self._hd_memory.get_context(query, max_chars=max_chars)
        except Exception as e:
            logger.debug(f"HD context retrieval failed: {e}")
            return ""

    # ===== SEMANTIC CACHE =====

    def enable_cache(
        self,
        max_entries: int = 500,
        similarity_threshold: float = 0.30,
        ttl_seconds: float = 3600,
    ) -> Dict[str, Any]:
        """
        Enable semantic caching.

        Args:
            max_entries: Maximum cache entries
            similarity_threshold: Min similarity for cache hit (0-1)
            ttl_seconds: Cache entry TTL

        Returns:
            Cache configuration
        """
        if not self._semantic_cache:
            try:
                from domains.inference.semantic_cache import SemanticCache

                self._semantic_cache = SemanticCache(
                    dim=10000,
                    max_entries=max_entries,
                    similarity_threshold=similarity_threshold,
                    ttl_seconds=ttl_seconds,
                )
            except Exception as e:
                return {"enabled": False, "error": str(e)}

        self._cache_enabled = True
        return {
            "enabled": True,
            "max_entries": max_entries,
            "similarity_threshold": similarity_threshold,
            "ttl_seconds": ttl_seconds,
        }

    def disable_cache(self) -> bool:
        """Disable semantic caching."""
        self._cache_enabled = False
        return True

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get semantic cache statistics.

        Returns:
            Cache stats including hit rate, entry count, etc.
        """
        if not self._semantic_cache:
            return {"enabled": False, "error": "Cache not initialized"}

        try:
            stats = self._semantic_cache.get_stats()
            stats["cache_enabled"] = self._cache_enabled
            return stats
        except Exception as e:
            return {"enabled": False, "error": str(e)}

    def clear_cache(self) -> int:
        """
        Clear semantic cache.

        Returns:
            Number of entries cleared
        """
        if not self._semantic_cache:
            return 0

        try:
            return self._semantic_cache.clear()
        except Exception as e:
            logger.debug(f"Cache clear failed: {e}")
            return 0

    def invalidate_cache_entry(self, query: str) -> bool:
        """
        Invalidate specific cache entry.

        Args:
            query: Query to invalidate

        Returns:
            True if entry was found and removed
        """
        if not self._semantic_cache:
            return False

        try:
            return self._semantic_cache.invalidate(query)
        except Exception as e:
            logger.debug(f"Cache invalidation failed: {e}")
            return False

    def learn(
        self,
        texts: List[str],
        soul_name: str = "assistant",
        epochs: int = 10,
        learning_rate: float = 0.001,
        vocab_size: int = 512,
        teacher_model=None,
        teacher_tokenizer=None,
        algo: str = "bpe",
    ) -> Dict[str, Any]:
        """Train this soul on text data using SloNet + tokenizer.

        This is the primary learning pathway.  It:
          1. Trains a BPE or Unigram tokenizer on the texts (via TokenizerManager)
          2. Creates a SloNet with matching vocabulary
          3. Runs the training loop (teacher-guided or self-supervised)
          4. Updates the soul profile with training metrics
          5. Stores the trained model + tokenizer for inference

        Args:
            texts: training texts (list of strings)
            soul_name: personality to use
            epochs: number of training epochs
            learning_rate: optimizer learning rate
            vocab_size: target vocabulary size
            teacher_model: optional HF teacher model (for distillation)
            teacher_tokenizer: optional HF teacher tokenizer
            algo: ``"bpe"`` (SloBPE, default) or ``"unigram"`` (SloUnigram)

        Returns:
            dict with keys: success, soul_name, epochs, loss, steps, model_type, vocab_size
        """
        import time as _time

        from domains.training.tokenizer_manager import get_tokenizer_manager
        from domains.training.slonet import (
            SloNet, SloEmbedding, SloLSTM,
            SloAdam, cross_entropy, tensor,
        )

        if not texts:
            return {"success": False, "error": "No training texts provided"}

        # 1. Train tokenizer on the texts (BPE or Unigram)
        mgr = get_tokenizer_manager()
        algo_kwargs = {}
        if algo == "unigram":
            algo_kwargs = {"seed_max_len": 6, "em_iters": 3}
        mgr.train(texts, vocab_size=vocab_size, min_frequency=2, lowercase=True, algo=algo, **algo_kwargs)
        tok = mgr.get_tokenizer()
        self._tokenizer = tok
        logger.info(f"{algo} tokenizer trained: vocab={tok.vocab_size}")

        # 2. Create SloNet with matching vocab
        traits = {}
        if hasattr(self._soul, 'personality') and self._soul.personality:
            traits = self._soul.personality.to_dict() if hasattr(self._soul.personality, 'to_dict') else {}

        net = SloNet(
            layers=[
                SloEmbedding(tok.vocab_size, 256),
                SloLSTM(tok.vocab_size, 256, 512, num_layers=2, dropout=0.0),
            ],
            soul_name=soul_name,
            soul_traits=traits,
            system_prompt=self._soul.system_prompt or f"You are a helpful {soul_name} assistant.",
            lineage="soulengine-learned",
        )
        self._model = net
        logger.info(f"SloNet created: vocab={tok.vocab_size}")

        # 3. Training loop
        optimizer = SloAdam(lr=learning_rate)
        total_loss = 0.0
        step = 0
        loss_history = []
        start = _time.perf_counter()

        for epoch in range(epochs):
            for text in texts:
                input_ids = tok.encode(text[:128])
                if len(input_ids) < 2:
                    continue

                seq_len = min(len(input_ids) - 1, 32)
                if seq_len < 1:
                    continue

                chunk_size = 8
                for i in range(0, seq_len, chunk_size):
                    x_chunk = input_ids[i : i + chunk_size]
                    y_chunk = input_ids[i + 1 : i + chunk_size + 1]
                    while len(x_chunk) < chunk_size:
                        x_chunk.append(tok.pad_id)
                    while len(y_chunk) < chunk_size:
                        y_chunk.append(tok.pad_id)

                    x = tensor([[x_chunk]], requires_grad=True)
                    y = tensor([[y_chunk]])

                    lstm_layer = net.layers[1]
                    hidden = lstm_layer.init_hidden()
                    logits, _ = lstm_layer.forward(x, hidden)
                    loss = cross_entropy(logits, y.reshape(-1))

                    loss.backward()
                    optimizer.step(net.parameters())

                    step += 1
                    total_loss += loss.data[()]
                    loss_history.append(loss.data[()])

        elapsed = _time.perf_counter() - start
        avg_loss = total_loss / max(step, 1)

        # 4. Update soul profile with training metrics
        self._soul.metadata["steps"] = step
        self._soul.metadata["avg_loss"] = round(avg_loss, 6)
        self._soul.metadata["epochs_trained"] = epochs
        self._soul.metadata["training_time_s"] = round(elapsed, 2)
        self._soul.metadata["vocab_size"] = tok.vocab_size
        self._soul.metadata["tokenizer_type"] = "soultok"
        self._soul.metadata["tokenizer_config"] = tok.to_dict()
        self._soul.name = soul_name
        self._soul.lineage = "soulengine-learned"
        self._soul.base_model = "slonet-lstm"

        logger.info(
            f"Training complete: {step} steps, loss={avg_loss:.4f}, "
            f"{elapsed:.1f}s, tokens/sec={step/elapsed:.0f}"
        )

        return {
            "success": True,
            "soul_name": soul_name,
            "epochs": epochs,
            "steps": step,
            "loss": round(avg_loss, 6),
            "loss_history": [round(float(v), 6) for v in loss_history[-20:]],
            "model_type": "slonet-lstm",
            "vocab_size": tok.vocab_size,
            "training_time_s": round(elapsed, 2),
            "tokens_per_sec": round(step / elapsed, 1) if elapsed > 0 else 0,
        }

    def generate_with_topk(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        topn: int = 5,
        **kwargs,
    ) -> List[Dict]:
        """Generate tokens one by one, returning top-k probabilities per step.

        Args:
            prompt: input text
            max_new_tokens: max tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus filtering
            topn: number of top candidates to return per step

        Returns:
            list of dicts with keys: token_id, token_text, prob, top_candidates
            where top_candidates is [{token_id, token_text, prob}, ...]
        """
        if self._model is None:
            return [{"token_id": 0, "token_text": "[no model]", "prob": 1.0, "top_candidates": []}]

        full_prompt = self._build_full_prompt(prompt, include_reasoning=True)
        idx = self._tokenize(full_prompt)
        from domains.training.slonet import SloLSTM, tensor
        lstm_layers = [l for l in self._model.layers if isinstance(l, SloLSTM)]
        if not lstm_layers:
            return [{"token_id": 0, "token_text": "[no LSTM]", "prob": 1.0, "top_candidates": []}]

        steps = []
        generated = list(idx.flatten())
        lstm_layer = lstm_layers[0]

        for step_i in range(max_new_tokens):
            seq = np.array([generated[-128:]], dtype=np.int64)
            in_t = tensor(seq, requires_grad=False)
            h = lstm_layer.init_hidden()
            logits_t, _ = lstm_layer.forward(in_t, h)
            logits = logits_t.data[0] / temperature
            logits = np.where(np.isfinite(logits), logits, -1e9)
            v = len(logits)

            if top_k and top_k < v:
                cutoff = np.partition(logits, -top_k)[-top_k]
                logits[logits < cutoff] = -1e9
            if top_p < 1.0:
                si = np.argsort(-logits)
                sv = logits[si]
                ps = np.exp(sv - sv.max())
                ps = ps / (ps.sum() + 1e-10)
                cs = np.cumsum(ps)
                logits[si[cs > top_p]] = -1e9

            ps = np.exp(logits - logits.max())
            ps = ps / (ps.sum() + 1e-10)
            nid = int(np.random.choice(len(ps), p=ps))
            generated.append(nid)

            # Top-n candidates
            top_indices = np.argsort(-ps)[:topn]
            candidates = []
            for ti in top_indices:
                t_text = self._detokenize(np.array([int(ti)], dtype=np.int64)).strip()
                candidates.append({
                    "token_id": int(ti),
                    "token_text": t_text,
                    "prob": round(float(ps[ti]), 4),
                })

            token_text = self._detokenize(np.array([nid], dtype=np.int64))
            steps.append({
                "token_id": nid,
                "token_text": token_text,
                "prob": round(float(ps[nid]), 4),
                "top_candidates": candidates,
            })

            if nid == self._tokenizer.pad_id:
                break

        return steps

    def status(self) -> Dict[str, Any]:
        """Return comprehensive soul + model + tokenizer status.

        This is the single status endpoint for the entire soul lifecycle.
        All frontends (API, CLI, TUI) call this instead of piecing together
        state from multiple places.

        Returns:
            dict with keys: soul, model, tokenizer, training
        """
        soul_info = {
            "name": self._soul.name,
            "lineage": self._soul.lineage,
            "base_model": self._soul.base_model,
            "description": self._soul.description,
        }
        if hasattr(self._soul, 'personality') and self._soul.personality:
            soul_info["personality"] = (
                self._soul.personality.to_dict()
                if hasattr(self._soul.personality, 'to_dict')
                else {}
            )

        model_info = {
            "loaded": self._model is not None,
            "type": type(self._model).__name__ if self._model else None,
            "device": self._device,
        }

        tok = self._tokenizer
        tokenizer_info = {
            "trained": tok is not None and tok.vocab_size > 0 if hasattr(tok, 'vocab_size') else False,
            "vocab_size": tok.vocab_size if tok and hasattr(tok, 'vocab_size') else 0,
            "merges": len(tok.merges) if tok and hasattr(tok, 'merges') else 0,
        }

        training_info = dict(self._soul.metadata) if hasattr(self._soul, 'metadata') and self._soul.metadata else {}

        return {
            "soul": soul_info,
            "model": model_info,
            "tokenizer": tokenizer_info,
            "training": training_info,
        }

    def __repr__(self) -> str:
        loaded = "loaded" if self._model else "no model"
        return f"SloEngine(soul={self._soul.name}, {loaded})"


__all__ = ["SloEngine", "GenerationContext"]
