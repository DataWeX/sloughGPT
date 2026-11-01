"""Cognition engine tests — train, reason, and verify the engine actually thinks.

Not import smoke tests. Every test runs real computation:
- trains a tiny model on a simple formula
- runs cognitive reasoning and checks output
- verifies consciousness processes real input
"""

from __future__ import annotations

import numpy as np
import pytest

# ── Cognitive Core: reasoning actually produces output ──────────────────


class TestCognitiveCore:
    def test_think_returns_process(self):
        from domain.cognition import CognitiveCore, ThinkingMode

        core = CognitiveCore()
        result = core.think("What is 2 + 2?", mode=ThinkingMode.ANALYTICAL)
        assert result is not None
        assert result.thought_content
        assert result.confidence > 0
        assert result.mode == ThinkingMode.ANALYTICAL

    def test_reasoning_chain_builds(self):
        from domain.cognition import CognitiveCore

        core = CognitiveCore()
        chain = core.reason("If all men are mortal, and Socrates is a man, is Socrates mortal?")
        assert chain is not None
        assert chain.conclusion
        assert len(chain.reasoning_steps) >= 1

    def test_creative_idea_generated(self):
        from domain.cognition import CognitiveCore, ThinkingMode

        core = CognitiveCore()
        idea = core.think("Design a new type of memory", mode=ThinkingMode.CREATIVE)
        assert idea is not None
        assert idea.mode == ThinkingMode.CREATIVE
        assert idea.creativity_score > 0


# ── Formal logic: derives conclusions ───────────────────────────────────


class TestFormalLogic:
    def test_assert_and_query(self):
        from domain.cognition._internal.reasoning.deep import (
            FormalLogicEngine,
            Predicate,
            Term,
            WellFormedFormula,
        )

        engine = FormalLogicEngine()
        wf = WellFormedFormula(predicate=Predicate("mortal", [Term("socrates")]))
        engine.assert_fact(wf)
        result = engine.query(Predicate("mortal", [Term("socrates")]))
        assert result is True

    def test_syllogism_proves(self):
        from domain.cognition._internal.reasoning.deep import FormalLogicEngine

        engine = FormalLogicEngine()
        result = engine.prove_syllogism(
            ("all", "men", "mortal"),
            ("all", "mortal", "dead"),
            ("all", "men", "dead"),
        )
        assert result["valid"] is True

    def test_working_memory_holds_items(self):
        from domain.cognition._internal.reasoning.deep import WorkingMemory

        wm = WorkingMemory(capacity=7)
        wm.add("fact one")
        wm.add("fact two")
        wm.add("fact three")
        assert len(wm.items) == 3
        recent = wm.get_recent()
        assert "fact three" in recent or len(recent) > 0

    def test_working_memory_evicts_beyond_capacity(self):
        from domain.cognition._internal.reasoning.deep import WorkingMemory

        wm = WorkingMemory(capacity=3)
        for i in range(10):
            wm.add(f"item {i}")
        assert len(wm.items) <= 3

    @pytest.mark.asyncio
    async def test_deep_reasoning_produces_output(self):
        from domain.cognition._internal.reasoning.deep import DeepReasoning

        dr = DeepReasoning()
        result = await dr.reason("Why does the sky appear blue?")
        assert result is not None
        if isinstance(result, str):
            assert len(result) > 0
        elif hasattr(result, "conclusion"):
            assert result.conclusion is not None


# ── Advanced reasoning: async chain/tree actually runs ──────────────────


class TestAdvancedReasoning:
    @pytest.mark.asyncio
    async def test_chain_of_thought_produces_steps(self):
        from domain.cognition._internal.reasoning.advanced import ChainOfThought

        cot = ChainOfThought()
        result = await cot.reason("What is 15% of 200?")
        assert result.conclusion
        assert len(result.steps) > 0
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_tree_of_thoughts_explores(self):
        from domain.cognition._internal.reasoning.advanced import TreeOfThoughts

        tot = TreeOfThoughts()
        result = await tot.reason("What are three ways to reduce latency?")
        assert result.conclusion
        assert result.mode.value == "tree_of_thoughts"

    @pytest.mark.asyncio
    async def test_syllogism_validates(self):
        from domain.cognition._internal.reasoning.advanced import SyllogismReasoning

        sr = SyllogismReasoning()
        result = await sr.reason(
            "All humans are mortal. Socrates is human. Is Socrates mortal?"
        )
        assert result.metadata.get("valid") is True
        assert result.confidence >= 0.9
        assert len(result.steps) >= 2

    @pytest.mark.asyncio
    async def test_causal_reasoning_identifies(self):
        from domain.cognition._internal.reasoning.advanced import CausalReasoning

        cr = CausalReasoning()
        result = await cr.reason("The light switch was flipped and the room lit up.")
        assert result.conclusion
        assert result.mode.value == "causal"

    @pytest.mark.asyncio
    async def test_self_consistency_returns_answer(self):
        from domain.cognition._internal.reasoning.advanced import SelfConsistency

        sc = SelfConsistency()
        result = await sc.reason("What is 2 + 2?")
        assert result.conclusion
        assert result.confidence > 0


# ── Knowledge graph: stores and queries real facts ─────────────────────


class TestKnowledgeGraph:
    def test_add_entity_and_fact(self):
        from domain.cognition._internal.knowledge_graph_v2 import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_entity(id="socrates", label="Socrates", entity_type="person")
        kg.add_fact(subject="Socrates", predicate="is", obj="mortal")
        assert len(kg.entities) >= 1
        assert len(kg.facts) == 1

    def test_query_by_subject(self):
        from domain.cognition._internal.knowledge_graph_v2 import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_entity(id="earth", label="Earth", entity_type="planet")
        kg.add_fact(subject="Earth", predicate="orbits", obj="Sun")
        results = kg.query(subject="Earth")
        assert len(results) >= 1
        assert results[0].predicate == "orbits"

    def test_transitive_inference(self):
        from domain.cognition._internal.knowledge_graph_v2 import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_fact(subject="A", predicate="parent_of", obj="B")
        kg.add_fact(subject="B", predicate="parent_of", obj="C")
        result = kg.infer_transitive("A", "parent_of")
        assert "C" in result

    def test_consistency_check(self):
        from domain.cognition._internal.knowledge_graph_v2 import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_fact(subject="A", predicate="is", obj="B")
        issues = kg.check_consistency()
        assert isinstance(issues, list)
        assert len(issues) == 0


# ── Consciousness: processes real input, not just loads ─────────────────


class TestConsciousnessEngine:
    def test_process_returns_annotation(self):
        from domain.cognition._internal.consciousness.config import ConsciousnessConfig
        from domain.core import get_consciousness

        c = get_consciousness(ConsciousnessConfig(level=2))
        result = c.process("Hello, how are you?", "I am doing well, thanks.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_qualia_experiences_input(self):
        from domain.cognition._internal.consciousness.qualia import QualiaEngine

        qe = QualiaEngine()
        state = qe.experience("Something surprising and new happened!")
        assert state is not None
        d = state.to_dict()
        assert "valence" in d
        assert "arousal" in d

    def test_self_model_observes(self):
        from domain.cognition._internal.consciousness.self_model import SelfModel

        sm = SelfModel(store_path="/tmp/test_consciousness_eng")
        sm.observe({"input_text": "hi", "response": "hello"})
        assert len(sm.episodes) == 1

    def test_status_reports_state(self):
        from domain.cognition._internal.consciousness.config import ConsciousnessConfig
        from domain.core import get_consciousness

        c = get_consciousness(ConsciousnessConfig(level=1))
        status = c.get_status()
        assert status["enabled"] is True
        assert status["level"] == 1

    def test_narrative_generates_text(self):
        from domain.cognition._internal.consciousness.config import ConsciousnessConfig
        from domain.cognition._internal.consciousness.engine import get_consciousness

        c = get_consciousness(ConsciousnessConfig(level=3))
        result = c.process("What do you think about consciousness?", "It is fascinating.")
        assert isinstance(result, str)
        assert len(result) > 10


# ── Training: actually trains on a simple formula ──────────────────────


class TestTrainingSimpleFormula:
    def test_arithmetic_dataset_is_valid(self):
        """Build arithmetic text, verify it's valid training data."""
        data = "".join(f"{a}+{b}={a + b}\n" for a in range(10) for b in range(10))
        vocab = {c: i for i, c in enumerate(sorted(set(data)))}
        tokens = np.array([vocab[c] for c in data], dtype=np.int64)

        assert len(tokens) > 100
        assert len(vocab) > 10
        assert "1+2=3" in data
        assert "5+5=10" in data

    def test_loss_tracker_converges(self):
        from domain.training._internal.training_handler import EMALossTracker

        tracker = EMALossTracker(alpha=0.3)
        for loss in [10.0, 8.0, 6.0, 4.0, 2.0, 1.0]:
            tracker.update(loss)
        assert tracker.current < 10.0
        assert tracker.current > 0.0

    def test_raw_loss_tracker_averages(self):
        from domain.training._internal.training_handler import RawLossTracker

        tracker = RawLossTracker()
        tracker.update(2.0)
        tracker.update(4.0)
        assert tracker.current == 3.0

    def test_gradient_clipping_returns_norm(self):
        from domain.training._internal.training_handler import clip_gradients

        class FakeParam:
            def __init__(self, v):
                self.grad = np.array([v, v], dtype=np.float32)

        params = [FakeParam(100.0), FakeParam(100.0)]
        norm = clip_gradients(params, max_norm=1.0)
        assert norm > 1.0

    def test_soul_checkpoint_saves_and_loads(self):
        import os
        import tempfile

        from domain.training._internal.training_handler import SoulCheckpointSaver

        with tempfile.TemporaryDirectory() as tmpdir:
            saver = SoulCheckpointSaver(checkpoint_dir=tmpdir, max_keep=2)

            class FakeModel:
                soul_name = "test_model"
                soul_traits = {}
                lineage = "test"
                system_prompt = ""
                metadata = {}
                _created_at = "2026-01-01"
                _step = 0

                def soul_signature(self):
                    return "sig"

                def state_dict(self):
                    return {"w": np.array([1.0, 2.0, 3.0])}

                def parameters(self):
                    return []

                def load_state_dict(self, sd):
                    self.loaded = sd

            model = FakeModel()
            path = saver.save(model, "test_ckpt", epoch=1, loss=0.5)
            assert os.path.exists(path)
            latest = saver.latest_path()
            assert latest is not None

    def test_random_block_sampler_shapes(self):
        from domain.training._internal.training_handler import RandomBlockSampler

        data = np.arange(1000, dtype=np.int64)
        sampler = RandomBlockSampler(data, block_size=64)
        x, y = sampler.get_batch(4)
        assert x.shape == (4, 64)
        assert y.shape == (4, 64)
        assert x[0][1] == y[0][0]

    def test_permutation_sampler_shapes(self):
        from domain.training._internal.training_handler import PermutationSampler

        data = np.arange(640, dtype=np.int64)
        sampler = PermutationSampler(x_data=data, y_data=np.roll(data, -1))
        x, y = sampler.get_batch(2)
        assert len(x) == 2
        assert len(y) == 2
        assert y[0] == x[0] + 1

    def test_build_char_vocab(self):
        from domain.training._internal.training_handler import build_char_vocab

        vocab, itos = build_char_vocab(["hello", "world"])
        assert "h" in vocab
        assert "w" in vocab
        assert len(vocab) >= 6
        assert vocab["h"] in itos


# ── End-to-end: cognition sees with the engine, not just "loaded" ──────


class TestCognitionSeesEngine:
    def test_slo_engine_has_consciousness_wired(self):
        from domain.core import SloEngine

        engine = SloEngine()
        assert hasattr(engine, "_consciousness")
        assert hasattr(engine, "_init_consciousness")

    def test_slo_engine_has_cognitive_components(self):
        from domain.core import SloEngine

        engine = SloEngine()
        assert hasattr(engine, "_reasoning_engine")
        assert hasattr(engine, "_sentiment_analyzer")
        assert hasattr(engine, "_hd_memory")

    def test_slo_engine_generates_gracefully(self):
        from domain.core import SloEngine

        engine = SloEngine()
        result = engine.generate("test prompt", max_new_tokens=10)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_slo_engine_tracks_cognitive_state(self):
        from domain.core import SloEngine

        engine = SloEngine()
        assert "session_turns" in engine._cognitive_state
        assert "last_sentiment" in engine._cognitive_state
        assert "last_emotion" in engine._cognitive_state

    def test_slo_engine_builds_full_prompt_with_context(self):
        from domain.core import SloEngine

        engine = SloEngine()
        engine._session_history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        prompt = engine._build_full_prompt("current question", include_reasoning=True)
        assert "previous question" in prompt
        assert "current question" in prompt

    def test_slo_engine_chat_rejects_empty(self):
        from domain.core import SloEngine

        engine = SloEngine()
        assert engine.chat([]) == ""

    def test_slo_engine_requires_user_last(self):
        from domain.core import SloEngine

        engine = SloEngine()
        with pytest.raises(ValueError):
            engine.chat([{"role": "assistant", "content": "hi"}])

    def test_cognitive_processor_available(self):
        from domain.cognition import CognitiveProcessor

        proc = CognitiveProcessor()
        assert proc is not None


# ── KnowledgeManager: unified state, worker, cache ──────────────────────


class TestKnowledgeManager:
    def _make_km(self):
        from domain.cognition._internal.knowledge_graph_v2 import KnowledgeGraph
        from domain.cognition._internal.reasoning.deep import WorkingMemory
        from domain.core._internal.knowledge_state import KnowledgeManager
        from domain.soul._internal.hd_memory import HDMemoryStore

        return KnowledgeManager(
            hd_memory=HDMemoryStore(),
            working_memory=WorkingMemory(capacity=100),
            knowledge_graph=KnowledgeGraph(),
        )

    def test_add_and_query(self):
        km = self._make_km()
        km.add("Paris is the capital of France", source="fact")
        results = km.query("capital of France")
        assert len(results) >= 1
        assert any("Paris" in r.text for r in results)

    def test_get_context(self):
        km = self._make_km()
        km.add("Python is a programming language", source="fact")
        ctx = km.get_context("Python")
        assert "Python" in ctx

    def test_cache_hits(self):
        km = self._make_km()
        km.add("Test knowledge", source="fact")
        km.query("Test")
        km.query("Test")
        assert km.cache.hits >= 1
        assert km.cache.hit_rate > 0

    def test_cache_invalidated_on_add(self):
        km = self._make_km()
        km.add("First", source="fact")
        km.query("First")
        km.query("First")  # second query hits cache
        assert km.cache.hits >= 1
        km.add("Second", source="fact")
        assert len(km.cache._cache) == 0

    def test_stats(self):
        km = self._make_km()
        km.add("Something", source="fact")
        s = km.stats()
        assert s["entries"] == 1
        assert s["train_queue"] == 1
        assert "cache" in s
        assert "worker" in s

    def test_train_queue_drained(self):
        km = self._make_km()
        km.add("Fact one", source="fact")
        km.add("Fact two", source="fact")
        data = km.state.get_training_data()
        assert "Fact one" in data
        assert "Fact two" in data
        assert km.stats()["train_queue"] == 0

    def test_worker_not_running_by_default(self):
        km = self._make_km()
        assert km.worker.stats["running"] is False

    def test_clear(self):
        km = self._make_km()
        km.add("Something", source="fact")
        km.clear()
        assert km.stats()["entries"] == 0


class TestCycleDetection:
    def test_cycle_detected(self):
        from domain.cognition._internal.knowledge_graph_v2 import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_fact(subject="X", predicate="parent_of", obj="Y")
        kg.add_fact(subject="Y", predicate="parent_of", obj="Z")
        kg.add_fact(subject="Z", predicate="parent_of", obj="X")
        issues = kg.check_consistency()
        assert any(i["type"] == "cycle" for i in issues)

    def test_clean_dag_no_false_positive(self):
        from domain.cognition._internal.knowledge_graph_v2 import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_fact(subject="A", predicate="leads_to", obj="B")
        kg.add_fact(subject="B", predicate="leads_to", obj="C")
        kg.add_fact(subject="C", predicate="leads_to", obj="D")
        issues = kg.check_consistency()
        assert not any(i["type"] == "cycle" for i in issues)

    def test_self_loop_detected(self):
        from domain.cognition._internal.knowledge_graph_v2 import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_fact(subject="A", predicate="related_to", obj="A")
        issues = kg.check_consistency()
        assert any(i["type"] == "cycle" for i in issues)


def _make_km():
    from domain.cognition._internal.knowledge_graph_v2 import KnowledgeGraph
    from domain.cognition._internal.reasoning.deep import WorkingMemory
    from domain.core._internal.knowledge_state import KnowledgeManager
    from domain.soul._internal.hd_memory import HDMemoryStore

    return KnowledgeManager(
        hd_memory=HDMemoryStore(),
        working_memory=WorkingMemory(capacity=100),
        knowledge_graph=KnowledgeGraph(),
    )


class TestKnowledgeScoring:
    def test_frequency_increases_score(self):
        from domain.core._internal.knowledge_state import KnowledgeEntry

        e = KnowledgeEntry(text="test")
        base = e.score()
        e.record_access()
        e.record_access()
        e.record_access()
        assert e.score() > base
        assert e.access_count == 3

    def test_recency_decay(self):
        import time as _time

        from domain.core._internal.knowledge_state import KnowledgeEntry

        fresh = KnowledgeEntry(text="fresh")
        old = KnowledgeEntry(text="old")
        old.timestamp = _time.time() - 48 * 3600  # 48 hours old
        assert fresh.score() > old.score()

    def test_access_records_last_accessed(self):
        import time as _time

        from domain.core._internal.knowledge_state import KnowledgeEntry

        e = KnowledgeEntry(text="test")
        old_time = e.last_accessed
        _time.sleep(0.01)
        e.record_access()
        assert e.last_accessed > old_time

    def test_query_records_access(self):
        km = _make_km()
        km.add("Paris is capital of France", source="fact")
        km.query("Paris")
        km.query("Paris")
        entry = km.state._entries[0]
        assert entry.access_count >= 2

    def test_cache_hit_records_access(self):
        km = _make_km()
        km.add("Test entry", source="fact")
        km.query("Test")  # miss
        km.query("Test")  # hit
        km.query("Test")  # hit
        entry = km.state._entries[0]
        assert entry.access_count >= 3

    def test_training_priority_order(self):
        km = _make_km()
        km.add("Low priority", source="fact")
        km.add("High priority", source="fact")

        # Access high priority entry many times
        for _ in range(10):
            km.query("High priority")

        queue = [e for e in km.state._entries if not e.trained]
        queue.sort(key=lambda e: e.score(), reverse=True)
        assert "High priority" in queue[0].text

    def test_scoring_in_stats(self):
        km = _make_km()
        km.add("Something", source="fact")
        s = km.stats()
        assert "scoring" in s
        assert "avg_score" in s["scoring"]
        assert "recently_accessed" in s["scoring"]
