"""Tests for domains.cognitive.reasoning.deep — DeepReasoning, FormalLogicEngine, WorkingMemory.

Covers: retrieval-augmented reasoning, self-correction, formal logic (unification,
modus ponens, resolution, syllogisms), working memory LRU eviction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.cognitive.reasoning.deep import (
    RetrievalSource,
    RetrievedKnowledge,
    DeepReasoningContext,
    DeepReasoning,
    FormalLogicEngine,
    LogicalOperator,
    Term,
    Predicate,
    WellFormedFormula,
    Substitution,
    WorkingMemory,
)
from domains.cognitive.reasoning.advanced import ThoughtStep


# ── Data classes ──────────────────────────────────────────────────────

class TestRetrievedKnowledge:
    def test_creation(self):
        k = RetrievedKnowledge(content="fact", source=RetrievalSource.VECTOR_STORE, relevance=0.9)
        assert k.content == "fact"
        assert k.source == RetrievalSource.VECTOR_STORE
        assert k.source_id is None


class TestDeepReasoningContext:
    def test_defaults(self):
        ctx = DeepReasoningContext(query="test")
        assert ctx.query == "test"
        assert ctx.retrieved_knowledge == []
        assert ctx.constraints == []
        assert ctx.assumptions == []


# ── Deep Reasoning ───────────────────────────────────────────────────

class TestDeepReasoning:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        dr = DeepReasoning()
        result = await dr.reason("What is cause and effect?")
        assert result.conclusion
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_with_context(self):
        dr = DeepReasoning()
        ctx = DeepReasoningContext(
            query="test",
            constraints=["must be fast"],
            assumptions=["data is valid"],
        )
        result = await dr.reason("Test problem", context=ctx)
        assert result.metadata["retrieved_count"] >= 0

    @pytest.mark.asyncio
    async def test_fallback_retrieval(self):
        dr = DeepReasoning()
        results = await dr._fallback_retrieval("The cause is because of rain. Therefore wet.")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_fallback_no_match(self):
        dr = DeepReasoning()
        results = await dr._fallback_retrieval("xyzzy plugh")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_build_context(self):
        dr = DeepReasoning()
        ctx = DeepReasoningContext(query="test", constraints=["c1"])
        retrieved = [RetrievedKnowledge("info", RetrievalSource.VECTOR_STORE, 0.8)]
        text = await dr._build_context("problem", retrieved, ctx)
        assert "Problem: problem" in text
        assert "info" in text
        assert "c1" in text

    @pytest.mark.asyncio
    async def test_custom_llm(self):
        calls = []
        async def mock_llm(prompt):
            calls.append(prompt)
            if "critique" in prompt.lower():
                return "No issues found."
            elif "final" in prompt.lower():
                return "Conclusion reached."
            return "Analysis step."
        dr = DeepReasoning(llm_call=mock_llm)
        result = await dr.reason("Test", max_depth=2)
        assert len(calls) > 0
        assert result.conclusion

    @pytest.mark.asyncio
    async def test_self_correction_detects_issues(self):
        async def mock_llm(prompt):
            if "critique" in prompt.lower():
                return "There is an error in this reasoning."
            return "Step analysis."
        dr = DeepReasoning(llm_call=mock_llm)
        ctx = DeepReasoningContext(query="test")
        reasoning = [ThoughtStep(0, "step 1", "analysis", 0.8)]
        result = await dr._self_correct(reasoning, "context", ctx)
        assert result["needs_revision"] is True
        assert len(result["corrections"]) > 0

    @pytest.mark.asyncio
    async def test_self_correction_clean(self):
        async def mock_llm(prompt):
            return "Looks good."
        dr = DeepReasoning(llm_call=mock_llm)
        ctx = DeepReasoningContext(query="test")
        reasoning = [ThoughtStep(0, "step 1", "analysis", 0.8)]
        result = await dr._self_correct(reasoning, "context", ctx)
        assert result["needs_revision"] is False


# ── Formal Logic Engine ──────────────────────────────────────────────

class TestFormalLogicEngine:
    def test_assert_and_query(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        q = Predicate(name="human", terms=[Term(name="socrates")])
        assert engine.query(q) is True

    def test_query_not_found(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        q = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert engine.query(q) is False

    def test_modus_ponens(self):
        engine = FormalLogicEngine()
        # Assert: human(socrates)
        engine.assert_predicate("human", "socrates")
        # Assert: human(X) → mortal(X)
        antecedent = WellFormedFormula(predicate=Predicate(
            name="human", terms=[Term(name="X", is_variable=True)]
        ))
        consequent = WellFormedFormula(predicate=Predicate(
            name="mortal", terms=[Term(name="X", is_variable=True)]
        ))
        implication = WellFormedFormula(
            operator=LogicalOperator.IMPLIES,
            left=antecedent,
            right=consequent,
        )
        engine.assert_fact(implication)
        q = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert engine.query(q) is True

    def test_unify_same(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="p", terms=[Term(name="a")])
        p2 = Predicate(name="p", terms=[Term(name="a")])
        subst = engine._unify(p1, p2)
        assert subst is not None

    def test_unify_variable(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="p", terms=[Term(name="X", is_variable=True)])
        p2 = Predicate(name="p", terms=[Term(name="a")])
        subst = engine._unify(p1, p2)
        assert subst is not None
        assert subst.mapping["X"].name == "a"

    def test_unify_different_names(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="p", terms=[Term(name="a")])
        p2 = Predicate(name="q", terms=[Term(name="a")])
        assert engine._unify(p1, p2) is None

    def test_unify_different_arity(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="p", terms=[Term(name="a")])
        p2 = Predicate(name="p", terms=[Term(name="a"), Term(name="b")])
        assert engine._unify(p1, p2) is None

    def test_occurs_check(self):
        engine = FormalLogicEngine()
        var = Term(name="X", is_variable=True)
        term = Term(name="f", is_function=True, arguments=[Term(name="X", is_variable=True)])
        subst = Substitution()
        assert engine._occurs_check(var, term, subst) is True

    def test_apply_substitution(self):
        engine = FormalLogicEngine()
        pred = Predicate(name="p", terms=[Term(name="X", is_variable=True)])
        subst = Substitution(mapping={"X": Term(name="a")})
        result = engine._apply_substitution(pred, subst)
        assert result.terms[0].name == "a"

    def test_prove_syllogism(self):
        engine = FormalLogicEngine()
        result = engine.prove_syllogism(
            premise1=("All", "are", "mortal"),
            premise2=("All", "are", "human"),
            conclusion=("All", "are", "mortal"),
        )
        assert result["valid"] is True
        assert result["figure"] == 1

    def test_resolution_proves(self):
        engine = FormalLogicEngine()
        # Assert: p(a)
        engine.assert_predicate("p", "a")
        # Assert: p(X) → q(X)
        antecedent = WellFormedFormula(predicate=Predicate(
            name="p", terms=[Term(name="X", is_variable=True)]
        ))
        consequent = WellFormedFormula(predicate=Predicate(
            name="q", terms=[Term(name="X", is_variable=True)]
        ))
        engine.assert_fact(WellFormedFormula(
            operator=LogicalOperator.IMPLIES,
            left=antecedent,
            right=consequent,
        ))
        goal = Predicate(name="q", terms=[Term(name="a")])
        assert engine.resolution(goal) is True


# ── Working Memory ───────────────────────────────────────────────────

class TestWorkingMemory:
    def test_add_and_get(self):
        wm = WorkingMemory(capacity=3)
        wm.add("a")
        wm.add("b")
        assert "a" in wm.items
        assert "b" in wm.items

    def test_lru_eviction(self):
        wm = WorkingMemory(capacity=2)
        wm.add("a")
        wm.add("b")
        wm.access("a")  # a now more used
        wm.add("c")  # should evict b (least used)
        assert "b" not in wm.items
        assert "a" in wm.items
        assert "c" in wm.items

    def test_get_recent(self):
        wm = WorkingMemory(capacity=5)
        wm.add("x")
        wm.add("y")
        wm.access("x")
        recent = wm.get_recent(2)
        assert recent[0] == "x"

    def test_clear(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.clear()
        assert len(wm.items) == 0
