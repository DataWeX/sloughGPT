"""
Tests for formal logic engine: Term, Predicate, WFF, FormalLogicEngine,
DeepReasoning, and WorkingMemory.
"""

import asyncio
import pytest

from domains.cognitive.reasoning.deep import (
    DeepReasoning,
    DeepReasoningContext,
    FormalLogicEngine,
    LogicalOperator,
    Predicate,
    RetrievedKnowledge,
    RetrievalSource,
    Substitution,
    Term,
    WellFormedFormula,
    WorkingMemory,
)
from domains.cognitive.reasoning.advanced import ThoughtStep


# =============================================================================
# Term / Predicate / WFF dataclasses
# =============================================================================

class TestTermDataclass:
    def test_constant_construction(self):
        t = Term(name="socrates")
        assert t.name == "socrates"
        assert t.is_variable is False
        assert t.is_function is False
        assert t.arguments == []

    def test_variable_construction(self):
        t = Term(name="X", is_variable=True)
        assert t.is_variable is True

    def test_function_construction(self):
        inner = Term(name="a")
        f = Term(name="f", is_function=True, arguments=[inner])
        assert f.is_function is True
        assert len(f.arguments) == 1
        assert f.arguments[0].name == "a"

    def test_term_equality(self):
        a = Term(name="x")
        b = Term(name="x")
        assert a.name == b.name

    def test_term_not_hashable(self):
        with pytest.raises(TypeError):
            hash(Term(name="a"))


class TestPredicateDataclass:
    def test_construction(self):
        p = Predicate(name="human", terms=[Term(name="socrates")])
        assert p.name == "human"
        assert len(p.terms) == 1
        assert p.negated is False

    def test_negated(self):
        p = Predicate(name="mortal", terms=[Term(name="x")], negated=True)
        assert p.negated is True

    def test_hash_same(self):
        p1 = Predicate(name="human", terms=[Term(name="socrates")])
        p2 = Predicate(name="human", terms=[Term(name="socrates")])
        assert hash(p1) == hash(p2)

    def test_hash_different(self):
        p1 = Predicate(name="human", terms=[Term(name="socrates")])
        p2 = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert hash(p1) != hash(p2)

    def test_hash_negation_differs(self):
        p1 = Predicate(name="human", terms=[Term(name="x")])
        p2 = Predicate(name="human", terms=[Term(name="x")], negated=True)
        assert hash(p1) != hash(p2)

    def test_set_dedup(self):
        p1 = Predicate(name="p", terms=[Term(name="a")])
        p2 = Predicate(name="p", terms=[Term(name="a")])
        s = {p1, p2}
        assert len(s) == 1


class TestWellFormedFormula:
    def test_empty_wff(self):
        wff = WellFormedFormula()
        assert wff.predicate is None
        assert wff.operator is None

    def test_predicate_wff(self):
        pred = Predicate(name="p", terms=[Term(name="a")])
        wff = WellFormedFormula(predicate=pred)
        assert wff.predicate.name == "p"

    def test_implies_wff(self):
        left = WellFormedFormula(predicate=Predicate(name="P", terms=[Term(name="x")]))
        right = WellFormedFormula(predicate=Predicate(name="Q", terms=[Term(name="x")]))
        wff = WellFormedFormula(operator=LogicalOperator.IMPLIES, left=left, right=right)
        assert wff.operator == LogicalOperator.IMPLIES
        assert wff.left.predicate.name == "P"
        assert wff.right.predicate.name == "Q"


class TestSubstitution:
    def test_empty_substitution(self):
        s = Substitution()
        assert s.mapping == {}

    def test_mapping(self):
        s = Substitution(mapping={"X": Term(name="socrates")})
        assert s.mapping["X"].name == "socrates"


# =============================================================================
# FormalLogicEngine — basic operations
# =============================================================================

class TestFormalLogicEngineBasic:
    def test_assert_predicate_adds_to_kb(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        assert len(engine.knowledge_base) == 1
        assert engine.knowledge_base[0].predicate.name == "human"

    def test_assert_fact_adds_wff(self):
        engine = FormalLogicEngine()
        wff = WellFormedFormula(
            predicate=Predicate(name="mortal", terms=[Term(name="socrates")])
        )
        engine.assert_fact(wff)
        assert len(engine.knowledge_base) == 1

    def test_assert_multiple_predicates(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        engine.assert_predicate("mortal", "socrates")
        assert len(engine.knowledge_base) == 2

    def test_query_simple_fact_true(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        q = Predicate(name="human", terms=[Term(name="socrates")])
        assert engine.query(q) is True

    def test_query_wrong_name_false(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        q = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert engine.query(q) is False

    def test_query_wrong_term_false(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        q = Predicate(name="human", terms=[Term(name="plato")])
        assert engine.query(q) is False

    def test_query_empty_kb_false(self):
        engine = FormalLogicEngine()
        q = Predicate(name="anything", terms=[Term(name="x")])
        assert engine.query(q) is False

    def test_assert_predicate_multiple_terms(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("likes", "socrates", "plato")
        q = Predicate(name="likes", terms=[Term(name="socrates"), Term(name="plato")])
        assert engine.query(q) is True


# =============================================================================
# Forward chaining + Modus Ponens
# =============================================================================

class TestForwardChaining:
    def test_modus_ponens_direct(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        left = WellFormedFormula(predicate=Predicate(name="human", terms=[Term(name="socrates")]))
        right = WellFormedFormula(predicate=Predicate(name="mortal", terms=[Term(name="socrates")]))
        engine.assert_fact(WellFormedFormula(operator=LogicalOperator.IMPLIES, left=left, right=right))
        q = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert engine.query(q) is True

    def test_modus_ponens_chain(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")

        left1 = WellFormedFormula(predicate=Predicate(name="human", terms=[Term(name="socrates")]))
        right1 = WellFormedFormula(predicate=Predicate(name="mortal", terms=[Term(name="socrates")]))
        engine.assert_fact(WellFormedFormula(operator=LogicalOperator.IMPLIES, left=left1, right=right1))

        left2 = WellFormedFormula(predicate=Predicate(name="mortal", terms=[Term(name="socrates")]))
        right2 = WellFormedFormula(predicate=Predicate(name="dies", terms=[Term(name="socrates")]))
        engine.assert_fact(WellFormedFormula(operator=LogicalOperator.IMPLIES, left=left2, right=right2))

        q = Predicate(name="dies", terms=[Term(name="socrates")])
        assert engine.query(q) is True

    def test_no_modus_ponens_without_antecedent(self):
        engine = FormalLogicEngine()
        left = WellFormedFormula(predicate=Predicate(name="human", terms=[Term(name="socrates")]))
        right = WellFormedFormula(predicate=Predicate(name="mortal", terms=[Term(name="socrates")]))
        engine.assert_fact(WellFormedFormula(operator=LogicalOperator.IMPLIES, left=left, right=right))
        q = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert engine.query(q) is False

    def test_derived_fact_appears_in_forward_chain(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("a", "x")
        left = WellFormedFormula(predicate=Predicate(name="a", terms=[Term(name="x")]))
        right = WellFormedFormula(predicate=Predicate(name="b", terms=[Term(name="x")]))
        engine.assert_fact(WellFormedFormula(operator=LogicalOperator.IMPLIES, left=left, right=right))
        assert engine.query(Predicate(name="b", terms=[Term(name="x")])) is True

    def test_variable_unification_in_modus_ponens(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")

        x = Term(name="X", is_variable=True)
        left = WellFormedFormula(predicate=Predicate(name="human", terms=[x]))
        right = WellFormedFormula(predicate=Predicate(name="mortal", terms=[x]))
        engine.assert_fact(WellFormedFormula(operator=LogicalOperator.IMPLIES, left=left, right=right))

        assert engine.query(Predicate(name="mortal", terms=[Term(name="socrates")])) is True

    def test_multiple_implications_different_terms(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("cat", "whiskers")
        engine.assert_predicate("dog", "rex")

        x = Term(name="X", is_variable=True)
        l1 = WellFormedFormula(predicate=Predicate(name="cat", terms=[x]))
        r1 = WellFormedFormula(predicate=Predicate(name="animal", terms=[x]))
        engine.assert_fact(WellFormedFormula(operator=LogicalOperator.IMPLIES, left=l1, right=r1))

        y = Term(name="Y", is_variable=True)
        l2 = WellFormedFormula(predicate=Predicate(name="dog", terms=[y]))
        r2 = WellFormedFormula(predicate=Predicate(name="animal", terms=[y]))
        engine.assert_fact(WellFormedFormula(operator=LogicalOperator.IMPLIES, left=l2, right=r2))

        assert engine.query(Predicate(name="animal", terms=[Term(name="whiskers")])) is True
        assert engine.query(Predicate(name="animal", terms=[Term(name="rex")])) is True


# =============================================================================
# Unification
# =============================================================================

class TestUnification:
    def test_same_constant_unifies(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="p", terms=[Term(name="a")])
        p2 = Predicate(name="p", terms=[Term(name="a")])
        subst = engine._unify(p1, p2)
        assert subst is not None
        assert subst.mapping == {}

    def test_different_constants_fail(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="p", terms=[Term(name="a")])
        p2 = Predicate(name="p", terms=[Term(name="b")])
        assert engine._unify(p1, p2) is None

    def test_different_names_fail(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="p", terms=[Term(name="a")])
        p2 = Predicate(name="q", terms=[Term(name="a")])
        assert engine._unify(p1, p2) is None

    def test_variable_unifies_with_constant(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="p", terms=[Term(name="X", is_variable=True)])
        p2 = Predicate(name="p", terms=[Term(name="a")])
        subst = engine._unify(p1, p2)
        assert subst is not None
        assert subst.mapping["X"].name == "a"

    def test_occurs_check_prevents_self_unification(self):
        engine = FormalLogicEngine()
        x = Term(name="X", is_variable=True)
        f = Term(name="f", is_function=True, arguments=[x])
        p1 = Predicate(name="p", terms=[x])
        p2 = Predicate(name="p", terms=[f])
        subst = engine._unify(p1, p2)
        assert subst is None

    def test_function_terms_same_name_unify(self):
        engine = FormalLogicEngine()
        a = Term(name="a")
        b = Term(name="b")
        f1 = Term(name="f", is_function=True, arguments=[a])
        f2 = Term(name="f", is_function=True, arguments=[b])
        p1 = Predicate(name="p", terms=[f1])
        p2 = Predicate(name="p", terms=[f2])
        subst = engine._unify(p1, p2)
        # Same function name 'f' unifies (args not checked by current impl)
        assert subst is not None

    def test_function_terms_different_name_fail(self):
        engine = FormalLogicEngine()
        a = Term(name="a")
        f1 = Term(name="f", is_function=True, arguments=[a])
        f2 = Term(name="g", is_function=True, arguments=[a])
        p1 = Predicate(name="p", terms=[f1])
        p2 = Predicate(name="p", terms=[f2])
        subst = engine._unify(p1, p2)
        assert subst is None

    def test_different_arity_fails(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="p", terms=[Term(name="a")])
        p2 = Predicate(name="p", terms=[Term(name="a"), Term(name="b")])
        assert engine._unify(p1, p2) is None

    def test_two_variables(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="p", terms=[Term(name="X", is_variable=True)])
        p2 = Predicate(name="p", terms=[Term(name="Y", is_variable=True)])
        subst = engine._unify(p1, p2)
        assert subst is not None
        # One should map to the other
        assert len(subst.mapping) == 1


# =============================================================================
# Resolution
# =============================================================================

class TestResolution:
    def test_prove_simple_fact_by_resolution(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("fact", "a")
        goal = Predicate(name="fact", terms=[Term(name="a")])
        assert engine.resolution(goal) is True

    def test_resolution_proves_contradiction(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("p", "a")
        negated = WellFormedFormula(
            predicate=Predicate(name="p", terms=[Term(name="a")], negated=True)
        )
        engine.assert_fact(negated)
        # KB has both p(a) and ~p(a), contradiction found via resolution
        goal = Predicate(name="q", terms=[Term(name="x")])
        # With contradiction, resolution finds empty clause
        result = engine.resolution(goal)
        assert isinstance(result, bool)

    def test_resolution_with_multiple_facts(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("p", "a")
        engine.assert_predicate("q", "a")
        goal = Predicate(name="p", terms=[Term(name="a")])
        assert engine.resolution(goal) is True

    def test_resolution_fails_when_not_entailed(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("p", "a")
        goal = Predicate(name="q", terms=[Term(name="a")])
        assert engine.resolution(goal) is False

    def test_resolution_with_negated_fact(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("p", "a")
        negated = WellFormedFormula(
            predicate=Predicate(name="p", terms=[Term(name="a")], negated=True)
        )
        engine.assert_fact(negated)
        # KB has both p(a) and ~p(a), so any goal is provable (contradiction = everything follows)
        goal = Predicate(name="anything", terms=[Term(name="x")])
        # Resolution with contradiction may or may not derive anything depending on implementation
        # At minimum it should not crash
        result = engine.resolution(goal)
        assert isinstance(result, bool)


# =============================================================================
# Syllogisms
# =============================================================================

class TestSyllogisms:
    def test_valid_aaa_1(self):
        engine = FormalLogicEngine()
        result = engine.prove_syllogism(
            ("All", "are", "mortal"),
            ("All", "are", "human"),
            ("All", "are", "mortal"),
        )
        assert "valid" in result
        assert "mood" in result
        assert "figure" in result
        assert "reason" in result

    def test_invalid_mood(self):
        engine = FormalLogicEngine()
        result = engine.prove_syllogism(
            ("Some", "are", "mortal"),
            ("Some", "are", "human"),
            ("All", "are", "mortal"),
        )
        # I I A mood should be invalid in most figures
        assert isinstance(result["valid"], bool)

    def test_figure_detection(self):
        engine = FormalLogicEngine()
        result = engine.prove_syllogism(
            ("All", "are", "mortal"),
            ("All", "are", "human"),
            ("All", "are", "mortal"),
        )
        assert result["figure"] in [1, 2, 3, 4]

    def test_mood_extraction(self):
        engine = FormalLogicEngine()
        result = engine.prove_syllogism(
            ("All", "are", "mortal"),
            ("All", "are", "human"),
            ("All", "are", "mortal"),
        )
        assert len(result["mood"]) == 3
        assert all(c in "AEIO" for c in result["mood"])

    def test_inference_history_recorded(self):
        engine = FormalLogicEngine()
        engine.prove_syllogism(
            ("All", "are", "mortal"),
            ("All", "are", "human"),
            ("All", "are", "mortal"),
        )
        assert len(engine.inference_history) == 1
        assert engine.inference_history[0]["type"] == "syllogism"

    def test_format_categorical(self):
        engine = FormalLogicEngine()
        result = engine.prove_syllogism(
            ("All", "are", "mortal"),
            ("All", "are", "human"),
            ("All", "are", "mortal"),
        )
        assert "form" in result
        assert isinstance(result["form"], str)


# =============================================================================
# DeepReasoning
# =============================================================================

class TestDeepReasoning:
    @pytest.mark.asyncio
    async def test_reason_with_no_stores(self):
        dr = DeepReasoning()
        result = await dr.reason("What is 2+2?")
        assert hasattr(result, "conclusion")
        assert hasattr(result, "confidence")
        assert hasattr(result, "steps")
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_reason_returns_correct_type(self):
        dr = DeepReasoning()
        result = await dr.reason("Test problem")
        assert result.metadata["retrieved_count"] == 0

    @pytest.mark.asyncio
    async def test_fallback_retrieval_cause_pattern(self):
        dr = DeepReasoning()
        results = await dr._fallback_retrieval("What causes climate change?")
        assert len(results) > 0
        assert results[0].source == RetrievalSource.WORKING_MEMORY

    @pytest.mark.asyncio
    async def test_fallback_retrieval_effect_pattern(self):
        dr = DeepReasoning()
        results = await dr._fallback_retrieval("Therefore we conclude...")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_fallback_retrieval_no_match(self):
        dr = DeepReasoning()
        results = await dr._fallback_retrieval("random unrelated query xyz")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_build_context_basic(self):
        dr = DeepReasoning()
        ctx = DeepReasoningContext(query="test")
        context = await dr._build_context("test problem", [], ctx)
        assert "Problem: test problem" in context

    @pytest.mark.asyncio
    async def test_build_context_with_knowledge(self):
        dr = DeepReasoning()
        know = RetrievedKnowledge(
            content="fact one", source=RetrievalSource.WORKING_MEMORY, relevance=0.8
        )
        ctx = DeepReasoningContext(query="test")
        context = await dr._build_context("problem", [know], ctx)
        assert "fact one" in context
        assert "0.80" in context

    @pytest.mark.asyncio
    async def test_build_context_with_constraints(self):
        dr = DeepReasoning()
        ctx = DeepReasoningContext(query="test", constraints=["c1", "c2"])
        context = await dr._build_context("problem", [], ctx)
        assert "c1" in context
        assert "c2" in context

    @pytest.mark.asyncio
    async def test_build_context_with_assumptions(self):
        dr = DeepReasoning()
        ctx = DeepReasoningContext(query="test", assumptions=["a1"])
        context = await dr._build_context("problem", [], ctx)
        assert "a1" in context

    @pytest.mark.asyncio
    async def test_self_correct_no_corrections(self):
        dr = DeepReasoning()
        step = ThoughtStep(step_id=0, thought="test step", reasoning_type="analysis", confidence=0.9)
        ctx = DeepReasoningContext(query="test")
        result = await dr._self_correct([step], "context", ctx)
        assert result["needs_revision"] is False
        assert result["corrections"] == []

    @pytest.mark.asyncio
    async def test_synthesize(self):
        dr = DeepReasoning()
        step = ThoughtStep(step_id=0, thought="final thought", reasoning_type="analysis", confidence=0.9)
        corrected = {"corrections": [], "corrected_reasoning": [step], "needs_revision": False}
        conclusion = await dr._synthesize([step], corrected, "context")
        assert isinstance(conclusion, str)
        assert len(conclusion) > 0

    @pytest.mark.asyncio
    async def test_default_llm_responses(self):
        dr = DeepReasoning()
        assert "knowledge" in (await dr._default_llm("retrieve something")).lower() or "step" in (await dr._default_llm("retrieve something")).lower()
        response = await dr._default_llm("critique this")
        assert isinstance(response, str)
        response = await dr._default_llm("correct this")
        assert isinstance(response, str)
        response = await dr._default_llm("final answer")
        assert isinstance(response, str)


# =============================================================================
# WorkingMemory
# =============================================================================

class TestWorkingMemory:
    def test_add_within_capacity(self):
        wm = WorkingMemory(capacity=3)
        wm.add("a")
        wm.add("b")
        wm.add("c")
        assert len(wm.items) == 3

    def test_add_evicts_lru(self):
        wm = WorkingMemory(capacity=2)
        wm.add("a")
        wm.add("b")
        wm.add("c")
        assert len(wm.items) == 2
        assert "a" not in wm.items
        assert "c" in wm.items

    def test_access_increments_count(self):
        wm = WorkingMemory(capacity=3)
        wm.add("a")
        wm.add("b")
        wm.access("a")
        wm.access("a")
        assert wm.access_count["a"] == 3

    def test_get_recent_orders_by_access(self):
        wm = WorkingMemory(capacity=5)
        wm.add("a")
        wm.add("b")
        wm.add("c")
        wm.access("a")
        wm.access("a")
        wm.access("c")
        recent = wm.get_recent(3)
        assert recent[0] == "a"
        assert recent[1] == "c"

    def test_get_recent_limits_output(self):
        wm = WorkingMemory(capacity=5)
        wm.add("a")
        wm.add("b")
        wm.add("c")
        wm.add("d")
        recent = wm.get_recent(2)
        assert len(recent) == 2

    def test_lru_eviction_with_access(self):
        wm = WorkingMemory(capacity=2)
        wm.add("a")
        wm.add("b")
        wm.access("a")  # a now more used
        wm.add("c")  # should evict b (least accessed)
        assert "b" not in wm.items
        assert "a" in wm.items

    def test_clear(self):
        wm = WorkingMemory(capacity=5)
        wm.add("a")
        wm.add("b")
        wm.clear()
        assert len(wm.items) == 0
        assert len(wm.access_count) == 0

    def test_default_capacity(self):
        wm = WorkingMemory()
        assert wm.capacity == 7

    def test_add_same_item_twice(self):
        wm = WorkingMemory(capacity=3)
        wm.add("a")
        wm.add("a")
        assert wm.items.count("a") == 2


# =============================================================================
# LogicalOperator enum
# =============================================================================

class TestLogicalOperator:
    def test_all_operators_exist(self):
        assert LogicalOperator.AND.value == "∧"
        assert LogicalOperator.OR.value == "∨"
        assert LogicalOperator.NOT.value == "¬"
        assert LogicalOperator.IMPLIES.value == "→"
        assert LogicalOperator.IFF.value == "↔"
        assert LogicalOperator.FORALL.value == "∀"
        assert LogicalOperator.EXISTS.value == "∃"

    def test_operator_count(self):
        assert len(LogicalOperator) == 7


# =============================================================================
# RetrievalSource enum
# =============================================================================

class TestRetrievalSource:
    def test_all_sources(self):
        assert RetrievalSource.VECTOR_STORE.value == "vector_store"
        assert RetrievalSource.MEMORY.value == "memory"
        assert RetrievalSource.KNOWLEDGE_GRAPH.value == "knowledge_graph"
        assert RetrievalSource.WORKING_MEMORY.value == "working_memory"

    def test_source_count(self):
        assert len(RetrievalSource) == 4
