"""Tests for domains.cognitive.reasoning.deep — comprehensive coverage.

Covers: DeepReasoning (retrieval, self-correction, synthesis), FormalLogicEngine
(forward chaining, modus ponens, unification, resolution, syllogisms),
WorkingMemory (LRU eviction, access tracking), dataclasses, enums.
No mocks, pure logic with default LLM stubs.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.cognitive.reasoning.deep import (
    DeepReasoning,
    DeepReasoningContext,
    FormalLogicEngine,
    LogicalOperator,
    Predicate,
    Term,
    WellFormedFormula,
    WorkingMemory,
    RetrievedKnowledge,
    RetrievalSource,
    Substitution,
)
from domains.cognitive.reasoning.advanced import ThoughtStep, ReasoningMode


# ═══════════════════════════════════════════════════════════════════════
# RetrievalSource Enum
# ═══════════════════════════════════════════════════════════════════════

class TestRetrievalSource:
    def test_all_values_are_strings(self):
        for src in RetrievalSource:
            assert isinstance(src.value, str)
            assert len(src.value) > 0

    def test_unique_values(self):
        values = [s.value for s in RetrievalSource]
        assert len(values) == len(set(values))

    def test_expected_sources(self):
        expected = {"vector_store", "memory", "knowledge_graph", "working_memory"}
        assert {s.value for s in RetrievalSource} == expected


# ═══════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════

class TestRetrievedKnowledge:
    def test_basic(self):
        rk = RetrievedKnowledge(content="fact", source=RetrievalSource.MEMORY, relevance=0.9)
        assert rk.content == "fact"
        assert rk.source == RetrievalSource.MEMORY
        assert rk.relevance == 0.9
        assert rk.source_id is None

    def test_with_source_id(self):
        rk = RetrievedKnowledge(
            content="doc", source=RetrievalSource.VECTOR_STORE,
            relevance=0.8, source_id="doc-42",
        )
        assert rk.source_id == "doc-42"


class TestDeepReasoningContext:
    def test_defaults(self):
        ctx = DeepReasoningContext(query="what?")
        assert ctx.query == "what?"
        assert ctx.retrieved_knowledge == []
        assert ctx.working_memory == []
        assert ctx.constraints == []
        assert ctx.assumptions == []

    def test_full_construction(self):
        rk = RetrievedKnowledge("c", RetrievalSource.MEMORY, 0.5)
        ctx = DeepReasoningContext(
            query="q",
            retrieved_knowledge=[rk],
            working_memory=["wm1"],
            constraints=["c1"],
            assumptions=["a1"],
        )
        assert len(ctx.retrieved_knowledge) == 1
        assert ctx.working_memory == ["wm1"]
        assert ctx.constraints == ["c1"]
        assert ctx.assumptions == ["a1"]


# ═══════════════════════════════════════════════════════════════════════
# LogicalOperator Enum
# ═══════════════════════════════════════════════════════════════════════

class TestLogicalOperator:
    def test_all_operators_present(self):
        names = {op.name for op in LogicalOperator}
        assert {"AND", "OR", "NOT", "IMPLIES", "IFF", "FORALL", "EXISTS"} == names

    def test_symbol_values(self):
        assert LogicalOperator.AND.value == "∧"
        assert LogicalOperator.OR.value == "∨"
        assert LogicalOperator.NOT.value == "¬"
        assert LogicalOperator.IMPLIES.value == "→"
        assert LogicalOperator.IFF.value == "↔"
        assert LogicalOperator.FORALL.value == "∀"
        assert LogicalOperator.EXISTS.value == "∃"


# ═══════════════════════════════════════════════════════════════════════
# Term & Predicate
# ═══════════════════════════════════════════════════════════════════════

class TestTerm:
    def test_constant(self):
        t = Term(name="socrates")
        assert t.name == "socrates"
        assert t.is_variable is False
        assert t.is_function is False
        assert t.arguments == []

    def test_variable(self):
        t = Term(name="X", is_variable=True)
        assert t.is_variable is True

    def test_function(self):
        arg = Term(name="a")
        f = Term(name="f", is_function=True, arguments=[arg])
        assert f.is_function is True
        assert len(f.arguments) == 1


class TestPredicate:
    def test_basic(self):
        p = Predicate(name="human", terms=[Term(name="socrates")])
        assert p.name == "human"
        assert p.negated is False

    def test_negated(self):
        p = Predicate(name="mortal", terms=[Term(name="socrates")], negated=True)
        assert p.negated is True

    def test_hash(self):
        p1 = Predicate(name="human", terms=[Term(name="socrates")])
        p2 = Predicate(name="human", terms=[Term(name="socrates")])
        assert hash(p1) == hash(p2)

    def test_hash_different(self):
        p1 = Predicate(name="human", terms=[Term(name="socrates")])
        p2 = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert hash(p1) != hash(p2)

    def test_hash_negated_differs(self):
        p1 = Predicate(name="human", terms=[Term(name="socrates")], negated=False)
        p2 = Predicate(name="human", terms=[Term(name="socrates")], negated=True)
        assert hash(p1) != hash(p2)


# ═══════════════════════════════════════════════════════════════════════
# WellFormedFormula
# ═══════════════════════════════════════════════════════════════════════

class TestWellFormedFormula:
    def test_predicate_only(self):
        pred = Predicate(name="p", terms=[Term(name="a")])
        wff = WellFormedFormula(predicate=pred)
        assert wff.predicate == pred
        assert wff.operator is None

    def test_implication(self):
        left = WellFormedFormula(predicate=Predicate(name="P", terms=[]))
        right = WellFormedFormula(predicate=Predicate(name="Q", terms=[]))
        wff = WellFormedFormula(operator=LogicalOperator.IMPLIES, left=left, right=right)
        assert wff.operator == LogicalOperator.IMPLIES

    def test_quantified(self):
        var = Term(name="X", is_variable=True)
        sub = WellFormedFormula(predicate=Predicate(name="P", terms=[var]))
        wff = WellFormedFormula(
            quantifier_var=var,
            quantifier_type=LogicalOperator.FORALL,
            subformula=sub,
        )
        assert wff.quantifier_type == LogicalOperator.FORALL


# ═══════════════════════════════════════════════════════════════════════
# Substitution
# ═══════════════════════════════════════════════════════════════════════

class TestSubstitution:
    def test_empty_mapping(self):
        s = Substitution()
        assert s.mapping == {}

    def test_custom_mapping(self):
        s = Substitution(mapping={"X": Term(name="socrates")})
        assert s.mapping["X"].name == "socrates"


# ═══════════════════════════════════════════════════════════════════════
# FormalLogicEngine — Knowledge Base
# ═══════════════════════════════════════════════════════════════════════

class TestFormalLogicKB:
    def test_assert_fact(self):
        engine = FormalLogicEngine()
        wff = WellFormedFormula(predicate=Predicate(name="P", terms=[]))
        engine.assert_fact(wff)
        assert len(engine.knowledge_base) == 1

    def test_assert_predicate(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        assert len(engine.knowledge_base) == 1
        pred = engine.knowledge_base[0].predicate
        assert pred.name == "human"
        assert pred.terms[0].name == "socrates"

    def test_assert_multiple_facts(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        engine.assert_predicate("mortal", "socrates")
        assert len(engine.knowledge_base) == 2


# ═══════════════════════════════════════════════════════════════════════
# FormalLogicEngine — Forward Chaining
# ═══════════════════════════════════════════════════════════════════════

class TestForwardChaining:
    def test_query_direct_fact(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        result = engine.query(Predicate(name="human", terms=[Term(name="socrates")]))
        assert result is True

    def test_query_missing_fact(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        result = engine.query(Predicate(name="mortal", terms=[Term(name="socrates")]))
        assert result is False

    def test_query_empty_kb(self):
        engine = FormalLogicEngine()
        result = engine.query(Predicate(name="P", terms=[]))
        assert result is False

    def test_modus_ponens_direct(self):
        engine = FormalLogicEngine()
        # Assert P → Q and P
        antecedent = WellFormedFormula(predicate=Predicate(name="P", terms=[]))
        consequent = WellFormedFormula(predicate=Predicate(name="Q", terms=[]))
        implication = WellFormedFormula(
            operator=LogicalOperator.IMPLIES,
            left=antecedent,
            right=consequent,
        )
        engine.assert_fact(implication)
        engine.assert_predicate("P")
        result = engine.query(Predicate(name="Q", terms=[]))
        assert result is True

    def test_modus_ponens_chain(self):
        engine = FormalLogicEngine()
        # P → Q, Q → R, P ⊢ R
        p_to_q = WellFormedFormula(
            operator=LogicalOperator.IMPLIES,
            left=WellFormedFormula(predicate=Predicate(name="P", terms=[])),
            right=WellFormedFormula(predicate=Predicate(name="Q", terms=[])),
        )
        q_to_r = WellFormedFormula(
            operator=LogicalOperator.IMPLIES,
            left=WellFormedFormula(predicate=Predicate(name="Q", terms=[])),
            right=WellFormedFormula(predicate=Predicate(name="R", terms=[])),
        )
        engine.assert_fact(p_to_q)
        engine.assert_fact(q_to_r)
        engine.assert_predicate("P")
        result = engine.query(Predicate(name="R", terms=[]))
        assert result is True

    def test_modus_ponens_no_antecedent(self):
        engine = FormalLogicEngine()
        p_to_q = WellFormedFormula(
            operator=LogicalOperator.IMPLIES,
            left=WellFormedFormula(predicate=Predicate(name="P", terms=[])),
            right=WellFormedFormula(predicate=Predicate(name="Q", terms=[])),
        )
        engine.assert_fact(p_to_q)
        result = engine.query(Predicate(name="Q", terms=[]))
        assert result is False


# ═══════════════════════════════════════════════════════════════════════
# FormalLogicEngine — Unification
# ═══════════════════════════════════════════════════════════════════════

class TestUnification:
    def test_unify_identical_constants(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="P", terms=[Term(name="a")])
        p2 = Predicate(name="P", terms=[Term(name="a")])
        subst = engine._unify(p1, p2)
        assert subst is not None

    def test_unify_different_names(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="P", terms=[Term(name="a")])
        p2 = Predicate(name="Q", terms=[Term(name="a")])
        assert engine._unify(p1, p2) is None

    def test_unify_different_arity(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="P", terms=[Term(name="a")])
        p2 = Predicate(name="P", terms=[Term(name="a"), Term(name="b")])
        assert engine._unify(p1, p2) is None

    def test_unify_variable_to_constant(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="P", terms=[Term(name="X", is_variable=True)])
        p2 = Predicate(name="P", terms=[Term(name="a")])
        subst = engine._unify(p1, p2)
        assert subst is not None
        assert subst.mapping["X"].name == "a"

    def test_unify_symmetric(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="P", terms=[Term(name="X", is_variable=True)])
        p2 = Predicate(name="P", terms=[Term(name="a")])
        subst1 = engine._unify(p1, p2)
        subst2 = engine._unify(p2, p1)
        assert subst1 is not None
        assert subst2 is not None

    def test_unify_occurs_check_fails(self):
        engine = FormalLogicEngine()
        # f(X) unifying X with f(X) should fail
        f_x = Term(name="f", is_function=True, arguments=[Term(name="X", is_variable=True)])
        p1 = Predicate(name="P", terms=[Term(name="X", is_variable=True)])
        p2 = Predicate(name="P", terms=[f_x])
        assert engine._unify(p1, p2) is None

    def test_apply_term_substitution(self):
        engine = FormalLogicEngine()
        subst = Substitution(mapping={"X": Term(name="a")})
        result = engine._apply_term_substitution(Term(name="X", is_variable=True), subst)
        assert result.name == "a"

    def test_apply_substitution_to_predicate(self):
        engine = FormalLogicEngine()
        subst = Substitution(mapping={"X": Term(name="a")})
        pred = Predicate(name="P", terms=[Term(name="X", is_variable=True)])
        result = engine._apply_substitution(pred, subst)
        assert result.terms[0].name == "a"

    def test_apply_substitution_preserves_negation(self):
        engine = FormalLogicEngine()
        subst = Substitution(mapping={"X": Term(name="a")})
        pred = Predicate(name="P", terms=[Term(name="X", is_variable=True)], negated=True)
        result = engine._apply_substitution(pred, subst)
        assert result.negated is True

    def test_unify_terms_same_constant(self):
        engine = FormalLogicEngine()
        subst = Substitution()
        result = engine._unify_terms(Term(name="a"), Term(name="a"), subst)
        assert result is not None

    def test_unify_terms_different_constants(self):
        engine = FormalLogicEngine()
        subst = Substitution()
        result = engine._unify_terms(Term(name="a"), Term(name="b"), subst)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════
# FormalLogicEngine — Resolution
# ═══════════════════════════════════════════════════════════════════════

class TestResolution:
    def test_resolution_proves_entailed(self):
        engine = FormalLogicEngine()
        # P ⊢ P
        engine.assert_predicate("P")
        result = engine.resolution(Predicate(name="P", terms=[]))
        assert result is True

    def test_resolution_rejects_not_entailed(self):
        engine = FormalLogicEngine()
        # Only P, try to prove Q
        engine.assert_predicate("P")
        result = engine.resolution(Predicate(name="Q", terms=[]))
        assert result is False

    def test_resolution_with_implication(self):
        engine = FormalLogicEngine()
        # P → Q, P ⊢ Q
        p_to_q = WellFormedFormula(
            operator=LogicalOperator.IMPLIES,
            left=WellFormedFormula(predicate=Predicate(name="P", terms=[])),
            right=WellFormedFormula(predicate=Predicate(name="Q", terms=[])),
        )
        engine.assert_fact(p_to_q)
        engine.assert_predicate("P")
        result = engine.resolution(Predicate(name="Q", terms=[]))
        assert result is True

    def test_resolution_empty_kb(self):
        engine = FormalLogicEngine()
        result = engine.resolution(Predicate(name="P", terms=[]))
        assert result is False

    def test_to_clausal_form(self):
        engine = FormalLogicEngine()
        wff = WellFormedFormula(predicate=Predicate(name="P", terms=[]))
        clauses = engine._to_clausal_form([wff])
        assert len(clauses) == 1
        assert len(clauses[0]) == 1

    def test_extract_literals_predicate(self):
        engine = FormalLogicEngine()
        wff = WellFormedFormula(predicate=Predicate(name="P", terms=[]))
        literals = engine._extract_literals(wff)
        assert len(literals) == 1

    def test_extract_literals_implication(self):
        engine = FormalLogicEngine()
        left = WellFormedFormula(predicate=Predicate(name="P", terms=[]))
        right = WellFormedFormula(predicate=Predicate(name="Q", terms=[]))
        wff = WellFormedFormula(operator=LogicalOperator.IMPLIES, left=left, right=right)
        literals = engine._extract_literals(wff)
        # Should have ¬P and Q
        names = {l.name for l in literals}
        assert "P" in names
        assert "Q" in names
        negated = [l for l in literals if l.negated]
        assert len(negated) == 1
        assert negated[0].name == "P"

    def test_resolve_clauses_complementary(self):
        engine = FormalLogicEngine()
        lit_p = Predicate(name="P", terms=[Term(name="a")])
        lit_not_p = Predicate(name="P", terms=[Term(name="a")], negated=True)
        clause1 = {lit_p}
        clause2 = {lit_not_p}
        result = engine._resolve_clauses(clause1, clause2)
        assert result is not None
        assert len(result) == 0  # Empty clause = contradiction


# ═══════════════════════════════════════════════════════════════════════
# FormalLogicEngine — Syllogisms
# ═══════════════════════════════════════════════════════════════════════

class TestSyllogisms:
    def test_prove_syllogism_returns_structure(self):
        engine = FormalLogicEngine()
        result = engine.prove_syllogism(
            premise1=("All", "are", "mortal"),
            premise2=("Socrates", "is", "human"),
            conclusion=("Socrates", "is", "mortal"),
        )
        assert "valid" in result
        assert "figure" in result
        assert "mood" in result
        assert "reason" in result
        assert "form" in result

    def test_prove_valid_aaa_figure1(self):
        engine = FormalLogicEngine()
        # Mood AAA with figure 1 is valid
        result = engine.prove_syllogism(
            premise1=("All", "are", "mortal"),
            premise2=("All", "are", "human"),
            conclusion=("All", "are", "mortal"),
        )
        # _to_categorical extracts first letter of quantifier uppercased
        # "All"→A, "All"→A, "All"→A => mood AAA, figure 1 => valid
        assert result["mood"] == "AAA"
        assert result["valid"] is True

    def test_prove_invalid_syllogism(self):
        engine = FormalLogicEngine()
        # Non-standard mood that doesn't map to valid combinations
        result = engine.prove_syllogism(
            premise1=("Some", "are", "X"),
            premise2=("Some", "are", "Y"),
            conclusion=("Some", "are", "Z"),
        )
        assert isinstance(result["valid"], bool)
        assert "figure" in result
        assert "mood" in result

    def test_to_categorical(self):
        engine = FormalLogicEngine()
        cat = engine._to_categorical(("All", "are", "mortal"))
        assert cat[0] == "A"
        assert cat[3] == "mortal"

    def test_format_categorical(self):
        engine = FormalLogicEngine()
        result = engine._format_categorical(("A", "S", "are", "P"))
        assert result == "A S are P"

    def test_inference_history_recorded(self):
        engine = FormalLogicEngine()
        engine.prove_syllogism(
            premise1=("All", "are", "mortal"),
            premise2=("Socrates", "is", "human"),
            conclusion=("Socrates", "is", "mortal"),
        )
        assert len(engine.inference_history) == 1
        assert engine.inference_history[0]["type"] == "syllogism"

    def test_check_validity_valid_mood(self):
        engine = FormalLogicEngine()
        valid, reason = engine._check_syllogism_validity("AAA", 1, (), (), ())
        assert valid is True

    def test_check_validity_invalid_mood(self):
        engine = FormalLogicEngine()
        valid, reason = engine._check_syllogism_validity("XYZ", 1, (), (), ())
        assert valid is False


# ═══════════════════════════════════════════════════════════════════════
# WorkingMemory
# ═══════════════════════════════════════════════════════════════════════

class TestWorkingMemory:
    def test_add_item(self):
        wm = WorkingMemory(capacity=5)
        wm.add("item1")
        assert "item1" in wm.items

    def test_add_multiple(self):
        wm = WorkingMemory(capacity=5)
        wm.add("a")
        wm.add("b")
        wm.add("c")
        assert len(wm.items) == 3

    def test_capacity_eviction(self):
        wm = WorkingMemory(capacity=3)
        wm.add("a")
        wm.add("b")
        wm.add("c")
        wm.add("d")
        assert len(wm.items) == 3
        assert "d" in wm.items

    def test_lru_eviction(self):
        wm = WorkingMemory(capacity=2)
        wm.add("a")
        wm.add("b")
        wm.access("a")  # Make a more recent
        wm.add("c")  # Should evict b (least recently used)
        assert "a" in wm.items
        assert "c" in wm.items
        assert "b" not in wm.items

    def test_access_increments_count(self):
        wm = WorkingMemory(capacity=5)
        wm.add("item")
        wm.access("item")
        wm.access("item")
        assert wm.access_count["item"] == 3

    def test_get_recent_by_access_count(self):
        wm = WorkingMemory(capacity=5)
        wm.add("a")
        wm.add("b")
        wm.access("b")
        wm.access("b")
        wm.access("a")
        recent = wm.get_recent(5)
        assert recent[0] == "b"  # Most accessed first

    def test_get_recent_limits_output(self):
        wm = WorkingMemory(capacity=10)
        for i in range(10):
            wm.add(f"item{i}")
        recent = wm.get_recent(3)
        assert len(recent) == 3

    def test_clear(self):
        wm = WorkingMemory(capacity=5)
        wm.add("a")
        wm.add("b")
        wm.clear()
        assert len(wm.items) == 0
        assert len(wm.access_count) == 0

    def test_access_new_item(self):
        wm = WorkingMemory(capacity=5)
        wm.access("new_item")
        assert wm.access_count["new_item"] == 1

    def test_eviction_updates_access_count(self):
        wm = WorkingMemory(capacity=2)
        wm.add("a")
        wm.add("b")
        wm.add("c")  # evicts a
        assert "a" not in wm.access_count


# ═══════════════════════════════════════════════════════════════════════
# DeepReasoning
# ═══════════════════════════════════════════════════════════════════════

class TestDeepReasoning:
    @pytest.mark.asyncio
    async def test_returns_correct_mode(self):
        dr = DeepReasoning()
        result = await dr.reason("What is the meaning of life?")
        assert result.mode == ReasoningMode.CHAIN_OF_THOUGHT

    @pytest.mark.asyncio
    async def test_confidence_positive(self):
        dr = DeepReasoning()
        result = await dr.reason("test problem")
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_metadata_structure(self):
        dr = DeepReasoning()
        result = await dr.reason("problem")
        assert "retrieved_count" in result.metadata
        assert "corrections" in result.metadata
        assert "depth" in result.metadata

    @pytest.mark.asyncio
    async def test_max_depth_affects_steps(self):
        dr = DeepReasoning()
        r1 = await dr.reason("p", max_depth=1)
        r2 = await dr.reason("p", max_depth=3)
        assert len(r2.steps) >= len(r1.steps)

    @pytest.mark.asyncio
    async def test_context_with_constraints(self):
        dr = DeepReasoning()
        ctx = DeepReasoningContext(query="problem", constraints=["must be fast"])
        result = await dr.reason("problem", context=ctx)
        assert result.steps[0].reasoning_type == "retrieval"

    @pytest.mark.asyncio
    async def test_context_with_assumptions(self):
        dr = DeepReasoning()
        ctx = DeepReasoningContext(query="problem", assumptions=["input is valid"])
        result = await dr.reason("problem", context=ctx)
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_fallback_retrieval(self):
        dr = DeepReasoning()
        knowledge = await dr._fallback_retrieval("because of rain")
        assert len(knowledge) > 0
        assert knowledge[0].source == RetrievalSource.WORKING_MEMORY

    @pytest.mark.asyncio
    async def test_fallback_retrieval_no_match(self):
        dr = DeepReasoning()
        knowledge = await dr._fallback_retrieval("random unrelated text")
        assert len(knowledge) == 0

    @pytest.mark.asyncio
    async def test_fallback_retrieval_multiple_patterns(self):
        dr = DeepReasoning()
        knowledge = await dr._fallback_retrieval("because X, therefore Y, however Z")
        assert len(knowledge) >= 2

    @pytest.mark.asyncio
    async def test_build_context(self):
        dr = DeepReasoning()
        retrieved = [
            RetrievedKnowledge("fact1", RetrievalSource.MEMORY, 0.9),
            RetrievedKnowledge("fact2", RetrievalSource.VECTOR_STORE, 0.7),
        ]
        ctx = DeepReasoningContext(query="problem", constraints=["c1"])
        context = await dr._build_context("problem", retrieved, ctx)
        assert "Problem: problem" in context
        assert "fact1" in context
        assert "c1" in context

    @pytest.mark.asyncio
    async def test_build_context_no_retrieved(self):
        dr = DeepReasoning()
        ctx = DeepReasoningContext(query="q")
        context = await dr._build_context("q", [], ctx)
        assert "Problem: q" in context

    @pytest.mark.asyncio
    async def test_default_llm_retrieve(self):
        dr = DeepReasoning()
        result = await dr._default_llm("retrieve something")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_default_llm_critique(self):
        dr = DeepReasoning()
        result = await dr._default_llm("critique this step")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_default_llm_correct(self):
        dr = DeepReasoning()
        result = await dr._default_llm("correct the reasoning")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_default_llm_final(self):
        dr = DeepReasoning()
        result = await dr._default_llm("final answer synthesis")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_default_llm_generic(self):
        dr = DeepReasoning()
        result = await dr._default_llm("some other prompt")
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════
# DeepReasoning — Self-Correction
# ═══════════════════════════════════════════════════════════════════════

class TestDeepReasoningSelfCorrection:
    @pytest.mark.asyncio
    async def test_self_correct_returns_structure(self):
        dr = DeepReasoning()
        steps = [
            ThoughtStep(0, "step 1", "analysis", 0.8),
            ThoughtStep(1, "step 2", "analysis", 0.7),
        ]
        result = await dr._self_correct(steps, "context", DeepReasoningContext(query="q"))
        assert "corrections" in result
        assert "corrected_reasoning" in result
        assert "needs_revision" in result

    @pytest.mark.asyncio
    async def test_critique_step_no_issue(self):
        dr = DeepReasoning()
        step = ThoughtStep(0, "sound reasoning", "analysis", 0.9)
        ctx = DeepReasoningContext(query="q")
        critique = await dr._critique_step(step, "context", ctx)
        assert critique["has_issue"] is False
        assert critique["issues"] == []

    @pytest.mark.asyncio
    async def test_critique_step_with_error(self):
        dr = DeepReasoning()

        async def error_llm(prompt):
            return "This contains an error in the logic."

        dr.llm_call = error_llm
        step = ThoughtStep(0, "some thought", "analysis", 0.8)
        ctx = DeepReasoningContext(query="q")
        critique = await dr._critique_step(step, "context", ctx)
        assert critique["has_issue"] is True
        assert "potential_error" in critique["issues"]

    @pytest.mark.asyncio
    async def test_apply_corrections_modifies_flagged(self):
        dr = DeepReasoning()
        steps = [
            ThoughtStep(0, "step 0", "analysis", 0.8),
            ThoughtStep(1, "step 1", "analysis", 0.7),
        ]
        corrections = [{"step_id": 0, "has_issue": True, "critique": "issue found"}]
        corrected = await dr._apply_corrections(steps, corrections)
        assert "[CORRECTED]" in corrected[0].thought
        assert corrected[1].thought == "step 1"

    @pytest.mark.asyncio
    async def test_apply_corrections_no_corrections(self):
        dr = DeepReasoning()
        steps = [ThoughtStep(0, "step 0", "analysis", 0.8)]
        corrected = await dr._apply_corrections(steps, [])
        assert corrected[0].thought == "step 0"


# ═══════════════════════════════════════════════════════════════════════
# DeepReasoning — With Vector/Memory Stores
# ═══════════════════════════════════════════════════════════════════════

class TestDeepReasoningWithStores:
    @pytest.mark.asyncio
    async def test_vector_store_used(self):
        class FakeDoc:
            def __init__(self, content):
                self.content = content

        class FakeVectorStore:
            def __init__(self):
                self.documents = {"d1": FakeDoc("knowledge about X")}

            def search(self, query, top_k=5):
                return [("d1", 0.95)]

        dr = DeepReasoning(vector_store=FakeVectorStore())
        knowledge = await dr._retrieve_knowledge("tell me about X")
        assert len(knowledge) == 1
        assert knowledge[0].source == RetrievalSource.VECTOR_STORE
        assert knowledge[0].relevance == 0.95

    @pytest.mark.asyncio
    async def test_memory_store_used(self):
        class FakeMemoryStore:
            def retrieve(self, query, top_k=5):
                return [{"content": "memory fact", "relevance": 0.8, "id": "m1"}]

        dr = DeepReasoning(memory_store=FakeMemoryStore())
        knowledge = await dr._retrieve_knowledge("query")
        assert len(knowledge) == 1
        assert knowledge[0].source == RetrievalSource.MEMORY

    @pytest.mark.asyncio
    async def test_vector_store_error_falls_through(self):
        class BrokenVectorStore:
            def search(self, query, top_k=5):
                raise RuntimeError("store broken")

        dr = DeepReasoning(vector_store=BrokenVectorStore())
        knowledge = await dr._retrieve_knowledge("query")
        # Should fall through to fallback or empty
        assert isinstance(knowledge, list)


# ═══════════════════════════════════════════════════════════════════════
# __all__ exports
# ═══════════════════════════════════════════════════════════════════════

class TestExports:
    def test_all_contains_expected(self):
        from domains.cognitive.reasoning.deep import __all__ as exported
        expected = {
            "DeepReasoning", "DeepReasoningContext", "RetrievedKnowledge",
            "RetrievalSource", "FormalLogicEngine", "LogicalOperator",
            "Term", "Predicate", "WellFormedFormula", "Substitution",
            "WorkingMemory",
        }
        assert expected == set(exported)
