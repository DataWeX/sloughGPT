"""Tests for domains.cognitive.reasoning.deep — LogicalOperator; domains.cognitive.reasoning.advanced — ReasoningMode, ThoughtStep."""

import pytest
from domains.cognitive.reasoning.deep import (
    LogicalOperator,
    Term,
    Predicate,
    WellFormedFormula,
    Substitution,
    RetrievalSource,
    RetrievedKnowledge,
    DeepReasoningContext,
    WorkingMemory,
    FormalLogicEngine,
)
from domains.cognitive.reasoning.advanced import (
    ReasoningMode,
    ThoughtStep,
    ReasoningResult,
)


# ── LogicalOperator ──────────────────────────────────────────────────────


class TestLogicalOperator:
    def test_all_members(self):
        assert len(LogicalOperator) == 7

    def test_values(self):
        assert LogicalOperator.AND.value == "∧"
        assert LogicalOperator.OR.value == "∨"
        assert LogicalOperator.NOT.value == "¬"
        assert LogicalOperator.IMPLIES.value == "→"
        assert LogicalOperator.FORALL.value == "∀"
        assert LogicalOperator.EXISTS.value == "∃"

    def test_iff_value(self):
        assert LogicalOperator.IFF.value == "↔"

    def test_members_are_enums(self):
        for m in LogicalOperator:
            assert isinstance(m, LogicalOperator)

    def test_member_names(self):
        names = {m.name for m in LogicalOperator}
        assert names == {"AND", "OR", "NOT", "IMPLIES", "IFF", "FORALL", "EXISTS"}

    def test_member_values_are_strings(self):
        for m in LogicalOperator:
            assert isinstance(m.value, str)

    def test_member_values_unique(self):
        values = [m.value for m in LogicalOperator]
        assert len(values) == len(set(values))

    def test_membership(self):
        assert LogicalOperator.AND in LogicalOperator
        assert LogicalOperator.XOR not in LogicalOperator if hasattr(LogicalOperator, 'XOR') else True

    def test_iteration(self):
        count = 0
        for m in LogicalOperator:
            count += 1
        assert count == 7

    def test_value_lookup(self):
        assert LogicalOperator("∧") is LogicalOperator.AND
        assert LogicalOperator("∨") is LogicalOperator.OR

    def test_name_lookup(self):
        assert LogicalOperator["AND"] is LogicalOperator.AND

    def test_eq_self(self):
        assert LogicalOperator.AND == LogicalOperator.AND

    def test_neq_different(self):
        assert LogicalOperator.AND != LogicalOperator.OR

    def test_hash_consistent(self):
        assert hash(LogicalOperator.AND) == hash(LogicalOperator.AND)

    def test_hash_different(self):
        assert hash(LogicalOperator.AND) != hash(LogicalOperator.OR)


# ── ReasoningMode ────────────────────────────────────────────────────────


class TestReasoningMode:
    def test_all_members(self):
        assert len(ReasoningMode) == 8

    def test_values(self):
        assert ReasoningMode.CHAIN_OF_THOUGHT.value == "chain_of_thought"
        assert ReasoningMode.TREE_OF_THOUGHTS.value == "tree_of_thoughts"
        assert ReasoningMode.REACT.value == "react"

    def test_self_consistency_value(self):
        assert ReasoningMode.SELF_CONSISTENCY.value == "self_consistency"

    def test_constitutional_value(self):
        assert ReasoningMode.CONSTITUTIONAL.value == "constitutional"

    def test_causal_value(self):
        assert ReasoningMode.CAUSAL.value == "causal"

    def test_counterfactual_value(self):
        assert ReasoningMode.COUNTERFACTUAL.value == "counterfactual"

    def test_syllogism_value(self):
        assert ReasoningMode.SYLLOGISM.value == "syllogism"

    def test_member_names(self):
        names = {m.name for m in ReasoningMode}
        assert names == {
            "CHAIN_OF_THOUGHT", "TREE_OF_THOUGHTS", "SELF_CONSISTENCY",
            "CONSTITUTIONAL", "REACT", "CAUSAL", "COUNTERFACTUAL", "SYLLOGISM",
        }

    def test_member_values_unique(self):
        values = [m.value for m in ReasoningMode]
        assert len(values) == len(set(values))

    def test_value_lookup(self):
        assert ReasoningMode("chain_of_thought") is ReasoningMode.CHAIN_OF_THOUGHT
        assert ReasoningMode("react") is ReasoningMode.REACT

    def test_name_lookup(self):
        assert ReasoningMode["CHAIN_OF_THOUGHT"] is ReasoningMode.CHAIN_OF_THOUGHT

    def test_eq_self(self):
        assert ReasoningMode.REACT == ReasoningMode.REACT

    def test_neq_different(self):
        assert ReasoningMode.REACT != ReasoningMode.CAUSAL

    def test_hash_consistent(self):
        assert hash(ReasoningMode.REACT) == hash(ReasoningMode.REACT)

    def test_hash_different(self):
        assert hash(ReasoningMode.REACT) != hash(ReasoningMode.CAUSAL)

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ReasoningMode("nonexistent")

    def test_invalid_name_raises(self):
        with pytest.raises(KeyError):
            _ = ReasoningMode["NONEXISTENT"]


# ── ThoughtStep ──────────────────────────────────────────────────────────


class TestThoughtStep:
    def test_fields(self):
        ts = ThoughtStep(step_id=1, thought="test", reasoning_type="deduction", confidence=0.9)
        assert ts.step_id == 1
        assert ts.thought == "test"
        assert ts.confidence == 0.9
        assert ts.parent_id is None
        assert ts.children_ids == []

    def test_defaults(self):
        ts = ThoughtStep(step_id=0, thought="", reasoning_type="", confidence=0.0)
        assert ts.value == 0.0
        assert ts.is_final is False

    def test_children_ids_list(self):
        ts = ThoughtStep(step_id=0, thought="a", reasoning_type="r", confidence=1.0, children_ids=[1, 2])
        assert ts.children_ids == [1, 2]

    def test_parent_id(self):
        ts = ThoughtStep(step_id=1, thought="child", reasoning_type="r", confidence=0.8, parent_id=0)
        assert ts.parent_id == 0

    def test_value_assignment(self):
        ts = ThoughtStep(step_id=0, thought="", reasoning_type="", confidence=0.0, value=3.5)
        assert ts.value == 3.5

    def test_is_final_flag(self):
        ts = ThoughtStep(step_id=0, thought="", reasoning_type="", confidence=0.0, is_final=True)
        assert ts.is_final is True

    def test_immutability_of_defaults(self):
        ts1 = ThoughtStep(step_id=0, thought="", reasoning_type="", confidence=0.0)
        ts2 = ThoughtStep(step_id=0, thought="", reasoning_type="", confidence=0.0)
        assert ts1.children_ids is not ts2.children_ids

    def test_equality(self):
        ts1 = ThoughtStep(step_id=1, thought="x", reasoning_type="r", confidence=0.9)
        ts2 = ThoughtStep(step_id=1, thought="x", reasoning_type="r", confidence=0.9)
        assert ts1 == ts2

    def test_inequality(self):
        ts1 = ThoughtStep(step_id=1, thought="x", reasoning_type="r", confidence=0.9)
        ts2 = ThoughtStep(step_id=2, thought="y", reasoning_type="r", confidence=0.9)
        assert ts1 != ts2

    def test_repr(self):
        ts = ThoughtStep(step_id=1, thought="hello", reasoning_type="deduction", confidence=0.85)
        r = repr(ts)
        assert "step_id" in r
        assert "thought" in r

    def test_many_children(self):
        children = list(range(20))
        ts = ThoughtStep(step_id=0, thought="", reasoning_type="", confidence=0.0, children_ids=children)
        assert len(ts.children_ids) == 20
        assert ts.children_ids == children

    def test_high_confidence(self):
        ts = ThoughtStep(step_id=0, thought="", reasoning_type="", confidence=1.0)
        assert ts.confidence == 1.0

    def test_zero_confidence(self):
        ts = ThoughtStep(step_id=0, thought="", reasoning_type="", confidence=0.0)
        assert ts.confidence == 0.0

    def test_negative_step_id(self):
        ts = ThoughtStep(step_id=-1, thought="", reasoning_type="", confidence=0.0)
        assert ts.step_id == -1


# ── ReasoningResult ──────────────────────────────────────────────────────


class TestReasoningResult:
    def test_creation(self):
        ts = ThoughtStep(step_id=0, thought="a", reasoning_type="r", confidence=0.9)
        rr = ReasoningResult(
            conclusion="yes", confidence=0.8, mode=ReasoningMode.CHAIN_OF_THOUGHT,
            steps=[ts], metadata={"k": "v"}, execution_time_ms=100.0,
        )
        assert rr.conclusion == "yes"
        assert rr.confidence == 0.8
        assert rr.mode == ReasoningMode.CHAIN_OF_THOUGHT
        assert rr.execution_time_ms == 100.0

    def test_metadata_dict(self):
        rr = ReasoningResult(
            conclusion="", confidence=0.0, mode=ReasoningMode.REACT,
            steps=[], metadata={"a": 1, "b": [2, 3]}, execution_time_ms=0.0,
        )
        assert rr.metadata == {"a": 1, "b": [2, 3]}

    def test_empty_steps(self):
        rr = ReasoningResult(
            conclusion="", confidence=0.0, mode=ReasoningMode.REACT,
            steps=[], metadata={}, execution_time_ms=0.0,
        )
        assert rr.steps == []

    def test_multiple_steps(self):
        steps = [
            ThoughtStep(step_id=i, thought=f"s{i}", reasoning_type="r", confidence=0.5)
            for i in range(5)
        ]
        rr = ReasoningResult(
            conclusion="c", confidence=0.7, mode=ReasoningMode.TREE_OF_THOUGHTS,
            steps=steps, metadata={}, execution_time_ms=50.0,
        )
        assert len(rr.steps) == 5


# ── Term ─────────────────────────────────────────────────────────────────


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
        f = Term(name="f", is_function=True, arguments=[Term(name="a"), Term(name="b")])
        assert f.is_function is True
        assert len(f.arguments) == 2

    def test_nested_function(self):
        inner = Term(name="g", is_function=True, arguments=[Term(name="x")])
        outer = Term(name="f", is_function=True, arguments=[inner])
        assert outer.arguments[0].is_function is True
        assert outer.arguments[0].name == "g"


# ── Predicate ────────────────────────────────────────────────────────────


class TestPredicate:
    def test_basic(self):
        p = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert p.name == "mortal"
        assert len(p.terms) == 1
        assert p.negated is False

    def test_negated(self):
        p = Predicate(name="mortal", terms=[Term(name="socrates")], negated=True)
        assert p.negated is True

    def test_hash(self):
        p1 = Predicate(name="mortal", terms=[Term(name="socrates")])
        p2 = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert hash(p1) == hash(p2)

    def test_hash_different(self):
        p1 = Predicate(name="mortal", terms=[Term(name="socrates")])
        p2 = Predicate(name="human", terms=[Term(name="socrates")])
        assert hash(p1) != hash(p2)

    def test_hash_negated_differs(self):
        p1 = Predicate(name="mortal", terms=[Term(name="socrates")], negated=False)
        p2 = Predicate(name="mortal", terms=[Term(name="socrates")], negated=True)
        assert hash(p1) != hash(p2)

    def test_multiple_terms(self):
        p = Predicate(name="loves", terms=[Term(name="a"), Term(name="b")])
        assert len(p.terms) == 2


# ── WellFormedFormula ────────────────────────────────────────────────────


class TestWellFormedFormula:
    def test_simple_predicate(self):
        p = Predicate(name="human", terms=[Term(name="socrates")])
        wff = WellFormedFormula(predicate=p)
        assert wff.predicate == p
        assert wff.operator is None

    def test_compound(self):
        left = WellFormedFormula(predicate=Predicate(name="A", terms=[]))
        right = WellFormedFormula(predicate=Predicate(name="B", terms=[]))
        wff = WellFormedFormula(operator=LogicalOperator.AND, left=left, right=right)
        assert wff.operator == LogicalOperator.AND
        assert wff.left == left
        assert wff.right == right

    def test_quantified(self):
        var = Term(name="X", is_variable=True)
        sub = WellFormedFormula(predicate=Predicate(name="human", terms=[var]))
        wff = WellFormedFormula(
            quantifier_var=var,
            quantifier_type=LogicalOperator.FORALL,
            subformula=sub,
        )
        assert wff.quantifier_type == LogicalOperator.FORALL


# ── Substitution ─────────────────────────────────────────────────────────


class TestSubstitution:
    def test_empty(self):
        s = Substitution()
        assert s.mapping == {}

    def test_with_mapping(self):
        s = Substitution(mapping={"X": Term(name="socrates")})
        assert "X" in s.mapping
        assert s.mapping["X"].name == "socrates"


# ── RetrievalSource & RetrievedKnowledge ─────────────────────────────────


class TestRetrievalSource:
    def test_all_members(self):
        members = {m.name for m in RetrievalSource}
        assert "VECTOR_STORE" in members
        assert "MEMORY" in members
        assert "KNOWLEDGE_GRAPH" in members
        assert "WORKING_MEMORY" in members

    def test_values(self):
        assert RetrievalSource.VECTOR_STORE.value == "vector_store"
        assert RetrievalSource.MEMORY.value == "memory"
        assert RetrievalSource.KNOWLEDGE_GRAPH.value == "knowledge_graph"
        assert RetrievalSource.WORKING_MEMORY.value == "working_memory"


class TestRetrievedKnowledge:
    def test_basic(self):
        rk = RetrievedKnowledge(
            content="fact", source=RetrievalSource.VECTOR_STORE, relevance=0.95,
        )
        assert rk.content == "fact"
        assert rk.relevance == 0.95
        assert rk.source_id is None

    def test_with_source_id(self):
        rk = RetrievedKnowledge(
            content="x", source=RetrievalSource.MEMORY, relevance=0.5, source_id="doc1",
        )
        assert rk.source_id == "doc1"


# ── DeepReasoningContext ─────────────────────────────────────────────────


class TestDeepReasoningContext:
    def test_defaults(self):
        ctx = DeepReasoningContext(query="why?")
        assert ctx.query == "why?"
        assert ctx.retrieved_knowledge == []
        assert ctx.working_memory == []
        assert ctx.constraints == []
        assert ctx.assumptions == []

    def test_with_data(self):
        rk = RetrievedKnowledge(content="c", source=RetrievalSource.MEMORY, relevance=0.5)
        ctx = DeepReasoningContext(
            query="q",
            retrieved_knowledge=[rk],
            working_memory=["a"],
            constraints=["c1"],
            assumptions=["a1"],
        )
        assert len(ctx.retrieved_knowledge) == 1
        assert ctx.working_memory == ["a"]
        assert ctx.constraints == ["c1"]
        assert ctx.assumptions == ["a1"]


# ── WorkingMemory ────────────────────────────────────────────────────────


class TestWorkingMemory:
    def test_add_within_capacity(self):
        wm = WorkingMemory(capacity=3)
        wm.add("a")
        wm.add("b")
        assert wm.items == ["a", "b"]

    def test_eviction(self):
        wm = WorkingMemory(capacity=2)
        wm.add("a")
        wm.add("b")
        wm.add("c")
        assert len(wm.items) == 2
        assert "a" not in wm.items

    def test_access_increments_count(self):
        wm = WorkingMemory(capacity=5)
        wm.add("a")
        wm.access("a")
        wm.access("a")
        assert wm.access_count["a"] == 3

    def test_get_recent(self):
        wm = WorkingMemory(capacity=5)
        wm.add("x")
        wm.add("y")
        wm.access("x")
        recent = wm.get_recent(2)
        assert recent[0] == "x"

    def test_clear(self):
        wm = WorkingMemory(capacity=5)
        wm.add("a")
        wm.add("b")
        wm.clear()
        assert wm.items == []
        assert wm.access_count == {}

    def test_get_recent_n_exceeds_items(self):
        wm = WorkingMemory(capacity=10)
        wm.add("only")
        recent = wm.get_recent(5)
        assert recent == ["only"]

    def test_eviction_uses_lru(self):
        wm = WorkingMemory(capacity=3)
        wm.add("a")
        wm.add("b")
        wm.add("c")
        wm.access("a")
        wm.add("d")
        assert "b" not in wm.items
        assert "a" in wm.items

    def test_custom_capacity(self):
        wm = WorkingMemory(capacity=1)
        wm.add("x")
        wm.add("y")
        assert wm.items == ["y"]


# ── FormalLogicEngine ────────────────────────────────────────────────────


class TestFormalLogicEngine:
    def test_init(self):
        e = FormalLogicEngine()
        assert e.knowledge_base == []
        assert e.inference_history == []

    def test_assert_fact(self):
        e = FormalLogicEngine()
        wff = WellFormedFormula(predicate=Predicate(name="human", terms=[Term(name="socrates")]))
        e.assert_fact(wff)
        assert len(e.knowledge_base) == 1

    def test_assert_predicate(self):
        e = FormalLogicEngine()
        e.assert_predicate("mortal", "socrates")
        assert len(e.knowledge_base) == 1
        pred = e.knowledge_base[0].predicate
        assert pred.name == "mortal"
        assert pred.terms[0].name == "socrates"

    def test_query_simple(self):
        e = FormalLogicEngine()
        e.assert_predicate("human", "socrates")
        q = Predicate(name="human", terms=[Term(name="socrates")])
        assert e.query(q) is True

    def test_query_not_found(self):
        e = FormalLogicEngine()
        e.assert_predicate("human", "socrates")
        q = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert e.query(q) is False

    def test_modus_ponens(self):
        e = FormalLogicEngine()
        human = WellFormedFormula(predicate=Predicate(name="human", terms=[Term(name="socrates")]))
        mortal = WellFormedFormula(predicate=Predicate(name="mortal", terms=[Term(name="socrates")]))
        implies = WellFormedFormula(
            operator=LogicalOperator.IMPLIES,
            left=WellFormedFormula(predicate=Predicate(name="human", terms=[Term(name="socrates")])),
            right=mortal,
        )
        e.assert_fact(human)
        e.assert_fact(implies)
        q = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert e.query(q) is True

    def test_unify_same(self):
        e = FormalLogicEngine()
        p1 = Predicate(name="human", terms=[Term(name="socrates")])
        p2 = Predicate(name="human", terms=[Term(name="socrates")])
        subst = e._unify(p1, p2)
        assert subst is not None

    def test_unify_different_name(self):
        e = FormalLogicEngine()
        p1 = Predicate(name="human", terms=[Term(name="socrates")])
        p2 = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert e._unify(p1, p2) is None

    def test_unify_variable(self):
        e = FormalLogicEngine()
        p1 = Predicate(name="human", terms=[Term(name="X", is_variable=True)])
        p2 = Predicate(name="human", terms=[Term(name="socrates")])
        subst = e._unify(p1, p2)
        assert subst is not None
        assert "X" in subst.mapping

    def test_resolution_entailed(self):
        e = FormalLogicEngine()
        e.assert_predicate("human", "socrates")
        e.assert_fact(WellFormedFormula(
            operator=LogicalOperator.IMPLIES,
            left=WellFormedFormula(predicate=Predicate(name="human", terms=[Term(name="socrates")])),
            right=WellFormedFormula(predicate=Predicate(name="mortal", terms=[Term(name="socrates")])),
        ))
        goal = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert e.resolution(goal) is True

    def test_resolution_not_entailed(self):
        e = FormalLogicEngine()
        e.assert_predicate("human", "socrates")
        goal = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert e.resolution(goal) is False

    def test_prove_syllogism(self):
        e = FormalLogicEngine()
        result = e.prove_syllogism(
            premise1=("All", "are", "mortal"),
            premise2=("All", "are", "human"),
            conclusion=("All", "are", "mortal"),
        )
        assert "valid" in result
        assert "figure" in result
        assert "mood" in result

    def test_inference_history_appended(self):
        e = FormalLogicEngine()
        e.prove_syllogism(("All", "are", "mortal"), ("All", "are", "human"), ("All", "are", "mortal"))
        assert len(e.inference_history) == 1

    def test_to_categorical(self):
        e = FormalLogicEngine()
        result = e._to_categorical(("All", "are", "mortal"))
        assert result[0] == "A"
        assert result[2] == "are"
        assert result[3] == "mortal"

    def test_check_valid_mood(self):
        e = FormalLogicEngine()
        valid, reason = e._check_syllogism_validity("AAA", 1, ("All", "S", "are", "P"), ("All", "S", "are", "M"), ("All", "S", "are", "P"))
        assert valid is True

    def test_check_invalid_mood(self):
        e = FormalLogicEngine()
        valid, reason = e._check_syllogism_validity("OOO", 1, ("All", "S", "are", "P"), ("All", "S", "are", "M"), ("All", "S", "are", "P"))
        assert valid is False

    def test_format_categorical(self):
        e = FormalLogicEngine()
        result = e._format_categorical(("All", "S", "are", "mortal"))
        assert result == "All S are mortal"

    def test_multiple_assertions(self):
        e = FormalLogicEngine()
        e.assert_predicate("a", "x")
        e.assert_predicate("b", "y")
        e.assert_predicate("c", "z")
        assert len(e.knowledge_base) == 3
        q = Predicate(name="a", terms=[Term(name="x")])
        assert e.query(q) is True

    def test_occurrs_check(self):
        e = FormalLogicEngine()
        var = Term(name="X", is_variable=True)
        t = Term(name="X")
        subst = Substitution()
        assert e._occurs_check(var, t, subst) is True

    def test_apply_substitution(self):
        e = FormalLogicEngine()
        subst = Substitution(mapping={"X": Term(name="socrates")})
        pred = Predicate(name="human", terms=[Term(name="X", is_variable=True)])
        result = e._apply_substitution(pred, subst)
        assert result.terms[0].name == "socrates"

    def test_empty_kb_query(self):
        e = FormalLogicEngine()
        q = Predicate(name="anything", terms=[Term(name="x")])
        assert e.query(q) is False
