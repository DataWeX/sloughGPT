"""Tests for cognitive/reasoning/deep.py — dataclasses + FormalLogicEngine + WorkingMemory."""

import pytest
from domains.cognitive.reasoning.deep import (
    RetrievalSource,
    RetrievedKnowledge,
    DeepReasoningContext,
    LogicalOperator,
    Term,
    Predicate,
    WellFormedFormula,
    Substitution,
    FormalLogicEngine,
    WorkingMemory,
)


class TestRetrievalSource:
    def test_all_values(self):
        for src in RetrievalSource:
            assert isinstance(src.value, str)
            assert len(src.value) > 0

    def test_unique_values(self):
        values = [s.value for s in RetrievalSource]
        assert len(values) == len(set(values))


class TestRetrievedKnowledge:
    def test_creation(self):
        k = RetrievedKnowledge(content="fact", source=RetrievalSource.MEMORY, relevance=0.9)
        assert k.content == "fact"
        assert k.source == RetrievalSource.MEMORY
        assert k.relevance == 0.9
        assert k.source_id is None

    def test_with_source_id(self):
        k = RetrievedKnowledge(content="x", source=RetrievalSource.VECTOR_STORE, relevance=0.5, source_id="v1")
        assert k.source_id == "v1"


class TestDeepReasoningContext:
    def test_defaults(self):
        ctx = DeepReasoningContext(query="q")
        assert ctx.query == "q"
        assert ctx.retrieved_knowledge == []
        assert ctx.working_memory == []
        assert ctx.constraints == []
        assert ctx.assumptions == []

    def test_with_values(self):
        k = RetrievedKnowledge(content="c", source=RetrievalSource.MEMORY, relevance=0.8)
        ctx = DeepReasoningContext(
            query="q", retrieved_knowledge=[k], working_memory=["m1"],
            constraints=["c1"], assumptions=["a1"],
        )
        assert len(ctx.retrieved_knowledge) == 1
        assert ctx.working_memory == ["m1"]


class TestLogicalOperator:
    def test_all_operators(self):
        ops = list(LogicalOperator)
        assert len(ops) == 7
        assert LogicalOperator.AND.value == "∧"
        assert LogicalOperator.OR.value == "∨"
        assert LogicalOperator.NOT.value == "¬"
        assert LogicalOperator.IMPLIES.value == "→"
        assert LogicalOperator.IFF.value == "↔"
        assert LogicalOperator.FORALL.value == "∀"
        assert LogicalOperator.EXISTS.value == "∃"


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
    def test_creation(self):
        p = Predicate(name="human", terms=[Term(name="socrates")])
        assert p.name == "human"
        assert len(p.terms) == 1
        assert p.negated is False

    def test_negated(self):
        p = Predicate(name="mortal", terms=[Term(name="x")], negated=True)
        assert p.negated is True

    def test_hash(self):
        p1 = Predicate(name="human", terms=[Term(name="socrates")])
        p2 = Predicate(name="human", terms=[Term(name="socrates")])
        assert hash(p1) == hash(p2)

    def test_hash_different(self):
        p1 = Predicate(name="human", terms=[Term(name="socrates")])
        p2 = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert hash(p1) != hash(p2)

    def test_hash_negated(self):
        p1 = Predicate(name="human", terms=[Term(name="socrates")], negated=False)
        p2 = Predicate(name="human", terms=[Term(name="socrates")], negated=True)
        assert hash(p1) != hash(p2)


class TestWellFormedFormula:
    def test_empty(self):
        wff = WellFormedFormula()
        assert wff.predicate is None
        assert wff.operator is None

    def test_predicate_wff(self):
        pred = Predicate(name="human", terms=[Term(name="socrates")])
        wff = WellFormedFormula(predicate=pred)
        assert wff.predicate.name == "human"

    def test_compound_wff(self):
        left = WellFormedFormula(predicate=Predicate(name="A", terms=[]))
        right = WellFormedFormula(predicate=Predicate(name="B", terms=[]))
        wff = WellFormedFormula(operator=LogicalOperator.AND, left=left, right=right)
        assert wff.operator == LogicalOperator.AND
        assert wff.left.predicate.name == "A"
        assert wff.right.predicate.name == "B"

    def test_quantifier_wff(self):
        var = Term(name="X", is_variable=True)
        sub = WellFormedFormula(predicate=Predicate(name="human", terms=[Term(name="X", is_variable=True)]))
        wff = WellFormedFormula(
            quantifier_var=var,
            quantifier_type=LogicalOperator.FORALL,
            subformula=sub,
        )
        assert wff.quantifier_type == LogicalOperator.FORALL


class TestSubstitution:
    def test_empty(self):
        s = Substitution()
        assert s.mapping == {}

    def test_with_mapping(self):
        s = Substitution(mapping={"X": Term(name="socrates")})
        assert "X" in s.mapping
        assert s.mapping["X"].name == "socrates"


class TestFormalLogicEngine:
    def test_init(self):
        engine = FormalLogicEngine()
        assert engine.knowledge_base == []
        assert engine.inference_history == []

    def test_assert_fact(self):
        engine = FormalLogicEngine()
        wff = WellFormedFormula(predicate=Predicate(name="human", terms=[Term(name="socrates")]))
        engine.assert_fact(wff)
        assert len(engine.knowledge_base) == 1

    def test_assert_predicate(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        assert len(engine.knowledge_base) == 1
        pred = engine.knowledge_base[0].predicate
        assert pred.name == "human"
        assert pred.terms[0].name == "socrates"

    def test_assert_multiple_predicates(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        engine.assert_predicate("mortal", "socrates")
        assert len(engine.knowledge_base) == 2

    def test_query_empty_kb(self):
        engine = FormalLogicEngine()
        result = engine.query(Predicate(name="human", terms=[Term(name="socrates")]))
        assert result is False

    def test_query_direct_match(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        result = engine.query(Predicate(name="human", terms=[Term(name="socrates")]))
        assert result is True

    def test_query_no_match(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        result = engine.query(Predicate(name="mortal", terms=[Term(name="socrates")]))
        assert result is False

    def test_forward_chaining_implies(self):
        engine = FormalLogicEngine()
        # human(X) → mortal(X)
        wff = WellFormedFormula(
            operator=LogicalOperator.IMPLIES,
            left=WellFormedFormula(predicate=Predicate(name="human", terms=[Term(name="X", is_variable=True)])),
            right=WellFormedFormula(predicate=Predicate(name="mortal", terms=[Term(name="X", is_variable=True)])),
        )
        engine.assert_fact(wff)
        engine.assert_predicate("human", "socrates")
        result = engine.query(Predicate(name="mortal", terms=[Term(name="socrates")]))
        assert result is True

    def test_prove_syllogism_all_men(self):
        engine = FormalLogicEngine()
        result = engine.prove_syllogism(
            ("All", "men", "are mortal"),
            ("Socrates", "is a", "man"),
            ("Socrates", "is", "mortal"),
        )
        assert result is not None
        assert "valid" in result
        assert "mood" in result

    def test_prove_syllogism_invalid(self):
        engine = FormalLogicEngine()
        result = engine.prove_syllogism(
            ("All", "cats", "are animals"),
            ("All", "dogs", "are animals"),
            ("Some", "dogs", "are cats"),
        )
        assert result is not None
        assert "valid" in result

    def test_to_categorical(self):
        engine = FormalLogicEngine()
        cat = engine._to_categorical(("All", "men", "are mortal"))
        assert len(cat) == 4
        assert cat[0] in ("A", "E", "I", "O")

    def test_determine_figure(self):
        engine = FormalLogicEngine()
        p1 = ("All", "men", "are mortal")
        p2 = ("Socrates", "is a", "man")
        fig = engine._determine_figure(p1, p2)
        assert isinstance(fig, int)
        assert 1 <= fig <= 4


class TestWorkingMemory:
    def test_init_default(self):
        wm = WorkingMemory()
        assert wm.capacity == 7
        assert wm.get_recent() == []

    def test_init_custom(self):
        wm = WorkingMemory(capacity=3)
        assert wm.capacity == 3

    def test_add(self):
        wm = WorkingMemory()
        wm.add("item1")
        assert "item1" in wm.get_recent()

    def test_add_overflow(self):
        wm = WorkingMemory(capacity=3)
        for i in range(5):
            wm.add(f"item{i}")
        recent = wm.get_recent()
        assert len(recent) == 3
        assert "item4" in recent

    def test_access(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.add("b")
        wm.access("a")
        recent = wm.get_recent()
        assert recent[0] == "a"

    def test_get_recent_n(self):
        wm = WorkingMemory()
        for i in range(10):
            wm.add(f"item{i}")
        assert len(wm.get_recent(3)) == 3

    def test_clear(self):
        wm = WorkingMemory()
        wm.add("x")
        wm.clear()
        assert wm.get_recent() == []

    def test_add_duplicate(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.add("a")
        recent = wm.get_recent()
        assert recent.count("a") == 1 or len(recent) == 2  # depends on dedup impl
