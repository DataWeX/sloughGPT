"""Tests for domains.cognitive.reasoning.deep — FormalLogicEngine, dataclasses."""

from domains.cognitive.reasoning.deep import (
    LogicalOperator, Term, Predicate, WellFormedFormula,
    Substitution, FormalLogicEngine, RetrievalSource,
)


class TestLogicalOperator:
    def test_all_members(self):
        assert len(LogicalOperator) == 7
    def test_values(self):
        assert LogicalOperator.AND.value == "∧"
        assert LogicalOperator.IMPLIES.value == "→"


class TestTerm:
    def test_constant(self):
        t = Term(name="socrates")
        assert t.is_variable is False
        assert t.is_function is False

    def test_variable(self):
        t = Term(name="X", is_variable=True)
        assert t.is_variable is True

    def test_function(self):
        f = Term(name="father", is_function=True, arguments=[Term(name="john")])
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


class TestWellFormedFormula:
    def test_simple(self):
        pred = Predicate(name="human", terms=[Term(name="socrates")])
        wff = WellFormedFormula(predicate=pred)
        assert wff.predicate.name == "human"
        assert wff.operator is None

    def test_implication(self):
        left = WellFormedFormula(predicate=Predicate(name="human", terms=[Term(name="X")], negated=False))
        right = WellFormedFormula(predicate=Predicate(name="mortal", terms=[Term(name="X")], negated=False))
        wff = WellFormedFormula(
            operator=LogicalOperator.IMPLIES,
            left=left,
            right=right,
        )
        assert wff.operator == LogicalOperator.IMPLIES


class TestSubstitution:
    def test_mapping(self):
        s = Substitution(mapping={"X": Term(name="socrates")})
        assert "X" in s.mapping


class TestFormalLogicEngine:
    def test_assert_and_query(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        q = Predicate(name="human", terms=[Term(name="socrates")])
        assert engine.query(q) is True

    def test_query_not_asserted(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        q = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert engine.query(q) is False

    def test_modus_ponens(self):
        engine = FormalLogicEngine()
        # human(X) → mortal(X)
        antecedent = WellFormedFormula(
            predicate=Predicate(name="human", terms=[Term(name="X", is_variable=True)])
        )
        consequent = WellFormedFormula(
            predicate=Predicate(name="mortal", terms=[Term(name="X", is_variable=True)])
        )
        rule = WellFormedFormula(
            operator=LogicalOperator.IMPLIES,
            left=antecedent,
            right=consequent,
        )
        engine.assert_fact(rule)
        engine.assert_predicate("human", "socrates")
        q = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert engine.query(q) is True


class TestRetrievalSource:
    def test_all_members(self):
        assert len(RetrievalSource) >= 3
