"""Tests for domains.cognitive.reasoning.deep — RetrievalSource, RetrievedKnowledge, DeepReasoningContext, Term, Predicate, WellFormedFormula, Substitution."""

from domains.cognitive.reasoning.deep import (
    RetrievalSource, RetrievedKnowledge, DeepReasoningContext,
    Term, Predicate, WellFormedFormula, Substitution, LogicalOperator,
)


class TestRetrievalSource:
    def test_all_members(self):
        assert len(RetrievalSource) == 4
    def test_values(self):
        assert RetrievalSource.VECTOR_STORE.value == "vector_store"
        assert RetrievalSource.KNOWLEDGE_GRAPH.value == "knowledge_graph"


class TestRetrievedKnowledge:
    def test_fields(self):
        rk = RetrievedKnowledge(content="fact", source=RetrievalSource.MEMORY, relevance=0.9)
        assert rk.content == "fact"
        assert rk.relevance == 0.9


class TestDeepReasoningContext:
    def test_defaults(self):
        ctx = DeepReasoningContext(query="why?")
        assert ctx.query == "why?"
        assert ctx.retrieved_knowledge == []
        assert ctx.constraints == []


class TestTerm:
    def test_constant(self):
        t = Term(name="cat")
        assert t.name == "cat"
        assert t.is_variable is False
    def test_variable(self):
        t = Term(name="X", is_variable=True)
        assert t.is_variable is True
    def test_function(self):
        t = Term(name="f", is_function=True, arguments=[Term(name="a"), Term(name="b")])
        assert t.is_function is True
        assert len(t.arguments) == 2


class TestPredicate:
    def test_fields(self):
        p = Predicate(name="likes", terms=[Term(name="cat"), Term(name="fish")])
        assert p.name == "likes"
        assert len(p.terms) == 2
        assert p.negated is False
    def test_hash(self):
        p1 = Predicate(name="likes", terms=[Term(name="cat")])
        p2 = Predicate(name="likes", terms=[Term(name="cat")])
        assert hash(p1) == hash(p2)


class TestWellFormedFormula:
    def test_simple(self):
        pred = Predicate(name="true", terms=[])
        wff = WellFormedFormula(predicate=pred)
        assert wff.predicate is pred
    def test_conjunction(self):
        left = WellFormedFormula(predicate=Predicate(name="a", terms=[]))
        right = WellFormedFormula(predicate=Predicate(name="b", terms=[]))
        wff = WellFormedFormula(operator=LogicalOperator.AND, left=left, right=right)
        assert wff.operator == LogicalOperator.AND


class TestSubstitution:
    def test_empty(self):
        s = Substitution()
        assert s.mapping == {}
    def test_with_mapping(self):
        s = Substitution(mapping={"X": Term(name="cat")})
        assert "X" in s.mapping
