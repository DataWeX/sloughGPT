"""Meaningful tests for FormalLogicEngine (unification, forward chaining, resolution) and WorkingMemory."""

from domains.cognitive.reasoning.deep import (
    FormalLogicEngine, LogicalOperator, Term, Predicate,
    WellFormedFormula, Substitution, WorkingMemory,
)


# ── WorkingMemory ──────────────────────────────────────────────────────

class TestWorkingMemory:
    def test_add_within_capacity(self):
        wm = WorkingMemory(capacity=5)
        wm.add("item1")
        wm.add("item2")
        assert wm.items == ["item1", "item2"]

    def test_add_evicts_lru(self):
        wm = WorkingMemory(capacity=2)
        wm.add("a")
        wm.add("b")
        wm.access("a")  # a accessed once, b accessed once
        wm.add("c")  # should evict one of a/b (both have access_count=1)
        # LRU is whichever comes first with min count
        assert len(wm.items) == 2
        assert "c" in wm.items

    def test_access_increments_count(self):
        wm = WorkingMemory(capacity=3)
        wm.add("x")
        wm.access("x")
        wm.access("x")
        assert wm.access_count["x"] == 3

    def test_get_recent(self):
        wm = WorkingMemory(capacity=10)
        wm.add("a")
        wm.add("b")
        wm.add("c")
        wm.access("a")
        wm.access("a")
        wm.access("c")
        recent = wm.get_recent(2)
        assert recent[0] == "a"  # highest access count
        assert recent[1] == "c"

    def test_clear(self):
        wm = WorkingMemory(capacity=5)
        wm.add("x")
        wm.clear()
        assert wm.items == []
        assert wm.access_count == {}

    def test_get_recent_empty(self):
        wm = WorkingMemory()
        assert wm.get_recent() == []


# ── Term / Predicate / WFF ────────────────────────────────────────────

class TestTerm:
    def test_constant(self):
        t = Term(name="socrates")
        assert t.is_variable is False

    def test_variable(self):
        t = Term(name="X", is_variable=True)
        assert t.is_variable is True

    def test_function(self):
        t = Term(name="f", is_function=True, arguments=[Term(name="a"), Term(name="b")])
        assert t.is_function is True
        assert len(t.arguments) == 2


class TestPredicate:
    def test_simple(self):
        p = Predicate(name="human", terms=[Term(name="socrates")])
        assert p.name == "human"
        assert p.negated is False

    def test_negated(self):
        p = Predicate(name="mortal", terms=[Term(name="X")], negated=True)
        assert p.negated is True


class TestWFF:
    def test_atomic(self):
        p = Predicate(name="human", terms=[Term(name="socrates")])
        wff = WellFormedFormula(predicate=p)
        assert wff.predicate == p
        assert wff.operator is None

    def test_implication(self):
        left = WellFormedFormula(predicate=Predicate(name="human", terms=[Term(name="X")]))
        right = WellFormedFormula(predicate=Predicate(name="mortal", terms=[Term(name="X")]))
        wff = WellFormedFormula(operator=LogicalOperator.IMPLIES, left=left, right=right)
        assert wff.operator == LogicalOperator.IMPLIES


# ── FormalLogicEngine ─────────────────────────────────────────────────

class TestFormalLogicEngineAssert:
    def test_assert_predicate(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        assert len(engine.knowledge_base) == 1
        assert engine.knowledge_base[0].predicate.name == "human"

    def test_assert_wff(self):
        engine = FormalLogicEngine()
        p = Predicate(name="mortal", terms=[Term(name="socrates")])
        wff = WellFormedFormula(predicate=p)
        engine.assert_fact(wff)
        assert len(engine.knowledge_base) == 1


class TestFormalLogicEngineQuery:
    def test_query_direct_fact(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        q = Predicate(name="human", terms=[Term(name="socrates")])
        assert engine.query(q) is True

    def test_query_missing_fact(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        q = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert engine.query(q) is False

    def test_query_modus_ponens(self):
        engine = FormalLogicEngine()
        # human(socrates)
        engine.assert_predicate("human", "socrates")
        # human(X) → mortal(X)
        left = WellFormedFormula(predicate=Predicate(name="human", terms=[Term(name="X", is_variable=True)]))
        right = WellFormedFormula(predicate=Predicate(name="mortal", terms=[Term(name="X", is_variable=True)]))
        implication = WellFormedFormula(operator=LogicalOperator.IMPLIES, left=left, right=right)
        engine.assert_fact(implication)
        # Query: mortal(socrates)
        q = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert engine.query(q) is True

    def test_query_modus_ponens_chain(self):
        engine = FormalLogicEngine()
        # human(socrates) → mortal(socrates) → finite(socrates)
        engine.assert_predicate("human", "socrates")
        # human(X) → mortal(X)
        left1 = WellFormedFormula(predicate=Predicate(name="human", terms=[Term(name="X", is_variable=True)]))
        right1 = WellFormedFormula(predicate=Predicate(name="mortal", terms=[Term(name="X", is_variable=True)]))
        engine.assert_fact(WellFormedFormula(operator=LogicalOperator.IMPLIES, left=left1, right=right1))
        # mortal(X) → finite(X)
        left2 = WellFormedFormula(predicate=Predicate(name="mortal", terms=[Term(name="X", is_variable=True)]))
        right2 = WellFormedFormula(predicate=Predicate(name="finite", terms=[Term(name="X", is_variable=True)]))
        engine.assert_fact(WellFormedFormula(operator=LogicalOperator.IMPLIES, left=left2, right=right2))
        # Query: finite(socrates)
        q = Predicate(name="finite", terms=[Term(name="socrates")])
        assert engine.query(q) is True


class TestFormalLogicEngineUnification:
    def test_unify_identical(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="human", terms=[Term(name="socrates")])
        p2 = Predicate(name="human", terms=[Term(name="socrates")])
        subst = engine._unify(p1, p2)
        assert subst is not None

    def test_unify_variable(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="human", terms=[Term(name="X", is_variable=True)])
        p2 = Predicate(name="human", terms=[Term(name="socrates")])
        subst = engine._unify(p1, p2)
        assert subst is not None
        assert subst.mapping["X"].name == "socrates"

    def test_unify_different_names(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="human", terms=[Term(name="socrates")])
        p2 = Predicate(name="mortal", terms=[Term(name="socrates")])
        subst = engine._unify(p1, p2)
        assert subst is None

    def test_unify_different_arity(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="human", terms=[Term(name="a")])
        p2 = Predicate(name="human", terms=[Term(name="a"), Term(name="b")])
        subst = engine._unify(p1, p2)
        assert subst is None

    def test_occurs_check(self):
        engine = FormalLogicEngine()
        var = Term(name="X", is_variable=True)
        func = Term(name="f", is_function=True, arguments=[Term(name="X", is_variable=True)])
        assert engine._occurs_check(var, func, Substitution()) is True

    def test_apply_substitution(self):
        engine = FormalLogicEngine()
        pred = Predicate(name="human", terms=[Term(name="X", is_variable=True)])
        subst = Substitution(mapping={"X": Term(name="socrates")})
        result = engine._apply_substitution(pred, subst)
        assert result.terms[0].name == "socrates"


class TestFormalLogicEngineResolution:
    def test_resolution_simple(self):
        engine = FormalLogicEngine()
        # human(socrates)
        engine.assert_predicate("human", "socrates")
        # human(X) → mortal(X)
        left = WellFormedFormula(predicate=Predicate(name="human", terms=[Term(name="X", is_variable=True)]))
        right = WellFormedFormula(predicate=Predicate(name="mortal", terms=[Term(name="X", is_variable=True)]))
        engine.assert_fact(WellFormedFormula(operator=LogicalOperator.IMPLIES, left=left, right=right))
        # Prove mortal(socrates)
        goal = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert engine.resolution(goal) is True

    def test_resolution_unprovable(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        goal = Predicate(name="divine", terms=[Term(name="socrates")])
        assert engine.resolution(goal) is False
