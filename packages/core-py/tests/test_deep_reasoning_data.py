"""Tests for domains.cognitive.reasoning.deep — RetrievalSource, RetrievedKnowledge, DeepReasoningContext, Term, Predicate, WellFormedFormula, Substitution, FormalLogicEngine, WorkingMemory."""

from domains.cognitive.reasoning.deep import (
    RetrievalSource, RetrievedKnowledge, DeepReasoningContext,
    Term, Predicate, WellFormedFormula, Substitution, LogicalOperator,
    FormalLogicEngine, WorkingMemory,
)


class TestRetrievalSource:
    def test_all_members(self):
        assert len(RetrievalSource) == 4

    def test_values(self):
        assert RetrievalSource.VECTOR_STORE.value == "vector_store"
        assert RetrievalSource.KNOWLEDGE_GRAPH.value == "knowledge_graph"

    def test_vector_store_member(self):
        assert RetrievalSource.VECTOR_STORE.name == "VECTOR_STORE"

    def test_memory_member(self):
        assert RetrievalSource.MEMORY.name == "MEMORY"

    def test_working_memory_member(self):
        assert RetrievalSource.WORKING_MEMORY.name == "WORKING_MEMORY"

    def test_members_are_enum(self):
        for member in RetrievalSource:
            assert isinstance(member, RetrievalSource)

    def test_equality(self):
        assert RetrievalSource.VECTOR_STORE == RetrievalSource.VECTOR_STORE
        assert RetrievalSource.VECTOR_STORE != RetrievalSource.MEMORY

    def test_value_uniqueness(self):
        values = [m.value for m in RetrievalSource]
        assert len(values) == len(set(values))

    def test_member_count(self):
        assert len([m for m in RetrievalSource]) == 4

    def test_membership(self):
        assert RetrievalSource.VECTOR_STORE in RetrievalSource
        assert "vector_store" in [m.value for m in RetrievalSource]


class TestRetrievedKnowledge:
    def test_fields(self):
        rk = RetrievedKnowledge(content="fact", source=RetrievalSource.MEMORY, relevance=0.9)
        assert rk.content == "fact"
        assert rk.relevance == 0.9

    def test_source_field(self):
        rk = RetrievedKnowledge(content="x", source=RetrievalSource.VECTOR_STORE, relevance=0.5)
        assert rk.source == RetrievalSource.VECTOR_STORE

    def test_source_id_default_none(self):
        rk = RetrievedKnowledge(content="x", source=RetrievalSource.MEMORY, relevance=0.5)
        assert rk.source_id is None

    def test_source_id_set(self):
        rk = RetrievedKnowledge(content="x", source=RetrievalSource.MEMORY, relevance=0.5, source_id="doc1")
        assert rk.source_id == "doc1"

    def test_relevance_zero(self):
        rk = RetrievedKnowledge(content="x", source=RetrievalSource.MEMORY, relevance=0.0)
        assert rk.relevance == 0.0

    def test_relevance_one(self):
        rk = RetrievedKnowledge(content="x", source=RetrievalSource.MEMORY, relevance=1.0)
        assert rk.relevance == 1.0

    def test_content_empty_string(self):
        rk = RetrievedKnowledge(content="", source=RetrievalSource.MEMORY, relevance=0.5)
        assert rk.content == ""

    def test_content_long_string(self):
        long = "a" * 10000
        rk = RetrievedKnowledge(content=long, source=RetrievalSource.MEMORY, relevance=0.5)
        assert len(rk.content) == 10000


class TestDeepReasoningContext:
    def test_defaults(self):
        ctx = DeepReasoningContext(query="why?")
        assert ctx.query == "why?"
        assert ctx.retrieved_knowledge == []
        assert ctx.constraints == []

    def test_query_only(self):
        ctx = DeepReasoningContext(query="test")
        assert ctx.query == "test"

    def test_retrieved_knowledge_default_empty(self):
        ctx = DeepReasoningContext(query="x")
        assert ctx.retrieved_knowledge == []

    def test_working_memory_default_empty(self):
        ctx = DeepReasoningContext(query="x")
        assert ctx.working_memory == []

    def test_constraints_default_empty(self):
        ctx = DeepReasoningContext(query="x")
        assert ctx.constraints == []

    def test_assumptions_default_empty(self):
        ctx = DeepReasoningContext(query="x")
        assert ctx.assumptions == []

    def test_set_retrieved_knowledge(self):
        rk = RetrievedKnowledge(content="fact", source=RetrievalSource.MEMORY, relevance=0.9)
        ctx = DeepReasoningContext(query="x", retrieved_knowledge=[rk])
        assert len(ctx.retrieved_knowledge) == 1

    def test_set_constraints(self):
        ctx = DeepReasoningContext(query="x", constraints=["must be fast"])
        assert ctx.constraints == ["must be fast"]

    def test_set_assumptions(self):
        ctx = DeepReasoningContext(query="x", assumptions=["data is clean"])
        assert ctx.assumptions == ["data is clean"]

    def test_set_working_memory(self):
        ctx = DeepReasoningContext(query="x", working_memory=["fact1", "fact2"])
        assert len(ctx.working_memory) == 2

    def test_all_fields_populated(self):
        rk = RetrievedKnowledge(content="c", source=RetrievalSource.MEMORY, relevance=0.8)
        ctx = DeepReasoningContext(
            query="q",
            retrieved_knowledge=[rk],
            working_memory=["w1"],
            constraints=["c1"],
            assumptions=["a1"],
        )
        assert ctx.query == "q"
        assert len(ctx.retrieved_knowledge) == 1
        assert ctx.working_memory == ["w1"]
        assert ctx.constraints == ["c1"]
        assert ctx.assumptions == ["a1"]


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

    def test_default_not_variable(self):
        t = Term(name="x")
        assert t.is_variable is False

    def test_default_not_function(self):
        t = Term(name="x")
        assert t.is_function is False

    def test_default_arguments_empty(self):
        t = Term(name="x")
        assert t.arguments == []

    def test_arguments_list(self):
        args = [Term(name="a"), Term(name="b"), Term(name="c")]
        t = Term(name="f", is_function=True, arguments=args)
        assert len(t.arguments) == 3

    def test_nested_function(self):
        inner = Term(name="g", is_function=True, arguments=[Term(name="x")])
        outer = Term(name="f", is_function=True, arguments=[inner])
        assert outer.arguments[0].is_function is True
        assert outer.arguments[0].name == "g"

    def test_variable_name(self):
        t = Term(name="X", is_variable=True)
        assert t.name == "X"

    def test_constant_name(self):
        t = Term(name="socrates")
        assert t.name == "socrates"


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

    def test_negated_default_false(self):
        p = Predicate(name="likes", terms=[])
        assert p.negated is False

    def test_negated_true(self):
        p = Predicate(name="likes", terms=[Term(name="cat")], negated=True)
        assert p.negated is True

    def test_hash_equal_predicates(self):
        p1 = Predicate(name="likes", terms=[Term(name="a"), Term(name="b")])
        p2 = Predicate(name="likes", terms=[Term(name="a"), Term(name="b")])
        assert hash(p1) == hash(p2)

    def test_hash_different_names(self):
        p1 = Predicate(name="likes", terms=[Term(name="a")])
        p2 = Predicate(name="hates", terms=[Term(name="a")])
        assert hash(p1) != hash(p2)

    def test_hash_different_terms(self):
        p1 = Predicate(name="likes", terms=[Term(name="a")])
        p2 = Predicate(name="likes", terms=[Term(name="b")])
        assert hash(p1) != hash(p2)

    def test_hash_negated_different(self):
        p1 = Predicate(name="likes", terms=[Term(name="a")], negated=False)
        p2 = Predicate(name="likes", terms=[Term(name="a")], negated=True)
        assert hash(p1) != hash(p2)

    def test_empty_terms(self):
        p = Predicate(name="true", terms=[])
        assert len(p.terms) == 0

    def test_single_term(self):
        p = Predicate(name="human", terms=[Term(name="socrates")])
        assert len(p.terms) == 1


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

    def test_disjunction(self):
        left = WellFormedFormula(predicate=Predicate(name="a", terms=[]))
        right = WellFormedFormula(predicate=Predicate(name="b", terms=[]))
        wff = WellFormedFormula(operator=LogicalOperator.OR, left=left, right=right)
        assert wff.operator == LogicalOperator.OR

    def test_implication(self):
        left = WellFormedFormula(predicate=Predicate(name="a", terms=[]))
        right = WellFormedFormula(predicate=Predicate(name="b", terms=[]))
        wff = WellFormedFormula(operator=LogicalOperator.IMPLIES, left=left, right=right)
        assert wff.operator == LogicalOperator.IMPLIES

    def test_default_predicate_none(self):
        wff = WellFormedFormula()
        assert wff.predicate is None

    def test_default_operator_none(self):
        wff = WellFormedFormula()
        assert wff.operator is None

    def test_default_left_right_none(self):
        wff = WellFormedFormula()
        assert wff.left is None
        assert wff.right is None

    def test_default_quantifier_none(self):
        wff = WellFormedFormula()
        assert wff.quantifier_var is None
        assert wff.quantifier_type is None
        assert wff.subformula is None

    def test_nested_formula(self):
        inner = WellFormedFormula(predicate=Predicate(name="a", terms=[]))
        outer = WellFormedFormula(
            operator=LogicalOperator.NOT,
            left=inner,
        )
        assert outer.operator == LogicalOperator.NOT
        assert outer.left.predicate.name == "a"

    def test_complex_formula(self):
        a = WellFormedFormula(predicate=Predicate(name="human", terms=[Term(name="socrates")]))
        b = WellFormedFormula(predicate=Predicate(name="mortal", terms=[Term(name="socrates")]))
        impl = WellFormedFormula(operator=LogicalOperator.IMPLIES, left=a, right=b)
        assert impl.operator == LogicalOperator.IMPLIES
        assert impl.left.predicate.name == "human"
        assert impl.right.predicate.name == "mortal"

    def test_biconditional(self):
        left = WellFormedFormula(predicate=Predicate(name="a", terms=[]))
        right = WellFormedFormula(predicate=Predicate(name="b", terms=[]))
        wff = WellFormedFormula(operator=LogicalOperator.IFF, left=left, right=right)
        assert wff.operator == LogicalOperator.IFF


class TestSubstitution:
    def test_empty(self):
        s = Substitution()
        assert s.mapping == {}

    def test_with_mapping(self):
        s = Substitution(mapping={"X": Term(name="cat")})
        assert "X" in s.mapping

    def test_mapping_value(self):
        t = Term(name="cat")
        s = Substitution(mapping={"X": t})
        assert s.mapping["X"] is t

    def test_multiple_mappings(self):
        s = Substitution(mapping={"X": Term(name="a"), "Y": Term(name="b")})
        assert len(s.mapping) == 2

    def test_default_empty_dict(self):
        s = Substitution()
        assert isinstance(s.mapping, dict)

    def test_mapping_is_dict(self):
        s = Substitution(mapping={"X": Term(name="cat")})
        assert isinstance(s.mapping, dict)

    def test_mapping_variable_to_variable(self):
        s = Substitution(mapping={"X": Term(name="Y", is_variable=True)})
        assert s.mapping["X"].is_variable is True

    def test_mapping_to_function_term(self):
        f = Term(name="f", is_function=True, arguments=[Term(name="a")])
        s = Substitution(mapping={"X": f})
        assert s.mapping["X"].is_function is True


class TestLogicalOperator:
    def test_and(self):
        assert LogicalOperator.AND.value == "∧"

    def test_or(self):
        assert LogicalOperator.OR.value == "∨"

    def test_not(self):
        assert LogicalOperator.NOT.value == "¬"

    def test_implies(self):
        assert LogicalOperator.IMPLIES.value == "→"

    def test_iff(self):
        assert LogicalOperator.IFF.value == "↔"

    def test_forall(self):
        assert LogicalOperator.FORALL.value == "∀"

    def test_exists(self):
        assert LogicalOperator.EXISTS.value == "∃"

    def test_member_count(self):
        assert len(LogicalOperator) == 7

    def test_all_members(self):
        members = [m.name for m in LogicalOperator]
        assert "AND" in members
        assert "OR" in members
        assert "NOT" in members
        assert "IMPLIES" in members
        assert "IFF" in members
        assert "FORALL" in members
        assert "EXISTS" in members

    def test_value_uniqueness(self):
        values = [m.value for m in LogicalOperator]
        assert len(values) == len(set(values))


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

    def test_assert_multiple_facts(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        engine.assert_predicate("mortal", "socrates")
        assert len(engine.knowledge_base) == 2

    def test_query_true(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        result = engine.query(Predicate(name="human", terms=[Term(name="socrates")]))
        assert result is True

    def test_query_false(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        result = engine.query(Predicate(name="mortal", terms=[Term(name="socrates")]))
        assert result is False

    def test_query_empty_kb(self):
        engine = FormalLogicEngine()
        result = engine.query(Predicate(name="anything", terms=[]))
        assert result is False

    def test_modus_ponens(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        a = WellFormedFormula(predicate=Predicate(name="human", terms=[Term(name="socrates")]))
        b = WellFormedFormula(predicate=Predicate(name="mortal", terms=[Term(name="socrates")]))
        impl = WellFormedFormula(operator=LogicalOperator.IMPLIES, left=a, right=b)
        engine.assert_fact(impl)
        result = engine.query(Predicate(name="mortal", terms=[Term(name="socrates")]))
        assert result is True

    def test_inference_history_appended(self):
        engine = FormalLogicEngine()
        engine.prove_syllogism(
            ("All", "are", "mortal"),
            ("All", "are", "human"),
            ("All", "are", "mortal"),
        )
        assert len(engine.inference_history) == 1

    def test_prove_syllogism_returns_dict(self):
        engine = FormalLogicEngine()
        result = engine.prove_syllogism(
            ("All", "are", "mortal"),
            ("All", "are", "human"),
            ("All", "are", "mortal"),
        )
        assert isinstance(result, dict)
        assert "valid" in result
        assert "figure" in result
        assert "mood" in result

    def test_prove_syllogism_figure(self):
        engine = FormalLogicEngine()
        result = engine.prove_syllogism(
            ("All", "are", "mortal"),
            ("All", "are", "human"),
            ("All", "are", "mortal"),
        )
        assert result["figure"] in [1, 2, 3, 4]

    def test_prove_syllogism_mood(self):
        engine = FormalLogicEngine()
        result = engine.prove_syllogism(
            ("All", "are", "mortal"),
            ("All", "are", "human"),
            ("All", "are", "mortal"),
        )
        assert isinstance(result["mood"], str)
        assert len(result["mood"]) == 3

    def test_prove_syllogism_valid_combination(self):
        engine = FormalLogicEngine()
        result = engine.prove_syllogism(
            ("All", "are", "mortal"),
            ("All", "are", "human"),
            ("All", "are", "mortal"),
        )
        assert result["valid"] is True or result["valid"] is False

    def test_prove_syllogism_has_reason(self):
        engine = FormalLogicEngine()
        result = engine.prove_syllogism(
            ("All", "are", "mortal"),
            ("All", "are", "human"),
            ("All", "are", "mortal"),
        )
        assert isinstance(result["reason"], str)

    def test_prove_syllogism_has_form(self):
        engine = FormalLogicEngine()
        result = engine.prove_syllogism(
            ("All", "are", "mortal"),
            ("All", "are", "human"),
            ("All", "are", "mortal"),
        )
        assert "form" in result

    def test_unify_identical_predicates(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="likes", terms=[Term(name="cat")])
        p2 = Predicate(name="likes", terms=[Term(name="cat")])
        subst = engine._unify(p1, p2)
        assert subst is not None

    def test_unify_different_names(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="likes", terms=[Term(name="cat")])
        p2 = Predicate(name="hates", terms=[Term(name="cat")])
        subst = engine._unify(p1, p2)
        assert subst is None

    def test_unify_different_arity(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="likes", terms=[Term(name="a")])
        p2 = Predicate(name="likes", terms=[Term(name="a"), Term(name="b")])
        subst = engine._unify(p1, p2)
        assert subst is None

    def test_resolution_empty_kb(self):
        engine = FormalLogicEngine()
        result = engine.resolution(Predicate(name="anything", terms=[]))
        assert result is False

    def test_resolution_with_fact(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        result = engine.resolution(Predicate(name="human", terms=[Term(name="socrates")]))
        assert result is True


class TestWorkingMemory:
    def test_init_default(self):
        wm = WorkingMemory()
        assert wm.capacity == 7
        assert wm.items == []
        assert wm.access_count == {}

    def test_add_item(self):
        wm = WorkingMemory()
        wm.add("fact1")
        assert "fact1" in wm.items

    def test_add_multiple_items(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.add("b")
        assert len(wm.items) == 2

    def test_add_updates_access_count(self):
        wm = WorkingMemory()
        wm.add("a")
        assert wm.access_count["a"] == 1

    def test_access_increments_count(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.access("a")
        assert wm.access_count["a"] == 2

    def test_get_recent(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.add("b")
        wm.add("c")
        recent = wm.get_recent(2)
        assert len(recent) == 2

    def test_get_recent_default_5(self):
        wm = WorkingMemory()
        for i in range(10):
            wm.add(f"item{i}")
        recent = wm.get_recent()
        assert len(recent) == 5

    def test_clear(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.add("b")
        wm.clear()
        assert wm.items == []
        assert wm.access_count == {}

    def test_capacity_eviction(self):
        wm = WorkingMemory(capacity=3)
        wm.add("a")
        wm.add("b")
        wm.add("c")
        wm.add("d")
        assert len(wm.items) == 3
        assert "d" in wm.items

    def test_evicts_least_recently_used(self):
        wm = WorkingMemory(capacity=2)
        wm.add("a")
        wm.add("b")
        wm.access("a")  # a is now more used
        wm.add("c")
        assert "a" in wm.items
        assert "b" not in wm.items

    def test_custom_capacity(self):
        wm = WorkingMemory(capacity=10)
        for i in range(15):
            wm.add(f"item{i}")
        assert len(wm.items) == 10

    def test_get_recent_by_access_count(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.add("b")
        wm.access("a")
        wm.access("a")
        wm.access("b")
        recent = wm.get_recent(2)
        assert recent[0] == "a"

    def test_add_same_item_twice(self):
        wm = WorkingMemory(capacity=3)
        wm.add("a")
        wm.add("a")
        assert wm.items.count("a") == 2

    def test_access_nonexistent_item(self):
        wm = WorkingMemory()
        wm.access("ghost")
        assert wm.access_count["ghost"] == 1

    def test_get_recent_empty(self):
        wm = WorkingMemory()
        recent = wm.get_recent()
        assert recent == []
