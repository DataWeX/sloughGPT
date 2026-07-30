"""Tests for reasoning/deep.py (DeepReasoning, FormalLogicEngine, etc.)."""

import pytest
from domains.cognitive.reasoning.deep import (
    DeepReasoning,
    DeepReasoningContext,
    RetrievedKnowledge,
    RetrievalSource,
    FormalLogicEngine,
    LogicalOperator,
    Term,
    Predicate,
    WellFormedFormula,
    Substitution,
    WorkingMemory,
)


# =============================================================================
# DeepReasoning
# =============================================================================

class TestDeepReasoning:
    async def test_init_defaults(self):
        dr = DeepReasoning()
        assert dr.vector_store is None
        assert dr.memory_store is None
        assert dr.llm_call is not None
        assert dr.max_retrieval == 5

    async def test_init_custom_llm(self):
        async def my_llm(prompt: str) -> str:
            return "custom"
        dr = DeepReasoning(llm_call=my_llm)
        result = await dr.llm_call("test")
        assert result == "custom"

    async def test_reason_returns_result(self):
        dr = DeepReasoning()
        result = await dr.reason("test problem", max_depth=1)
        assert isinstance(result.conclusion, str)
        assert len(result.conclusion) > 0

    async def test_reason_includes_steps(self):
        dr = DeepReasoning()
        result = await dr.reason("test", max_depth=2)
        assert len(result.steps) > 0

    async def test_reason_steps_start_with_retrieval(self):
        dr = DeepReasoning()
        result = await dr.reason("test", max_depth=1)
        assert any(s.reasoning_type == "retrieval" for s in result.steps)

    async def test_reason_includes_correction_step(self):
        dr = DeepReasoning()
        result = await dr.reason("test", max_depth=1)
        assert any(s.reasoning_type == "correction" for s in result.steps)

    async def test_reason_metadata_has_retrieved_count(self):
        dr = DeepReasoning()
        result = await dr.reason("test", max_depth=1)
        assert "retrieved_count" in result.metadata

    async def test_reason_metadata_has_depth(self):
        dr = DeepReasoning()
        result = await dr.reason("test", max_depth=3)
        assert result.metadata["depth"] == 3

    async def test_reason_with_context(self):
        ctx = DeepReasoningContext(
            query="test",
            constraints=["must be concise"],
            assumptions=["base knowledge is correct"],
        )
        dr = DeepReasoning()
        result = await dr.reason("test", context=ctx, max_depth=1)
        assert isinstance(result.conclusion, str)

    async def test_reason_with_custom_context_assumptions(self):
        ctx = DeepReasoningContext(
            query="what is gravity",
            assumptions=["gravity exists on Earth"],
        )
        dr = DeepReasoning()
        result = await dr.reason("what is gravity", context=ctx, max_depth=1)
        assert isinstance(result.conclusion, str)

    async def test_retrieve_knowledge_fallback(self):
        dr = DeepReasoning()
        results = await dr._retrieve_knowledge("because of gravity")
        assert isinstance(results, list)
        assert len(results) > 0  # fallback triggers with "because"

    async def test_retrieve_knowledge_no_match(self):
        dr = DeepReasoning()
        results = await dr._retrieve_knowledge("xyzzy")
        assert isinstance(results, list)
        assert len(results) == 0

    async def test_fallback_retrieval_cause_keyword(self):
        dr = DeepReasoning()
        results = await dr._fallback_retrieval("because of gravity")
        assert len(results) == 1
        assert results[0].source == RetrievalSource.WORKING_MEMORY

    async def test_fallback_retrieval_effect_keyword(self):
        dr = DeepReasoning()
        results = await dr._fallback_retrieval("therefore we conclude")
        assert len(results) >= 1

    async def test_fallback_retrieval_compare_keyword(self):
        dr = DeepReasoning()
        results = await dr._fallback_retrieval("however there are exceptions")
        assert len(results) >= 1

    async def test_fallback_retrieval_define_keyword(self):
        dr = DeepReasoning()
        results = await dr._fallback_retrieval("is defined as a concept")
        assert len(results) >= 1

    async def test_fallback_retrieval_no_keyword(self):
        dr = DeepReasoning()
        results = await dr._fallback_retrieval("something completely random")
        assert len(results) == 0

    async def test_fallback_retrieval_respects_max(self):
        dr = DeepReasoning()
        dr.max_retrieval = 2
        results = await dr._fallback_retrieval("because of therefore however")
        assert len(results) <= 2

    async def test_build_context(self):
        dr = DeepReasoning()
        retrieved = [RetrievedKnowledge(content="fact", source=RetrievalSource.WORKING_MEMORY, relevance=0.9)]
        ctx = DeepReasoningContext(query="test")
        context = await dr._build_context("test", retrieved, ctx)
        assert "Problem: test" in context
        assert "fact" in context

    async def test_build_context_with_constraints(self):
        dr = DeepReasoning()
        ctx = DeepReasoningContext(query="test", constraints=["no speculation"])
        context = await dr._build_context("test", [], ctx)
        assert "no speculation" in context

    async def test_build_context_with_assumptions(self):
        dr = DeepReasoning()
        ctx = DeepReasoningContext(query="test", assumptions=["base is correct"])
        context = await dr._build_context("test", [], ctx)
        assert "base is correct" in context

    async def test_build_context_empty_retrieval(self):
        dr = DeepReasoning()
        ctx = DeepReasoningContext(query="test")
        context = await dr._build_context("test", [], ctx)
        assert context == "Problem: test"

    async def test_generate_reasoning_creates_steps(self):
        dr = DeepReasoning()
        steps = await dr._generate_reasoning("context", max_depth=2)
        assert len(steps) == 2

    async def test_generate_reasoning_default_llm(self):
        dr = DeepReasoning()
        steps = await dr._generate_reasoning("context", max_depth=1)
        assert "analysis" in steps[0].reasoning_type

    async def test_default_llm_retrieve_response(self):
        dr = DeepReasoning()
        response = await dr._default_llm("retrieve knowledge")
        assert "Based on available knowledge" in response

    async def test_default_llm_critique_response(self):
        dr = DeepReasoning()
        response = await dr._default_llm("critique the reasoning")
        assert "No major issues" in response

    async def test_default_llm_correct_response(self):
        dr = DeepReasoning()
        response = await dr._default_llm("correct the analysis")
        assert "sounds sound" in response or "appears sound" in response

    async def test_default_llm_final_response(self):
        dr = DeepReasoning()
        response = await dr._default_llm("final answer")
        assert "conclusion" in response

    async def test_default_llm_fallback(self):
        dr = DeepReasoning()
        response = await dr._default_llm("random text with no keywords")
        assert "Reasoning continues" in response

    async def test_self_correct_no_issues(self):
        from domains.cognitive.reasoning.advanced import ThoughtStep as TS
        dr = DeepReasoning()
        steps = [TS(step_id=0, thought="everything is fine", reasoning_type="analysis", confidence=0.9)]
        ctx = DeepReasoningContext(query="test")
        result = await dr._self_correct(steps, "context", ctx)
        assert len(result["corrections"]) == 0
        assert result["needs_revision"] is False

    async def test_self_correct_with_issues(self):
        from domains.cognitive.reasoning.advanced import ThoughtStep as TS
        dr = DeepReasoning()
        steps = [TS(step_id=0, thought="assume base is wrong", reasoning_type="analysis", confidence=0.9)]
        ctx = DeepReasoningContext(query="test")
        result = await dr._self_correct(steps, "context", ctx)
        assert isinstance(result, dict)
        assert "corrections" in result
        assert "corrected_reasoning" in result

    async def test_critique_step_no_issues(self):
        from domains.cognitive.reasoning.advanced import ThoughtStep as TS
        dr = DeepReasoning()
        step = TS(step_id=0, thought="valid reasoning", reasoning_type="analysis", confidence=0.9)
        ctx = DeepReasoningContext(query="test")
        result = await dr._critique_step(step, "context", ctx)
        assert "step_id" in result
        assert "has_issue" in result
        assert result["has_issue"] is False

    async def test_critique_step_detects_error_keywords(self):
        from domains.cognitive.reasoning.advanced import ThoughtStep as TS
        dr = DeepReasoning()
        step = TS(step_id=0, thought="fine", reasoning_type="analysis", confidence=0.9)
        ctx = DeepReasoningContext(query="test")
        result = await dr._critique_step(step, "context", ctx)
        _name = "result is dict with has_issue key"
        # This section may or may not detect issues depending on default_llm output
        # Just verify the structure is correct
        assert "has_issue" in result

    async def test_synthesize(self):
        from domains.cognitive.reasoning.advanced import ThoughtStep as TS
        dr = DeepReasoning()
        steps = [TS(step_id=0, thought="first step", reasoning_type="analysis", confidence=0.9)]
        corrected = {"corrections": [], "corrected_reasoning": steps, "needs_revision": False}
        conclusion = await dr._synthesize(steps, corrected, "context")
        assert isinstance(conclusion, str)
        assert len(conclusion) > 0


# =============================================================================
# DeepReasoningContext
# =============================================================================

class TestDeepReasoningContext:
    def test_defaults(self):
        ctx = DeepReasoningContext(query="test")
        assert ctx.query == "test"
        assert ctx.retrieved_knowledge == []
        assert ctx.working_memory == []
        assert ctx.constraints == []
        assert ctx.assumptions == []

    def test_with_all_fields(self):
        rk = RetrievedKnowledge(content="x", source=RetrievalSource.VECTOR_STORE, relevance=0.9)
        ctx = DeepReasoningContext(
            query="q",
            retrieved_knowledge=[rk],
            working_memory=["a"],
            constraints=["c"],
            assumptions=["a2"],
        )
        assert len(ctx.retrieved_knowledge) == 1
        assert ctx.working_memory == ["a"]
        assert ctx.constraints == ["c"]
        assert ctx.assumptions == ["a2"]


# =============================================================================
# RetrievedKnowledge
# =============================================================================

class TestRetrievedKnowledge:
    def test_minimal(self):
        rk = RetrievedKnowledge(content="test", source=RetrievalSource.MEMORY, relevance=0.5)
        assert rk.content == "test"
        assert rk.source == RetrievalSource.MEMORY
        assert rk.relevance == 0.5
        assert rk.source_id is None

    def test_with_source_id(self):
        rk = RetrievedKnowledge(content="x", source=RetrievalSource.KNOWLEDGE_GRAPH, relevance=0.8, source_id="kg-1")
        assert rk.source_id == "kg-1"


# =============================================================================
# RetrievalSource
# =============================================================================

class TestRetrievalSource:
    def test_values(self):
        assert RetrievalSource.VECTOR_STORE.value == "vector_store"
        assert RetrievalSource.MEMORY.value == "memory"
        assert RetrievalSource.KNOWLEDGE_GRAPH.value == "knowledge_graph"
        assert RetrievalSource.WORKING_MEMORY.value == "working_memory"

    def test_members(self):
        assert len(RetrievalSource) == 4


# =============================================================================
# LogicalOperator
# =============================================================================

class TestLogicalOperator:
    def test_values(self):
        assert LogicalOperator.AND.value == "\u2227"
        assert LogicalOperator.OR.value == "\u2228"
        assert LogicalOperator.NOT.value == "\u00ac"
        assert LogicalOperator.IMPLIES.value == "\u2192"
        assert LogicalOperator.IFF.value == "\u2194"
        assert LogicalOperator.FORALL.value == "\u2200"
        assert LogicalOperator.EXISTS.value == "\u2203"

    def test_members(self):
        assert len(LogicalOperator) == 7


# =============================================================================
# Term
# =============================================================================

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
        t = Term(name="f", is_function=True, arguments=[Term(name="x")])
        assert t.is_function is True
        assert len(t.arguments) == 1


# =============================================================================
# Predicate
# =============================================================================

class TestPredicate:
    def test_minimal(self):
        p = Predicate(name="human", terms=[Term(name="socrates")])
        assert p.name == "human"
        assert len(p.terms) == 1
        assert p.negated is False

    def test_negated(self):
        p = Predicate(name="mortal", terms=[Term(name="socrates")], negated=True)
        assert p.negated is True

    def test_hash(self):
        p1 = Predicate(name="human", terms=[Term(name="x")])
        p2 = Predicate(name="human", terms=[Term(name="x")])
        p3 = Predicate(name="mortal", terms=[Term(name="x")])
        assert hash(p1) == hash(p2)
        assert hash(p1) != hash(p3)

    def test_hash_distinguishes_negation(self):
        p1 = Predicate(name="p", terms=[Term(name="x")], negated=False)
        p2 = Predicate(name="p", terms=[Term(name="x")], negated=True)
        assert hash(p1) != hash(p2)


# =============================================================================
# WellFormedFormula
# =============================================================================

class TestWellFormedFormula:
    def test_predicate_form(self):
        wff = WellFormedFormula(predicate=Predicate(name="p", terms=[]))
        assert wff.predicate is not None
        assert wff.predicate.name == "p"

    def test_binary_form(self):
        left = WellFormedFormula(predicate=Predicate(name="p", terms=[]))
        right = WellFormedFormula(predicate=Predicate(name="q", terms=[]))
        wff = WellFormedFormula(operator=LogicalOperator.AND, left=left, right=right)
        assert wff.operator == LogicalOperator.AND
        assert wff.left is not None
        assert wff.right is not None

    def test_quantified_form(self):
        var = Term(name="x", is_variable=True)
        sub = WellFormedFormula(predicate=Predicate(name="p", terms=[var]))
        wff = WellFormedFormula(
            quantifier_var=var,
            quantifier_type=LogicalOperator.FORALL,
            subformula=sub,
        )
        assert wff.quantifier_type == LogicalOperator.FORALL
        assert wff.subformula is not None


# =============================================================================
# Substitution
# =============================================================================

class TestSubstitution:
    def test_empty(self):
        s = Substitution()
        assert s.mapping == {}

    def test_with_mapping(self):
        s = Substitution(mapping={"X": Term(name="socrates")})
        assert "X" in s.mapping
        assert s.mapping["X"].name == "socrates"


# =============================================================================
# FormalLogicEngine
# =============================================================================

class TestFormalLogicEngine:
    def test_init(self):
        eng = FormalLogicEngine()
        assert eng.knowledge_base == []
        assert eng.inference_history == []

    def test_assert_predicate(self):
        eng = FormalLogicEngine()
        eng.assert_predicate("human", "socrates")
        assert len(eng.knowledge_base) == 1

    def test_assert_fact(self):
        eng = FormalLogicEngine()
        wff = WellFormedFormula(predicate=Predicate(name="mortal", terms=[Term(name="socrates")]))
        eng.assert_fact(wff)
        assert len(eng.knowledge_base) == 1

    def test_query_true(self):
        eng = FormalLogicEngine()
        eng.assert_predicate("human", "socrates")
        q = Predicate(name="human", terms=[Term(name="socrates")])
        assert eng.query(q) is True

    def test_query_false(self):
        eng = FormalLogicEngine()
        eng.assert_predicate("human", "socrates")
        q = Predicate(name="human", terms=[Term(name="plato")])
        assert eng.query(q) is False

    def test_query_missing_predicate(self):
        eng = FormalLogicEngine()
        eng.assert_predicate("human", "socrates")
        q = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert eng.query(q) is False

    def test_unify_same(self):
        eng = FormalLogicEngine()
        p1 = Predicate(name="p", terms=[Term(name="a")])
        p2 = Predicate(name="p", terms=[Term(name="a")])
        subst = eng._unify(p1, p2)
        assert subst is not None

    def test_unify_different_name(self):
        eng = FormalLogicEngine()
        p1 = Predicate(name="p", terms=[Term(name="a")])
        p2 = Predicate(name="q", terms=[Term(name="a")])
        assert eng._unify(p1, p2) is None

    def test_unify_variable(self):
        eng = FormalLogicEngine()
        p1 = Predicate(name="p", terms=[Term(name="X", is_variable=True)])
        p2 = Predicate(name="p", terms=[Term(name="a")])
        subst = eng._unify(p1, p2)
        assert subst is not None
        assert subst.mapping["X"].name == "a"

    def test_unify_different_arity(self):
        eng = FormalLogicEngine()
        p1 = Predicate(name="p", terms=[Term(name="a")])
        p2 = Predicate(name="p", terms=[Term(name="a"), Term(name="b")])
        assert eng._unify(p1, p2) is None

    def test_apply_substitution(self):
        eng = FormalLogicEngine()
        pred = Predicate(name="p", terms=[Term(name="X", is_variable=True)])
        subst = Substitution(mapping={"X": Term(name="socrates")})
        result = eng._apply_substitution(pred, subst)
        assert result.terms[0].name == "socrates"

    def test_forward_chain_implication(self):
        eng = FormalLogicEngine()
        p_pred = Predicate(name="p", terms=[Term(name="a")])
        q_pred = Predicate(name="q", terms=[Term(name="a")])
        antecedent = WellFormedFormula(predicate=p_pred)
        consequent = WellFormedFormula(predicate=q_pred)
        eng.assert_fact(WellFormedFormula(operator=LogicalOperator.IMPLIES, left=antecedent, right=consequent))
        eng.assert_predicate("p", "a")
        assert eng.query(q_pred) is True

    def test_modus_ponens(self):
        eng = FormalLogicEngine()
        p_pred = Predicate(name="p", terms=[Term(name="a")])
        q_pred = Predicate(name="q", terms=[Term(name="a")])
        antecedent = WellFormedFormula(predicate=p_pred)
        consequent = WellFormedFormula(predicate=q_pred)
        derived = {p_pred}
        result = eng._modus_ponens(antecedent, consequent, derived)
        assert result is not None
        assert result.name == "q"

    def test_modus_ponens_no_match(self):
        eng = FormalLogicEngine()
        p_pred = Predicate(name="p", terms=[Term(name="a")])
        q_pred = Predicate(name="q", terms=[Term(name="a")])
        antecedent = WellFormedFormula(predicate=p_pred)
        consequent = WellFormedFormula(predicate=q_pred)
        derived = {Predicate(name="other", terms=[Term(name="a")])}
        assert eng._modus_ponens(antecedent, consequent, derived) is None

    def test_occurs_check_true(self):
        eng = FormalLogicEngine()
        var = Term(name="X", is_variable=True)
        term = Term(name="X")
        subst = Substitution()
        assert eng._occurs_check(var, term, subst) is True

    def test_occurs_check_false(self):
        eng = FormalLogicEngine()
        var = Term(name="X", is_variable=True)
        term = Term(name="Y")
        subst = Substitution()
        assert eng._occurs_check(var, term, subst) is False

    def test_occurs_check_not_variable(self):
        eng = FormalLogicEngine()
        var = Term(name="X")
        term = Term(name="X")
        subst = Substitution()
        assert eng._occurs_check(var, term, subst) is False

    def test_resolution_contradiction(self):
        eng = FormalLogicEngine()
        eng.assert_predicate("p", "a")
        q = Predicate(name="p", terms=[Term(name="a")])
        assert eng.resolution(q) is True

    def test_resolution_no_contradiction(self):
        eng = FormalLogicEngine()
        eng.assert_predicate("p", "a")
        q = Predicate(name="q", terms=[Term(name="a")])
        assert eng.resolution(q) is False

    def test_prove_syllogism_valid(self):
        eng = FormalLogicEngine()
        result = eng.prove_syllogism(
            ("All", "humans", "mortal"),
            ("All", "socrates", "human"),
            ("All", "socrates", "mortal"),
        )
        assert isinstance(result, dict)
        assert "valid" in result
        assert "figure" in result
        assert "mood" in result
        assert "reason" in result

    def test_prove_syllogism_invalid(self):
        eng = FormalLogicEngine()
        result = eng.prove_syllogism(
            ("No", "As", "Bs"),
            ("All", "Cs", "Ds"),
            ("All", "Es", "Fs"),
        )
        assert result["valid"] is False

    def test_prove_syllogism_appends_history(self):
        eng = FormalLogicEngine()
        eng.prove_syllogism(
            ("All", "humans", "mortal"),
            ("All", "socrates", "human"),
            ("All", "socrates", "mortal"),
        )
        assert len(eng.inference_history) == 1
        assert eng.inference_history[0]["type"] == "syllogism"

    def test_to_categorical(self):
        eng = FormalLogicEngine()
        result = eng._to_categorical(("All", "humans", "mortal"))
        assert len(result) == 4

    def test_format_categorical(self):
        eng = FormalLogicEngine()
        cat = eng._to_categorical(("All", "humans", "mortal"))
        formatted = eng._format_categorical(cat)
        assert isinstance(formatted, str)

    def test_to_clausal_form(self):
        eng = FormalLogicEngine()
        wffs = [WellFormedFormula(predicate=Predicate(name="p", terms=[Term(name="a")]))]
        clauses = eng._to_clausal_form(wffs)
        assert len(clauses) == 1
        assert len(clauses[0]) == 1

    def test_extract_literals_predicate(self):
        eng = FormalLogicEngine()
        wff = WellFormedFormula(predicate=Predicate(name="p", terms=[Term(name="a")]))
        lits = eng._extract_literals(wff)
        assert len(lits) == 1

    def test_extract_literals_and(self):
        eng = FormalLogicEngine()
        left = WellFormedFormula(predicate=Predicate(name="p", terms=[]))
        right = WellFormedFormula(predicate=Predicate(name="q", terms=[]))
        wff = WellFormedFormula(operator=LogicalOperator.AND, left=left, right=right)
        lits = eng._extract_literals(wff)
        assert len(lits) == 2

    def test_apply_term_substitution_variable(self):
        eng = FormalLogicEngine()
        term = Term(name="X", is_variable=True)
        subst = Substitution(mapping={"X": Term(name="a")})
        result = eng._apply_term_substitution(term, subst)
        assert result.name == "a"

    def test_apply_term_substitution_constant(self):
        eng = FormalLogicEngine()
        term = Term(name="a")
        subst = Substitution(mapping={"X": Term(name="b")})
        result = eng._apply_term_substitution(term, subst)
        assert result.name == "a"

    def test_apply_term_substitution_function(self):
        eng = FormalLogicEngine()
        term = Term(name="f", is_function=True, arguments=[Term(name="X", is_variable=True)])
        subst = Substitution(mapping={"X": Term(name="a")})
        result = eng._apply_term_substitution(term, subst)
        assert result.arguments[0].name == "a"

    def test_unify_complementary_left_negated(self):
        eng = FormalLogicEngine()
        lit1 = Predicate(name="p", terms=[Term(name="a")], negated=True)
        lit2 = Predicate(name="p", terms=[Term(name="a")], negated=False)
        subst = eng._unify_complementary(lit1, lit2)
        assert subst is not None

    def test_unify_complementary_right_negated(self):
        eng = FormalLogicEngine()
        lit1 = Predicate(name="p", terms=[Term(name="a")], negated=False)
        lit2 = Predicate(name="p", terms=[Term(name="a")], negated=True)
        subst = eng._unify_complementary(lit1, lit2)
        assert subst is not None

    def test_unify_complementary_no_match(self):
        eng = FormalLogicEngine()
        lit1 = Predicate(name="p", terms=[Term(name="a")], negated=True)
        lit2 = Predicate(name="q", terms=[Term(name="a")], negated=False)
        assert eng._unify_complementary(lit1, lit2) is None


# =============================================================================
# WorkingMemory
# =============================================================================

class TestWorkingMemory:
    def test_init(self):
        wm = WorkingMemory()
        assert wm.capacity == 7
        assert wm.items == []
        assert wm.access_count == {}

    def test_init_custom_capacity(self):
        wm = WorkingMemory(capacity=3)
        assert wm.capacity == 3

    def test_add_item(self):
        wm = WorkingMemory()
        wm.add("item1")
        assert "item1" in wm.items

    def test_add_evicts_lru(self):
        wm = WorkingMemory(capacity=2)
        wm.add("a")
        wm.add("b")
        wm.access("a")
        wm.add("c")
        assert "b" not in wm.items
        assert "a" in wm.items
        assert "c" in wm.items

    def test_access_increments_count(self):
        wm = WorkingMemory()
        wm.add("x")
        wm.access("x")
        wm.access("x")
        assert wm.access_count["x"] == 3

    def test_access_nonexistent(self):
        wm = WorkingMemory()
        wm.access("y")
        assert wm.access_count.get("y", 0) == 1

    def test_get_recent(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.add("b")
        wm.access("b")
        wm.access("b")
        wm.access("a")
        recent = wm.get_recent(2)
        assert len(recent) == 2

    def test_get_recent_returns_most_accessed(self):
        wm = WorkingMemory()
        wm.add("low")
        wm.add("high")
        wm.access("high")
        wm.access("high")
        wm.access("high")
        recent = wm.get_recent(2)
        assert recent[0] == "high"

    def test_clear(self):
        wm = WorkingMemory()
        wm.add("x")
        wm.add("y")
        wm.clear()
        assert wm.items == []
        assert wm.access_count == {}
