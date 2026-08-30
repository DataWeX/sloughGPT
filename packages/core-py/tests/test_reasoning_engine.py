"""Tests for domains.cognitive.reasoning.__init__ — ReasoningEngine."""

import asyncio
import pytest
from domains.cognitive.reasoning import (
    ReasoningEngine, ReasoningMode, WorkingMemory,
    ChainOfThought, TreeOfThoughts, SelfConsistency,
    ConstitutionalAI, CausalReasoning, SyllogismReasoning,
    ReActReasoning, ReasoningResult, ThoughtStep,
    DeepReasoning, DeepReasoningContext, RetrievedKnowledge, RetrievalSource,
    FormalLogicEngine, LogicalOperator, Term, Predicate,
    WellFormedFormula, Substitution,
    advanced_reasoning,
)


# ---------------------------------------------------------------------------
# ReasoningEngine
# ---------------------------------------------------------------------------

class TestReasoningEngine:
    def test_init(self):
        engine = ReasoningEngine()
        assert engine.mode == ReasoningMode.CHAIN_OF_THOUGHT
        assert engine.reasoning_history == []

    def test_assert_and_query(self):
        engine = ReasoningEngine()
        engine.assert_fact("human", "socrates")
        assert engine.query("human", "socrates") is True

    def test_query_not_asserted(self):
        engine = ReasoningEngine()
        assert engine.query("human", "socrates") is False

    def test_set_mode(self):
        engine = ReasoningEngine()
        asyncio.run(engine.set_mode(ReasoningMode.TREE_OF_THOUGHTS))
        assert engine.mode == ReasoningMode.TREE_OF_THOUGHTS

    def test_get_history_empty(self):
        engine = ReasoningEngine()
        history = asyncio.run(engine.get_history())
        assert history == []

    def test_reason(self):
        engine = ReasoningEngine()
        result = asyncio.run(engine.reason("What is 2+2?", {}))
        assert isinstance(result, str)
        assert len(engine.reasoning_history) == 1

    def test_reason_multiple_queries(self):
        engine = ReasoningEngine()
        asyncio.run(engine.reason("Problem 1", {}))
        asyncio.run(engine.reason("Problem 2", {}))
        assert len(engine.reasoning_history) == 2

    def test_assert_multiple_facts(self):
        engine = ReasoningEngine()
        engine.assert_fact("human", "socrates")
        engine.assert_fact("human", "plato")
        assert engine.query("human", "socrates") is True
        assert engine.query("human", "plato") is True

    def test_query_wrong_value(self):
        engine = ReasoningEngine()
        engine.assert_fact("human", "socrates")
        assert engine.query("human", "plato") is False

    def test_set_mode_all_modes(self):
        engine = ReasoningEngine()
        for mode in ReasoningMode:
            asyncio.run(engine.set_mode(mode))
            assert engine.mode == mode

    def test_deep_reason(self):
        engine = ReasoningEngine()
        result = asyncio.run(engine.deep_reason("Problem"))
        assert isinstance(result, ReasoningResult)

    def test_logical_proof(self):
        engine = ReasoningEngine()
        result = asyncio.run(engine.logical_proof(
            ("All", "are", "mortal"),
            ("Some", "are", "human"),
            ("Some", "are", "mortal"),
        ))
        assert "valid" in result

    def test_assert_and_query_chain(self):
        engine = ReasoningEngine()
        engine.assert_fact("mortal", "socrates")
        engine.assert_fact("human", "socrates")
        assert engine.query("mortal", "socrates") is True
        assert engine.query("human", "socrates") is True

    def test_query_after_reason(self):
        engine = ReasoningEngine()
        asyncio.run(engine.reason("Problem", {}))
        assert len(engine.reasoning_history) == 1
        engine.assert_fact("test", "value")
        assert engine.query("test", "value") is True

    def test_mode_persists_across_reason(self):
        engine = ReasoningEngine()
        asyncio.run(engine.set_mode(ReasoningMode.CAUSAL))
        asyncio.run(engine.reason("Problem", {}))
        assert engine.mode == ReasoningMode.CAUSAL

    def test_history_contains_reasoning_results(self):
        engine = ReasoningEngine()
        asyncio.run(engine.reason("Problem", {}))
        assert isinstance(engine.reasoning_history[0], ReasoningResult)


# ---------------------------------------------------------------------------
# WorkingMemory
# ---------------------------------------------------------------------------

class TestWorkingMemory:
    def test_init(self):
        wm = WorkingMemory()
        assert wm is not None
        assert wm.capacity == 7
        assert wm.items == []

    def test_add_item(self):
        wm = WorkingMemory()
        wm.add("fact1")
        assert "fact1" in wm.items

    def test_add_multiple_items(self):
        wm = WorkingMemory()
        for i in range(5):
            wm.add(f"item_{i}")
        assert len(wm.items) == 5

    def test_capacity_eviction(self):
        wm = WorkingMemory(capacity=3)
        wm.add("a")
        wm.add("b")
        wm.add("c")
        wm.add("d")
        assert len(wm.items) == 3
        assert "a" not in wm.items

    def test_access_increments_count(self):
        wm = WorkingMemory()
        wm.add("item")
        wm.access("item")
        wm.access("item")
        assert wm.access_count["item"] == 3

    def test_get_recent(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.add("b")
        wm.add("c")
        recent = wm.get_recent(2)
        assert len(recent) == 2

    def test_get_recent_respects_access_count(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.add("b")
        wm.access("a")
        wm.access("a")
        wm.access("a")
        recent = wm.get_recent(1)
        assert recent[0] == "a"

    def test_clear(self):
        wm = WorkingMemory()
        wm.add("item1")
        wm.add("item2")
        wm.clear()
        assert wm.items == []
        assert wm.access_count == {}

    def test_lru_eviction_policy(self):
        wm = WorkingMemory(capacity=2)
        wm.add("x")
        wm.add("y")
        wm.access("x")
        wm.add("z")
        assert "x" in wm.items
        assert "y" not in wm.items

    def test_custom_capacity(self):
        wm = WorkingMemory(capacity=1)
        wm.add("only")
        wm.add("second")
        assert len(wm.items) == 1
        assert wm.items[0] == "second"

    def test_add_at_capacity_evicts_lru(self):
        wm = WorkingMemory(capacity=3)
        wm.add("a")
        wm.add("b")
        wm.add("c")
        wm.access("a")
        wm.access("a")
        wm.add("d")
        assert "a" in wm.items
        assert "b" not in wm.items

    def test_get_recent_empty(self):
        wm = WorkingMemory()
        recent = wm.get_recent(5)
        assert recent == []

    def test_get_recent_n_larger_than_items(self):
        wm = WorkingMemory()
        wm.add("x")
        recent = wm.get_recent(10)
        assert recent == ["x"]

    def test_access_unknown_item(self):
        wm = WorkingMemory()
        wm.access("unknown")
        assert wm.access_count["unknown"] == 1

    def test_capacity_zero_raises(self):
        wm = WorkingMemory(capacity=0)
        with pytest.raises(ValueError):
            wm.add("item")

    def test_capacity_one(self):
        wm = WorkingMemory(capacity=1)
        wm.add("first")
        wm.add("second")
        assert wm.items == ["second"]

    def test_add_same_item_twice(self):
        wm = WorkingMemory(capacity=5)
        wm.add("x")
        wm.add("x")
        assert wm.items.count("x") == 2


# ---------------------------------------------------------------------------
# ThoughtStep
# ---------------------------------------------------------------------------

class TestThoughtStep:
    def test_defaults(self):
        step = ThoughtStep(step_id=0, thought="test", reasoning_type="decomp", confidence=0.8)
        assert step.parent_id is None
        assert step.children_ids == []
        assert step.value == 0.0
        assert step.is_final is False

    def test_custom_fields(self):
        step = ThoughtStep(
            step_id=1, thought="analysis", reasoning_type="analysis",
            confidence=0.9, parent_id=0, children_ids=[2, 3],
            value=0.7, is_final=True,
        )
        assert step.parent_id == 0
        assert step.children_ids == [2, 3]
        assert step.is_final is True

    def test_step_id_zero(self):
        step = ThoughtStep(step_id=0, thought="a", reasoning_type="root", confidence=1.0)
        assert step.step_id == 0

    def test_negative_confidence(self):
        step = ThoughtStep(step_id=0, thought="a", reasoning_type="test", confidence=-0.5)
        assert step.confidence == -0.5

    def test_confidence_boundary(self):
        step = ThoughtStep(step_id=0, thought="a", reasoning_type="test", confidence=0.0)
        assert step.confidence == 0.0
        step2 = ThoughtStep(step_id=1, thought="b", reasoning_type="test", confidence=1.0)
        assert step2.confidence == 1.0

    def test_children_ids_custom(self):
        step = ThoughtStep(step_id=5, thought="x", reasoning_type="branch", confidence=0.6, children_ids=[6, 7, 8])
        assert step.children_ids == [6, 7, 8]

    def test_value_default(self):
        step = ThoughtStep(step_id=0, thought="a", reasoning_type="root", confidence=1.0)
        assert step.value == 0.0

    def test_value_custom(self):
        step = ThoughtStep(step_id=0, thought="a", reasoning_type="root", confidence=1.0, value=0.95)
        assert step.value == 0.95


# ---------------------------------------------------------------------------
# ReasoningResult
# ---------------------------------------------------------------------------

class TestReasoningResult:
    def test_construction(self):
        result = ReasoningResult(
            conclusion="42", confidence=0.95,
            mode=ReasoningMode.CHAIN_OF_THOUGHT,
            steps=[], metadata={}, execution_time_ms=1.0,
        )
        assert result.conclusion == "42"
        assert result.confidence == 0.95

    def test_metadata_dict(self):
        result = ReasoningResult(
            conclusion="x", confidence=0.5,
            mode=ReasoningMode.CAUSAL,
            steps=[], metadata={"key": "value"}, execution_time_ms=0.1,
        )
        assert result.metadata["key"] == "value"

    def test_mode_stored(self):
        result = ReasoningResult(
            conclusion="c", confidence=0.5,
            mode=ReasoningMode.REACT,
            steps=[], metadata={}, execution_time_ms=0.0,
        )
        assert result.mode == ReasoningMode.REACT

    def test_steps_stored(self):
        steps = [ThoughtStep(0, "a", "root", 1.0)]
        result = ReasoningResult(
            conclusion="c", confidence=0.5,
            mode=ReasoningMode.CHAIN_OF_THOUGHT,
            steps=steps, metadata={}, execution_time_ms=0.0,
        )
        assert len(result.steps) == 1

    def test_execution_time_ms(self):
        result = ReasoningResult(
            conclusion="c", confidence=0.5,
            mode=ReasoningMode.CHAIN_OF_THOUGHT,
            steps=[], metadata={}, execution_time_ms=123.45,
        )
        assert result.execution_time_ms == 123.45

    def test_empty_metadata(self):
        result = ReasoningResult(
            conclusion="c", confidence=0.5,
            mode=ReasoningMode.CHAIN_OF_THOUGHT,
            steps=[], metadata={}, execution_time_ms=0.0,
        )
        assert result.metadata == {}

    def test_multiple_steps(self):
        steps = [
            ThoughtStep(0, "a", "decomp", 0.8),
            ThoughtStep(1, "b", "analysis", 0.9),
            ThoughtStep(2, "c", "synthesis", 0.95),
        ]
        result = ReasoningResult(
            conclusion="c", confidence=0.9,
            mode=ReasoningMode.CHAIN_OF_THOUGHT,
            steps=steps, metadata={}, execution_time_ms=50.0,
        )
        assert len(result.steps) == 3


# ---------------------------------------------------------------------------
# ChainOfThought
# ---------------------------------------------------------------------------

class TestChainOfThought:
    def test_basic_reason(self):
        cot = ChainOfThought()
        result = asyncio.run(cot.reason("What is 1+1?"))
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.CHAIN_OF_THOUGHT

    def test_custom_llm(self):
        async def my_llm(prompt):
            return "therefore the answer is 42"
        cot = ChainOfThought(llm_call=my_llm)
        result = asyncio.run(cot.reason("Problem"))
        assert "42" in result.conclusion

    def test_evaluate_confidence_high(self):
        cot = ChainOfThought()
        conf = cot._evaluate_confidence("therefore the result. it is correct.")
        assert conf >= 0.7

    def test_evaluate_confidence_low(self):
        cot = ChainOfThought()
        conf = cot._evaluate_confidence("maybe")
        assert conf == 0.5

    def test_extract_subproblem_remaining(self):
        cot = ChainOfThought()
        sub = cot._extract_subproblem("We need to compute: remaining: the final sum")
        assert sub == "the final sum"

    def test_extract_subproblem_now(self):
        cot = ChainOfThought()
        sub = cot._extract_subproblem("now we need to verify the result")
        assert sub is not None

    def test_extract_subproblem_none(self):
        cot = ChainOfThought()
        sub = cot._extract_subproblem("no subproblem here")
        assert sub is None

    def test_extract_conclusion_therefore(self):
        cot = ChainOfThought()
        conc = cot._extract_conclusion("therefore the answer is correct")
        assert "answer is correct" in conc

    def test_extract_conclusion_answer(self):
        cot = ChainOfThought()
        conc = cot._extract_conclusion("answer: 42")
        assert "42" in conc

    def test_extract_conclusion_fallback(self):
        cot = ChainOfThought()
        conc = cot._extract_conclusion("just some text")
        assert conc == "just some text"

    def test_max_steps_limit(self):
        async def slow_llm(prompt):
            return "thinking step"
        cot = ChainOfThought(llm_call=slow_llm)
        result = asyncio.run(cot.reason("Problem", max_steps=2))
        assert len(result.steps) <= 2

    def test_confidence_threshold_early_stop(self):
        async def confident_llm(prompt):
            return "therefore the solution is found"
        cot = ChainOfThought(llm_call=confident_llm)
        result = asyncio.run(cot.reason("Problem", confidence_threshold=0.5))
        assert result.steps[-1].is_final is True

    def test_steps_populated(self):
        cot = ChainOfThought()
        result = asyncio.run(cot.reason("Problem"))
        assert len(result.steps) > 0

    def test_evaluate_confidence_multiple_indicators(self):
        cot = ChainOfThought()
        conf = cot._evaluate_confidence("therefore thus hence. so conclude.")
        assert conf >= 0.9

    def test_evaluate_confidence_single_sentence(self):
        cot = ChainOfThought()
        conf = cot._evaluate_confidence("simple thought")
        assert conf == 0.5

    def test_extract_subproblem_next(self):
        cot = ChainOfThought()
        sub = cot._extract_subproblem("next: compute the sum")
        assert sub == "compute the sum"

    def test_extract_conclusion_thus(self):
        cot = ChainOfThought()
        conc = cot._extract_conclusion("thus the answer is 5")
        assert "answer is 5" in conc

    def test_extract_conclusion_hence(self):
        cot = ChainOfThought()
        conc = cot._extract_conclusion("hence the result is found")
        assert "result is found" in conc

    def test_extract_conclusion_so(self):
        cot = ChainOfThought()
        conc = cot._extract_conclusion("so the final answer is 7")
        assert "final answer is 7" in conc

    def test_extract_conclusion_colon(self):
        cot = ChainOfThought()
        conc = cot._extract_conclusion("conclusion: done")
        assert "done" in conc

    def test_result_metadata_solved(self):
        async def confident_llm(prompt):
            return "therefore the solution is found"
        cot = ChainOfThought(llm_call=confident_llm)
        result = asyncio.run(cot.reason("Problem", confidence_threshold=0.5))
        assert result.metadata["solved"] is True


# ---------------------------------------------------------------------------
# TreeOfThoughts
# ---------------------------------------------------------------------------

class TestTreeOfThoughts:
    def test_basic_reason(self):
        tot = TreeOfThoughts()
        result = asyncio.run(tot.reason("Problem"))
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.TREE_OF_THOUGHTS

    def test_beam_width(self):
        tot = TreeOfThoughts(beam_width=2)
        assert tot.beam_width == 2

    def test_evaluate_node_value(self):
        tot = TreeOfThoughts()
        val = tot._evaluate_node("therefore the conclusion is X")
        assert val >= 0.5

    def test_evaluate_node_no_keywords(self):
        tot = TreeOfThoughts()
        val = tot._evaluate_node("just some text")
        assert val == 0.5

    def test_is_solution_true(self):
        tot = TreeOfThoughts()
        assert tot._is_solution("answer: 42") is True

    def test_is_solution_false(self):
        tot = TreeOfThoughts()
        assert tot._is_solution("thinking about it") is False

    def test_prune_nodes(self):
        tot = TreeOfThoughts(beam_width=2)
        tot.nodes = {
            0: ThoughtStep(0, "a", "branch", 0.5, value=0.5),
            1: ThoughtStep(1, "b", "branch", 0.5, value=0.9),
            2: ThoughtStep(2, "c", "branch", 0.5, value=0.1),
        }
        pruned = tot._prune_nodes([0, 1, 2], 0.3)
        assert len(pruned) == 2
        assert 1 in pruned

    def test_get_path(self):
        tot = TreeOfThoughts()
        tot.nodes = {
            0: ThoughtStep(0, "root", "root", 1.0, parent_id=None),
            1: ThoughtStep(1, "child", "branch", 0.8, parent_id=0),
        }
        path = tot._get_path(1)
        assert path == [0, 1]

    def test_nodes_created(self):
        tot = TreeOfThoughts(beam_width=2)
        result = asyncio.run(tot.reason("Problem", max_depth=2))
        assert len(tot.nodes) > 1

    def test_is_solution_solution_keyword(self):
        tot = TreeOfThoughts()
        assert tot._is_solution("solution: 42") is True

    def test_is_solution_therefore(self):
        tot = TreeOfThoughts()
        assert tot._is_solution("therefore done") is True

    def test_is_solution_conclusion(self):
        tot = TreeOfThoughts()
        assert tot._is_solution("conclusion reached") is True

    def test_evaluate_node_solution_keyword(self):
        tot = TreeOfThoughts()
        val = tot._evaluate_node("the solution is found")
        assert val >= 0.6

    def test_get_path_root(self):
        tot = TreeOfThoughts()
        tot.nodes = {
            0: ThoughtStep(0, "root", "root", 1.0, parent_id=None),
        }
        path = tot._get_path(0)
        assert path == [0]

    def test_get_path_deep(self):
        tot = TreeOfThoughts()
        tot.nodes = {
            0: ThoughtStep(0, "root", "root", 1.0, parent_id=None),
            1: ThoughtStep(1, "a", "branch", 0.8, parent_id=0),
            2: ThoughtStep(2, "b", "branch", 0.7, parent_id=1),
            3: ThoughtStep(3, "c", "branch", 0.6, parent_id=2),
        }
        path = tot._get_path(3)
        assert path == [0, 1, 2, 3]

    def test_prune_nodes_all_below_threshold(self):
        tot = TreeOfThoughts(beam_width=5)
        tot.nodes = {
            0: ThoughtStep(0, "a", "branch", 0.5, value=0.1),
            1: ThoughtStep(1, "b", "branch", 0.5, value=0.2),
        }
        pruned = tot._prune_nodes([0, 1], 0.5)
        assert len(pruned) == 0

    def test_beam_width_one(self):
        tot = TreeOfThoughts(beam_width=1)
        result = asyncio.run(tot.reason("Problem", max_depth=1))
        assert isinstance(result, ReasoningResult)


# ---------------------------------------------------------------------------
# SelfConsistency
# ---------------------------------------------------------------------------

class TestSelfConsistency:
    def test_basic_reason(self):
        sc = SelfConsistency(num_paths=3)
        result = asyncio.run(sc.reason("Problem"))
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.SELF_CONSISTENCY

    def test_extract_conclusion_answer(self):
        sc = SelfConsistency()
        conc = sc._extract_conclusion("The answer: 42 is correct")
        assert "42" in conc

    def test_extract_conclusion_fallback(self):
        sc = SelfConsistency()
        conc = sc._extract_conclusion("short")
        assert isinstance(conc, str)

    def test_majority_vote(self):
        sc = SelfConsistency()
        vote = sc._majority_vote(["A", "A", "B", "C", "A"])
        assert vote == "A"

    def test_num_paths(self):
        sc = SelfConsistency(num_paths=5)
        assert sc.num_paths == 5

    def test_majority_vote_single(self):
        sc = SelfConsistency()
        vote = sc._majority_vote(["only_one"])
        assert vote == "only_one"

    def test_majority_vote_tie(self):
        sc = SelfConsistency()
        vote = sc._majority_vote(["A", "B"])
        assert vote in ["A", "B"]

    def test_extract_conclusion_no_answer_keyword(self):
        sc = SelfConsistency()
        conc = sc._extract_conclusion("this is a long reasoning text that goes beyond one hundred characters to test the fallback path properly")
        assert isinstance(conc, str)

    def test_metadata_num_paths(self):
        sc = SelfConsistency(num_paths=3)
        result = asyncio.run(sc.reason("Problem"))
        assert result.metadata["num_paths"] == 3

    def test_steps_populated(self):
        sc = SelfConsistency(num_paths=2)
        result = asyncio.run(sc.reason("Problem"))
        assert len(result.steps) > 0


# ---------------------------------------------------------------------------
# ConstitutionalAI
# ---------------------------------------------------------------------------

class TestConstitutionalAI:
    def test_basic_reason(self):
        ai = ConstitutionalAI()
        result = asyncio.run(ai.reason("Problem"))
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.CONSTITUTIONAL

    def test_custom_principles(self):
        ai = ConstitutionalAI()
        result = asyncio.run(ai.reason("Problem", custom_principles=["Be helpful"]))
        assert result.metadata["principles_used"] == 1

    def test_default_principles(self):
        ai = ConstitutionalAI()
        assert len(ai.principles) == 5

    def test_review_history_init(self):
        ai = ConstitutionalAI()
        assert ai.review_history == []

    def test_principles_copy(self):
        ai = ConstitutionalAI()
        original = ai.principles.copy()
        asyncio.run(ai.reason("Problem", custom_principles=["New principle"]))
        assert ai.principles == ["New principle"]
        assert original != ai.principles

    def test_steps_three(self):
        ai = ConstitutionalAI()
        result = asyncio.run(ai.reason("Problem"))
        assert len(result.steps) == 3

    def test_confidence_high(self):
        ai = ConstitutionalAI()
        result = asyncio.run(ai.reason("Problem"))
        assert result.confidence == 0.9

    def test_custom_principles_multiple(self):
        ai = ConstitutionalAI()
        result = asyncio.run(ai.reason("Problem", custom_principles=["A", "B", "C"]))
        assert result.metadata["principles_used"] == 3


# ---------------------------------------------------------------------------
# CausalReasoning
# ---------------------------------------------------------------------------

class TestCausalReasoning:
    def test_basic_reason(self):
        cr = CausalReasoning()
        result = asyncio.run(cr.reason("Because X, therefore Y"))
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.CAUSAL

    def test_identify_causes(self):
        cr = CausalReasoning()
        causes = cr._identify_causes("It happened because of rain")
        assert len(causes) > 0
        assert "rain" in causes[0]

    def test_identify_causes_due_to(self):
        cr = CausalReasoning()
        causes = cr._identify_causes("Failure due to low battery")
        assert len(causes) > 0

    def test_identify_effects(self):
        cr = CausalReasoning()
        effects = cr._identify_effects("It rained, therefore the ground is wet")
        assert len(effects) > 0

    def test_identify_relationships(self):
        cr = CausalReasoning()
        rels = cr._identify_relationships("because X therefore Y")
        assert len(rels) > 0

    def test_build_causal_conclusion_empty(self):
        cr = CausalReasoning()
        conc = cr._build_causal_conclusion([], [], [])
        assert "Unable" in conc

    def test_build_causal_conclusion(self):
        cr = CausalReasoning()
        conc = cr._build_causal_conclusion(["A"], ["B"], [("A", "B", 0.8)])
        assert "A" in conc
        assert "B" in conc

    def test_identify_causes_caused_by(self):
        cr = CausalReasoning()
        causes = cr._identify_causes("The failure was caused by overheating")
        assert len(causes) > 0

    def test_identify_causes_leads_to(self):
        cr = CausalReasoning()
        causes = cr._identify_causes("Pressure leads to deformation")
        assert len(causes) > 0

    def test_identify_causes_results_in(self):
        cr = CausalReasoning()
        causes = cr._identify_causes("Heat results in expansion")
        assert len(causes) > 0

    def test_identify_effects_thus(self):
        cr = CausalReasoning()
        effects = cr._identify_effects("The input changed, thus the output changed")
        assert len(effects) > 0

    def test_identify_effects_consequently(self):
        cr = CausalReasoning()
        effects = cr._identify_effects("It rained, consequently the road is wet")
        assert len(effects) > 0

    def test_build_causal_conclusion_multiple(self):
        cr = CausalReasoning()
        conc = cr._build_causal_conclusion(["A", "B"], ["C", "D"], [("A", "C", 0.8), ("B", "D", 0.9)])
        assert "A" in conc
        assert "C" in conc

    def test_metadata_causes_count(self):
        cr = CausalReasoning()
        result = asyncio.run(cr.reason("Because X, therefore Y"))
        assert "causes" in result.metadata
        assert "effects" in result.metadata

    def test_causal_graph_init(self):
        cr = CausalReasoning()
        assert cr.causal_graph == {}


# ---------------------------------------------------------------------------
# SyllogismReasoning
# ---------------------------------------------------------------------------

class TestSyllogismReasoning:
    def test_basic_reason(self):
        sr = SyllogismReasoning()
        result = asyncio.run(sr.reason("All humans are mortal. Socrates is human."))
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.SYLLOGISM

    def test_parse_premises(self):
        sr = SyllogismReasoning()
        premises = sr._parse_premises("All A are B are long enough text. Some C are D are also long. No E is F has length.")
        assert len(premises) >= 2

    def test_identify_figure(self):
        sr = SyllogismReasoning()
        fig = sr._identify_figure(["All A are B", "Some C are D"])
        assert fig in [1, 2, 3, 4]

    def test_identify_mood(self):
        sr = SyllogismReasoning()
        mood = sr._identify_mood(["All A are B", "Some C are D", "Therefore X"])
        assert mood == "AAA"

    def test_apply_rules_valid(self):
        sr = SyllogismReasoning()
        valid, _ = sr._apply_syllogistic_rules(1, "AAA")
        assert valid is True

    def test_apply_rules_invalid(self):
        sr = SyllogismReasoning()
        valid, _ = sr._apply_syllogistic_rules(1, "OOO")
        assert valid is False

    def test_derive_conclusion(self):
        sr = SyllogismReasoning()
        conc = sr._derive_conclusion(["Premise 1", "Premise 2"])
        assert "Therefore" in conc

    def test_derive_conclusion_insufficient(self):
        sr = SyllogismReasoning()
        conc = sr._derive_conclusion([])
        assert "Insufficient" in conc

    def test_parse_premises_short(self):
        sr = SyllogismReasoning()
        premises = sr._parse_premises("short")
        assert len(premises) >= 0

    def test_parse_premises_default(self):
        sr = SyllogismReasoning()
        premises = sr._parse_premises("")
        assert len(premises) >= 0

    def test_identify_mood_two_premises(self):
        sr = SyllogismReasoning()
        mood = sr._identify_mood(["All A are B", "Some C are D"])
        assert mood in ["AAA", "AA"]

    def test_apply_rules_valid_eae(self):
        sr = SyllogismReasoning()
        valid, _ = sr._apply_syllogistic_rules(1, "EAE")
        assert valid is True

    def test_apply_rules_valid_aii(self):
        sr = SyllogismReasoning()
        valid, _ = sr._apply_syllogistic_rules(1, "AII")
        assert valid is True

    def test_apply_rules_figure2(self):
        sr = SyllogismReasoning()
        valid, _ = sr._apply_syllogistic_rules(2, "EAE")
        assert valid is True

    def test_apply_rules_figure3(self):
        sr = SyllogismReasoning()
        valid, _ = sr._apply_syllogistic_rules(3, "AAI")
        assert valid is True

    def test_apply_rules_figure4(self):
        sr = SyllogismReasoning()
        valid, _ = sr._apply_syllogistic_rules(4, "AAI")
        assert valid is True

    def test_derive_conclusion_single(self):
        sr = SyllogismReasoning()
        conc = sr._derive_conclusion(["Only one premise"])
        assert "Insufficient" in conc

    def test_metadata_validity(self):
        sr = SyllogismReasoning()
        result = asyncio.run(sr.reason("All humans are mortal. Socrates is human."))
        assert "valid" in result.metadata
        assert "figure" in result.metadata
        assert "mood" in result.metadata


# ---------------------------------------------------------------------------
# ReActReasoning
# ---------------------------------------------------------------------------

class TestReActReasoning:
    def test_basic_reason(self):
        rr = ReActReasoning()
        result = asyncio.run(rr.reason("Problem"))
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.REACT

    def test_with_tools(self):
        tools = {"search": lambda q: f"Result for {q}"}
        rr = ReActReasoning(tool_registry=tools)
        result = asyncio.run(rr.reason("Problem"))
        assert result.metadata["actions"] >= 0

    def test_is_solved_true(self):
        rr = ReActReasoning()
        assert rr._is_solved("answer: 42") is True

    def test_is_solved_false(self):
        rr = ReActReasoning()
        assert rr._is_solved("thinking") is False

    def test_max_steps(self):
        rr = ReActReasoning()
        result = asyncio.run(rr.reason("Problem", max_steps=2))
        assert len(result.steps) <= 4  # 2 think + 2 act

    def test_is_solved_solution(self):
        rr = ReActReasoning()
        assert rr._is_solved("solution: found") is True

    def test_is_solved_conclusion(self):
        rr = ReActReasoning()
        assert rr._is_solved("conclusion: done") is True

    def test_is_solved_final_answer(self):
        rr = ReActReasoning()
        assert rr._is_solved("final answer is 42") is True

    def test_no_tools(self):
        rr = ReActReasoning()
        result = asyncio.run(rr.reason("Problem"))
        assert result.metadata["actions"] == 0

    def test_multiple_tools(self):
        tools = {"search": lambda q: "result1", "calc": lambda q: "result2"}
        rr = ReActReasoning(tool_registry=tools)
        result = asyncio.run(rr.reason("Problem", max_steps=3))
        assert isinstance(result, ReasoningResult)

    def test_steps_count(self):
        rr = ReActReasoning()
        result = asyncio.run(rr.reason("Problem", max_steps=3))
        assert len(result.steps) > 0

    def test_action_history_init(self):
        rr = ReActReasoning()
        assert rr.action_history == []

    def test_tool_registry_init(self):
        rr = ReActReasoning()
        assert rr.tool_registry == {}


# ---------------------------------------------------------------------------
# advanced_reasoning factory
# ---------------------------------------------------------------------------

class TestAdvancedReasoning:
    @pytest.mark.asyncio
    async def test_cot_mode(self):
        result = await advanced_reasoning("Problem", ReasoningMode.CHAIN_OF_THOUGHT)
        assert result.mode == ReasoningMode.CHAIN_OF_THOUGHT

    @pytest.mark.asyncio
    async def test_tot_mode(self):
        result = await advanced_reasoning("Problem", ReasoningMode.TREE_OF_THOUGHTS)
        assert result.mode == ReasoningMode.TREE_OF_THOUGHTS

    @pytest.mark.asyncio
    async def test_sc_mode(self):
        result = await advanced_reasoning("Problem", ReasoningMode.SELF_CONSISTENCY)
        assert result.mode == ReasoningMode.SELF_CONSISTENCY

    @pytest.mark.asyncio
    async def test_constitutional_mode(self):
        result = await advanced_reasoning("Problem", ReasoningMode.CONSTITUTIONAL)
        assert result.mode == ReasoningMode.CONSTITUTIONAL

    @pytest.mark.asyncio
    async def test_causal_mode(self):
        result = await advanced_reasoning("Because X therefore Y", ReasoningMode.CAUSAL)
        assert result.mode == ReasoningMode.CAUSAL

    @pytest.mark.asyncio
    async def test_syllogism_mode(self):
        result = await advanced_reasoning("All A are B. Some C are D.", ReasoningMode.SYLLOGISM)
        assert result.mode == ReasoningMode.SYLLOGISM

    @pytest.mark.asyncio
    async def test_react_mode(self):
        result = await advanced_reasoning("Problem", ReasoningMode.REACT)
        assert result.mode == ReasoningMode.REACT

    @pytest.mark.asyncio
    async def test_counterfactuMode(self):
        result = await advanced_reasoning("Problem", ReasoningMode.COUNTERFACTUAL)
        assert result.mode == ReasoningMode.CHAIN_OF_THOUGHT

    @pytest.mark.asyncio
    async def test_custom_llm(self):
        async def my_llm(prompt):
            return "therefore answer is 42"
        result = await advanced_reasoning("Problem", ReasoningMode.CHAIN_OF_THOUGHT, llm_call=my_llm)
        assert "42" in result.conclusion

    @pytest.mark.asyncio
    async def test_tot_beam_width(self):
        result = await advanced_reasoning("Problem", ReasoningMode.TREE_OF_THOUGHTS, beam_width=2)
        assert result.mode == ReasoningMode.TREE_OF_THOUGHTS

    @pytest.mark.asyncio
    async def test_sc_num_paths(self):
        result = await advanced_reasoning("Problem", ReasoningMode.SELF_CONSISTENCY, num_paths=3)
        assert result.metadata["num_paths"] == 3


# ---------------------------------------------------------------------------
# FormalLogicEngine — Term, Predicate, WellFormedFormula
# ---------------------------------------------------------------------------

class TestTerm:
    def test_basic_term(self):
        t = Term(name="socrates")
        assert t.name == "socrates"
        assert t.is_variable is False

    def test_variable_term(self):
        t = Term(name="X", is_variable=True)
        assert t.is_variable is True

    def test_function_term(self):
        t = Term(name="f", is_function=True, arguments=[Term("a"), Term("b")])
        assert t.is_function is True
        assert len(t.arguments) == 2

    def test_term_default_not_variable(self):
        t = Term(name="a")
        assert t.is_variable is False
        assert t.is_function is False

    def test_term_arguments_default(self):
        t = Term(name="a")
        assert t.arguments == []


class TestPredicate:
    def test_basic_predicate(self):
        p = Predicate(name="human", terms=[Term("socrates")])
        assert p.name == "human"
        assert p.negated is False

    def test_negated_predicate(self):
        p = Predicate(name="mortal", terms=[Term("socrates")], negated=True)
        assert p.negated is True

    def test_hash(self):
        p1 = Predicate(name="human", terms=[Term("socrates")])
        p2 = Predicate(name="human", terms=[Term("socrates")])
        assert hash(p1) == hash(p2)

    def test_hash_different(self):
        p1 = Predicate(name="human", terms=[Term("socrates")])
        p2 = Predicate(name="mortal", terms=[Term("socrates")])
        assert hash(p1) != hash(p2)

    def test_predicate_multiple_terms(self):
        p = Predicate(name="loves", terms=[Term("socrates"), Term("plato")])
        assert len(p.terms) == 2

    def test_hash_negated(self):
        p1 = Predicate(name="human", terms=[Term("socrates")], negated=True)
        p2 = Predicate(name="human", terms=[Term("socrates")], negated=True)
        assert hash(p1) == hash(p2)

    def test_hash_negated_vs_nonnegated(self):
        p1 = Predicate(name="human", terms=[Term("socrates")], negated=False)
        p2 = Predicate(name="human", terms=[Term("socrates")], negated=True)
        assert hash(p1) != hash(p2)


class TestWellFormedFormula:
    def test_predicate_formula(self):
        pred = Predicate(name="human", terms=[Term("socrates")])
        wff = WellFormedFormula(predicate=pred)
        assert wff.predicate == pred

    def test_implication_formula(self):
        left = WellFormedFormula(predicate=Predicate(name="human", terms=[Term("x")]))
        right = WellFormedFormula(predicate=Predicate(name="mortal", terms=[Term("x")]))
        wff = WellFormedFormula(operator=LogicalOperator.IMPLIES, left=left, right=right)
        assert wff.operator == LogicalOperator.IMPLIES

    def test_and_formula(self):
        left = WellFormedFormula(predicate=Predicate(name="a", terms=[Term("x")]))
        right = WellFormedFormula(predicate=Predicate(name="b", terms=[Term("x")]))
        wff = WellFormedFormula(operator=LogicalOperator.AND, left=left, right=right)
        assert wff.operator == LogicalOperator.AND

    def test_not_formula(self):
        pred = Predicate(name="human", terms=[Term("socrates")])
        wff = WellFormedFormula(operator=LogicalOperator.NOT, left=WellFormedFormula(predicate=pred))
        assert wff.operator == LogicalOperator.NOT

    def test_formula_defaults(self):
        wff = WellFormedFormula()
        assert wff.predicate is None
        assert wff.operator is None
        assert wff.left is None
        assert wff.right is None


class TestSubstitution:
    def test_basic_substitution(self):
        s = Substitution(mapping={"X": Term("socrates")})
        assert s.mapping["X"].name == "socrates"

    def test_empty_substitution(self):
        s = Substitution()
        assert len(s.mapping) == 0

    def test_multiple_mappings(self):
        s = Substitution(mapping={"X": Term("a"), "Y": Term("b")})
        assert len(s.mapping) == 2
        assert s.mapping["Y"].name == "b"

    def test_substitution_with_variable(self):
        s = Substitution(mapping={"X": Term("Y", is_variable=True)})
        assert s.mapping["X"].is_variable is True


class TestLogicalOperator:
    def test_all_operators(self):
        ops = [LogicalOperator.AND, LogicalOperator.OR, LogicalOperator.NOT,
               LogicalOperator.IMPLIES, LogicalOperator.IFF,
               LogicalOperator.FORALL, LogicalOperator.EXISTS]
        assert len(ops) == 7

    def test_operator_values(self):
        assert LogicalOperator.AND.value == "∧"
        assert LogicalOperator.OR.value == "∨"
        assert LogicalOperator.NOT.value == "¬"
        assert LogicalOperator.IMPLIES.value == "→"
        assert LogicalOperator.IFF.value == "↔"
        assert LogicalOperator.FORALL.value == "∀"
        assert LogicalOperator.EXISTS.value == "∃"


# ---------------------------------------------------------------------------
# FormalLogicEngine
# ---------------------------------------------------------------------------

class TestFormalLogicEngine:
    def test_init(self):
        engine = FormalLogicEngine()
        assert engine.knowledge_base == []

    def test_assert_fact(self):
        engine = FormalLogicEngine()
        wff = WellFormedFormula(
            predicate=Predicate(name="human", terms=[Term("socrates")])
        )
        engine.assert_fact(wff)
        assert len(engine.knowledge_base) == 1

    def test_assert_predicate(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        assert len(engine.knowledge_base) == 1

    def test_query_direct(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        result = engine.query(Predicate(name="human", terms=[Term("socrates")]))
        assert result is True

    def test_query_not_found(self):
        engine = FormalLogicEngine()
        result = engine.query(Predicate(name="mortal", terms=[Term("socrates")]))
        assert result is False

    def test_modus_ponens(self):
        engine = FormalLogicEngine()
        human_x = WellFormedFormula(predicate=Predicate(name="human", terms=[Term("x", is_variable=True)]))
        mortal_x = WellFormedFormula(predicate=Predicate(name="mortal", terms=[Term("x", is_variable=True)]))
        implication = WellFormedFormula(
            operator=LogicalOperator.IMPLIES,
            left=human_x,
            right=mortal_x,
        )
        engine.assert_fact(implication)
        engine.assert_predicate("human", "socrates")
        result = engine.query(Predicate(name="mortal", terms=[Term("socrates")]))
        assert result is True

    def test_unify_same_predicates(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="human", terms=[Term("socrates")])
        p2 = Predicate(name="human", terms=[Term("socrates")])
        subst = engine._unify(p1, p2)
        assert subst is not None

    def test_unify_different_names(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="human", terms=[Term("socrates")])
        p2 = Predicate(name="mortal", terms=[Term("socrates")])
        subst = engine._unify(p1, p2)
        assert subst is None

    def test_unify_variable(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="human", terms=[Term("X", is_variable=True)])
        p2 = Predicate(name="human", terms=[Term("socrates")])
        subst = engine._unify(p1, p2)
        assert subst is not None
        assert subst.mapping["X"].name == "socrates"

    def test_unify_terms_same(self):
        engine = FormalLogicEngine()
        subst = engine._unify_terms(Term("a"), Term("a"), Substitution())
        assert subst is not None

    def test_unify_terms_different(self):
        engine = FormalLogicEngine()
        subst = engine._unify_terms(Term("a"), Term("b"), Substitution())
        assert subst is None

    def test_occurs_check_true(self):
        engine = FormalLogicEngine()
        var = Term("X", is_variable=True)
        term = Term("f", is_function=True, arguments=[Term("X", is_variable=True)])
        assert engine._occurs_check(var, term, Substitution()) is True

    def test_occurs_check_false(self):
        engine = FormalLogicEngine()
        var = Term("X", is_variable=True)
        term = Term("a")
        assert engine._occurs_check(var, term, Substitution()) is False

    def test_apply_term_substitution(self):
        engine = FormalLogicEngine()
        subst = Substitution(mapping={"X": Term("socrates")})
        result = engine._apply_term_substitution(Term("X", is_variable=True), subst)
        assert result.name == "socrates"

    def test_apply_substitution_to_predicate(self):
        engine = FormalLogicEngine()
        subst = Substitution(mapping={"X": Term("socrates")})
        pred = Predicate(name="human", terms=[Term("X", is_variable=True)])
        result = engine._apply_substitution(pred, subst)
        assert result.terms[0].name == "socrates"

    def test_resolution_simple(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        result = engine.resolution(Predicate(name="human", terms=[Term("socrates")]))
        assert result is True

    def test_resolution_contradiction(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        result = engine.resolution(Predicate(name="human", terms=[Term("plato")]))
        assert result is False

    def test_prove_syllogism(self):
        engine = FormalLogicEngine()
        result = engine.prove_syllogism(
            ("All", "are", "mortal"),
            ("Some", "are", "human"),
            ("Some", "are", "mortal"),
        )
        assert "valid" in result
        assert "figure" in result
        assert "mood" in result

    def test_to_categorical(self):
        engine = FormalLogicEngine()
        cat = engine._to_categorical(("All", "are", "mortal"))
        assert cat[0] == "A"

    def test_determine_figure(self):
        engine = FormalLogicEngine()
        fig = engine._determine_figure(("A", "S", "are", "P"), ("A", "S", "are", "P"))
        assert fig in [1, 2, 3, 4]

    def test_check_validity_valid(self):
        engine = FormalLogicEngine()
        valid, reason = engine._check_syllogism_validity("AAA", 1, (), (), ())
        assert valid is True

    def test_check_validity_invalid(self):
        engine = FormalLogicEngine()
        valid, reason = engine._check_syllogism_validity("OOO", 1, (), (), ())
        assert valid is False

    def test_inference_history(self):
        engine = FormalLogicEngine()
        engine.prove_syllogism(("All", "are", "mortal"), ("Some", "are", "human"), ("Some", "are", "mortal"))
        assert len(engine.inference_history) == 1

    def test_multiple_assertions(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        engine.assert_predicate("mortal", "socrates")
        assert len(engine.knowledge_base) == 2

    def test_forward_chain_multiple(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("a", "x")
        engine.assert_predicate("b", "x")
        result_a = engine.query(Predicate(name="a", terms=[Term("x")]))
        result_b = engine.query(Predicate(name="b", terms=[Term("x")]))
        assert result_a is True
        assert result_b is True

    def test_apply_term_substitution_not_variable(self):
        engine = FormalLogicEngine()
        subst = Substitution(mapping={"X": Term("socrates")})
        result = engine._apply_term_substitution(Term("a"), subst)
        assert result.name == "a"

    def test_apply_substitution_negated(self):
        engine = FormalLogicEngine()
        subst = Substitution(mapping={"X": Term("socrates")})
        pred = Predicate(name="human", terms=[Term("X", is_variable=True)], negated=True)
        result = engine._apply_substitution(pred, subst)
        assert result.negated is True
        assert result.terms[0].name == "socrates"

    def test_query_empty_kb(self):
        engine = FormalLogicEngine()
        result = engine.query(Predicate(name="anything", terms=[Term("x")]))
        assert result is False

    def test_prove_syllogism_records_history(self):
        engine = FormalLogicEngine()
        engine.prove_syllogism(("All", "are", "mortal"), ("Some", "are", "human"), ("Some", "are", "mortal"))
        engine.prove_syllogism(("No", "are", "mortal"), ("Some", "are", "human"), ("Some", "are", "mortal"))
        assert len(engine.inference_history) == 2

    def test_format_categorical(self):
        engine = FormalLogicEngine()
        result = engine._format_categorical(("A", "S", "are", "P"))
        assert "A" in result
        assert "S" in result

    def test_to_categorical_quantifier(self):
        engine = FormalLogicEngine()
        cat = engine._to_categorical(("Some", "are", "mortal"))
        assert cat[0] == "S"

    def test_check_validity_figure2(self):
        engine = FormalLogicEngine()
        valid, _ = engine._check_syllogism_validity("EAE", 2, (), (), ())
        assert valid is True

    def test_check_validity_figure3(self):
        engine = FormalLogicEngine()
        valid, _ = engine._check_syllogism_validity("AAI", 3, (), (), ())
        assert valid is True

    def test_check_validity_figure4(self):
        engine = FormalLogicEngine()
        valid, _ = engine._check_syllogism_validity("AAI", 4, (), (), ())
        assert valid is True

    def test_occurs_check_function_nested(self):
        engine = FormalLogicEngine()
        var = Term("X", is_variable=True)
        inner = Term("Y", is_variable=True)
        term = Term("f", is_function=True, arguments=[Term("g", is_function=True, arguments=[inner])])
        assert engine._occurs_check(var, term, Substitution()) is False

    def test_unify_different_arity(self):
        engine = FormalLogicEngine()
        p1 = Predicate(name="human", terms=[Term("a"), Term("b")])
        p2 = Predicate(name="human", terms=[Term("a")])
        subst = engine._unify(p1, p2)
        assert subst is None


# ---------------------------------------------------------------------------
# DeepReasoning
# ---------------------------------------------------------------------------

class TestDeepReasoning:
    def test_basic_reason(self):
        dr = DeepReasoning()
        result = asyncio.run(dr.reason("Problem"))
        assert isinstance(result, ReasoningResult)

    def test_fallback_retrieval_cause(self):
        dr = DeepReasoning()
        results = asyncio.run(dr._fallback_retrieval("because of this"))
        assert len(results) > 0

    def test_fallback_retrieval_effect(self):
        dr = DeepReasoning()
        results = asyncio.run(dr._fallback_retrieval("therefore something"))
        assert len(results) > 0

    def test_fallback_retrieval_compare(self):
        dr = DeepReasoning()
        results = asyncio.run(dr._fallback_retrieval("however this differs"))
        assert len(results) > 0

    def test_fallback_retrieval_empty(self):
        dr = DeepReasoning()
        results = asyncio.run(dr._fallback_retrieval("random text"))
        assert len(results) == 0

    def test_build_context(self):
        dr = DeepReasoning()
        ctx = DeepReasoningContext(query="test", constraints=["c1"], assumptions=["a1"])
        context = asyncio.run(dr._build_context("Problem", [], ctx))
        assert "Problem" in context

    def test_build_context_with_knowledge(self):
        dr = DeepReasoning()
        know = RetrievedKnowledge(content="info", source=RetrievalSource.WORKING_MEMORY, relevance=0.8)
        ctx = DeepReasoningContext(query="test")
        context = asyncio.run(dr._build_context("Problem", [know], ctx))
        assert "info" in context

    def test_self_correct_no_issues(self):
        dr = DeepReasoning()
        steps = [ThoughtStep(0, "test", "analysis", 0.8)]
        ctx = DeepReasoningContext(query="test")
        result = asyncio.run(dr._self_correct(steps, "context", ctx))
        assert result["needs_revision"] is False

    def test_critique_step(self):
        dr = DeepReasoning()
        step = ThoughtStep(0, "thinking", "analysis", 0.8)
        ctx = DeepReasoningContext(query="test")
        critique = asyncio.run(dr._critique_step(step, "context", ctx))
        assert "has_issue" in critique

    def test_max_retrieval_default(self):
        dr = DeepReasoning()
        assert dr.max_retrieval == 5

    def test_fallback_retrieval_define(self):
        dr = DeepReasoning()
        results = asyncio.run(dr._fallback_retrieval("is defined as a concept"))
        assert len(results) > 0

    def test_build_context_with_constraints(self):
        dr = DeepReasoning()
        ctx = DeepReasoningContext(query="test", constraints=["c1", "c2"])
        context = asyncio.run(dr._build_context("Problem", [], ctx))
        assert "c1" in context
        assert "c2" in context

    def test_build_context_with_assumptions(self):
        dr = DeepReasoning()
        ctx = DeepReasoningContext(query="test", assumptions=["a1"])
        context = asyncio.run(dr._build_context("Problem", [], ctx))
        assert "a1" in context

    def test_build_context_with_knowledge_and_constraints(self):
        dr = DeepReasoning()
        know = RetrievedKnowledge(content="fact1", source=RetrievalSource.WORKING_MEMORY, relevance=0.9)
        ctx = DeepReasoningContext(query="test", constraints=["limit"])
        context = asyncio.run(dr._build_context("Problem", [know], ctx))
        assert "fact1" in context
        assert "limit" in context

    def test_reasoning_steps_populated(self):
        dr = DeepReasoning()
        result = asyncio.run(dr.reason("Problem"))
        assert len(result.steps) > 0

    def test_reason_metadata(self):
        dr = DeepReasoning()
        result = asyncio.run(dr.reason("Problem"))
        assert "retrieved_count" in result.metadata
        assert "depth" in result.metadata


# ---------------------------------------------------------------------------
# RetrievedKnowledge
# ---------------------------------------------------------------------------

class TestRetrievedKnowledge:
    def test_construction(self):
        rk = RetrievedKnowledge(
            content="fact", source=RetrievalSource.VECTOR_STORE,
            relevance=0.9, source_id="doc1",
        )
        assert rk.content == "fact"
        assert rk.source == RetrievalSource.VECTOR_STORE
        assert rk.relevance == 0.9

    def test_source_enum(self):
        assert RetrievalSource.VECTOR_STORE.value == "vector_store"
        assert RetrievalSource.MEMORY.value == "memory"

    def test_source_id_optional(self):
        rk = RetrievedKnowledge(content="x", source=RetrievalSource.MEMORY, relevance=0.5)
        assert rk.source_id is None

    def test_knowledge_graph_source(self):
        rk = RetrievedKnowledge(content="x", source=RetrievalSource.KNOWLEDGE_GRAPH, relevance=0.7)
        assert rk.source == RetrievalSource.KNOWLEDGE_GRAPH

    def test_working_memory_source(self):
        rk = RetrievedKnowledge(content="x", source=RetrievalSource.WORKING_MEMORY, relevance=0.6)
        assert rk.source == RetrievalSource.WORKING_MEMORY


# ---------------------------------------------------------------------------
# DeepReasoningContext
# ---------------------------------------------------------------------------

class TestDeepReasoningContext:
    def test_defaults(self):
        ctx = DeepReasoningContext(query="test")
        assert ctx.retrieved_knowledge == []
        assert ctx.working_memory == []
        assert ctx.constraints == []
        assert ctx.assumptions == []

    def test_with_lists(self):
        ctx = DeepReasoningContext(
            query="q", constraints=["c1"], assumptions=["a1"],
            working_memory=["w1"],
        )
        assert len(ctx.constraints) == 1
        assert len(ctx.assumptions) == 1

    def test_retrieved_knowledge_populated(self):
        rk = RetrievedKnowledge(content="x", source=RetrievalSource.MEMORY, relevance=0.5)
        ctx = DeepReasoningContext(query="q", retrieved_knowledge=[rk])
        assert len(ctx.retrieved_knowledge) == 1

    def test_working_memory_multiple(self):
        ctx = DeepReasoningContext(query="q", working_memory=["w1", "w2", "w3"])
        assert len(ctx.working_memory) == 3

    def test_constraints_multiple(self):
        ctx = DeepReasoningContext(query="q", constraints=["c1", "c2"])
        assert len(ctx.constraints) == 2

    def test_assumptions_multiple(self):
        ctx = DeepReasoningContext(query="q", assumptions=["a1", "a2", "a3"])
        assert len(ctx.assumptions) == 3

    def test_query_stored(self):
        ctx = DeepReasoningContext(query="my question")
        assert ctx.query == "my question"
