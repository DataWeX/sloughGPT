"""Tests for advanced reasoning module."""

from __future__ import annotations

import pytest
from domains.cognitive.reasoning.advanced import (
    ReasoningMode,
    ThoughtStep,
    ReasoningResult,
    ChainOfThought,
    TreeOfThoughts,
    SelfConsistency,
    ConstitutionalAI,
    CausalReasoning,
    SyllogismReasoning,
    ReActReasoning,
    advanced_reasoning,
)


class TestReasoningMode:
    def test_values(self):
        assert ReasoningMode.CHAIN_OF_THOUGHT.value == "chain_of_thought"
        assert ReasoningMode.TREE_OF_THOUGHTS.value == "tree_of_thoughts"
        assert ReasoningMode.SELF_CONSISTENCY.value == "self_consistency"
        assert ReasoningMode.CONSTITUTIONAL.value == "constitutional"
        assert ReasoningMode.REACT.value == "react"
        assert ReasoningMode.CAUSAL.value == "causal"
        assert ReasoningMode.COUNTERFACTUAL.value == "counterfactual"
        assert ReasoningMode.SYLLOGISM.value == "syllogism"


class TestThoughtStep:
    def test_minimal(self):
        s = ThoughtStep(step_id=0, thought="test", reasoning_type="reasoning", confidence=0.8)
        assert s.step_id == 0
        assert s.thought == "test"
        assert s.parent_id is None
        assert s.children_ids == []
        assert s.is_final is False

    def test_full(self):
        s = ThoughtStep(
            step_id=1, thought="step", reasoning_type="decomposition",
            confidence=0.9, parent_id=0, children_ids=[2, 3], value=0.7, is_final=True,
        )
        assert s.parent_id == 0
        assert s.children_ids == [2, 3]
        assert s.value == 0.7
        assert s.is_final is True


class TestReasoningResult:
    def test_fields(self):
        step = ThoughtStep(0, "thought", "reasoning", 0.8)
        r = ReasoningResult(
            conclusion="done", confidence=0.9, mode=ReasoningMode.CHAIN_OF_THOUGHT,
            steps=[step], metadata={"key": "val"}, execution_time_ms=10.5,
        )
        assert r.conclusion == "done"
        assert r.confidence == 0.9
        assert len(r.steps) == 1
        assert r.metadata == {"key": "val"}


class TestChainOfThought:
    async def test_reason_returns_result(self):
        cot = ChainOfThought()
        result = await cot.reason("solve 2+2", max_steps=3)
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.CHAIN_OF_THOUGHT
        assert len(result.steps) > 0

    async def test_reason_adds_steps(self):
        cot = ChainOfThought()
        result = await cot.reason("solve x+5=10", max_steps=5)
        assert len(cot.steps) > 0
        assert len(result.steps) == len(cot.steps)

    async def test_confidence_with_indicators(self):
        cot = ChainOfThought()
        high = cot._evaluate_confidence("therefore the answer is 5")
        low = cot._evaluate_confidence("maybe")
        assert high > low

    async def test_extract_subproblem(self):
        cot = ChainOfThought()
        sub = cot._extract_subproblem("remaining: solve for x")
        assert sub == "solve for x"
        assert cot._extract_subproblem("done.") is None

    async def test_extract_conclusion(self):
        cot = ChainOfThought()
        assert "42" in cot._extract_conclusion("therefore 42")
        assert cot._extract_conclusion("answer: 7") == "7"

    async def test_default_reasoning_completes(self):
        cot = ChainOfThought()
        result = await cot.reason("test problem")
        assert isinstance(result.conclusion, str)
        assert len(result.conclusion) > 0


class TestTreeOfThoughts:
    async def test_reason_returns_result(self):
        tot = TreeOfThoughts(beam_width=2)
        result = await tot.reason("solve problem", max_depth=2)
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.TREE_OF_THOUGHTS
        assert len(result.steps) > 0

    async def test_generates_multiple_nodes(self):
        tot = TreeOfThoughts(beam_width=3)
        result = await tot.reason("test", max_depth=2)
        assert len(tot.nodes) > 1

    async def test_evaluate_node(self):
        tot = TreeOfThoughts()
        v1 = tot._evaluate_node("therefore the answer is 5")
        v2 = tot._evaluate_node("maybe not sure")
        assert v1 > v2

    async def test_is_solution(self):
        tot = TreeOfThoughts()
        assert tot._is_solution("answer: 42")
        assert tot._is_solution("solution: done")
        assert not tot._is_solution("thinking about it")

    async def test_prune_nodes(self):
        tot = TreeOfThoughts()
        tot.nodes = {0: ThoughtStep(0, "a", "root", 1.0, value=0.9)}
        tot.nodes[1] = ThoughtStep(1, "b", "branch", 0.8, value=0.2)
        tot.nodes[2] = ThoughtStep(2, "c", "branch", 0.8, value=0.8)
        tot.nodes[3] = ThoughtStep(3, "d", "branch", 0.8, value=0.4)
        tot.beam_width = 2
        pruned = tot._prune_nodes([1, 2, 3], 0.3)
        assert 2 in pruned
        assert len(pruned) <= 2

    async def test_get_path(self):
        tot = TreeOfThoughts()
        tot.nodes = {0: ThoughtStep(0, "root", "root", 1.0)}
        tot.nodes[1] = ThoughtStep(1, "child", "branch", 0.8, parent_id=0)
        tot.nodes[2] = ThoughtStep(2, "grandchild", "branch", 0.8, parent_id=1)
        path = tot._get_path(2)
        assert path == [0, 1, 2]


class TestSelfConsistency:
    async def test_reason_returns_result(self):
        sc = SelfConsistency(num_paths=3)
        result = await sc.reason("test problem")
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.SELF_CONSISTENCY
        assert result.confidence > 0

    async def test_generates_multiple_paths(self):
        sc = SelfConsistency(num_paths=2)
        await sc.reason("test")
        assert len(sc.reasoning_paths) == 2

    async def test_majority_vote(self):
        sc = SelfConsistency()
        winner = sc._majority_vote(["a", "b", "a"])
        assert winner == "a"

    async def test_extract_conclusion(self):
        sc = SelfConsistency()
        assert sc._extract_conclusion("the answer: 42") == "42"
        extracted = sc._extract_conclusion("some long text here")
        assert len(extracted) <= 100


class TestConstitutionalAI:
    async def test_reason_returns_result(self):
        cai = ConstitutionalAI()
        result = await cai.reason("solve problem")
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.CONSTITUTIONAL
        assert len(result.steps) == 3

    async def test_custom_principles(self):
        cai = ConstitutionalAI()
        custom = ["Be concise.", "Be accurate."]
        result = await cai.reason("problem", custom_principles=custom)
        assert result.metadata["principles_used"] == 2

    async def test_default_principles(self):
        cai = ConstitutionalAI()
        assert len(cai.PRINCIPLES) >= 3

    async def test_generate_initial(self):
        cai = ConstitutionalAI()
        resp = await cai._generate_initial("test")
        assert isinstance(resp, str)


class TestCausalReasoning:
    async def test_reason_returns_result(self):
        cr = CausalReasoning()
        result = await cr.reason("X causes Y because of Z")
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.CAUSAL
        assert len(result.steps) > 0

    async def test_identify_causes(self):
        cr = CausalReasoning()
        causes = cr._identify_causes("This happened because of rain due to weather")
        assert len(causes) >= 1

    async def test_identify_effects(self):
        cr = CausalReasoning()
        effects = cr._identify_effects("therefore the result was good")
        assert len(effects) >= 1

    async def test_identify_relationships(self):
        cr = CausalReasoning()
        rels = cr._identify_relationships("A because of B, therefore C")
        assert len(rels) >= 1

    async def test_build_causal_conclusion(self):
        cr = CausalReasoning()
        conclusion = cr._build_causal_conclusion(["rain"], ["flood"], [("rain", "flood", 0.8)])
        assert "rain" in conclusion
        assert "flood" in conclusion

    async def test_unknown_causes_returns_fallback(self):
        cr = CausalReasoning()
        causes = cr._identify_causes("no causal words here")
        assert causes == ["Unknown cause"]


class TestSyllogismReasoning:
    async def test_reason_valid(self):
        sr = SyllogismReasoning()
        result = await sr.reason("All humans are mortal. Socrates is human.")
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.SYLLOGISM
        assert result.conclusion is not None

    async def test_parse_premises(self):
        sr = SyllogismReasoning()
        premises = sr._parse_premises("All A are B. All B are C. Some D is E.")
        assert len(premises) >= 2

    async def test_identify_figure(self):
        sr = SyllogismReasoning()
        assert sr._identify_figure(["p1", "p2"]) == 1

    async def test_identify_mood(self):
        sr = SyllogismReasoning()
        assert sr._identify_mood(["p1", "p2"]) == "AAA"

    async def test_apply_syllogistic_rules_valid(self):
        sr = SyllogismReasoning()
        valid, explanation = sr._apply_syllogistic_rules(1, "AAA")
        assert valid is True

    async def test_apply_syllogistic_rules_invalid(self):
        sr = SyllogismReasoning()
        valid, explanation = sr._apply_syllogistic_rules(1, "XYZ")
        assert valid is False

    async def test_derive_conclusion_two_premises(self):
        sr = SyllogismReasoning()
        c = sr._derive_conclusion(["All A are B", "All B are C"])
        assert "B" in c

    async def test_empty_premises_falls_back(self):
        sr = SyllogismReasoning()
        premises = sr._parse_premises("")
        assert len(premises) > 0


class TestReActReasoning:
    async def test_reason_returns_result(self):
        rr = ReActReasoning()
        result = await rr.reason("solve problem", max_steps=3)
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.REACT

    async def test_reason_with_tools(self):
        def dummy_tool(query: str) -> str:
            return "result from tool"
        rr = ReActReasoning(tool_registry={"search": dummy_tool})
        result = await rr.reason("find answer", max_steps=3)
        assert result.metadata["actions"] > 0

    async def test_is_solved(self):
        rr = ReActReasoning()
        assert rr._is_solved("the answer: 42")
        assert rr._is_solved("final answer is 10")
        assert not rr._is_solved("thinking about it")

    async def test_act_returns_result(self):
        rr = ReActReasoning(tool_registry={"calc": lambda x: "42"})
        tool, result = await rr._act("what is 2+2")
        assert tool == "calc"
        assert "Result" in result


class TestAdvancedReasoning:
    async def test_chain_of_thought_mode(self):
        result = await advanced_reasoning("test", ReasoningMode.CHAIN_OF_THOUGHT)
        assert result.mode == ReasoningMode.CHAIN_OF_THOUGHT

    async def test_tree_of_thoughts_mode(self):
        result = await advanced_reasoning("test", ReasoningMode.TREE_OF_THOUGHTS, beam_width=2)
        assert result.mode == ReasoningMode.TREE_OF_THOUGHTS

    async def test_self_consistency_mode(self):
        result = await advanced_reasoning("test", ReasoningMode.SELF_CONSISTENCY, num_paths=2)
        assert result.mode == ReasoningMode.SELF_CONSISTENCY

    async def test_constitutional_mode(self):
        result = await advanced_reasoning("test", ReasoningMode.CONSTITUTIONAL)
        assert result.mode == ReasoningMode.CONSTITUTIONAL

    async def test_causal_mode(self):
        result = await advanced_reasoning("X causes Y", ReasoningMode.CAUSAL)
        assert result.mode == ReasoningMode.CAUSAL

    async def test_syllogism_mode(self):
        result = await advanced_reasoning("All A are B. All B are C.", ReasoningMode.SYLLOGISM)
        assert result.mode == ReasoningMode.SYLLOGISM

    async def test_react_mode(self):
        result = await advanced_reasoning("solve", ReasoningMode.REACT)
        assert result.mode == ReasoningMode.REACT

    async def test_unknown_mode_falls_back_to_cot(self):
        result = await advanced_reasoning("test", "unknown_mode")
        assert result.mode == ReasoningMode.CHAIN_OF_THOUGHT
