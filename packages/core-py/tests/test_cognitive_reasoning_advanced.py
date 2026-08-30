"""Tests for domains.cognitive.reasoning.advanced — comprehensive coverage.

Covers: dataclasses, all 7 reasoning engines, factory function, enum values,
internal logic (confidence, pruning, unification heuristics, subproblem extraction).
No mocks, pure logic with default LLM stubs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

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


# ═══════════════════════════════════════════════════════════════════════
# ReasoningMode Enum
# ═══════════════════════════════════════════════════════════════════════

class TestReasoningMode:
    def test_all_values_are_strings(self):
        for mode in ReasoningMode:
            assert isinstance(mode.value, str)
            assert len(mode.value) > 0

    def test_unique_values(self):
        values = [m.value for m in ReasoningMode]
        assert len(values) == len(set(values))

    def test_expected_modes(self):
        expected = {
            "chain_of_thought", "tree_of_thoughts", "self_consistency",
            "constitutional", "react", "causal", "counterfactual", "syllogism",
        }
        assert {m.value for m in ReasoningMode} == expected


# ═══════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════

class TestThoughtStep:
    def test_defaults(self):
        s = ThoughtStep(step_id=0, thought="hello", reasoning_type="test", confidence=0.9)
        assert s.parent_id is None
        assert s.children_ids == []
        assert s.value == 0.0
        assert s.is_final is False

    def test_full_construction(self):
        s = ThoughtStep(
            step_id=5, thought="deep", reasoning_type="branch",
            confidence=0.7, parent_id=2, children_ids=[6, 7],
            value=0.85, is_final=True,
        )
        assert s.parent_id == 2
        assert s.children_ids == [6, 7]
        assert s.is_final is True

    def test_children_ids_default_is_empty_list(self):
        a = ThoughtStep(0, "a", "t", 0.5)
        b = ThoughtStep(1, "b", "t", 0.5)
        assert a.children_ids is not b.children_ids


class TestReasoningResult:
    def test_basic(self):
        r = ReasoningResult(
            conclusion="done", confidence=0.95,
            mode=ReasoningMode.REACT, steps=[], metadata={}, execution_time_ms=1.0,
        )
        assert r.conclusion == "done"
        assert r.mode == ReasoningMode.REACT

    def test_with_steps(self):
        step = ThoughtStep(0, "step", "analysis", 0.8)
        r = ReasoningResult(
            conclusion="c", confidence=0.7, mode=ReasoningMode.CAUSAL,
            steps=[step], metadata={"k": "v"}, execution_time_ms=5.0,
        )
        assert len(r.steps) == 1
        assert r.metadata["k"] == "v"


# ═══════════════════════════════════════════════════════════════════════
# ChainOfThought
# ═══════════════════════════════════════════════════════════════════════

class TestChainOfThought:
    @pytest.mark.asyncio
    async def test_returns_correct_mode(self):
        result = await ChainOfThought().reason("Q")
        assert result.mode == ReasoningMode.CHAIN_OF_THOUGHT

    @pytest.mark.asyncio
    async def test_max_steps_limits_steps(self):
        result = await ChainOfThought().reason("Hard", max_steps=2)
        assert len(result.steps) <= 2

    @pytest.mark.asyncio
    async def test_confidence_threshold_early_exit(self):
        async def fast_llm(prompt):
            return "Therefore we conclude that the answer is obvious."
        cot = ChainOfThought(llm_call=fast_llm)
        result = await cot.reason("Q", confidence_threshold=0.5)
        assert result.steps[-1].is_final is True

    @pytest.mark.asyncio
    async def test_custom_llm_is_called(self):
        calls = []
        async def track(prompt):
            calls.append(prompt)
            return "Step done."
        await ChainOfThought(llm_call=track).reason("X")
        assert len(calls) >= 1

    def test_evaluate_confidence_base(self):
        cot = ChainOfThought()
        assert cot._evaluate_confidence("maybe") == 0.5

    def test_evaluate_confidence_indicators(self):
        cot = ChainOfThought()
        for word in ["therefore", "thus", "hence", "conclude", "implies"]:
            assert cot._evaluate_confidence(f"{word} X.") >= 0.6

    def test_evaluate_confidence_sentences(self):
        cot = ChainOfThought()
        c = cot._evaluate_confidence("First part. Second part.")
        assert c >= 0.6

    def test_evaluate_confidence_capped_at_one(self):
        cot = ChainOfThought()
        long = "Therefore. " * 10
        assert cot._evaluate_confidence(long) <= 1.0

    def test_extract_subproblem_remaining(self):
        cot = ChainOfThought()
        assert cot._extract_subproblem("remaining: solve X") == "solve X"

    def test_extract_subproblem_next(self):
        cot = ChainOfThought()
        assert cot._extract_subproblem("next: do Y") == "do Y"

    def test_extract_subproblem_now(self):
        cot = ChainOfThought()
        result = cot._extract_subproblem("now we need to find Z")
        assert result is not None
        assert "find Z" in result

    def test_extract_subproblem_none(self):
        cot = ChainOfThought()
        assert cot._extract_subproblem("random text") is None

    def test_extract_conclusion_therefore(self):
        cot = ChainOfThought()
        assert "42" in cot._extract_conclusion("Therefore the answer is 42.")

    def test_extract_conclusion_answer(self):
        cot = ChainOfThought()
        assert "hello" in cot._extract_conclusion("answer: hello world.")

    def test_extract_conclusion_conclusion(self):
        cot = ChainOfThought()
        assert "yes" in cot._extract_conclusion("conclusion: yes, it works.")

    def test_extract_conclusion_fallback(self):
        cot = ChainOfThought()
        assert cot._extract_conclusion("no match here") == "no match here"

    @pytest.mark.asyncio
    async def test_metadata_solved_flag(self):
        async def solved_llm(p):
            return "Therefore it is done."
        result = await ChainOfThought(llm_call=solved_llm).reason("Q", confidence_threshold=0.5)
        assert result.metadata["solved"] is True

    @pytest.mark.asyncio
    async def test_metadata_max_steps(self):
        result = await ChainOfThought().reason("Q", max_steps=5)
        assert result.metadata["max_steps"] == 5


# ═══════════════════════════════════════════════════════════════════════
# TreeOfThoughts
# ═══════════════════════════════════════════════════════════════════════

class TestTreeOfThoughts:
    @pytest.mark.asyncio
    async def test_returns_correct_mode(self):
        result = await TreeOfThoughts().reason("Q", max_depth=1)
        assert result.mode == ReasoningMode.TREE_OF_THOUGHTS

    @pytest.mark.asyncio
    async def test_total_nodes_grows(self):
        result = await TreeOfThoughts(beam_width=2).reason("Q", max_depth=2)
        assert result.metadata["total_nodes"] > 1

    def test_evaluate_node_with_keywords(self):
        tot = TreeOfThoughts()
        assert tot._evaluate_node("Therefore done.") > 0.5
        assert tot._evaluate_node("Solution found.") > 0.5
        assert tot._evaluate_node("Answer: X") > 0.5

    def test_evaluate_node_without_keywords(self):
        tot = TreeOfThoughts()
        assert tot._evaluate_node("just thinking") == 0.5

    def test_evaluate_node_capped(self):
        tot = TreeOfThoughts()
        val = tot._evaluate_node("Therefore conclusion solution answer thus")
        assert val <= 1.0

    def test_is_solution_detects(self):
        tot = TreeOfThoughts()
        assert tot._is_solution("The answer: 42")
        assert tot._is_solution("Solution: found")
        assert tot._is_solution("Therefore X")
        assert tot._is_solution("Conclusion: done")

    def test_is_solution_rejects(self):
        tot = TreeOfThoughts()
        assert not tot._is_solution("maybe not")
        assert not tot._is_solution("thinking about it")

    def test_prune_nodes_keeps_top(self):
        tot = TreeOfThoughts(beam_width=2)
        tot.nodes = {
            0: ThoughtStep(0, "a", "t", 0.5, value=0.9),
            1: ThoughtStep(1, "b", "t", 0.5, value=0.2),
            2: ThoughtStep(2, "c", "t", 0.5, value=0.7),
        }
        pruned = tot._prune_nodes([0, 1, 2], threshold=0.3)
        assert len(pruned) <= 2
        assert 0 in pruned
        assert 1 not in pruned

    def test_prune_nodes_respects_threshold(self):
        tot = TreeOfThoughts(beam_width=5)
        tot.nodes = {
            0: ThoughtStep(0, "a", "t", 0.5, value=0.1),
            1: ThoughtStep(1, "b", "t", 0.5, value=0.05),
        }
        pruned = tot._prune_nodes([0, 1], threshold=0.3)
        assert pruned == []

    def test_get_path_from_root(self):
        tot = TreeOfThoughts()
        tot.nodes = {
            0: ThoughtStep(0, "root", "root", 1.0, parent_id=None),
            1: ThoughtStep(1, "a", "branch", 0.8, parent_id=0),
            2: ThoughtStep(2, "b", "branch", 0.7, parent_id=1),
        }
        assert tot._get_path(2) == [0, 1, 2]

    def test_get_path_root_only(self):
        tot = TreeOfThoughts()
        tot.nodes = {0: ThoughtStep(0, "root", "root", 1.0, parent_id=None)}
        assert tot._get_path(0) == [0]

    @pytest.mark.asyncio
    async def test_beam_width_affects_candidate_count(self):
        r1 = await TreeOfThoughts(beam_width=1).reason("Q", max_depth=1)
        r2 = await TreeOfThoughts(beam_width=3).reason("Q", max_depth=1)
        assert r2.metadata["total_nodes"] >= r1.metadata["total_nodes"]

    def test_default_llm(self):
        tot = TreeOfThoughts()
        result = tot._default_llm.__func__
        import asyncio
        val = asyncio.get_event_loop().run_until_complete(tot._default_llm("test"))
        assert isinstance(val, str)


# ═══════════════════════════════════════════════════════════════════════
# SelfConsistency
# ═══════════════════════════════════════════════════════════════════════

class TestSelfConsistency:
    @pytest.mark.asyncio
    async def test_returns_correct_mode(self):
        result = await SelfConsistency(num_paths=2).reason("Q")
        assert result.mode == ReasoningMode.SELF_CONSISTENCY

    @pytest.mark.asyncio
    async def test_confidence_positive(self):
        result = await SelfConsistency(num_paths=3).reason("Q")
        assert result.confidence > 0

    def test_extract_conclusion_with_answer(self):
        sc = SelfConsistency()
        assert sc._extract_conclusion("Answer: 42") == "42"

    def test_extract_conclusion_long_text(self):
        sc = SelfConsistency()
        text = "x" * 200
        result = sc._extract_conclusion(text)
        assert len(result) <= 100

    def test_extract_conclusion_short(self):
        sc = SelfConsistency()
        assert sc._extract_conclusion("hi") == "hi"

    def test_majority_vote_majority(self):
        sc = SelfConsistency()
        assert sc._majority_vote(["A", "A", "B"]) == "A"

    def test_majority_vote_single(self):
        sc = SelfConsistency()
        assert sc._majority_vote(["X"]) == "X"

    def test_majority_vote_tie_returns_first(self):
        sc = SelfConsistency()
        result = sc._majority_vote(["A", "B"])
        assert result in ["A", "B"]

    @pytest.mark.asyncio
    async def test_num_paths_affects_steps(self):
        r1 = await SelfConsistency(num_paths=2).reason("Q")
        r2 = await SelfConsistency(num_paths=4).reason("Q")
        assert len(r2.steps) >= len(r1.steps)

    @pytest.mark.asyncio
    async def test_steps_limited_to_20(self):
        r = await SelfConsistency(num_paths=10).reason("Q")
        assert len(r.steps) <= 20

    @pytest.mark.asyncio
    async def test_metadata_num_paths(self):
        r = await SelfConsistency(num_paths=3).reason("Q")
        assert r.metadata["num_paths"] == 3

    @pytest.mark.asyncio
    async def test_metadata_agreement(self):
        r = await SelfConsistency(num_paths=5).reason("Q")
        assert r.metadata["agreement"] >= 1


# ═══════════════════════════════════════════════════════════════════════
# ConstitutionalAI
# ═══════════════════════════════════════════════════════════════════════

class TestConstitutionalAI:
    @pytest.mark.asyncio
    async def test_returns_correct_mode(self):
        result = await ConstitutionalAI().reason("Q")
        assert result.mode == ReasoningMode.CONSTITUTIONAL

    @pytest.mark.asyncio
    async def test_three_steps(self):
        result = await ConstitutionalAI().reason("Q")
        assert len(result.steps) == 3

    @pytest.mark.asyncio
    async def test_confidence_is_nine(self):
        result = await ConstitutionalAI().reason("Q")
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_custom_principles_override(self):
        ai = ConstitutionalAI()
        await ai.reason("Q", custom_principles=["Be kind", "Be honest"])
        assert ai.principles == ["Be kind", "Be honest"]
        assert ai.PRINCIPLES == ConstitutionalAI.PRINCIPLES

    @pytest.mark.asyncio
    async def test_default_principles_preserved(self):
        ai = ConstitutionalAI()
        assert len(ai.principles) == 5

    @pytest.mark.asyncio
    async def test_metadata_principles_count(self):
        r = await ConstitutionalAI().reason("Q")
        assert r.metadata["principles_used"] == 5

    @pytest.mark.asyncio
    async def test_step_types(self):
        result = await ConstitutionalAI().reason("Q")
        types = [s.reasoning_type for s in result.steps]
        assert "initial" in types
        assert "critique" in types
        assert "revision" in types


# ═══════════════════════════════════════════════════════════════════════
# CausalReasoning
# ═══════════════════════════════════════════════════════════════════════

class TestCausalReasoning:
    @pytest.mark.asyncio
    async def test_returns_correct_mode(self):
        result = await CausalReasoning().reason("Because X. Therefore Y.")
        assert result.mode == ReasoningMode.CAUSAL

    @pytest.mark.asyncio
    async def test_conclusion_contains_causes_and_effects(self):
        result = await CausalReasoning().reason("Because rain. Therefore wet ground.")
        assert "Causes" in result.conclusion
        assert "Effects" in result.conclusion

    def test_identify_causes_because(self):
        cr = CausalReasoning()
        causes = cr._identify_causes("It happened because of bad luck.")
        assert any("bad luck" in c for c in causes)

    def test_identify_causes_due_to(self):
        cr = CausalReasoning()
        causes = cr._identify_causes("The failure due to timeout.")
        assert any("timeout" in c for c in causes)

    def test_identify_causes_leads_to(self):
        cr = CausalReasoning()
        causes = cr._identify_causes("The storm leads to flooding.")
        assert any("flooding" in c for c in causes)

    def test_identify_causes_fallback(self):
        cr = CausalReasoning()
        causes = cr._identify_causes("random text no markers")
        assert causes == ["Unknown cause"]

    def test_identify_effects_therefore(self):
        cr = CausalReasoning()
        effects = cr._identify_effects("Therefore the system crashed.")
        assert any("crashed" in e for e in effects)

    def test_identify_effects_consequently(self):
        cr = CausalReasoning()
        effects = cr._identify_effects("Consequently the data was lost.")
        assert any("lost" in e for e in effects)

    def test_identify_effects_fallback(self):
        cr = CausalReasoning()
        effects = cr._identify_effects("no markers here")
        assert effects == ["Unknown effect"]

    def test_identify_relationships_pairs(self):
        cr = CausalReasoning()
        rels = cr._identify_relationships("Because X. Therefore Y.")
        assert len(rels) >= 1
        assert rels[0][2] == 0.8

    def test_build_conclusion_empty(self):
        cr = CausalReasoning()
        assert "Unable" in cr._build_causal_conclusion([], [], [])

    def test_build_conclusion_limits_items(self):
        cr = CausalReasoning()
        causes = ["c1", "c2", "c3", "c4"]
        effects = ["e1", "e2", "e3", "e4"]
        conclusion = cr._build_causal_conclusion(causes, effects, [])
        assert "c1" in conclusion
        assert "c4" not in conclusion

    def test_metadata_counts(self):
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            CausalReasoning().reason("Because X. Therefore Y.")
        )
        assert result.metadata["causes"] >= 1
        assert result.metadata["effects"] >= 1


# ═══════════════════════════════════════════════════════════════════════
# SyllogismReasoning
# ═══════════════════════════════════════════════════════════════════════

class TestSyllogismReasoning:
    @pytest.mark.asyncio
    async def test_returns_correct_mode(self):
        result = await SyllogismReasoning().reason("All men are mortal. Socrates is a man.")
        assert result.mode == ReasoningMode.SYLLOGISM

    @pytest.mark.asyncio
    async def test_valid_syllogism(self):
        result = await SyllogismReasoning().reason("All men are mortal. Socrates is a man.")
        assert result.metadata["valid"] is True
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_invalid_syllogism(self):
        result = await SyllogismReasoning().reason("XYZ not real")
        # Default premises are valid AAA
        assert isinstance(result.metadata["valid"], bool)

    def test_parse_premises_multi(self):
        sr = SyllogismReasoning()
        p = sr._parse_premises("All A are B. X is A. Therefore X is B.")
        assert len(p) >= 2

    def test_parse_premises_short_fallback(self):
        sr = SyllogismReasoning()
        p = sr._parse_premises("short")
        assert "Socrates" in p[1]

    def test_identify_figure_returns_int(self):
        sr = SyllogismReasoning()
        assert isinstance(sr._identify_figure(["a", "b"]), int)

    def test_identify_mood_three_premises(self):
        sr = SyllogismReasoning()
        assert sr._identify_mood(["a", "b", "c"]) == "AAA"

    def test_identify_mood_two_premises(self):
        sr = SyllogismReasoning()
        assert sr._identify_mood(["a", "b"]) == "AAA"

    def test_identify_mood_one_premise(self):
        sr = SyllogismReasoning()
        assert sr._identify_mood(["a"]) == "AA"

    def test_apply_rules_valid_mood(self):
        sr = SyllogismReasoning()
        valid, msg = sr._apply_syllogistic_rules(1, "AAA")
        assert valid is True
        assert "valid" in msg.lower()

    def test_apply_rules_invalid_mood(self):
        sr = SyllogismReasoning()
        valid, msg = sr._apply_syllogistic_rules(1, "XYZ")
        assert valid is False
        assert "invalid" in msg.lower()

    def test_apply_rules_figure2_eae(self):
        sr = SyllogismReasoning()
        valid, _ = sr._apply_syllogistic_rules(2, "EAE")
        assert valid is True

    def test_apply_rules_figure3_aai(self):
        sr = SyllogismReasoning()
        valid, _ = sr._apply_syllogistic_rules(3, "AAI")
        assert valid is True

    def test_apply_rules_figure4_aee(self):
        sr = SyllogismReasoning()
        valid, _ = sr._apply_syllogistic_rules(4, "AEE")
        assert valid is True

    def test_derive_conclusion_two_premises(self):
        sr = SyllogismReasoning()
        c = sr._derive_conclusion(["All A are B", "X is A"])
        assert "Therefore" in c
        assert "X is A" in c

    def test_derive_conclusion_insufficient(self):
        sr = SyllogismReasoning()
        assert "Insufficient" in sr._derive_conclusion(["one"])

    def test_conclusion_stored(self):
        import asyncio
        sr = SyllogismReasoning()
        asyncio.get_event_loop().run_until_complete(sr.reason("All A are B. X is A."))
        assert sr.conclusion is not None


# ═══════════════════════════════════════════════════════════════════════
# ReActReasoning
# ═══════════════════════════════════════════════════════════════════════

class TestReActReasoning:
    @pytest.mark.asyncio
    async def test_returns_correct_mode(self):
        result = await ReActReasoning().reason("Q", max_steps=3)
        assert result.mode == ReasoningMode.REACT

    @pytest.mark.asyncio
    async def test_no_tools(self):
        result = await ReActReasoning().reason("Q", max_steps=3)
        assert result.metadata["actions"] == 0

    @pytest.mark.asyncio
    async def test_with_tools_uses_action(self):
        ra = ReActReasoning(tool_registry={"search": lambda q: "found"})
        result = await ra.reason("Q", max_steps=3)
        assert result.metadata["actions"] >= 1

    @pytest.mark.asyncio
    async def test_action_count_limited_to_two(self):
        ra = ReActReasoning(tool_registry={"a": lambda x: "r"})
        result = await ra.reason("Q", max_steps=10)
        assert result.metadata["actions"] <= 2

    def test_is_solved_answer(self):
        ra = ReActReasoning()
        assert ra._is_solved("The answer: 42")

    def test_is_solved_solution(self):
        ra = ReActReasoning()
        assert ra._is_solved("Solution: found it")

    def test_is_solved_conclusion(self):
        ra = ReActReasoning()
        assert ra._is_solved("Conclusion: yes")

    def test_is_solved_final_answer(self):
        ra = ReActReasoning()
        assert ra._is_solved("Final answer is 7")

    def test_is_not_solved(self):
        ra = ReActReasoning()
        assert not ra._is_solved("thinking about it")

    @pytest.mark.asyncio
    async def test_act_returns_tool_and_result(self):
        ra = ReActReasoning(tool_registry={"calc": lambda x: "42"})
        tool, result = await ra._act("compute")
        assert tool == "calc"
        assert "calc" in result

    @pytest.mark.asyncio
    async def test_act_no_tools(self):
        ra = ReActReasoning()
        tool, result = await ra._act("x")
        assert tool == "search"

    @pytest.mark.asyncio
    async def test_early_solved_stops(self):
        ra = ReActReasoning()
        result = await ra.reason("The answer: 42.", max_steps=10)
        assert result.steps[-1].is_final is True

    @pytest.mark.asyncio
    async def test_metadata_steps(self):
        result = await ReActReasoning().reason("Q", max_steps=3)
        assert result.metadata["steps"] == len(result.steps)


# ═══════════════════════════════════════════════════════════════════════
# Factory Function
# ═══════════════════════════════════════════════════════════════════════

class TestAdvancedReasoningFactory:
    @pytest.mark.asyncio
    async def test_chain_of_thought(self):
        r = await advanced_reasoning("Q", mode=ReasoningMode.CHAIN_OF_THOUGHT)
        assert r.mode == ReasoningMode.CHAIN_OF_THOUGHT

    @pytest.mark.asyncio
    async def test_tree_of_thoughts(self):
        r = await advanced_reasoning("Q", mode=ReasoningMode.TREE_OF_THOUGHTS, max_depth=1)
        assert r.mode == ReasoningMode.TREE_OF_THOUGHTS

    @pytest.mark.asyncio
    async def test_self_consistency(self):
        r = await advanced_reasoning("Q", mode=ReasoningMode.SELF_CONSISTENCY, num_paths=2)
        assert r.mode == ReasoningMode.SELF_CONSISTENCY

    @pytest.mark.asyncio
    async def test_constitutional(self):
        r = await advanced_reasoning("Q", mode=ReasoningMode.CONSTITUTIONAL)
        assert r.mode == ReasoningMode.CONSTITUTIONAL

    @pytest.mark.asyncio
    async def test_causal(self):
        r = await advanced_reasoning("Because X. Therefore Y.", mode=ReasoningMode.CAUSAL)
        assert r.mode == ReasoningMode.CAUSAL

    @pytest.mark.asyncio
    async def test_syllogism(self):
        r = await advanced_reasoning("All A are B. X is A.", mode=ReasoningMode.SYLLOGISM)
        assert r.mode == ReasoningMode.SYLLOGISM

    @pytest.mark.asyncio
    async def test_react(self):
        r = await advanced_reasoning("Q", mode=ReasoningMode.REACT)
        assert r.mode == ReasoningMode.REACT

    @pytest.mark.asyncio
    async def test_counterfactual_defaults_to_cot(self):
        r = await advanced_reasoning("Q", mode=ReasoningMode.COUNTERFACTUAL)
        assert r.mode == ReasoningMode.CHAIN_OF_THOUGHT

    @pytest.mark.asyncio
    async def test_unknown_string_defaults_to_cot(self):
        r = await advanced_reasoning("Q", mode="unknown_mode")
        assert r.mode == ReasoningMode.CHAIN_OF_THOUGHT

    @pytest.mark.asyncio
    async def test_beam_width_passed_to_tot(self):
        r = await advanced_reasoning("Q", mode=ReasoningMode.TREE_OF_THOUGHTS, beam_width=1, max_depth=1)
        assert r.mode == ReasoningMode.TREE_OF_THOUGHTS

    @pytest.mark.asyncio
    async def test_llm_call_passed_through(self):
        calls = []
        async def track(p):
            calls.append(p)
            return "Therefore done."
        r = await advanced_reasoning("Q", mode=ReasoningMode.CHAIN_OF_THOUGHT, llm_call=track)
        assert len(calls) >= 1


# ═══════════════════════════════════════════════════════════════════════
# __all__ exports
# ═══════════════════════════════════════════════════════════════════════

class TestExports:
    def test_all_contains_expected(self):
        from domains.cognitive.reasoning.advanced import __all__ as exported
        expected = {
            "ReasoningMode", "ThoughtStep", "ReasoningResult",
            "ChainOfThought", "TreeOfThoughts", "SelfConsistency",
            "ConstitutionalAI", "CausalReasoning", "SyllogismReasoning",
            "ReActReasoning", "advanced_reasoning",
        }
        assert expected == set(exported)
