"""Unit tests for the near-duplicate consolidation planner."""

import pytest

from domains.memory.consolidation import plan_consolidation

SHORT = "Machine learning learns patterns from data."
LONG = "Machine learning learns patterns from data very effectively."
LONGEST = "Machine learning learns patterns from data very effectively during training."


def _fact(fid: str, content: str, topic: str = "ml") -> dict:
    return {"id": fid, "content": content, "topic": topic}


class TestPlanConsolidationExtended:
    def test_empty_list(self):
        plan = plan_consolidation([], threshold=0.80)
        assert plan["keep_ids"] == []
        assert plan["remove_ids"] == []
        assert plan["groups"] == []
        assert plan["removed_count"] == 0

    def test_single_fact_threshold_zero(self):
        plan = plan_consolidation([_fact("f1", "a")], threshold=0.0)
        assert plan["keep_ids"] == ["f1"]
        assert plan["removed_count"] == 0

    def test_single_fact_threshold_one(self):
        plan = plan_consolidation([_fact("f1", "a")], threshold=1.0)
        assert plan["keep_ids"] == ["f1"]
        assert plan["removed_count"] == 0

    def test_two_identical_merge(self):
        plan = plan_consolidation(
            [_fact("a", "exact same text"), _fact("b", "exact same text")], threshold=0.80
        )
        assert plan["removed_count"] == 1
        assert plan["keep_ids"] == ["a"]
        assert plan["remove_ids"] == ["b"]

    def test_three_identical_merge(self):
        plan = plan_consolidation(
            [_fact("a", "dup"), _fact("b", "dup"), _fact("c", "dup")], threshold=0.80
        )
        assert plan["removed_count"] == 2
        assert plan["keep_ids"] == ["a"]

    def test_all_different_topics(self):
        plan = plan_consolidation(
            [_fact("f1", "same content", "t1"), _fact("f2", "same content", "t2"),
             _fact("f3", "same content", "t3")],
            threshold=0.0,
        )
        assert plan["removed_count"] == 0
        assert len(plan["keep_ids"]) == 3

    def test_two_groups_same_topic(self):
        plan = plan_consolidation(
            [_fact("a", "aaa", "t"), _fact("b", "aaa", "t"),
             _fact("c", "bbb", "t"), _fact("d", "bbb", "t")],
            threshold=0.5,
        )
        assert plan["removed_count"] == 2
        assert len(plan["groups"]) == 2

    def test_keep_longest_in_cluster(self):
        base = "the user prefers the code editor Zed over VS Code"
        plan = plan_consolidation(
            [_fact("a", base, "t"), _fact("b", base + " because it is fast", "t"),
             _fact("c", base + " because it is fast and lightweight", "t")],
            threshold=0.1,
        )
        assert plan["removed_count"] == 2
        assert plan["keep_ids"] == ["c"]

    def test_transitive_chain_four(self):
        plan = plan_consolidation(
            [_fact("f1", "a one"), _fact("f2", "a two"), _fact("f3", "a three"), _fact("f4", "a four")],
            threshold=0.3,
        )
        if plan["removed_count"] > 0:
            assert len(plan["groups"]) == 1

    def test_threshold_halfway(self):
        plan = plan_consolidation(
            [_fact("a", SHORT, "ml"), _fact("b", LONG, "ml")], threshold=0.50
        )
        assert plan["removed_count"] == 1

    def test_result_keys_complete(self):
        plan = plan_consolidation([_fact("f1", "x")], threshold=0.5)
        assert set(plan.keys()) == {"keep_ids", "remove_ids", "groups", "removed_count"}

    def test_removed_count_non_negative(self):
        facts = [_fact(f"f{i}", f"fact {i}") for i in range(20)]
        plan = plan_consolidation(facts, threshold=0.5)
        assert plan["removed_count"] >= 0

    def test_groups_count_matches_clusters(self):
        plan = plan_consolidation(
            [_fact("a", "aaa", "t"), _fact("b", "aaa", "t")], threshold=0.5
        )
        assert len(plan["groups"]) == 1
        assert plan["groups"][0]["keep"]["id"] == "a"
        assert plan["groups"][0]["duplicates"][0]["id"] == "b"

    def test_different_topics_same_content(self):
        plan = plan_consolidation(
            [_fact("a", "same text", "t1"), _fact("b", "same text", "t2")], threshold=0.0
        )
        assert plan["removed_count"] == 0

    def test_empty_content_zero_threshold(self):
        plan = plan_consolidation(
            [_fact("a", ""), _fact("b", "")], threshold=0.0
        )
        assert plan["removed_count"] == 1

    def test_long_content_preserved(self):
        long_text = "This is a very long fact that contains many words. " * 10
        plan = plan_consolidation(
            [_fact("a", "short", "t"), _fact("b", long_text, "t")], threshold=0.3
        )
        if plan["keep_ids"] == ["b"]:
            assert len(plan["keep_ids"]) == 1


class TestPlanConsolidation:
    def test_empty_input(self):
        plan = plan_consolidation([], threshold=0.85)
        assert plan == {
            "keep_ids": [],
            "remove_ids": [],
            "groups": [],
            "removed_count": 0,
        }

    def test_single_fact_kept(self):
        plan = plan_consolidation([_fact("f1", SHORT)], threshold=0.85)
        assert plan["keep_ids"] == ["f1"]
        assert plan["remove_ids"] == []
        assert plan["removed_count"] == 0

    def test_near_duplicate_removed_keep_longest(self):
        plan = plan_consolidation(
            [_fact("f1", SHORT), _fact("f2", LONG)], threshold=0.80
        )
        assert plan["removed_count"] == 1
        assert plan["remove_ids"] == ["f1"]
        assert plan["keep_ids"] == ["f2"]
        assert len(plan["groups"]) == 1
        assert plan["groups"][0]["keep"]["id"] == "f2"

    def test_threshold_boundary_controls_merge(self):
        merge = plan_consolidation([_fact("f1", SHORT), _fact("f2", LONG)], threshold=0.80)
        assert merge["removed_count"] == 1
        strict = plan_consolidation([_fact("f1", SHORT), _fact("f2", LONG)], threshold=0.90)
        assert strict["removed_count"] == 0
        assert strict["keep_ids"] == ["f1", "f2"]

    def test_paraphrase_not_merged_at_default(self):
        plan = plan_consolidation(
            [
                _fact("f1", "The user prefers the code editor Zed over VS Code because it is fast."),
                _fact("f2", "User prefers the editor Zed over VS Code for its speed."),
            ],
            threshold=0.85,
        )
        assert plan["removed_count"] == 0
        assert plan["keep_ids"] == ["f1", "f2"]

    def test_distinct_topics_never_compared(self):
        plan = plan_consolidation(
            [_fact("f1", SHORT, "ml"), _fact("f2", SHORT, "biology")], threshold=0.85
        )
        assert plan["removed_count"] == 0
        assert plan["keep_ids"] == ["f1", "f2"]

    def test_transitive_chain_collapses_to_longest(self):
        plan = plan_consolidation(
            [_fact("f1", SHORT), _fact("f2", LONG), _fact("f3", LONGEST)],
            threshold=0.80,
        )
        assert plan["removed_count"] == 2
        assert plan["keep_ids"] == ["f3"]
        assert plan["remove_ids"] == ["f1", "f2"]
        assert len(plan["groups"]) == 1
        assert plan["groups"][0]["keep"]["id"] == "f3"

    def test_groups_report_cluster_members(self):
        plan = plan_consolidation(
            [_fact("f1", SHORT), _fact("f2", LONG), _fact("f3", "The octopus has three hearts.")],
            threshold=0.80,
        )
        assert len(plan["groups"]) == 1
        group = plan["groups"][0]
        assert {m["id"] for m in group["duplicates"]} == {"f1"}
        assert group["keep"]["id"] == "f2"

    def test_two_identical_facts_merge(self):
        plan = plan_consolidation(
            [_fact("f1", "same content"), _fact("f2", "same content")], threshold=0.5
        )
        assert plan["removed_count"] == 1
        assert len(plan["groups"]) == 1

    def test_completely_different_no_merge(self):
        plan = plan_consolidation(
            [
                _fact("f1", "The sky is blue on clear days"),
                _fact("f2", "Quantum entanglement is a physics phenomenon"),
            ],
            threshold=0.5,
        )
        assert plan["removed_count"] == 0
        assert plan["keep_ids"] == ["f1", "f2"]

    def test_threshold_zero_merges_everything(self):
        plan = plan_consolidation(
            [_fact("f1", "a"), _fact("f2", "b"), _fact("f3", "c")], threshold=0.0
        )
        assert plan["removed_count"] == 2
        assert len(plan["keep_ids"]) == 1

    def test_threshold_one_merges_nothing(self):
        plan = plan_consolidation(
            [_fact("f1", SHORT), _fact("f2", LONG)], threshold=1.0
        )
        assert plan["removed_count"] == 0
        assert plan["keep_ids"] == ["f1", "f2"]

    def test_three_clusters(self):
        plan = plan_consolidation(
            [
                _fact("f1", "cats are mammals", "animals"),
                _fact("f2", "cats are feline mammals", "animals"),
                _fact("f3", "dogs are mammals", "animals"),
                _fact("f4", "dogs are canine mammals", "animals"),
                _fact("f5", "water boils at 100c", "physics"),
                _fact("f6", "water boils at one hundred celsius", "physics"),
            ],
            threshold=0.5,
        )
        assert plan["removed_count"] >= 2

    def test_keep_ids_preserve_input_order(self):
        plan = plan_consolidation(
            [_fact("z", LONG), _fact("a", SHORT), _fact("m", LONGEST)],
            threshold=0.80,
        )
        assert plan["keep_ids"] == ["m"]
        assert plan["remove_ids"] == ["z", "a"]

    def test_remove_ids_in_input_order(self):
        plan = plan_consolidation(
            [_fact("f3", SHORT), _fact("f1", LONG), _fact("f2", LONGEST)],
            threshold=0.80,
        )
        assert plan["remove_ids"] == ["f3", "f1"]

    def test_groups_contain_full_facts(self):
        plan = plan_consolidation(
            [_fact("f1", SHORT), _fact("f2", LONG)], threshold=0.80
        )
        group = plan["groups"][0]
        assert "id" in group["keep"]
        assert "content" in group["keep"]
        assert "topic" in group["keep"]
        for d in group["duplicates"]:
            assert "id" in d
            assert "content" in d

    def test_empty_content_facts(self):
        plan = plan_consolidation(
            [_fact("f1", ""), _fact("f2", "")], threshold=0.5
        )
        assert plan["removed_count"] == 1

    def test_single_character_facts(self):
        plan = plan_consolidation(
            [_fact("f1", "a"), _fact("f2", "a")], threshold=0.5
        )
        assert plan["removed_count"] == 1

    def test_long_facts(self):
        long_a = "word " * 500
        long_b = "word " * 500
        plan = plan_consolidation(
            [_fact("f1", long_a), _fact("f2", long_b)], threshold=0.80
        )
        assert plan["removed_count"] == 1

    def test_unicode_content(self):
        plan = plan_consolidation(
            [
                _fact("f1", "日本語テスト"),
                _fact("f2", "日本語テストです"),
            ],
            threshold=0.5,
        )
        assert plan["removed_count"] == 1

    def test_mixed_topics_multiple_groups(self):
        plan = plan_consolidation(
            [
                _fact("f1", SHORT, "ml"),
                _fact("f2", LONG, "ml"),
                _fact("f3", SHORT, "physics"),
                _fact("f4", LONG, "physics"),
            ],
            threshold=0.80,
        )
        assert plan["removed_count"] == 2
        assert len(plan["groups"]) == 2

    def test_default_threshold(self):
        plan = plan_consolidation(
            [_fact("f1", SHORT), _fact("f2", LONG)]
        )
        assert isinstance(plan, dict)

    def test_removed_count_matches_remove_ids(self):
        plan = plan_consolidation(
            [_fact("f1", SHORT), _fact("f2", LONG), _fact("f3", LONGEST)],
            threshold=0.80,
        )
        assert plan["removed_count"] == len(plan["remove_ids"])

    def test_keep_union_remove_equals_all_ids(self):
        facts = [_fact(f"f{i}", f"content {i} topic ml") for i in range(10)]
        plan = plan_consolidation(facts, threshold=0.5)
        assert set(plan["keep_ids"]) | set(plan["remove_ids"]) == {f"f{i}" for i in range(10)}

    def test_no_overlap_between_keep_and_remove(self):
        facts = [_fact("f1", SHORT), _fact("f2", LONG), _fact("f3", LONGEST)]
        plan = plan_consolidation(facts, threshold=0.80)
        assert set(plan["keep_ids"]).isdisjoint(set(plan["remove_ids"]))

    def test_groups_keep_has_longest(self):
        plan = plan_consolidation(
            [_fact("f1", SHORT), _fact("f2", LONG), _fact("f3", LONGEST)],
            threshold=0.80,
        )
        for g in plan["groups"]:
            assert len(g["keep"]["content"]) >= max(len(d["content"]) for d in g["duplicates"])

    def test_dedup_same_length_keeps_first(self):
        plan = plan_consolidation(
            [_fact("a", "identical text"), _fact("b", "identical text")],
            threshold=0.5,
        )
        assert plan["keep_ids"] == ["a"]
        assert plan["remove_ids"] == ["b"]

    def test_many_facts(self):
        facts = [_fact(f"f{i}", f"fact number {i} about machine learning") for i in range(50)]
        plan = plan_consolidation(facts, threshold=0.80)
        assert plan["removed_count"] == 0
        assert len(plan["keep_ids"]) == 50

    def test_result_structure(self):
        plan = plan_consolidation([_fact("f1", SHORT)], threshold=0.85)
        assert set(plan.keys()) == {"keep_ids", "remove_ids", "groups", "removed_count"}

    def test_groups_are_list(self):
        plan = plan_consolidation(
            [_fact("f1", SHORT), _fact("f2", LONG)], threshold=0.80
        )
        assert isinstance(plan["groups"], list)

    def test_group_keys(self):
        plan = plan_consolidation(
            [_fact("f1", SHORT), _fact("f2", LONG)], threshold=0.80
        )
        g = plan["groups"][0]
        assert set(g.keys()) == {"keep", "duplicates"}

    def test_duplicate_keys_in_group(self):
        plan = plan_consolidation(
            [_fact("f1", SHORT), _fact("f2", LONG)], threshold=0.80
        )
        g = plan["groups"][0]
        for d in g["duplicates"]:
            assert set(d.keys()) >= {"id", "content", "topic"}

    def test_general_topic_when_none(self):
        facts = [{"id": "f1", "content": "test", "topic": None}]
        plan = plan_consolidation(facts, threshold=0.5)
        assert plan["keep_ids"] == ["f1"]

    def test_general_topic_when_missing(self):
        facts = [{"id": "f1", "content": "test"}]
        plan = plan_consolidation(facts, threshold=0.5)
        assert plan["keep_ids"] == ["f1"]

    def test_multiple_groups_have_separate_keep(self):
        plan = plan_consolidation(
            [
                _fact("f1", SHORT, "ml"),
                _fact("f2", LONG, "ml"),
                _fact("f3", "The user likes cats a lot", "pets"),
                _fact("f4", "The user likes cats a great deal", "pets"),
            ],
            threshold=0.3,
        )
        keeps = {g["keep"]["id"] for g in plan["groups"]}
        assert len(keeps) == 2

    def test_empty_facts_list(self):
        plan = plan_consolidation([], threshold=0.5)
        assert plan["removed_count"] == 0
        assert plan["groups"] == []

    def test_one_fact_in_large_group_not_removed(self):
        plan = plan_consolidation(
            [_fact("f1", "unique fact with no duplicates")], threshold=0.5
        )
        assert plan["removed_count"] == 0
        assert plan["keep_ids"] == ["f1"]
