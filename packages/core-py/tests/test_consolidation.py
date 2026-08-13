"""Unit tests for the near-duplicate consolidation planner."""

import pytest

from domains.memory.consolidation import plan_consolidation

SHORT = "Machine learning learns patterns from data."
LONG = "Machine learning learns patterns from data very effectively."
LONGEST = "Machine learning learns patterns from data very effectively during training."


def _fact(fid: str, content: str, topic: str = "ml") -> dict:
    return {"id": fid, "content": content, "topic": topic}


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
